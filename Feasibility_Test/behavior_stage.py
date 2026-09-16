# behavior_stage.py
import re,time
from pathlib import Path
from collections import deque

import cv2
import torch
import torch.nn as nn
from torchvision.models.shufflenetv2 import ShuffleNetV2

from config import IMAGE_DIR,NUM_FRAMES,BEHAVIOR_CLASSES


def _clean(state): 
    return {k.removeprefix("module.").removeprefix("_orig_mod."):v for k,v in state.items()}

def _load(path):
    ckpt=torch.load(path,map_location="cpu",weights_only=True)
    return _clean(ckpt.get("model_state_dict",ckpt))

def _last_linear(state,prefix):
    ids=[int(m.group(1)) for k,v in state.items() 
         if (m:=re.match(rf"^{re.escape(prefix)}\.(\d+)\.weight$",k)) and isinstance(v,torch.Tensor) and v.ndim==2]
    return max(ids) if ids else None

def _repeats(state):
    out=[]
    for s in (2,3,4):
        ids=[int(m.group(1)) for k in state if (m:=re.match(rf"^backbone\.{s}\.(\d+)\.",k))]
        if not ids: raise RuntimeError(f"Cannot infer backbone stage {s}")
        out.append(max(ids)+1)
    return out

def _channels(state):
    keys=["backbone.0.0.weight","backbone.2.0.branch1.2.weight","backbone.3.0.branch1.2.weight",
          "backbone.4.0.branch1.2.weight","backbone.5.0.weight"]
    return [
        int(state[keys[0]].shape[0]),
        2*int(state[keys[1]].shape[0]),
        2*int(state[keys[2]].shape[0]),
        2*int(state[keys[3]].shape[0]),
        int(state[keys[4]].shape[0])
    ]

def _head(inp,hidden,out,final_index,bias=True,dropout=0.5):
    layers=[nn.Linear(inp,hidden),nn.ReLU(inplace=True)]
    if final_index==3: layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden,out,bias=bias))
    return nn.Sequential(*layers)


class TemporalAttention(nn.Module):
    def __init__(self,hidden,attn_hidden,final_index,bias):
        super().__init__()
        self.attn=_head(hidden,attn_hidden,1,final_index,bias)

    def forward(self,x):
        w=torch.softmax(self.attn(x),dim=1)
        return (w*x).sum(dim=1)


class BehaviorModel(nn.Module):
    def __init__(self,spec):
        super().__init__()

        backbone=ShuffleNetV2(
            stages_repeats=spec["stage_repeats"],
            stages_out_channels=spec["stage_channels"],
            num_classes=1000
        )

        self.hidden=spec["hidden"]
        self.fusion_in=spec["fusion_in"]
        self.backbone=nn.Sequential(*list(backbone.children())[:-1])
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.projection=nn.Sequential(nn.Linear(spec["feature"],spec["hidden"]),nn.ReLU(inplace=True))
        self.gru=nn.GRU(spec["hidden"],spec["hidden"],batch_first=True)
        self.temporal_attn=TemporalAttention(spec["hidden"],spec["attn_hidden"],spec["attn_final_index"],spec["attn_bias"])
        self.fusion=nn.Sequential(nn.Linear(spec["fusion_in"],spec["fusion_out"]),nn.ReLU(inplace=True))
        self.behavior_head=_head(spec["fusion_out"],spec["behavior_hidden"],spec["num_classes"],
                                 spec["behavior_final_index"],spec["behavior_bias"])

    def forward(self,x):
        B,T,C,H,W=x.shape
        x=self.backbone(x.reshape(B*T,C,H,W))
        x=self.pool(x).flatten(1)
        x=self.projection(x).reshape(B,T,self.hidden)
        gru,_=self.gru(x)
        attn=self.temporal_attn(gru)

        if self.fusion_in==2*self.hidden: x=torch.cat((attn,gru[:,-1,:]),dim=1)
        elif self.fusion_in==self.hidden: x=attn
        else: raise RuntimeError(f"Unsupported fusion input: {self.fusion_in}")

        return self.behavior_head(self.fusion(x))


def infer_model_spec(state):
    channels=_channels(state)
    p=state["projection.0.weight"]
    a=state["temporal_attn.attn.0.weight"]
    f=state["fusion.0.weight"]
    b0=state["behavior_head.0.weight"]
    ai=_last_linear(state,"temporal_attn.attn")
    bi=_last_linear(state,"behavior_head")

    if ai is None or bi is None: raise RuntimeError("Could not infer attention/behavior heads")

    spec={
        "stage_repeats":_repeats(state),
        "stage_channels":channels,
        "feature":int(p.shape[1]),
        "hidden":int(p.shape[0]),
        "attn_hidden":int(a.shape[0]),
        "attn_final_index":ai,
        "attn_bias":f"temporal_attn.attn.{ai}.bias" in state,
        "fusion_in":int(f.shape[1]),
        "fusion_out":int(f.shape[0]),
        "behavior_hidden":int(b0.shape[0]),
        "behavior_final_index":bi,
        "behavior_bias":f"behavior_head.{bi}.bias" in state,
        "num_classes":int(state[f"behavior_head.{bi}.weight"].shape[0])
    }

    return spec


def load_behavior_model(model_path,int8=False):
    path=Path(model_path)
    state=_load(path)
    spec=infer_model_spec(state)

    model=BehaviorModel(spec)

    model.load_state_dict(state,strict=True)
    model.cpu().eval()

    if int8:
        model=torch.ao.quantization.quantize_dynamic(model,{nn.Linear,nn.GRU},dtype=torch.qint8,inplace=False)
        model.cpu().eval()

    return model


def _crop(frame,box,size=90):
    h,w=frame.shape[:2]
    size=min(size,h,w)
    x1,y1,x2,y2=map(int,box)
    cx=(x1+x2)//2
    cy=(y1+y2)//2
    x=max(0,min(cx-size//2,w-size))
    y=max(0,min(cy-size//2,h-size))
    return frame[y:y+size,x:x+size]

def _tensor(crop):
    crop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(crop).permute(2,0,1).float()/255.0


class BehaviorBuffer:
    def __init__(self,n=NUM_FRAMES): 
        self.n=n
        self.buffers={}

    def ready(self,track_id): 
        return int(track_id) in self.buffers and len(self.buffers[int(track_id)])==self.n

    def remove_missing_tracks(self,valid):
        valid={int(x) for x in valid}
        for tid in list(self.buffers):
            if tid not in valid: 
                self.buffers.pop(tid,None)

    def update_from_history(self,image_path,track_id,box):
        paths=sorted(IMAGE_DIR.glob("*.jpg"))
        current=Path(image_path)
        index=paths.index(current)

        paths=paths[max(0,index-self.n+1):index+1]
        if len(paths)<self.n: 
            return False

        seq=[]
        for path in paths:
            frame=cv2.imread(str(path))
            if frame is None: 
                return False
            crop=_crop(frame,box)
            if crop.size==0: 
                return False
            seq.append(_tensor(crop))

        self.buffers[int(track_id)]=deque(seq,maxlen=self.n)
        return True

    @torch.inference_mode()
    def infer(self,model,track_id):
        tid=int(track_id)
        if not self.ready(tid): 
            return None

        x=torch.stack(list(self.buffers[tid])).unsqueeze(0).cpu()
        start=time.perf_counter()
        logits=model(x)
        runtime=(time.perf_counter()-start)*1000.0
        probs=torch.softmax(logits,dim=1)[0]
        idx=int(probs.argmax())

        return {
            "track_id":tid,
            "behavior":BEHAVIOR_CLASSES[idx],
            "behavior_confidence":float(probs[idx]),
            "runtime_ms":runtime,
            "sequence_length":self.n
        }