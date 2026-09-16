# camot_stage.py
import time
from dataclasses import dataclass,field

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models,transforms

from config import (CAMOT_ASSOC_MAX_FEAT,CAMOT_DET_SIZE,CAMOT_SCHEDULER,CAMOT_TASK_DEADLINE_MS,CAMOT_TASK_PERIOD_MS,
                    CAMOT_TTC_CRITICAL_SEC,CAMOT_WCET_ASSOC,CAMOT_WCET_DET,TARGET_LABELS,YOLO_CONF,YOLO_IOU,YOLO_MAX_DET)
from ttc_stage import estimate_ttc_from_box

DEVICE= "cpu"


# Appearance feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        net=models.mobilenet_v2(weights=None)
        self.features=net.features
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.proj=nn.Linear(1280,128)

    def forward(self,x):
        return self.proj(self.pool(self.features(x)).flatten(1))

FEATURE_TRANSFORM=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# Track state
@dataclass
class TrackedObject:
    track_id:int
    agent_type:str
    pos:np.ndarray=field(default_factory=lambda:np.zeros(2))
    vel:np.ndarray=field(default_factory=lambda:np.zeros(2))
    box:tuple|None=None
    feature:torch.Tensor|None=None
    feature_age:int=0
    pos_age:int=0
    miss_count:int=0
    ttc:float=999.0
    initial_ttc:float=999.0

# Scheduler task
@dataclass
class MOTTask:
    track_id:int
    period:float=CAMOT_TASK_PERIOD_MS
    deadline:float=CAMOT_TASK_DEADLINE_MS
    age_det:int=0
    age_assoc:int=0
    last_s:str="L"
    last_f:str="L"

    def record(self,s,f):
        if s in ("M","H"): 
            self.age_det+=1
        if f in ("M","H"): 
            self.age_assoc+=1
        self.last_s=s
        self.last_f=f


# Helpers
def iou(a,b):
    ax1,ay1,ax2,ay2=a
    bx1,by1,bx2,by2=b

    x1=max(ax1,bx1)
    y1=max(ay1,by1)
    x2=min(ax2,bx2)
    y2=min(ay2,by2)
    if x2<=x1 or y2<=y1:
        return 0.0
    inter=(x2-x1)*(y2-y1)
    area_a=(ax2-ax1)*(ay2-ay1)
    area_b=(bx2-bx1)*(by2-by1)
    return inter/(area_a+area_b-inter+1e-6)


def feature_similarity(a,b):
    a=a/(a.norm()+1e-6)
    b=b/(b.norm()+1e-6)
    return float(torch.dot(a,b).clamp(0,1))


@torch.no_grad()
def extract_feature(extractor,frame,box):
    h,w=frame.shape[:2]
    x1,y1,x2,y2=map(int,box)

    x1=max(0,x1)
    y1=max(0,y1)
    x2=min(w,x2)
    y2=min(h,y2)

    crop=frame[y1:y2,x1:x2]
    if crop.size==0:
        return None
    x=FEATURE_TRANSFORM(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)

    return extractor(x).squeeze(0).cpu()


# Detection
def run_detection(model,frame,level,region=None):
    fh,fw=frame.shape[:2]
    size=CAMOT_DET_SIZE[level]
    inp=cv2.resize(frame,(size,size))
    sx=fw/size
    sy=fh/size
    start=time.perf_counter()
    result=model(inp,imgsz=size,conf=YOLO_CONF,iou=YOLO_IOU,agnostic_nms=True,max_det=YOLO_MAX_DET,verbose=False)[0]
    runtime=(time.perf_counter()-start)*1000.0
    out=[]
    if result.boxes is not None:
        for b in result.boxes:
            label=model.names[int(b.cls[0])]
            if label not in TARGET_LABELS:
                continue
            x1,y1,x2,y2=map(float,b.xyxy[0])
            out.append({
                "label":label,
                "confidence":float(b.conf[0]),
                "box":(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy))})
    return out,runtime

# Critical region
def identify_critical_region(tracks):
    boxes=[t.box for t in tracks if (t.box is not None and t.ttc<CAMOT_TTC_CRITICAL_SEC)]
    if not boxes:
        return None
    return (min(x[0] for x in boxes),min(x[1] for x in boxes),max(x[2] for x in boxes),max(x[3] for x in boxes))

# EDF-Slack
def slack_edf_slack(task,all_tasks,t_cur):
    c=(CAMOT_WCET_DET["L"]+CAMOT_WCET_ASSOC["L"])
    if not all_tasks:
        return 0.0
    d1=task.deadline
    U=(c/min(t.period for t in all_tasks)+sum(c/t.period for t in all_tasks))
    p=0.0
    for t in sorted(all_tasks,key=lambda x:x.deadline,reverse=True):
        if t.deadline<=d1:
            continue
        U-=c/t.period
        max_exec=(1.0-U)*(t.deadline-d1)
        q=max(0.0,c-max_exec)
        den=t.deadline-d1
        if den>1e-6:
            U=min(1.0,U+(c-q)/den)
        p+=q
    return max(0.0,d1-t_cur-p)


def slack_edf_be(task,active,t_cur):
    if len(active)!=1:
        return 0.0
    c=(CAMOT_WCET_DET["L"]+CAMOT_WCET_ASSOC["L"])
    return max(0.0,task.deadline-t_cur-c)


# CA-MOT fidelity selection
def determine_options(task,slack):
    if slack<=0:
        return "L","L"
    def pick_det(v):
        if v>=(CAMOT_WCET_DET["H"]-CAMOT_WCET_DET["L"]):
            return "H"
        if v>=(CAMOT_WCET_DET["M"]-CAMOT_WCET_DET["L"]):
            return "M"
        return "L"

    def pick_assoc(v):
        if v>=(CAMOT_WCET_ASSOC["H"]-CAMOT_WCET_ASSOC["L"]):
            return "H"
        if v>=(CAMOT_WCET_ASSOC["M"]-CAMOT_WCET_ASSOC["L"]):
            return "M"
        return "L"
    if task.age_det<=task.age_assoc:
        rem=slack-(CAMOT_WCET_DET["H"]-CAMOT_WCET_DET["L"])
        if rem>=0:
            return "H",pick_assoc(rem)
        return pick_det(slack),"L"
    rem=slack-(CAMOT_WCET_ASSOC["H"]-CAMOT_WCET_ASSOC["L"])
    if rem>=0:
        return pick_det(rem),"H"
    return "L",pick_assoc(slack)

# Find lower D/A combination when original choice doesn't fit
def fitting_camot_option(task,planned_s,planned_f,current):
    rank={"L":0,"M":1,"H":2}
    candidates=[]
    for s in ("L","M","H"):
        if rank[s]>rank[planned_s]:
            continue
        for f in ("L","M","H"):
            if rank[f]>rank[planned_f]:
                continue
            cost=(CAMOT_WCET_DET[s]+CAMOT_WCET_ASSOC[f])
            if current+cost>task.deadline:
                continue
            if task.age_det<=task.age_assoc:
                priority=(rank[s]+rank[f],rank[s],rank[f])
            else:
                priority=(rank[s]+rank[f],rank[f],rank[s])
            candidates.append((priority,s,f,cost))
    if not candidates:
        return None
    _,s,f,cost=max(candidates,key=lambda x:x[0])
    return s,f,cost


# Scheduler
def camot_schedule(tasks,mode=CAMOT_SCHEDULER):
    edf=sorted(tasks,key=lambda x:x.deadline)
    out=[]
    current=0.0
    minimum_cost=(CAMOT_WCET_DET["L"]+CAMOT_WCET_ASSOC["L"])

    for i,t in enumerate(edf):
        remaining_tasks=edf[i:]
        # Even L/L cannot fit
        if current+minimum_cost>t.deadline:
            continue
        #EDF-Slack.
        if mode=="EDF-BE":
            slack=slack_edf_be(t,remaining_tasks,current)
        else:
            slack=slack_edf_slack(t,remaining_tasks,current)

        planned_s,planned_f=determine_options(t,slack)

        planned_cost=(CAMOT_WCET_DET[planned_s]+CAMOT_WCET_ASSOC[planned_f])

        # Selected pair fits
        if current+planned_cost<=t.deadline:
            s=planned_s
            f=planned_f
            cost=planned_cost

        # Selected pair doesn't fit : downgrade
        else:
            chosen=fitting_camot_option(t,planned_s,planned_f,current)
            if chosen is None:
                continue
            s,f,cost=chosen
        out.append((t,s,f))
        current+=cost
    return out


# Runtime
class CAMOTRuntime:

    def __init__(self):
        self.extractor=FeatureExtractor().to(DEVICE).eval()

        self.tracks=[]
        self.tasks={}
        self.next_id=1


    # Initial
    def _bootstrap(self,detections):
        for d in detections:
            x1,y1,x2,y2=d["box"]
            pos=np.array([(x1+x2)/2.0,(y1+y2)/2.0],dtype=float)

            ttc=estimate_ttc_from_box(d["box"])

            self.tracks.append(
                TrackedObject(
                    track_id=self.next_id,
                    agent_type=d["label"],
                    pos=pos,
                    box=d["box"],
                    ttc=ttc,
                    initial_ttc=ttc))
            self.next_id+=1


    # Predict existing tracks before scheduled detection
    def _predict_tracks(self,dt):
        dt=max(float(dt),1e-3)
        for t in self.tracks:
            shift=t.vel*dt
            t.pos=t.pos+shift
            if t.box is not None:
                x1,y1,x2,y2=t.box
                dx,dy=shift
                t.box=(int(x1+dx),int(y1+dy),int(x2+dx),int(y2+dy))

            if t.ttc<999.0:
                t.ttc=max(0.0,t.ttc-dt)

            t.pos_age+=1
            t.feature_age+=1
            t.miss_count+=1

        self.tracks=[t for t in self.tracks if t.miss_count<=10]

    # Association
    def _match(self,frame,track,detections,level):

        candidates=[d for d in detections if d["label"]==track.agent_type]
        if not candidates:
            return None,None

        max_feat=CAMOT_ASSOC_MAX_FEAT[level]

        # Appearance association
        if (max_feat>0 and track.feature is not None):
            best=None
            for d in candidates[:max_feat]:
                feature=extract_feature(self.extractor,frame,d["box"])
                if feature is None:
                    continue
                score=feature_similarity(feature,track.feature)
                if (best is None or score>best[0]):
                    best=(score,d,feature)
            if (best is not None and best[0]>=0.30):
                return best[1],best[2]

        # IoU fallback
        d=max(candidates,key=lambda x:iou(x["box"],track.box))
        if iou(d["box"],track.box)<0.30:
            return None,None

        feature=(extract_feature(self.extractor,frame,d["box"])
            if max_feat>0
            else None)

        return d,feature


    # One complete CA-MOT cycle
    def step(self,frame,yolo_model,dt):
        total_start=time.perf_counter()
        bootstrap_runtime=0.0
        if not self.tracks:
            detections,bootstrap_runtime=run_detection(yolo_model,frame,"L")
            self._bootstrap(detections)

        else:
            # No detector
            self._predict_tracks(dt)

        # Maintain scheduler tasks
        active={t.track_id for t in self.tracks}

        for tid in list(self.tasks):
            if tid not in active:
                self.tasks.pop(tid,None)
        for t in self.tracks:
            if t.track_id not in self.tasks:
                self.tasks[t.track_id]=MOTTask(t.track_id)
        tasks=[self.tasks[t.track_id] for t in self.tracks]
        # Scheduler
        # Budget begins at ZERO
        scheduler_start=time.perf_counter()
        schedule=camot_schedule(tasks)
        sched_ms=(time.perf_counter()-scheduler_start)*1000.0
        track_map={t.track_id:t for t in self.tracks}
        decisions={}
        completion={}
        executed_schedule=[]
        det_ms=0.0
        assoc_ms=0.0
        # Computation budget covers selected D+A operations
        compute_start=time.perf_counter()
        # Execute selected jobs
        for task,s,f in schedule:
            track=track_map.get(task.track_id)
            if track is None:
                continue
            # Previous actual jobs may have taken longer
            actual_used=(time.perf_counter()-compute_start)*1000.0
            required=(CAMOT_WCET_DET[s]+CAMOT_WCET_ASSOC[f])
            # Planned choice no longer fits actual remaining time
            if actual_used+required>task.deadline:
                chosen=fitting_camot_option(task,s,f,actual_used)
                if chosen is None:
                    continue
                s,f,required=chosen
            # Final budget check.
            actual_used=(time.perf_counter()-compute_start)*1000.0
            if actual_used+required>task.deadline:
                continue
            # Selected detector:
            dets,runtime=run_detection(yolo_model,frame,s)
            det_ms+=runtime
            # Selected association:
            assoc_start=time.perf_counter()
            d,feature=self._match(frame,track,dets,f)
            assoc_runtime=(time.perf_counter()-assoc_start)*1000.0
            assoc_ms+=assoc_runtime
            # Detection/association succeeded
            if d is not None:
                x1,y1,x2,y2=d["box"]
                new_pos=np.array([(x1+x2)/2.0,(y1+y2)/2.0],dtype=float)
                if dt>1e-6:
                    track.vel=(new_pos-track.pos)/dt
                track.pos=new_pos
                track.box=d["box"]
                track.ttc=estimate_ttc_from_box(track.box)
                track.pos_age=0
                track.miss_count=0
                if feature is not None:
                    track.feature=feature
                    track.feature_age=0
                elif f=="L":
                    track.feature_age+=1
            task.record(s,f)
            decisions[task.track_id]=(s,f)
            completion[task.track_id]=(time.perf_counter()-compute_start)*1000.0
            executed_schedule.append((task,s,f))
        # Export track information
        h,w=frame.shape[:2]
        executed_ids={task.track_id for task,_,_ in executed_schedule}
        objects=[]
        for t in self.tracks:
            if t.box is None:
                continue
            if t.track_id in decisions:
                s,f=decisions[t.track_id]
            else:
                s=""
                f=""
            task=self.tasks[t.track_id]
            completion_ms=completion.get(t.track_id,"")
            objects.append({
                "track_id":t.track_id,
                "job_uid":f"track_{t.track_id:04d}",
                "label":t.agent_type,
                "box":list(t.box),
                "ttc":round(t.ttc,3),
                "initial_ttc":round(t.initial_ttc,3),
                "track_source":("DETECTED" if t.miss_count==0 else "PREDICTED"),
                "miss_count":t.miss_count,
                "image_width":w,
                "image_height":h,
                "detection_level":s,
                "association_level":f,
                "feature_age":t.feature_age,
                "position_age":t.pos_age,
                "age_det":task.age_det,
                "age_assoc":task.age_assoc,
                "job_status":("PROCESSED" if t.track_id in executed_ids else "DELAYED"),
                "completion_ms":(round(completion_ms,3) if completion_ms!="" else ""),
                "deadline_met":(completion_ms<=task.deadline if completion_ms!="" else "")
            })

        compute_ms=(time.perf_counter()-compute_start)*1000.0

        total_ms=(time.perf_counter()-total_start)*1000.0
        return {
            "objects":sorted(objects,key=lambda x:x["ttc"]),
            "jobs_considered":len(tasks),
            "jobs_admitted":len(schedule),
            "jobs_processed":len(executed_schedule),
            "jobs_delayed":len(tasks)-len(executed_schedule),
            "scheduler_runtime_ms":sched_ms,
            "detection_runtime_ms":det_ms,
            "association_runtime_ms":assoc_ms,
            "bootstrap_runtime_ms":bootstrap_runtime,
            "compute_runtime_ms":compute_ms,
            "total_runtime_ms":total_ms
        }