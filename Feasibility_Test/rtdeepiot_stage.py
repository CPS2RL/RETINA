# rtdeepiot_stage.py
import json
import math
import time
from pathlib import Path
from typing import Dict,Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from config import (RTDEEPIOT_BUDGET_MS,RTDEEPIOT_CHECKPOINT_PATH,RTDEEPIOT_CONFIDENCE_EMA_ALPHA,RTDEEPIOT_DEFAULT_SEED_CONFIDENCE,
                    RTDEEPIOT_MAX_DEPTH,RTDEEPIOT_REWARD_QUANTIZATION_DELTA,RTDEEPIOT_UTILITY_HEURISTIC,RTDEEPIOT_WCET_PATH)

from rtdeepiot.models.multi_exit_resnet import build_model
from rtdeepiot.scheduler.dp_scheduler import (predict_task_rewards,load_wcet_profile)

from rtdeepiot.data.road_behavior_dataset import (CLASS_NAMES as ROAD_CLASS_NAMES,NORM_MEAN,NORM_STD)


DEVICE="cpu"

#ROAD-behavior preprocessing
CLASSIFY_TRANSFORM=T.Compose([
    T.ToPILImage(),
    T.Resize((32,32)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN,NORM_STD)
])


# Per-track confidence memory
actor_confidence_ema:Dict[int,float]={}

def reset_confidence_ema(): actor_confidence_ema.clear()

# WCET loading
def _load_wcet(path):
    path=Path(path)

    try:
        with open(path,"r") as f:
            data=json.load(f)

        if isinstance(data,dict):
            keys=[
                "stage_1_wcet_ms",
                "stage_2_wcet_ms",
                "stage_3_wcet_ms"
            ]

            if all(k in data for k in keys):
                return [
                    float(data[keys[0]]),
                    float(data[keys[1]]),
                    float(data[keys[2]])
                ]

    except Exception:
        pass

    return [
        float(x)
        for x in load_wcet_profile(path)
    ]


# model loading
def load_rtdeepiot_runtime(checkpoint_path=RTDEEPIOT_CHECKPOINT_PATH,wcet_path=RTDEEPIOT_WCET_PATH):
    model=build_model("road_behavior").to(DEVICE).eval()
    checkpoint=torch.load(checkpoint_path,map_location=DEVICE)
    if isinstance(checkpoint,dict) and "model_state" in checkpoint:
        state=checkpoint["model_state"]
    else:
        state=checkpoint
    model.load_state_dict(state,strict=True)
    wcet=_load_wcet(wcet_path)
    wcet=wcet[:3]

    return model,DEVICE,wcet


# Frame loading
def read_frame(image_path):
    frame=cv2.imread(str(image_path))
    return frame


# One classifier execution
@torch.no_grad()
def classify_crop(model,device,frame_bgr,box,exit_after):
    if frame_bgr is None:
        return None
    exit_after=int(max(1,min(RTDEEPIOT_MAX_DEPTH,int(exit_after))))
    h,w=frame_bgr.shape[:2]
    x1,y1,x2,y2=map(int,box)
    x1=max(0,min(x1,w-1))
    y1=max(0,min(y1,h-1))
    x2=max(0,min(x2,w))
    y2=max(0,min(y2,h))
    if x2<=x1 or y2<=y1:
        return None
    crop=frame_bgr[y1:y2,x1:x2]
    if crop.size==0:
        return None
    total_start=time.perf_counter()
    tensor=CLASSIFY_TRANSFORM(
        cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    model_start=time.perf_counter()
    outputs=model(tensor,exit_after=exit_after)
    model_ms=(time.perf_counter()-model_start)*1000.0
    total_ms=(time.perf_counter()-total_start)*1000.0
    final=outputs[-1]
    class_id=int(final["predicted"].item())
    if not (0<=class_id<len(ROAD_CLASS_NAMES)):
        return None
    return {
        "class_id":class_id,
        "label":ROAD_CLASS_NAMES[class_id],
        "confidence":float(final["confidence"].item()),
        "stage1_confidence":float(outputs[0]["confidence"].item()),
        "exit_after":exit_after,
        "runtime_ms":model_ms,
        "total_runtime_ms":total_ms
    }


# Job deadline
def job_deadline_ms(obj,wcet):
    ttc_ms=float(obj.get("ttc",999.0))*1000.0
    return max(float(wcet[0]),min(ttc_ms,float(RTDEEPIOT_BUDGET_MS)))


# Sort by TTC: smallest TTC first.
# Each admitted job reserves P1.
def _admit_stage1(objects,wcet):
    p1=float(wcet[0])
    budget=float(RTDEEPIOT_BUDGET_MS)

    ordered=sorted(
        objects,
        key=lambda o:float(o.get("ttc",999.0)))
    admitted=[]
    delayed=[]
    used=0.0

    for obj in ordered:
        finish=used+p1
        deadline=job_deadline_ms(obj,wcet)
        # Must fit shared cycle budget AND its effective deadline.
        if finish<=budget and finish<=deadline:
            admitted.append(int(obj["track_id"]))
            used=finish
        else:
            delayed.append(int(obj["track_id"]))
    return admitted,delayed,used


# Quantized reward
def _reward(v):
    delta=float(RTDEEPIOT_REWARD_QUANTIZATION_DELTA)
    if delta<=0:
        return float(v)
    return round(float(v)/delta)*delta


# Shared-budget depth DP
# Mandatory P1 is already reserved/executed.

def _choose_depths(admitted_ids,stage1_results,wcet,remaining_budget):
    if not admitted_ids:
        return {}
    p2=float(wcet[1])
    p3=float(wcet[2])
    budget=max(0,int(math.floor(remaining_budget)))
    # state:
    # used_ms -> (total_reward, assignment)
    states={0:(0.0,{})}

    for tid in admitted_ids:
        s1=stage1_results.get(tid,{})
        confidence=float(s1.get("stage1_confidence",RTDEEPIOT_DEFAULT_SEED_CONFIDENCE))
        rewards=predict_task_rewards(wcet,seed_confidence=confidence,heuristic=RTDEEPIOT_UTILITY_HEURISTIC)
        rewards=[_reward(x) for x in rewards]
        options=[(1,0,rewards[0]),
                 (2,int(math.ceil(p2)),rewards[1]),
                 (3,int(math.ceil(p3)),rewards[2])]
        new_states={}
        for used,(score,assignment) in states.items():
            for depth,cost,reward in options:
                new_used=used+cost
                if new_used>budget:
                    continue
                new_score=score+reward
                old=new_states.get(new_used)
                if (old is None or new_score>old[0]):
                    new_assignment=dict(assignment)
                    new_assignment[tid]=depth
                    new_states[new_used]=(new_score,new_assignment)
        states=new_states
    if not states:
        return {tid:1 for tid in admitted_ids}
    # Maximum reward.
    # If equal reward, prefer lower compute usage.
    best_used,best=sorted(
        states.items(),
        key=lambda x:(-x[1][0],x[0]))[0]

    return best[1]

# Two-phase hardware scheduler
def run_rtdeepiot_schedule(model,device,wcet,frame,objects):
    schedule_start=time.perf_counter()

    if not objects:
        return {
            "initial_assignment":{},
            "final_assignment":{},
            "admitted_ids":set(),
            "delayed_ids":set(),
            "stage1_results":{},
            "stage1_runtime_ms":0.0,
            "predicted_cost_ms":0.0,
            "scheduler_runtime_ms":0.0}

    # Admission based on mandatory P1.
    admitted,delayed,reserved_stage1_ms=_admit_stage1(objects,wcet)
    admitted_set=set(admitted)
    delayed_set=set(delayed)
    # Initial assignment is depth1 because P1 has been reserved.
    initial_assignment={tid:1for tid in admitted}

    # Run mandatory stage1 ONLY for admitted jobs.

    object_map={int(o["track_id"]):o for o in objects}
    stage1_results={}
    stage1_total_runtime=0.0
    for tid in admitted:
        obj=object_map[tid]
        result=classify_crop(model,device,frame,obj["box"],1)
        if result is None:
            # Failed crop/inference.
            delayed_set.add(tid)
            admitted_set.discard(tid)
            continue
        stage1_total_runtime+=float(result["runtime_ms"])
        result["completion_ms"]=(time.perf_counter()-schedule_start)*1000.0
        stage1_results[tid]=result
        conf=float(result["stage1_confidence"])
        previous=actor_confidence_ema.get(tid,conf)
        actor_confidence_ema[tid]=(RTDEEPIOT_CONFIDENCE_EMA_ALPHA*conf+(1.0-RTDEEPIOT_CONFIDENCE_EMA_ALPHA)*previous)

    admitted=[tid for tid in admitted if tid in admitted_set]
    # Remaining predicted budget after mandatory stage1.
    reserved_stage1_ms=(len(admitted)*float(wcet[0]))
    remaining=max(0.0,float(RTDEEPIOT_BUDGET_MS)-reserved_stage1_ms)
    # Choose final depths under remaining shared budget.
    final_assignment=_choose_depths(admitted,stage1_results,wcet,remaining)
    # Predicted total cost for diagnostics.
    predicted_cost=reserved_stage1_ms
    for tid,depth in final_assignment.items():
        if depth==2:
            predicted_cost+=float(wcet[1])
        elif depth==3:
            predicted_cost+=float(wcet[2])
    total_ms=(time.perf_counter()-schedule_start)*1000.0
    return {
        "initial_assignment":initial_assignment,
        "final_assignment":final_assignment,
        "admitted_ids":set(admitted),
        "delayed_ids":delayed_set,
        "stage1_results":stage1_results,
        "stage1_runtime_ms":stage1_total_runtime,
        "predicted_cost_ms":predicted_cost,
        "scheduler_runtime_ms":total_ms}