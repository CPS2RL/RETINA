# tracking_stage.py

from dataclasses import dataclass
from config import TRACK_IOU_THRESHOLD,TRACK_MAX_MISSES

# measure the overlap between two box
def iou(a,b):
    # Top and left coordinate
    ax1,ay1,ax2,ay2=map(float,a)
    # Right and bottom coordinate
    bx1,by1,bx2,by2=map(float,b)
    x1,y1,x2,y2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    # check overlapping
    if x2<=x1 or y2<=y1: 
        return 0.0
    inter=(x2-x1)*(y2-y1)
    aa=(ax2-ax1)*(ay2-ay1)
    ab=(bx2-bx1)*(by2-by1)
    return inter/max(aa+ab-inter,1e-6)

# Store information for each obejct
@dataclass
class Track:
    track_id:int
    label:str
    box:list
    miss_count:int=0

# Track obejcts across the frames
class HardwareTracker:
    def __init__(self,threshold=TRACK_IOU_THRESHOLD,max_misses=TRACK_MAX_MISSES):
        self.threshold=threshold
        self.max_misses=max_misses
        self.next_id=1
        self.tracks={}

    # all current objetcs
    def active_track_ids(self): 
        return set(self.tracks)
    # Update tracking
    def update(self,detections):
        unmatched_tracks=set(self.tracks)
        unmatched_dets=set(range(len(detections)))
        matches={}
        candidates=[]
        #Loop through each detection and compare with tracke dobjects
        for di,obj in enumerate(detections):
            for tid,trk in self.tracks.items():
                if obj["label"]!=trk.label: 
                    continue
                score=iou(obj["box"],trk.box)
                if score>=self.threshold: 
                    candidates.append((score,di,tid))

        for _,di,tid in sorted(candidates,reverse=True):
            if di not in unmatched_dets or tid not in unmatched_tracks: continue
            matches[di]=tid
            unmatched_dets.remove(di)
            unmatched_tracks.remove(tid)

        for di,tid in matches.items():
            self.tracks[tid].box=list(detections[di]["box"])
            self.tracks[tid].label=detections[di]["label"]
            self.tracks[tid].miss_count=0

        # Assign new tracking id to unmatched objects
        for di in unmatched_dets:
            tid=self.next_id; self.next_id+=1
            self.tracks[tid]=Track(tid,detections[di]["label"],list(detections[di]["box"]))
            matches[di]=tid

        for tid in unmatched_tracks:
            if tid in self.tracks: 
                self.tracks[tid].miss_count+=1

        # Remove old tracks
        for tid in [tid for tid,t in self.tracks.items() if t.miss_count>self.max_misses]: 
            self.tracks.pop(tid,None)

        # Add tracking to detected objects
        out=[]
        for di,obj in enumerate(detections):
            tid=matches[di]
            out.append({**obj,"track_id":tid,"track_uid":f"track_{tid:04d}","job_uid":f"track_{tid:04d}"})

        return sorted(out,key=lambda o:float(o["ttc"]))