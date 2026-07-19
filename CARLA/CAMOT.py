import carla
import math
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import sys
import os
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

sys.path.insert(0, './yolov5')


TM_PORT     = 8000
YOLO_CONF   = 0.4
CAM_W       = 800
CAM_H       = 600
RECORD_FPS  = 10
RECORD_DIR  = "recordings"
LOG_DIR     = "logs"
BRAKE_DIST  = 20.0

TOTAL_PEDS        = 20
TOTAL_CARS        = 15
TOTAL_MOTORCYCLES = 5
TOTAL_BICYCLES    = 5
TOTAL_BUSES       = 3
TOTAL_TRUCKS      = 5
PED_SPREAD_MIN    = 10.0
PED_SPREAD_MAX    = 120.0
UNSTICK_INTERVAL  = 3.0

TTC_CRITICAL_SEC  = 2.0   # objects with TTC below this are "critical"

# Detection input resolution per workload level (Low/Medium/High fidelity)
DET_SIZE = {'L': 256, 'M': 416, 'H': 672}

# Max detections given a real feature-based re-id match at each level
ASSOC_MAX_FEAT = {'L': 0, 'M': 3, 'H': 99}

# Profiled worst-case execution times (ms) per level, used by the scheduler
WCET_DET   = {'L': 43.6,  'M': 53.5,  'H': 67.6}
WCET_ASSOC = {'L': 11.3,  'M': 74.0,  'H': 125.2}

TASK_PERIOD_MS   = 150.0   # per-object tracking task period/deadline
TASK_DEADLINE_MS = 150.0

CAMOT_SCHEDULER = 'EDF-Slack'   # or 'EDF-BE' -- see camot_schedule()

YOLO_CLASS_MAP = {0:"Ped", 1:"Cyc", 2:"Car", 3:"Mobike", 5:"Bus", 7:"LarVeh"}
YOLO_CLASSES   = list(YOLO_CLASS_MAP.keys())

AGENT_COLORS = {
    "Ped":(0,255,127), "Car":(0,128,255), "Cyc":(255,191,0),
    "Mobike":(255,0,200), "Bus":(0,200,255), "LarVeh":(255,80,0),
}
CRITICAL_COLOR    = (0, 0, 255)
NONCRITICAL_COLOR = (0, 220, 60)

VEHICLE_BP_FILTERS = {
    "Car"   :['vehicle.tesla.model3','vehicle.audi.a2','vehicle.bmw.grandtourer',
              'vehicle.chevrolet.impala','vehicle.dodge.charger_2020'],
    "Mobike":['vehicle.harley-davidson.low_rider','vehicle.kawasaki.ninja',
              'vehicle.vespa.zx125','vehicle.yamaha.yzf'],
    "Cyc"   :['vehicle.bh.crossbike','vehicle.diamondback.century','vehicle.gazelle.omafiets'],
    "Bus"   :['vehicle.mitsubishi.fusorosa','vehicle.volkswagen.t2'],
    "LarVeh":['vehicle.carlamotors.carlacola','vehicle.ford.ambulance','vehicle.mercedes.sprinter'],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE]   Using {DEVICE}")

os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

latest_frame: Optional[np.ndarray] = None

def camera_callback(image):
    global latest_frame
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    latest_frame = arr[:, :, :3].copy()


print("[YOLO]     Loading YOLOv5n...")
yolo_model = torch.hub.load('./yolov5', 'custom', path='yolov5n.pt', source='local')
yolo_model.conf    = YOLO_CONF
yolo_model.classes = YOLO_CLASSES
print("[YOLO]     Ready")


# MobileNetV2-based feature extractor -- produces a 128-d embedding per
# cropped detection, used for appearance-based re-identification (only
# actually run for objects scheduled at Medium/High association level).
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        backbone      = tvm.mobilenet_v2(pretrained=False)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.proj     = nn.Linear(1280, 128)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

print("[FEAT]     Building feature extractor...")
feat_extractor = FeatureExtractor().to(DEVICE).eval()
print("[FEAT]     Ready")

from torchvision import transforms
crop_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@torch.no_grad()
def extract_feature(frame_bgr: np.ndarray, box: tuple) -> Optional[torch.Tensor]:
    # Crop, preprocess, and embed one detection box -- used only when the
    # scheduler's chosen association level (M/H) allows a feature match.
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    t = crop_transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    t = t.unsqueeze(0).to(DEVICE)
    return feat_extractor(t).squeeze(0).cpu()


# One tracked object across frames (CA-MOT's own tracker state, separate
# from the ground-truth CARLA actor it corresponds to).
@dataclass
class TrackedObject:
    track_id    : int
    actor_id    : int
    agent_type  : str
    pos         : np.ndarray = field(default_factory=lambda: np.zeros(2))
    vel         : np.ndarray = field(default_factory=lambda: np.zeros(2))
    box         : Optional[tuple] = None
    feature     : Optional[torch.Tensor] = None
    feature_age : int = 0   # cycles since this track's feature was last refreshed
    pos_age     : int = 0   # cycles since its position was last updated by a real match
    miss_count  : int = 0   # consecutive missed frames (dropped after 10, see run_association)
    is_critical : bool = False


# Per-object real-time task: tracks how "stale" its detection/association
# have gotten (age_det/age_assoc), used by determine_options() to decide
# which fidelity level to bump it to next.
@dataclass
class MOTTask:
    actor_id  : int
    period    : float = TASK_PERIOD_MS
    deadline  : float = TASK_DEADLINE_MS
    age_det   : int   = 0
    age_assoc : int   = 0
    last_s    : str   = 'L'
    last_f    : str   = 'L'

    def record(self, s: str, f: str):
        if s in ('M','H'): self.age_det   += 1
        if f in ('M','H'): self.age_assoc += 1
        self.last_s = s; self.last_f = f


def iou(a: tuple, b: tuple) -> float:
    # Intersection-over-union of two boxes (position-only association metric)
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter=(ix2-ix1)*(iy2-iy1)
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6)

def feat_sim(fa: torch.Tensor, fb: torch.Tensor) -> float:
    # Cosine similarity between two feature embeddings (appearance-based
    # association metric, used when a track has a stored feature)
    fa=fa/(fa.norm()+1e-6); fb=fb/(fb.norm()+1e-6)
    return float(torch.dot(fa,fb).clamp(0,1))


def identify_critical_region(
        tracked: List[TrackedObject],
        ttc_map: Dict[int,float]) -> Optional[Tuple[int,int,int,int]]:
    # Bounding box enclosing every currently-critical (TTC < threshold)
    # tracked object -- this is where Low/Medium-level detection crops
    # the frame to, so critical objects stay covered even at low fidelity.
    boxes = [t.box for t in tracked
             if ttc_map.get(t.actor_id, 999) < TTC_CRITICAL_SEC
             and t.box is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def run_detection(frame: np.ndarray,
                  s_level: str,
                  critical_region: Optional[Tuple]) -> Tuple[List[dict], np.ndarray]:
    # Runs YOLO at a resolution/crop determined by s_level (detection
    # fidelity chosen by the scheduler for this cycle):
    #   'H' -> full frame, resized to DET_SIZE['H']
    #   'L'/'M' -> only the critical region (or frame center if none yet)
    #              is cropped and resized -- cheaper but only covers a
    #              sub-area of the scene.
    fh, fw    = frame.shape[:2]
    tsz       = DET_SIZE[s_level]
    annotated = frame.copy()

    if s_level == 'H':
        inp  = cv2.resize(frame, (tsz, tsz))
        sx, sy, ox, oy = fw/tsz, fh/tsz, 0, 0
    else:
        if critical_region is not None:
            rx1, ry1, rx2, ry2 = critical_region
        else:
            rx1, ry1 = fw//4, fh//4
            rx2, ry2 = 3*fw//4, 3*fh//4
        cw, ch = rx2-rx1, ry2-ry1
        if cw < tsz:
            pad=(tsz-cw)//2; rx1=max(0,rx1-pad); rx2=min(fw,rx2+pad)
        if ch < tsz:
            pad=(tsz-ch)//2; ry1=max(0,ry1-pad); ry2=min(fh,ry2+pad)
        crop = frame[ry1:ry2, rx1:rx2]
        inp  = cv2.resize(crop, (tsz, tsz))
        sx, sy = (rx2-rx1)/tsz, (ry2-ry1)/tsz
        ox, oy = rx1, ry1
        cv2.rectangle(annotated,(rx1,ry1),(rx2,ry2),(200,200,0),1)
        cv2.putText(annotated,f"RoI-{s_level}",(rx1+2,ry1+14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,0),1)

    results = yolo_model(inp[:,:,::-1].copy())
    dets    = []
    for *box, conf, cls in results.xyxy[0].tolist():
        ci = int(cls)
        if ci not in YOLO_CLASS_MAP: continue
        bx1=int(box[0]*sx+ox); by1=int(box[1]*sy+oy)
        bx2=int(box[2]*sx+ox); by2=int(box[3]*sy+oy)
        lbl = YOLO_CLASS_MAP[ci]
        dets.append({'box':(bx1,by1,bx2,by2),
                     'center':((bx1+bx2)//2,(by1+by2)//2),
                     'conf':float(conf),'label':lbl})
        col = AGENT_COLORS.get(lbl,(200,200,200))
        cv2.rectangle(annotated,(bx1,by1),(bx2,by2),col,2)
        cv2.putText(annotated,f"{lbl} {conf:.2f}",(bx1,max(by1-8,8)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)
    return dets, annotated


def run_association(frame: np.ndarray,
                    f_level: str,
                    detections: List[dict],
                    tracked: List[TrackedObject],
                    next_tid: List[int]) -> List[TrackedObject]:
    # Match this cycle's detections to existing tracks, at fidelity f_level:
    # 1) feature-based match (cosine similarity) for tracks/detections that
    #    have an embedded feature (limited by ASSOC_MAX_FEAT[f_level])
    # 2) IoU-based match for anything left unmatched
    # 3) unmatched tracks get their position extrapolated by last velocity
    # 4) unmatched detections become new tracks
    # 5) tracks missed >10 cycles in a row are dropped
    max_feats = ASSOC_MAX_FEAT[f_level]

    det_feats: List[Optional[torch.Tensor]] = []
    for i, det in enumerate(detections):
        if i < max_feats:
            det_feats.append(extract_feature(frame, det['box']))
        else:
            det_feats.append(None)

    matched_dets  : set = set()
    matched_tracks: set = set()

    tracks_with_feat = [t for t in tracked if t.feature is not None]
    if tracks_with_feat:
        sims = {}
        for di, (det, df) in enumerate(zip(detections, det_feats)):
            if df is None: continue
            for ti, trk in enumerate(tracks_with_feat):
                sims[(di,ti)] = feat_sim(df, trk.feature)
        for (di,ti), s in sorted(sims.items(), key=lambda x: x[1], reverse=True):
            if s < 0.3: break
            if di in matched_dets or tracks_with_feat[ti].track_id in matched_tracks:
                continue
            trk = tracks_with_feat[ti]
            matched_dets.add(di); matched_tracks.add(trk.track_id)
            bx1,by1,bx2,by2 = detections[di]['box']
            new_pos = np.array([(bx1+bx2)/2,(by1+by2)/2],dtype=float)
            trk.vel = new_pos - trk.pos
            trk.pos = new_pos; trk.box = detections[di]['box']
            trk.pos_age=0; trk.miss_count=0
            if det_feats[di] is not None:
                trk.feature=det_feats[di]; trk.feature_age=0
            else:
                trk.feature_age+=1

    unm_dets   = [i for i in range(len(detections)) if i not in matched_dets]
    unm_tracks = [t for t in tracked
                  if t.track_id not in matched_tracks and t.box is not None]
    if unm_dets and unm_tracks:
        ious = {}
        for di in unm_dets:
            for ti,trk in enumerate(unm_tracks):
                ious[(di,ti)] = iou(detections[di]['box'], trk.box)
        for (di,ti), v in sorted(ious.items(), key=lambda x: x[1], reverse=True):
            if v < 0.3: break
            if di in matched_dets or unm_tracks[ti].track_id in matched_tracks:
                continue
            trk = unm_tracks[ti]
            matched_dets.add(di); matched_tracks.add(trk.track_id)
            bx1,by1,bx2,by2 = detections[di]['box']
            new_pos = np.array([(bx1+bx2)/2,(by1+by2)/2],dtype=float)
            trk.vel = new_pos - trk.pos
            trk.pos = new_pos; trk.box = detections[di]['box']
            trk.pos_age=0; trk.miss_count=0; trk.feature_age+=1

    for trk in tracked:
        if trk.track_id in matched_tracks: continue
        trk.pos = trk.pos + trk.vel
        trk.pos_age+=1; trk.feature_age+=1; trk.miss_count+=1
        if trk.box is not None:
            bx1,by1,bx2,by2 = trk.box
            w=bx2-bx1; h=by2-by1
            cx=int(trk.pos[0]); cy=int(trk.pos[1])
            trk.box=(cx-w//2, cy-h//2, cx+w//2, cy+h//2)

    updated = list(tracked)
    for di in range(len(detections)):
        if di in matched_dets: continue
        det = detections[di]
        bx1,by1,bx2,by2 = det['box']
        new_trk = TrackedObject(
            track_id  = next_tid[0],
            actor_id  = -1,
            agent_type= det['label'],
            pos       = np.array([(bx1+bx2)/2,(by1+by2)/2],dtype=float),
            box       = det['box'],
            feature   = det_feats[di],
            feature_age = 0 if det_feats[di] is not None else 1,
        )
        next_tid[0]+=1
        updated.append(new_trk)

    updated = [t for t in updated if t.miss_count <= 10]
    return updated


def is_schedulable(tasks: List[MOTTask]) -> bool:
    # Rough utilization check at the cheapest (L,L) workload: true if
    # running every task at minimum fidelity still fits within the period.
    if not tasks: return True
    c_ll = WCET_DET['L'] + WCET_ASSOC['L']
    lhs  = (c_ll / TASK_PERIOD_MS) + len(tasks) * (c_ll / TASK_PERIOD_MS)
    return lhs <= 1.0


def slack_edf_slack(task_k: MOTTask,
                    all_tasks: List[MOTTask],
                    t_cur: float) -> float:
    # EDF-Slack: how much extra execution time task_k can be given (above
    # the minimum (L,L) cost) without any task with a later deadline
    # missing its own deadline. Higher slack -> can afford a higher
    # detection/association fidelity level this cycle.
    c_ll = WCET_DET['L'] + WCET_ASSOC['L']
    if not all_tasks: return 0.0
    d1 = task_k.deadline
    min_period = min(t.period for t in all_tasks)
    U  = (c_ll / min_period) + sum(c_ll / t.period for t in all_tasks)
    p  = 0.0
    for ti in sorted(all_tasks, key=lambda t: t.deadline, reverse=True):
        di = ti.deadline
        if di <= d1: continue
        U -= c_ll / ti.period
        max_exec = (1.0 - U) * (di - d1)
        RC = c_ll
        qi = max(0.0, RC - max_exec)
        denom = di - d1
        if denom > 1e-6:
            U = min(1.0, U + (RC - qi) / denom)
        p += qi
    return max(0.0, d1 - t_cur - p)


def slack_edf_be(task_k: MOTTask,
                 active: List[MOTTask],
                 t_cur: float) -> float:
    # EDF-BE (best-effort): simpler slack estimate, only nonzero when
    # task_k is the sole active task this cycle (no contention to reason about).
    if len(active) != 1: return 0.0
    c_ll = WCET_DET['L'] + WCET_ASSOC['L']
    return max(0.0, task_k.deadline - t_cur - c_ll)


def determine_options(task: MOTTask, slack: float) -> Tuple[str, str]:
    # Spend available slack on whichever of detection/association is
    # more stale (higher age_det vs age_assoc), preferring to bump that
    # one all the way to 'H' if slack allows, else partially upgrade it.
    if slack <= 0:
        return ('L', 'L')

    def pick_s(budget):
        if budget >= WCET_DET['H']  - WCET_DET['L']:  return 'H'
        if budget >= WCET_DET['M']  - WCET_DET['L']:  return 'M'
        return 'L'

    def pick_f(budget):
        if budget >= WCET_ASSOC['H'] - WCET_ASSOC['L']: return 'H'
        if budget >= WCET_ASSOC['M'] - WCET_ASSOC['L']: return 'M'
        return 'L'

    if task.age_det <= task.age_assoc:
        rem = slack - (WCET_DET['H'] - WCET_DET['L'])
        if rem >= 0:
            return ('H', pick_f(rem))
        else:
            return (pick_s(slack), 'L')
    else:
        rem = slack - (WCET_ASSOC['H'] - WCET_ASSOC['L'])
        if rem >= 0:
            return (pick_s(rem), 'H')
        else:
            return ('L', pick_f(slack))


def camot_schedule(tasks: List[MOTTask],
                   mode: str) -> List[Tuple[MOTTask,str,str]]:
    # Order tasks by EDF (earliest deadline first), then for each one
    # compute its available slack and pick a (detection, association)
    # fidelity level via determine_options(). t_sim accumulates the
    # simulated execution time so later tasks in EDF order see less slack.
    if not tasks: return []
    edf   = sorted(tasks, key=lambda t: t.deadline)
    out   = []
    t_sim = 0.0
    for idx, task in enumerate(edf):
        remaining = edf[idx:]
        if mode == 'EDF-BE':
            sl = slack_edf_be(task, remaining, t_sim)
        else:
            sl = slack_edf_slack(task, remaining, t_sim)
        s, f = determine_options(task, sl)
        out.append((task, s, f))
        t_sim += WCET_DET[s] + WCET_ASSOC[f]
    return out


def get_dist(a, b) -> float:
    # Euclidean distance between two CARLA actors
    l1=a.get_location(); l2=b.get_location()
    return math.sqrt((l1.x-l2.x)**2+(l1.y-l2.y)**2+(l1.z-l2.z)**2)

def get_speed(actor) -> float:
    # Actor speed in m/s
    v=actor.get_velocity()
    return math.sqrt(v.x**2+v.y**2+v.z**2)

def is_on_road(world, loc):
    wp=world.get_map().get_waypoint(loc,project_to_road=False,lane_type=carla.LaneType.Any)
    return wp is not None and wp.lane_type==carla.LaneType.Driving

def nav_sidewalk(world, n=50):
    # Sample a random pedestrian-navigable point that's off the road
    for _ in range(n):
        loc=world.get_random_location_from_navigation()
        if loc and not is_on_road(world,loc): return loc
    return None

def world_to_pixel(loc, vehicle, fov=90):
    # Project a 3D world location into the ego camera's pixel space
    # (pinhole model, camera assumed 2.5m fwd / 1.5m up from vehicle origin)
    vt=vehicle.get_transform(); yaw=math.radians(vt.rotation.yaw)
    cx=vt.location.x+math.cos(yaw)*2.5
    cy_=vt.location.y+math.sin(yaw)*2.5
    cz=vt.location.z+1.5
    dx=loc.x-cx; dy=loc.y-cy_; dz=loc.z-cz
    fwd=dx*math.cos(yaw)+dy*math.sin(yaw)
    if fwd<=0.1: return None
    rgt=dx*math.sin(yaw)-dy*math.cos(yaw)
    f=(CAM_W/2)/math.tan(math.radians(fov/2))
    u=int(CAM_W/2-(rgt/fwd)*f); v=int(CAM_H/2-(dz/fwd)*f)
    return (u,v) if 0<=u<CAM_W and 0<=v<CAM_H else None

def actor_to_det(actor, vehicle, dets, thr=80):
    # Match a known CARLA actor to a YOLO detection box by pixel projection
    proj=world_to_pixel(actor.get_location(), vehicle)
    if proj is None: return None
    u,v=proj
    for det in dets:
        x1,y1,x2,y2=det['box']
        if x1-20<=u<=x2+20 and y1-20<=v<=y2+20:
            return det
    return None

def compute_ttc(vehicle, actors) -> Dict[int,float]:
    # TTC approximation per actor: distance / closing speed (blends ego
    # speed against half the actor's own speed if it's roughly ahead)
    spd=max(get_speed(vehicle),0.5)
    etf=vehicle.get_transform(); ef=etf.get_forward_vector()
    ttc={}
    for actor in actors:
        dist=get_dist(vehicle,actor)
        try:   asp=get_speed(actor)
        except: asp=0.0
        to_a=actor.get_location()-etf.location
        dot=ef.x*to_a.x+ef.y*to_a.y
        if dot>-0.3:
            closing=max(spd-asp*0.5,0.5)
            ttc[actor.id]=dist/closing
        else:
            ttc[actor.id]=dist/spd
    return ttc


def spawn_ped(world, bp_lib, vehicle, fwd, beh, idx):
    # Spawn one pedestrian near a sidewalk point ~fwd metres ahead of ego,
    # with behaviour 'standing'/'walking'/'crossing' (crossing pedestrians
    # start stationary; main loop triggers the actual crossing walk later)
    vt=vehicle.get_transform(); yaw=math.radians(vt.rotation.yaw)
    tgt=carla.Location(x=vt.location.x+math.cos(yaw)*fwd,
                       y=vt.location.y+math.sin(yaw)*fwd,
                       z=vt.location.z)
    best=None; bd=float('inf')
    for _ in range(80):
        nav=world.get_random_location_from_navigation()
        if nav is None or is_on_road(world,nav): continue
        d=math.sqrt((nav.x-tgt.x)**2+(nav.y-tgt.y)**2)
        if d<bd: bd=d; best=nav
    if best is None: return None
    sl=carla.Location(x=best.x,y=best.y,z=best.z+0.5)
    if is_on_road(world,sl): return None
    pbp=random.choice(bp_lib.filter('walker.pedestrian.*'))
    cbp=bp_lib.find('controller.ai.walker')
    ped=world.try_spawn_actor(pbp,carla.Transform(sl))
    if ped is None:
        sl.z+=0.5; ped=world.try_spawn_actor(pbp,carla.Transform(sl))
    if ped is None: return None
    ctrl=world.spawn_actor(cbp,carla.Transform(),attach_to=ped)
    world.tick(); world.tick(); world.tick()
    ctrl.start()
    d1=nav_sidewalk(world)
    cross_dest=None; cstart=False; cdone=False
    if beh=='standing':
        ctrl.set_max_speed(0.0)
        if d1: ctrl.go_to_location(d1)
    elif beh=='walking':
        ctrl.set_max_speed(random.uniform(0.8,1.2))
        if d1: ctrl.go_to_location(d1)
    elif beh=='crossing':
        ctrl.set_max_speed(0.0)
        cross_dest=world.get_random_location_from_navigation()
    print(f"[PED {idx:02d}]  {beh:10s} fwd={fwd:.0f}m id={ped.id}")
    return {'ped':ped,'ctrl':ctrl,'behaviour':beh,'cross_dest':cross_dest,
            'cross_started':cstart,'cross_done':cdone,'spawn_time':time.time()}

def destroy_ped(e):
    try: e['ctrl'].stop(); e['ctrl'].destroy()
    except: pass
    try: e['ped'].destroy()
    except: pass

def spawn_all_peds(world, bp_lib, vehicle):
    peds=[]
    dists=sorted([random.uniform(PED_SPREAD_MIN,PED_SPREAD_MAX) for _ in range(TOTAL_PEDS)])
    behs=['standing']*4+['walking']*4+['crossing']*2
    random.shuffle(behs)
    print(f"\n[SPAWN]    {TOTAL_PEDS} pedestrians")
    for i,(d,b) in enumerate(zip(dists,behs)):
        e=spawn_ped(world,bp_lib,vehicle,d,b,i+1)
        if e: peds.append(e)
    print(f"[SPAWN]    {len(peds)} peds OK")
    return peds

def _valid_road_spawn(cmap, sp):
    # A usable vehicle spawn point: on a driving lane, not at a junction,
    # with at least 20m of road ahead
    wp = cmap.get_waypoint(sp.location, project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    return wp is not None and not wp.is_junction and bool(wp.next(20.0))

def spawn_all_vehicles(world, bp_lib, vehicle, tm):
    spawned=[]
    cmap=world.get_map()
    spts=[sp for sp in cmap.get_spawn_points() if _valid_road_spawn(cmap, sp)]
    eloc=vehicle.get_location()
    spts=[s for s in spts if
          math.sqrt((s.location.x-eloc.x)**2+(s.location.y-eloc.y)**2)>15]
    random.shuffle(spts); pt=0
    plan=[("Car",TOTAL_CARS),("Mobike",TOTAL_MOTORCYCLES),
          ("Cyc",TOTAL_BICYCLES),("Bus",TOTAL_BUSES),("LarVeh",TOTAL_TRUCKS)]
    print(f"\n[SPAWN]    Vehicles ({len(spts)} road points)")
    for atype,count in plan:
        ok=0
        for _ in range(count):
            if pt>=len(spts): break
            cands=[]
            for nm in VEHICLE_BP_FILTERS.get(atype,[]):
                try:
                    bp=bp_lib.find(nm)
                    if bp: cands.append(bp)
                except: pass
            if not cands:
                fb={'Cyc':'vehicle.*bicycle*','Mobike':'vehicle.*moto*',
                    'Bus':'vehicle.*bus*','LarVeh':'vehicle.*truck*'}.get(atype,'vehicle.tesla.*')
                cands=list(bp_lib.filter(fb))
            if not cands: continue
            actor=world.try_spawn_actor(random.choice(cands),spts[pt]); pt+=1
            tries=0
            while actor is None and tries<3 and pt<len(spts):
                actor=world.try_spawn_actor(random.choice(cands),spts[pt]); pt+=1; tries+=1
            if actor:
                actor.set_autopilot(True,TM_PORT)
                tm.ignore_lights_percentage(actor,0)
                tm.ignore_signs_percentage(actor,0)
                tm.distance_to_leading_vehicle(actor,3.0)
                tm.auto_lane_change(actor,True)
                tm.vehicle_percentage_speed_difference(actor,random.uniform(0,20))
                actor.apply_control(carla.VehicleControl(throttle=0.6))
                spawned.append({'actor':actor,'type':atype,
                                'spawn_time':time.time(),'last_moved':time.time()})
                ok+=1
                print(f"  [OK] {atype:8s} id={actor.id}")
        print(f"  {atype}: {ok}/{count}")
    time.sleep(2.0)
    return spawned

def unstick(rvehicles, tm, world):
    # Nudge a vehicle that's been idle >8s and isn't legitimately stopped
    # (not at a junction, not waiting on a red/yellow light)
    now=time.time(); cmap=world.get_map()
    for rv in rvehicles:
        try:
            a=rv['actor']
            if get_speed(a)>0.5: rv['last_moved']=now; continue
            wp=cmap.get_waypoint(a.get_location(),project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
            at_j=wp is not None and wp.is_junction
            red=False
            try:
                tl=a.get_traffic_light()
                if tl: red=tl.get_state() in (carla.TrafficLightState.Red,
                                               carla.TrafficLightState.Yellow)
            except: pass
            if at_j or red: rv['last_moved']=now; continue
            if now-rv.get('last_moved',rv['spawn_time'])>8.0:
                a.set_autopilot(False,TM_PORT)
                a.apply_control(carla.VehicleControl(throttle=0.7))
                time.sleep(0.15)
                a.set_autopilot(True,TM_PORT)
                tm.distance_to_leading_vehicle(a,3.0)
                tm.auto_lane_change(a,True)
                rv['last_moved']=now
                print(f"  [UNSTICK] id={a.id}")
        except: pass


def draw_hud(frame, speed, n_dets, n_tracked, n_crit,
             elapsed, brake, state, scheduler,
             avg_age_det, avg_age_assoc, sched_ms, lines):
    # On-screen overlay: speed/state/counts + per-track scheduling info panel
    if frame is None: return None
    h,w=frame.shape[:2]; d=frame.copy()
    info=[
        (f"CA-MOT  [{scheduler}]",               (255,220,0)),
        (f"Speed      : {speed:.2f} m/s",         (0,255,0)),
        (f"State      : {state}",                 (0,0,255) if brake else (0,255,0)),
        (f"Detections : {n_dets}",                (255,255,0)),
        (f"Tracked    : {n_tracked}",             (200,200,255)),
        (f"Critical   : {n_crit} (TTC<{TTC_CRITICAL_SEC:.0f}s)",(0,80,255)),
        (f"Age Det    : {avg_age_det:.1f}",        (100,255,180)),
        (f"Age Assoc  : {avg_age_assoc:.1f}",      (180,255,100)),
        (f"Sched ms   : {sched_ms:.1f}",           (200,200,200)),
        (f"Elapsed    : {elapsed:.1f}s",           (200,200,200)),
        (f"Deadline   : {TASK_DEADLINE_MS:.0f}ms", (255,165,0)),
    ]
    for i,(txt,col) in enumerate(info):
        cv2.putText(d,txt,(10,24+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.50,col,2)
    px=w-380; ph=18+len(lines)*20
    cv2.rectangle(d,(px-6,4),(w-4,ph),(20,20,20),-1)
    cv2.rectangle(d,(px-6,4),(w-4,ph),(80,80,80),1)
    cv2.putText(d,"TRACKS  [D-lvl A-lvl TTC fa]",
                (px,18),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)
    for i,ln in enumerate(lines):
        cv2.putText(d,ln,(px,36+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,220,60),1)
    if brake:
        cv2.rectangle(d,(0,h-50),(w,h),(0,0,150),-1)
        cv2.putText(d,"!!! PROXIMITY BRAKE !!!",
                    (10,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    return d

def draw_tracks(frame, tracked, ttc_map):
    # Draw each track's box (red if critical/TTC<threshold, green otherwise)
    if frame is None: return frame
    d=frame.copy()
    for t in tracked:
        if t.box is None: continue
        crit=ttc_map.get(t.actor_id,999)<TTC_CRITICAL_SEC
        col=CRITICAL_COLOR if crit else NONCRITICAL_COLOR
        x1,y1,x2,y2=t.box
        cv2.rectangle(d,(x1,y1),(x2,y2),col,3 if crit else 2)
        tag=f"{'[!]' if crit else ''}T{t.track_id} {t.agent_type} fa={t.feature_age}"
        cv2.putText(d,tag,(x1,max(y1-6,8)),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1)
    return d

def draw_crit_region(frame, reg):
    # Semi-transparent red overlay marking the critical region used to
    # crop Low/Medium-fidelity detection (see identify_critical_region)
    if frame is None or reg is None: return frame
    d=frame.copy(); x1,y1,x2,y2=reg
    ov=d.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),(0,0,180),-1)
    cv2.addWeighted(ov,0.18,d,0.82,0,d)
    cv2.rectangle(d,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.putText(d,"CRITICAL REGION",(x1+4,y1+18),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,60,255),2)
    return d


def main():
    # Entry point: sets up CARLA world/ego/sensors/scene, then runs the
    # detect -> track/associate -> schedule -> brake loop per frame.
    client=carla.Client('localhost',2000); client.set_timeout(10.0)
    world=client.get_world(); bp_lib=world.get_blueprint_library()
    spts=world.get_map().get_spawn_points()
    world.set_weather(carla.WeatherParameters.ClearNoon)
    print(f"[WORLD]    {world.get_map().name}")

    vehicle=None; peds=[]; rvehicles=[]
    camera=None; col_sensor=None; csv_file=None; vw=None
    rec_path=""; log_path=""

    try:
        vbp=bp_lib.filter('vehicle.tesla.model3')[0]
        vsp=random.choice(spts)
        vehicle=world.spawn_actor(vbp,vsp)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=1.0))
        print(f"[VEHICLE]  id={vehicle.id}")

        cbp=bp_lib.find('sensor.camera.rgb')
        cbp.set_attribute('image_size_x',str(CAM_W))
        cbp.set_attribute('image_size_y',str(CAM_H))
        cbp.set_attribute('fov','90')
        camera=world.spawn_actor(cbp,carla.Transform(carla.Location(x=2.5,z=1.5)),
                                  attach_to=vehicle)
        camera.listen(camera_callback)

        col_bp=bp_lib.find('sensor.other.collision')
        col_sensor=world.spawn_actor(col_bp,carla.Transform(),attach_to=vehicle)
        col_sensor.listen(lambda e: print(f"\n  !! COLLISION {e.other_actor.type_id}\n"))

        tm=client.get_trafficmanager(TM_PORT)
        tm.set_synchronous_mode(False); tm.set_random_device_seed(42)
        tm.set_global_distance_to_leading_vehicle(3.0)
        tm.global_percentage_speed_difference(10.0)

        peds=spawn_all_peds(world,bp_lib,vehicle)
        rvehicles=spawn_all_vehicles(world,bp_lib,vehicle,tm)

        time.sleep(1.0)
        vehicle.set_autopilot(True,TM_PORT)
        tm.ignore_lights_percentage(vehicle,0)
        tm.distance_to_leading_vehicle(vehicle,5.0)
        tm.auto_lane_change(vehicle,False)
        tm.vehicle_percentage_speed_difference(vehicle,20)

        rec_path=os.path.join(RECORD_DIR,time.strftime("camot_%Y%m%d_%H%M%S.avi"))
        vw=cv2.VideoWriter(rec_path,cv2.VideoWriter_fourcc(*'XVID'),RECORD_FPS,(CAM_W,CAM_H))

        log_path=os.path.join(LOG_DIR,time.strftime("camot_%Y%m%d_%H%M%S.csv"))
        csv_file=open(log_path,'w',newline='')
        csw=csv.writer(csv_file)
        csw.writerow([
            'elapsed_s','n_tracked','n_critical',
            'n_det_L','n_det_M','n_det_H',
            'n_assoc_L','n_assoc_M','n_assoc_H',
            'avg_age_det','avg_age_assoc',
            'overall_acc_proxy','critical_acc_proxy',
            'sched_ms','scheduler','speed_ms','brake',
            'ttc_values','det_workloads','assoc_workloads',
            'track_ids','feature_ages',
        ])

        spectator=world.get_spectator()
        cv2.namedWindow("CA-MOT",cv2.WINDOW_AUTOSIZE)

        print("\n"+"="*65)
        print(f"  CA-MOT | {CAMOT_SCHEDULER}")
        print(f"  TTC threshold : {TTC_CRITICAL_SEC}s")
        print(f"  Det  L/M/H px : {DET_SIZE['L']}/{DET_SIZE['M']}/{DET_SIZE['H']}")
        print(f"  Assoc L/M/H   : 0/{ASSOC_MAX_FEAT['M']}/all")
        print(f"  WCET det ms   : {WCET_DET['L']}/{WCET_DET['M']}/{WCET_DET['H']}")
        print(f"  WCET assoc ms : {WCET_ASSOC['L']}/{WCET_ASSOC['M']}/{WCET_ASSOC['H']}")
        print(f"  Deadline      : {TASK_DEADLINE_MS}ms")
        print("  Press Q to quit")
        print("="*65+"\n")

        start=time.time()
        was_braking=False; brake_hold=False
        ped_counter=TOTAL_PEDS+1; last_unstick=time.time()

        mot_tasks  : Dict[int,MOTTask]       = {}
        tracked    : List[TrackedObject]     = []
        next_tid   : List[int]               = [1]
        ttc_map    : Dict[int,float]         = {}
        trk_map    : Dict[int,TrackedObject] = {}

        while True:
            now=time.time(); elapsed=now-start

            if now-last_unstick>UNSTICK_INTERVAL:
                last_unstick=now; unstick(rvehicles,tm,world)

            for e in peds:
                if (e['behaviour']=='crossing' and not e['cross_started']
                        and now-e['spawn_time']>1.0):
                    e['cross_started']=True
                    dest=e['cross_dest'] or world.get_random_location_from_navigation()
                    e['cross_dest']=dest
                    if dest:
                        e['ctrl'].go_to_location(dest)
                        e['ctrl'].set_max_speed(random.uniform(1.2,1.6))
            for e in peds:
                if (e['behaviour']=='crossing' and e['cross_started']
                        and not e['cross_done'] and now-e['spawn_time']>3.0
                        and get_speed(e['ped'])<0.1):
                    e['cross_done']=True; e['ctrl'].set_max_speed(0.0)
            gone=[]
            for e in peds:
                try:
                    if get_dist(vehicle,e['ped'])>100: gone.append(e)
                except: gone.append(e)
            for e in gone:
                destroy_ped(e); peds.remove(e)
                ne=spawn_ped(world,bp_lib,vehicle,
                             random.uniform(60,100),
                             random.choice(['standing','walking','crossing']),
                             ped_counter)
                if ne: peds.append(ne)
                ped_counter+=1

            vt=vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                vt.location+carla.Location(x=-10,z=5),
                carla.Rotation(pitch=-15,yaw=vt.rotation.yaw)))

            speed=get_speed(vehicle); brake=False; state="DRIVING"

            if latest_frame is None:
                time.sleep(0.05); continue

            frame=latest_frame.copy()

            all_actors=([(e['ped'],'Ped') for e in peds]+
                        [(rv['actor'],rv['type']) for rv in rvehicles])

            # TTC per actor (informational + drives critical-region detection)
            ttc_map=compute_ttc(vehicle,[a for a,_ in all_actors])

            # Always run a cheap 'L' pass first, just to see which known
            # actors are currently visible at all (before fidelity scheduling)
            dets_L, display=run_detection(frame,'L',critical_region=None)

            visible=[]
            for actor,atype in all_actors:
                det=actor_to_det(actor,vehicle,dets_L)
                if det is not None:
                    visible.append((actor,atype,det))

            if not visible:
                cv2.imshow("CA-MOT",display)
                if cv2.waitKey(1)&0xFF==ord('q'): break
                time.sleep(0.05); continue

            # Ensure every visible actor has a task + track (create on first sight)
            for actor,atype,det in visible:
                if actor.id not in mot_tasks:
                    mot_tasks[actor.id]=MOTTask(actor_id=actor.id)
                if actor.id not in trk_map:
                    bx1,by1,bx2,by2=det['box']
                    new_t=TrackedObject(
                        track_id=next_tid[0],actor_id=actor.id,agent_type=atype,
                        pos=np.array([(bx1+bx2)/2,(by1+by2)/2],dtype=float),
                        box=det['box'])
                    next_tid[0]+=1; tracked.append(new_t); trk_map[actor.id]=new_t

            crit_region=identify_critical_region(
                [trk_map[a.id] for a,_,_ in visible if a.id in trk_map],
                ttc_map)
            display=draw_crit_region(display,crit_region)

            tasks_frame=[mot_tasks[a.id] for a,_,_ in visible]
            if not is_schedulable(tasks_frame):
                print(f"  [WARN] Not schedulable at C(L,L) n={len(tasks_frame)}")

            # CA-MOT's core scheduling decision: EDF order + per-task
            # detection/association fidelity level (see camot_schedule)
            t0=time.time()
            schedule=camot_schedule(tasks_frame,CAMOT_SCHEDULER)
            sched_ms=(time.time()-t0)*1000

            wl={'det':defaultdict(int),'assoc':defaultdict(int)}
            det_wls=[]; assoc_wls=[]
            n_crit=0; acc_all=[]; acc_crit=[]
            track_lines=[]

            print(f"\n[{elapsed:6.1f}s] spd={speed:.2f}m/s vis={len(visible)}"
                  f" crit={'yes' if crit_region else 'no'}"
                  f" sched={sched_ms:.1f}ms [{CAMOT_SCHEDULER}]")
            print(f"  {'─'*65}")

            # For each scheduled task, re-run detection/association at its
            # assigned (s, f) fidelity level and update its track + a rough
            # accuracy proxy (higher fidelity -> assumed higher accuracy)
            for (task,s,f),(actor,atype,det_l) in zip(schedule,visible):
                ttc=ttc_map.get(actor.id,999.0)
                is_crit=ttc<TTC_CRITICAL_SEC
                if is_crit: n_crit+=1
                wl['det'][s]+=1; wl['assoc'][f]+=1
                det_wls.append(s); assoc_wls.append(f)

                dets_task,_=run_detection(frame,s,crit_region)
                best_det=actor_to_det(actor,vehicle,dets_task) or det_l

                trk=trk_map[actor.id]
                max_feats=ASSOC_MAX_FEAT[f]
                if max_feats>0:
                    feat=extract_feature(frame,best_det['box'])
                    if feat is not None:
                        trk.feature=feat; trk.feature_age=0
                    else:
                        trk.feature_age+=1
                else:
                    trk.feature_age+=1

                bx1,by1,bx2,by2=best_det['box']
                new_pos=np.array([(bx1+bx2)/2,(by1+by2)/2],dtype=float)
                trk.vel=new_pos-trk.pos; trk.pos=new_pos
                trk.box=best_det['box']; trk.pos_age=0; trk.miss_count=0
                trk.is_critical=is_crit; trk.agent_type=atype

                task.record(s,f)

                # Accuracy proxy (no real model here): higher detection/
                # association fidelity assumed to yield higher accuracy --
                # used only for CSV logging/comparison against RETINA's
                # real measured accuracy.
                acc={'H':1.0,'M':0.85,'L':0.70}[s]+{'H':0.05,'M':0.02,'L':0.0}[f]
                acc=min(1.0,acc)
                acc_all.append(acc)
                if is_crit: acc_crit.append(acc)

                print(f"  {'[CRIT!]' if is_crit else '       '}"
                      f" id={actor.id:4d} {atype:7s} | ttc={ttc:5.1f}s |"
                      f" D={s} A={f} | age=({task.age_det},{task.age_assoc})"
                      f" | fa={trk.feature_age}")

                track_lines.append(
                    f"{'[!]' if is_crit else '   '}"
                    f"T{trk.track_id} {atype[:4]:4s}"
                    f" D{s}A{f} ttc={ttc:.1f}s fa={trk.feature_age}")

            for actor_id,trk in trk_map.items():
                if actor_id not in [a.id for a,_,_ in visible]:
                    trk.pos=trk.pos+trk.vel
                    trk.pos_age+=1; trk.feature_age+=1; trk.miss_count+=1

            print(f"  {'─'*65}")
            ad=sum(t.age_det for t in tasks_frame)/max(1,len(tasks_frame))
            aa=sum(t.age_assoc for t in tasks_frame)/max(1,len(tasks_frame))
            ov=sum(acc_all)/len(acc_all) if acc_all else 0.0
            cr=sum(acc_crit)/len(acc_crit) if acc_crit else 0.0
            print(f"  OvAcc={ov:.3f} CritAcc={cr:.3f} | AgeD={ad:.1f} AgeA={aa:.1f}"
                  f" | D(L/M/H)={wl['det']['L']}/{wl['det']['M']}/{wl['det']['H']}"
                  f" A(L/M/H)={wl['assoc']['L']}/{wl['assoc']['M']}/{wl['assoc']['H']}")

            # Simple in-lane proximity/TTC brake check -- independent of
            # tracking/scheduling; brakes if any visible actor is directly
            # ahead and either very close or closing fast (TTC<2.5s).
            etf=vehicle.get_transform(); ef=etf.get_forward_vector()
            for actor,_,_ in visible:
                try:
                    dist=get_dist(vehicle,actor)
                    to_a=actor.get_location()-etf.location
                    tl=math.sqrt(to_a.x**2+to_a.y**2)
                    if tl<0.1: continue
                    dfwd=ef.x*to_a.x+ef.y*to_a.y
                    dlat=abs(ef.y*to_a.x-ef.x*to_a.y)
                    if dfwd<=0 or dlat>=2.5: continue
                    if dist<BRAKE_DIST: brake=True; state="BRAKING"; break
                    asp=get_speed(actor); cs=speed-asp
                    if cs>0.5 and dist/cs<2.5: brake=True; state="BRAKING"; break
                except: pass

            if brake and not was_braking: print("\n  *** BRAKING ***")
            if not brake and was_braking: print("\n  *** RESUMING ***")
            was_braking=brake
            if brake: brake_hold=True
            elif brake_hold:
                if not any(ttc_map.get(a.id,999)<TTC_CRITICAL_SEC for a,_,_ in visible):
                    brake_hold=False
            if brake_hold:
                vehicle.set_autopilot(False,TM_PORT)
                vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=1.0,steer=0.0))
                state="STOPPED"
            else:
                vehicle.set_autopilot(True,TM_PORT)

            csw.writerow([
                f"{elapsed:.2f}",len(tracked),n_crit,
                wl['det']['L'],wl['det']['M'],wl['det']['H'],
                wl['assoc']['L'],wl['assoc']['M'],wl['assoc']['H'],
                f"{ad:.2f}",f"{aa:.2f}",f"{ov:.4f}",f"{cr:.4f}",
                f"{sched_ms:.2f}",CAMOT_SCHEDULER,f"{speed:.2f}",int(brake),
                ';'.join(f"{ttc_map.get(a.id,999):.2f}" for a,_,_ in visible),
                ';'.join(det_wls),';'.join(assoc_wls),
                ';'.join(str(trk_map[a.id].track_id) for a,_,_ in visible if a.id in trk_map),
                ';'.join(str(trk_map[a.id].feature_age) for a,_,_ in visible if a.id in trk_map),
            ])
            csv_file.flush()

            display=draw_tracks(display,tracked,ttc_map)
            hud=draw_hud(display,speed,len(dets_L),len(tracked),n_crit,
                         elapsed,brake,state,CAMOT_SCHEDULER,ad,aa,sched_ms,track_lines)
            if hud is not None:
                cv2.imshow("CA-MOT",hud)
                if vw: vw.write(hud)

            if cv2.waitKey(1)&0xFF==ord('q'): print("Quit."); break
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        try: csv_file.close(); print(f"  [OK] Log: {log_path}")
        except: pass
        try:
            if vw: vw.release(); print(f"  [OK] Video: {rec_path}")
        except: pass
        if camera:     camera.stop(); camera.destroy()
        if col_sensor: col_sensor.stop(); col_sensor.destroy()
        for e in peds: destroy_ped(e)
        for rv in rvehicles:
            try: rv['actor'].set_autopilot(False); rv['actor'].destroy()
            except: pass
        if vehicle:
            vehicle.set_autopilot(False); vehicle.destroy()
        print("Done.")


if __name__ == '__main__':
    main()
