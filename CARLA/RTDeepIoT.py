import carla
import math
import time
import random
import numpy as np
import cv2
import torch
import sys
import os
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

sys.path.insert(0, './yolov5')
sys.path.insert(0, '.')  # so `rtdeepiot` package resolves

from rtdeepiot.models.multi_exit_resnet import build_model
from rtdeepiot.scheduler.dp_scheduler import (
    Task, dp_schedule, predict_task_rewards, load_wcet_profile,
)
from rtdeepiot.data.road_crop_dataset import CLASS_NAMES as ROAD_CLASS_NAMES
from rtdeepiot.data.road_crop_dataset import NORM_MEAN, NORM_STD
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
# Constants -- EXACT copies from carla_camot_fail.py where applicable
# ─────────────────────────────────────────────────────────────────────────────

TM_PORT          = 8000
YOLO_CONF        = 0.4
CAM_W, CAM_H     = 800, 600
RECORD_FPS       = 10
RECORD_DIR       = "recordings"
LOG_DIR          = "logs"
UNSTICK_INTERVAL = 3.0

TOTAL_PEDS        = 30
TOTAL_CARS        = 10
TOTAL_MOTORCYCLES = 5
TOTAL_BICYCLES    = 5
TOTAL_BUSES       = 1
TOTAL_TRUCKS      = 5

PED_SPREAD_MIN    = 10.0
PED_SPREAD_MAX    = 120.0

# RTDeepIoT parameters (replaces CA-MOT's TTC_CRITICAL_SEC/DET_SIZE/
# WCET_DET/WCET_ASSOC/TASK_DEADLINE_MS/CAMOT_SCHEDULER)
CHECKPOINT_PATH   = './rtdeepiot/checkpoints/best_road.pth'
WCET_PATH         = './rtdeepiot/checkpoints/wcet_road.json'
UTILITY_HEURISTIC = 'exp'
REWARD_QUANTIZATION_DELTA = 0.01
CONFIDENCE_EMA_ALPHA = 0.3
DEFAULT_SEED_CONFIDENCE = 0.5
RTDEEPIOT_SCHEDULER = 'RTDeepIoT-DP'   # analogous to CAMOT_SCHEDULER label

VEHICLE_BP_FILTERS = {
    "Car"   :['vehicle.tesla.model3','vehicle.audi.a2',
              'vehicle.bmw.grandtourer','vehicle.chevrolet.impala',
              'vehicle.dodge.charger_2020','vehicle.lincoln.mkz_2020',
              'vehicle.mercedes.coupe_2020','vehicle.mini.cooper_s'],
    "Mobike":['vehicle.harley-davidson.low_rider','vehicle.kawasaki.ninja',
              'vehicle.vespa.zx125','vehicle.yamaha.yzf'],
    "Cyc"   :['vehicle.bh.crossbike','vehicle.diamondback.century',
              'vehicle.gazelle.omafiets'],
    "Bus"   :['vehicle.mitsubishi.fusorosa','vehicle.volkswagen.t2'],
    "LarVeh":['vehicle.carlamotors.carlacola','vehicle.carlamotors.firetruck',
              'vehicle.ford.ambulance','vehicle.mercedes.sprinter'],
}

YOLO_CLASS_MAP = {0:"Ped",1:"Cyc",2:"Car",3:"Mobike",5:"Bus",7:"LarVeh"}
YOLO_CLASSES   = list(YOLO_CLASS_MAP.keys())
AGENT_COLORS   = {
    "Ped":(0,255,127),"Car":(0,128,255),"Cyc":(255,191,0),
    "Mobike":(255,0,200),"Bus":(0,200,255),"LarVeh":(255,80,0),
}
CRITICAL_COLOR    = (0, 0, 255)
NONCRITICAL_COLOR = (0, 220, 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE]   {DEVICE}")
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

latest_frame: Optional[np.ndarray] = None

def camera_callback(image):
    global latest_frame
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    latest_frame = arr[:, :, :3].copy()


# ─────────────────────────────────────────────────────────────────────────────
# YOLO -- localization only, same as carla_camot_fail.py
# ─────────────────────────────────────────────────────────────────────────────

print("[YOLO]     Loading YOLOv5n...")
yolo_model = torch.hub.load('./yolov5','custom',path='yolov5n.pt',source='local')
yolo_model.conf    = YOLO_CONF
yolo_model.classes = YOLO_CLASSES
print("[YOLO]     Ready")


# ─────────────────────────────────────────────────────────────────────────────
# RTDeepIoT multi-exit ResNet -- replaces CA-MOT's MOTTask/TrackedObject/
# camot_schedule entirely
# ─────────────────────────────────────────────────────────────────────────────

print("[RTDeepIoT] Loading multi-exit ResNet...")
rtdeepiot_model = build_model('road').to(DEVICE).eval()
_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
rtdeepiot_model.load_state_dict(_ckpt['model_state'])
print(f"[RTDeepIoT] Loaded checkpoint (epoch {_ckpt.get('epoch','?')}, "
      f"best_acc={_ckpt.get('best_acc','?')})")

WCET_ROAD = load_wcet_profile(WCET_PATH)  # [P^1, P^2, P^3] cumulative ms
print(f"[RTDeepIoT] WCET (cumulative ms): "
      f"stage1={WCET_ROAD[0]:.3f}  stage2={WCET_ROAD[1]:.3f}  stage3={WCET_ROAD[2]:.3f}")

classify_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

# actor_id -> running average of that actor's own previously-observed
# stage-1 confidence, used to seed reward prediction (\u00a7II-D)
actor_confidence_ema: Dict[int, float] = {}


@torch.no_grad()
def classify_crop(frame_bgr: np.ndarray, box: tuple,
                  exit_after: int) -> Optional[dict]:
    """Identical logic to rtdeepiot_carla.py's classify_crop()."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    t = classify_transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    t = t.unsqueeze(0).to(DEVICE)

    outputs = rtdeepiot_model(t, exit_after=exit_after)
    final = outputs[-1]
    label = ROAD_CLASS_NAMES[final['predicted'].item()]
    confidence = final['confidence'].item()
    stage1_confidence = outputs[0]['confidence'].item()

    return {
        'label': label, 'confidence': confidence,
        'exit_after': exit_after, 'stage1_confidence': stage1_confidence,
    }


def run_rtdeepiot_schedule(visible_with_dets: list,
                           ttc_map: Dict[int, float],
                           frame: np.ndarray) -> Dict[int, int]:
    """
    Two-phase DP scheduling (\u00a7II-D/\u00a7II-E), identical logic to
    rtdeepiot_carla.py's run_rtdeepiot_schedule(). Replaces CA-MOT's
    camot_schedule() entirely.

    Args:
        visible_with_dets : list of (actor, atype, det) tuples
        ttc_map            : dict[actor_id] -> TTC seconds
        frame               : current camera frame (for running mandatory
                              stage-1 classification in Phase B)

    Returns: dict[actor_id] -> final assigned depth (1, 2, or 3)
    """
    if not visible_with_dets:
        return {}

    # Phase A: plan using seeded/predicted rewards
    tasks = []
    for actor, atype, det in visible_with_dets:
        seed = actor_confidence_ema.get(actor.id, DEFAULT_SEED_CONFIDENCE)
        rewards = predict_task_rewards(WCET_ROAD, seed_confidence=seed,
                                       heuristic=UTILITY_HEURISTIC)
        deadline_ms = max(ttc_map.get(actor.id, 999.0) * 1000.0, WCET_ROAD[0])
        tasks.append(Task(
            job_id=str(actor.id), deadline_ms=deadline_ms,
            omega=1, max_depth=3,
            predicted_rewards=rewards, exec_times=WCET_ROAD,
        ))
    dp_schedule(tasks, delta=REWARD_QUANTIZATION_DELTA)  # initial plan (unused directly)

    # Phase B: run mandatory stage 1 for real, revise
    real_stage1: Dict[str, float] = {}
    for actor, atype, det in visible_with_dets:
        result = classify_crop(frame, det['box'], exit_after=1)
        if result is None:
            real_stage1[str(actor.id)] = actor_confidence_ema.get(
                actor.id, DEFAULT_SEED_CONFIDENCE)
            continue
        real_conf = result['stage1_confidence']
        real_stage1[str(actor.id)] = real_conf
        prev = actor_confidence_ema.get(actor.id, real_conf)
        actor_confidence_ema[actor.id] = (
            CONFIDENCE_EMA_ALPHA * real_conf + (1 - CONFIDENCE_EMA_ALPHA) * prev
        )

    for t in tasks:
        seed = real_stage1.get(t.job_id, t.predicted_rewards[0])
        t.predicted_rewards = predict_task_rewards(
            WCET_ROAD, seed_confidence=seed, heuristic=UTILITY_HEURISTIC)

    final_assignment = dp_schedule(tasks, delta=REWARD_QUANTIZATION_DELTA)
    return {int(jid): depth for jid, depth in final_assignment.items()}


def world_to_pixel(loc, vehicle, fov=90):
    """
    Projects a 3D world location into 2D camera-frame pixel coordinates.
    Reused from rtdeepiot_carla.py to enable correct actor-to-detection
    matching (each pedestrian gets ITS OWN detected box, not a shared
    placeholder).
    """
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
    """
    Matches one CARLA actor to the YOLO detection whose box contains
    its projected pixel position -- so each actor gets its OWN crop,
    not a shared/duplicated one.
    """
    proj=world_to_pixel(actor.get_location(), vehicle)
    if proj is None: return None
    u,v=proj
    for det in dets:
        if det['label']!='Ped': continue
        x1,y1,x2,y2=det['box']
        if x1-thr<=u<=x2+thr and y1-thr<=v<=y2+thr:
            return det
    return None


def run_detection(frame):
    """
    YOLO detection over the FULL frame -- RTDeepIoT does not scale
    detection cost the way CA-MOT does (no L/M/H region cropping),
    since only classification DEPTH is adaptive in RTDeepIoT. Returns
    (dets, annotated_frame), matching run_detection()'s signature
    shape from carla_camot_fail.py but without the s_level/crit_region
    parameters (not applicable here).
    """
    if frame is None: return [], frame
    ann = frame.copy()
    results = yolo_model(frame[:,:,::-1].copy())
    dets = []
    for *box,conf,cls in results.xyxy[0].tolist():
        ci=int(cls)
        if ci not in YOLO_CLASS_MAP: continue
        bx1,by1,bx2,by2=map(int,box)
        lbl=YOLO_CLASS_MAP[ci]
        col=AGENT_COLORS.get(lbl,(200,200,200))
        dets.append({'box':(bx1,by1,bx2,by2),'label':lbl,'conf':float(conf)})
        cv2.rectangle(ann,(bx1,by1),(bx2,by2),col,2)
        cv2.putText(ann,f"{lbl}{conf:.2f}",
                    (bx1,max(by1-6,8)),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
    return dets, ann


# ─────────────────────────────────────────────────────────────────────────────
# CARLA helpers -- EXACT copies from carla_camot_fail.py
# ─────────────────────────────────────────────────────────────────────────────

def get_distance(a, b):
    l1=a.get_location(); l2=b.get_location()
    return math.sqrt((l1.x-l2.x)**2+(l1.y-l2.y)**2+(l1.z-l2.z)**2)

def get_vehicle_speed_ms(v):
    vel=v.get_velocity()
    return math.sqrt(vel.x**2+vel.y**2+vel.z**2)

def get_ped_speed(p):
    vel=p.get_velocity()
    return math.sqrt(vel.x**2+vel.y**2+vel.z**2)

def is_on_road(world, location):
    wp=world.get_map().get_waypoint(
        location,project_to_road=False,lane_type=carla.LaneType.Any)
    return wp is not None and wp.lane_type==carla.LaneType.Driving

def get_sidewalk_nav_location(world, attempts=50):
    for _ in range(attempts):
        loc=world.get_random_location_from_navigation()
        if loc and not is_on_road(world,loc): return loc
    return None

def compute_ttc(vehicle, actor):
    dist    = get_distance(vehicle,actor)
    ego_spd = max(get_vehicle_speed_ms(vehicle),0.1)
    try:    asp=get_vehicle_speed_ms(actor)
    except: asp=get_ped_speed(actor)
    etf=vehicle.get_transform(); ef=etf.get_forward_vector()
    to_a=actor.get_location()-etf.location
    dot=ef.x*to_a.x+ef.y*to_a.y
    if dot>-0.3:
        closing=max(ego_spd-asp*0.3,0.1)
        return dist/closing
    return dist/ego_spd


# ─────────────────────────────────────────────────────────────────────────────
# Pedestrian spawn -- EXACT copy from carla_camot_fail.py
# ─────────────────────────────────────────────────────────────────────────────

def spawn_single_ped(world, bp_lib, vehicle, fwd_dist, behaviour, ped_index):
    vt      = vehicle.get_transform()
    yaw_rad = math.radians(vt.rotation.yaw)
    target  = carla.Location(
        x=vt.location.x+math.cos(yaw_rad)*fwd_dist,
        y=vt.location.y+math.sin(yaw_rad)*fwd_dist,
        z=vt.location.z)

    best_loc=None; best_dist=float('inf')
    for _ in range(80):
        nav_loc=world.get_random_location_from_navigation()
        if nav_loc is None: continue
        if is_on_road(world,nav_loc): continue
        d=math.sqrt((nav_loc.x-target.x)**2+(nav_loc.y-target.y)**2)
        if d<best_dist: best_dist=d; best_loc=nav_loc

    if best_loc is None: return None
    spawn_loc=carla.Location(x=best_loc.x,y=best_loc.y,z=best_loc.z+0.5)
    if is_on_road(world,spawn_loc): return None

    ped_bp  = random.choice(bp_lib.filter('walker.pedestrian.*'))
    ctrl_bp = bp_lib.find('controller.ai.walker')

    ped=world.try_spawn_actor(ped_bp,carla.Transform(spawn_loc))
    if ped is None:
        spawn_loc.z+=0.5
        ped=world.try_spawn_actor(ped_bp,carla.Transform(spawn_loc))
    if ped is None: return None

    ctrl=world.spawn_actor(ctrl_bp,carla.Transform(),attach_to=ped)
    world.tick(); world.tick(); world.tick()
    ctrl.start()

    dest_1=get_sidewalk_nav_location(world)
    dest_2=get_sidewalk_nav_location(world)
    cross_dest=None; cross_started=False; cross_done=False

    if behaviour=='standing':
        ctrl.set_max_speed(0.0)
        if dest_1: ctrl.go_to_location(dest_1)
    elif behaviour=='walking':
        ctrl.set_max_speed(random.uniform(0.8,1.2))
        if dest_1: ctrl.go_to_location(dest_1)
    elif behaviour=='crossing':
        ctrl.set_max_speed(0.0)
        cross_dest=world.get_random_location_from_navigation()
    elif behaviour=='crossed':
        ctrl.set_max_speed(0.5)
        if dest_1: ctrl.go_to_location(dest_1)
        cross_done=True

    print(f"[PED {ped_index:02d}]  {behaviour:10s} fwd={fwd_dist:.0f}m id={ped.id}")
    return {'ped':ped,'ctrl':ctrl,'behaviour':behaviour,
            'cross_dest':cross_dest,'cross_started':cross_started,
            'cross_done':cross_done,'spawn_time':time.time(),'nav_dest_2':dest_2}

def spawn_all_pedestrians(world, bp_lib, vehicle):
    pedestrians=[]
    fwd_distances=sorted([random.uniform(PED_SPREAD_MIN,PED_SPREAD_MAX)
                          for _ in range(TOTAL_PEDS)])
    behaviours=(['crossing']*15+['walking']*10+['standing']*5)
    random.shuffle(behaviours)
    print(f"\n[SPAWN]    Spawning {TOTAL_PEDS} pedestrians")
    for i,(fwd,beh) in enumerate(zip(fwd_distances,behaviours)):
        entry=spawn_single_ped(world,bp_lib,vehicle,fwd,beh,i+1)
        if entry: pedestrians.append(entry)
    print(f"[SPAWN]    {len(pedestrians)} pedestrians spawned")
    return pedestrians

def destroy_ped(entry):
    try: entry['ctrl'].stop(); entry['ctrl'].destroy()
    except: pass
    try: entry['ped'].destroy()
    except: pass

def respawn_single_ahead(world, bp_lib, vehicle, ped_index, pedestrians):
    fwd_dist =random.uniform(60.0,100.0)
    behaviour=random.choice(['crossing','crossing','crossing','walking','walking','standing'])
    entry=spawn_single_ped(world,bp_lib,vehicle,fwd_dist,behaviour,ped_index)
    if entry:
        pedestrians.append(entry)
        print(f"[RESPAWN]  ped {fwd_dist:.0f}m ahead | {behaviour}")


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle spawn -- EXACT copy from carla_camot_fail.py (unused here since
# road_vehicles=[] in main(), kept for parity / future use)
# ─────────────────────────────────────────────────────────────────────────────

def _valid_road_spawn(cmap, sp):
    wp=cmap.get_waypoint(sp.location,project_to_road=True,
                          lane_type=carla.LaneType.Driving)
    return wp is not None and not wp.is_junction and bool(wp.next(20.0))

def spawn_all_vehicles(world, bp_lib, vehicle, tm):
    spawned=[]; cmap=world.get_map()
    spts=[sp for sp in cmap.get_spawn_points() if _valid_road_spawn(cmap,sp)]
    eloc=vehicle.get_location()
    spts=[sp for sp in spts if
          math.sqrt((sp.location.x-eloc.x)**2+(sp.location.y-eloc.y)**2)>15]
    random.shuffle(spts); pt=0
    plan=[("Car",TOTAL_CARS),("Mobike",TOTAL_MOTORCYCLES),
          ("Cyc",TOTAL_BICYCLES),("Bus",TOTAL_BUSES),("LarVeh",TOTAL_TRUCKS)]
    print(f"\n[SPAWN]    Vehicles ({len(spts)} road pts)")
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
                actor=world.try_spawn_actor(random.choice(cands),spts[pt])
                pt+=1; tries+=1
            if actor:
                actor.set_autopilot(True,TM_PORT)
                tm.ignore_lights_percentage(actor,0)
                tm.distance_to_leading_vehicle(actor,3.0)
                tm.auto_lane_change(actor,True)
                tm.vehicle_percentage_speed_difference(actor,random.uniform(0,20))
                actor.apply_control(carla.VehicleControl(throttle=0.6))
                spawned.append({'actor':actor,'type':atype,
                                'spawn_time':time.time(),'last_moved':time.time()})
                ok+=1
        print(f"  {atype}: {ok}/{count}")
    time.sleep(2.0)
    return spawned

def unstick_vehicles(road_vehicles, vehicle, tm, world):
    now=time.time(); cmap=world.get_map()
    for rv in road_vehicles:
        try:
            actor=rv['actor']
            if get_vehicle_speed_ms(actor)>0.5: rv['last_moved']=now; continue
            wp=cmap.get_waypoint(actor.get_location(),project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
            at_j=wp is not None and wp.is_junction
            red=False
            try:
                tl=actor.get_traffic_light()
                if tl: red=tl.get_state() in (carla.TrafficLightState.Red,
                                               carla.TrafficLightState.Yellow)
            except: pass
            if at_j or red: rv['last_moved']=now; continue
            if now-rv.get('last_moved',rv['spawn_time'])>8.0:
                actor.set_autopilot(False,TM_PORT)
                actor.apply_control(carla.VehicleControl(throttle=0.7))
                time.sleep(0.15)
                actor.set_autopilot(True,TM_PORT)
                tm.distance_to_leading_vehicle(actor,3.0)
                rv['last_moved']=now
                print(f"  [UNSTICK] id={actor.id}")
        except: pass


# ─────────────────────────────────────────────────────────────────────────────
# HUD -- EXACT copy from carla_camot_fail.py (speed only)
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame, speed):
    """Speed only -- black background box, green text."""
    if frame is None: return None
    d = frame.copy()
    txt = f"Speed: {speed:.2f} m/s"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(d, (8, 8), (14 + tw, 22 + th), (0, 0, 0), -1)
    cv2.putText(d, txt, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    client    = carla.Client('localhost',2000)
    client.set_timeout(15.0)
    world     = client.get_world()
    bp_lib    = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points()
    world.set_weather(carla.WeatherParameters.ClearNoon)
    print(f"[WORLD]    {world.get_map().name}  ({len(spawn_pts)} spawn pts)")

    vehicle=None; pedestrians=[]; road_vehicles=[]
    camera=None; col_sensor=None; csv_file=None; vw=None
    rec_path=""; log_path=""

    try:
        # Ego vehicle -- EXACT same spawn point as carla_camot_fail.py
        EGO_SPAWN_IDX = 80
        vbp    = bp_lib.filter('vehicle.tesla.model3')[0]
        v_sp   = spawn_pts[EGO_SPAWN_IDX]
        vehicle= world.spawn_actor(vbp,v_sp)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=1.0))
        print(f"[VEHICLE]  id={vehicle.id}"
              f" ({v_sp.location.x:.1f},{v_sp.location.y:.1f})")

        # Camera
        cbp=bp_lib.find('sensor.camera.rgb')
        cbp.set_attribute('image_size_x','800')
        cbp.set_attribute('image_size_y','600')
        cbp.set_attribute('fov','90')
        camera=world.spawn_actor(cbp,
            carla.Transform(carla.Location(x=2.5,z=1.5)),
            attach_to=vehicle)
        camera.listen(camera_callback)

        # Collision sensor -- logs only, same as carla_camot_fail.py
        col_bp    =bp_lib.find('sensor.other.collision')
        col_sensor=world.spawn_actor(col_bp,carla.Transform(),attach_to=vehicle)
        col_sensor.listen(
            lambda e: print(f"\n  !! COLLISION {e.other_actor.type_id} !!\n"))

        # Traffic manager -- EXACT same settings as carla_camot_fail.py
        tm=client.get_trafficmanager(TM_PORT)
        tm.set_synchronous_mode(False)
        tm.set_random_device_seed(42)
        tm.set_global_distance_to_leading_vehicle(3.0)
        tm.global_percentage_speed_difference(10.0)

        # Spawn actors
        pedestrians   = spawn_all_pedestrians(world,bp_lib,vehicle)
        road_vehicles = []   # no other vehicles

        time.sleep(1.0)

        # Release ego -- EXACT same as carla_camot_fail.py, INCLUDING
        # ignore_walkers_percentage(100): the vehicle will NOT
        # automatically avoid pedestrians via CARLA's own autopilot.
        # Combined with the hardcoded brake=0 below, this means
        # NOTHING in this script prevents a collision -- intentional,
        # matching CA-MOT's reference file exactly.
        vehicle.set_autopilot(True,TM_PORT)
        tm.ignore_lights_percentage(vehicle,0)
        tm.ignore_signs_percentage(vehicle,0)
        tm.ignore_vehicles_percentage(vehicle,0)
        tm.ignore_walkers_percentage(vehicle,100)
        tm.distance_to_leading_vehicle(vehicle,5.0)
        tm.auto_lane_change(vehicle,True)
        tm.vehicle_percentage_speed_difference(vehicle,10)
        tm.keep_right_rule_percentage(vehicle,100)
        print("[VEHICLE]  Ego released")

        # Recording
        rec_path=os.path.join(RECORD_DIR,
            time.strftime("rtdeepiot_%Y%m%d_%H%M%S.avi"))
        vw=cv2.VideoWriter(rec_path,cv2.VideoWriter_fourcc(*'XVID'),
                           RECORD_FPS,(CAM_W,CAM_H))

        # CSV -- schema mirrors carla_camot_fail.py's column STYLE, but
        # with RTDeepIoT-specific fields (depth distribution, confidence,
        # labels) replacing CA-MOT's det/assoc workload/age fields.
        log_path=os.path.join(LOG_DIR,
            time.strftime("rtdeepiot_%Y%m%d_%H%M%S.csv"))
        csv_file=open(log_path,'w',newline='')
        csw=csv.writer(csv_file)
        csw.writerow([
            'elapsed_s','ego_speed_ms','n_visible','n_critical',
            'n_depth1','n_depth2','n_depth3',
            'avg_depth','sched_ms',
            'scheduler','brake',
            'ttc_values','depths','labels','confidences',
        ])

        spectator=world.get_spectator()
        cv2.namedWindow("CARLA - RTDeepIoT",cv2.WINDOW_AUTOSIZE)

        print("\n"+"="*65)
        print(f"  RTDeepIoT  |  {RTDEEPIOT_SCHEDULER}  |  {TOTAL_PEDS} pedestrians")
        print(f"  WCET stage1/2/3 ms: {WCET_ROAD[0]:.2f}/{WCET_ROAD[1]:.2f}/{WCET_ROAD[2]:.2f}")
        print(f"  Utility heuristic  : {UTILITY_HEURISTIC}")
        print(f"  NOTE: ignore_walkers=100%, brake hardcoded to 0 --")
        print(f"        this is a pure metrics-logging pass, matching")
        print(f"        carla_camot_fail.py exactly. No collision")
        print(f"        avoidance of any kind is active.")
        print("  Press Q to quit")
        print("="*65+"\n")

        start=time.time()
        last_record_time = time.time()
        ped_counter=TOTAL_PEDS+1; last_unstick=time.time()

        while True:
            now=time.time(); elapsed=now-start

            # Maintenance (no other vehicles to unstick)

            # Crossing trigger -- EXACT from carla_camot_fail.py
            for entry in pedestrians:
                if (entry['behaviour']=='crossing' and
                        not entry['cross_started'] and
                        now-entry['spawn_time']>1.0):
                    entry['cross_started']=True
                    dest=entry['cross_dest']
                    if dest is None:
                        dest=world.get_random_location_from_navigation()
                        entry['cross_dest']=dest
                    if dest:
                        entry['ctrl'].go_to_location(dest)
                        entry['ctrl'].set_max_speed(random.uniform(1.2,1.6))

            # Crossing done check -- EXACT from carla_camot_fail.py
            for entry in pedestrians:
                if (entry['behaviour']=='crossing' and
                        entry['cross_started'] and
                        not entry['cross_done']):
                    if now-entry['spawn_time']>3.0 and get_ped_speed(entry['ped'])<0.1:
                        entry['cross_done']=True
                        entry['ctrl'].set_max_speed(0.0)

            # Respawn far peds -- EXACT from carla_camot_fail.py
            gone=[]
            for entry in pedestrians:
                try:
                    if get_distance(vehicle,entry['ped'])>100.0: gone.append(entry)
                except: gone.append(entry)
            for entry in gone:
                destroy_ped(entry); pedestrians.remove(entry)
                respawn_single_ahead(world,bp_lib,vehicle,ped_counter,pedestrians)
                ped_counter+=1

            # Spectator
            vt=vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                vt.location+carla.Location(x=-10,z=5),
                carla.Rotation(pitch=-15,yaw=vt.rotation.yaw)))

            if latest_frame is None:
                time.sleep(0.05); continue

            frame=latest_frame.copy(); speed=get_vehicle_speed_ms(vehicle)

            # Build visible actor list (within 60m) -- EXACT threshold
            # from carla_camot_fail.py
            visible_actors=[]
            ttc_map={}
            for e in pedestrians:
                try:
                    ttc=compute_ttc(vehicle,e['ped'])
                    dist=get_distance(vehicle,e['ped'])
                    if dist<60.0:
                        visible_actors.append((e['ped'],'Ped',e.get('cross_done',False)))
                        ttc_map[e['ped'].id]=ttc
                except: pass

            if not visible_actors:
                _,display=run_detection(frame)
                hud=draw_hud(display,speed)
                if hud is not None:
                    cv2.imshow("CARLA - RTDeepIoT",hud)
                    if vw: vw.write(hud)
                if cv2.waitKey(1)&0xFF==ord('q'): break
                time.sleep(0.05); continue

            # Detect (full-frame YOLO, no L/M/H region cropping --
            # only classification depth is adaptive in RTDeepIoT)
            dets, display = run_detection(frame)

            # Match visible CARLA actors to their OWN YOLO detection via
            # world-to-pixel projection -- each pedestrian gets its own
            # crop, not a shared/duplicated placeholder box.
            visible_with_dets=[]
            for actor,atype,cdone in visible_actors:
                det=actor_to_det(actor, vehicle, dets)
                if det is not None:
                    visible_with_dets.append((actor,atype,det))

            if not visible_with_dets:
                hud=draw_hud(display,speed)
                if hud is not None:
                    cv2.imshow("CARLA - RTDeepIoT",hud)
                    if vw: vw.write(hud)
                if cv2.waitKey(1)&0xFF==ord('q'): break
                time.sleep(0.05); continue

            # ---- RTDeepIoT scheduling (replaces camot_schedule) ----
            t0=time.time()
            depth_assignment=run_rtdeepiot_schedule(visible_with_dets,ttc_map,frame)
            sched_ms=(time.time()-t0)*1000

            depth_counts={1:0,2:0,3:0}
            depths_log=[]; labels_log=[]; conf_log=[]; ttc_vals=[]
            n_critical=0

            for actor,atype,det in visible_with_dets:
                ttc=ttc_map.get(actor.id,999.0)
                ttc_vals.append(ttc)
                is_crit = ttc < 2.0  # informational count only, matching
                                     # carla_camot_fail.py's n_critical
                                     # style count (not used for gating,
                                     # since RTDeepIoT has no threshold)
                if is_crit: n_critical+=1

                depth=depth_assignment.get(actor.id,1)
                result=classify_crop(frame,det['box'],exit_after=depth)
                if result is None: continue

                depth_counts[depth]+=1
                depths_log.append(str(depth))
                labels_log.append(result['label'])
                conf_log.append(f"{result['confidence']:.3f}")

            avg_depth=sum(depth_counts[d]*d for d in (1,2,3))/max(1,len(visible_with_dets))

            # brake hardcoded to 0 -- EXACT match to carla_camot_fail.py.
            # No braking logic of any kind is implemented in this script.
            csw.writerow([
                f"{elapsed:.2f}",f"{speed:.2f}",
                len(visible_with_dets),n_critical,
                depth_counts[1],depth_counts[2],depth_counts[3],
                f"{avg_depth:.3f}",f"{sched_ms:.2f}",
                RTDEEPIOT_SCHEDULER,0,
                ';'.join(f"{t:.2f}" for t in ttc_vals),
                ';'.join(depths_log),';'.join(labels_log),';'.join(conf_log),
            ])
            csv_file.flush()

            hud=draw_hud(display,speed)
            if hud is not None:
                cv2.imshow("CARLA - RTDeepIoT",hud)
                if vw and (now - last_record_time) >= (1.0 / RECORD_FPS):
                    vw.write(hud)
                    last_record_time = now

            if cv2.waitKey(1)&0xFF==ord('q'):
                print("Quit."); break

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
        for e in pedestrians: destroy_ped(e)
        print(f"  [OK] {len(pedestrians)} peds destroyed")
        if vehicle:
            vehicle.set_autopilot(False); vehicle.destroy()
        print("Done.")


if __name__=='__main__':
    main()
