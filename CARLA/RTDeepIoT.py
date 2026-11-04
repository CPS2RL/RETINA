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
from typing import Optional, Dict

import torchvision.transforms as T

sys.path.insert(0, './yolov5')
sys.path.insert(0, '.')

from rtdeepiot.models.multi_exit_resnet import build_model
from rtdeepiot.scheduler.dp_scheduler import (
    Task, dp_schedule, predict_task_rewards, load_wcet_profile,
)
from rtdeepiot.data.road_crop_dataset import CLASS_NAMES as ROAD_CLASS_NAMES
from rtdeepiot.data.road_crop_dataset import NORM_MEAN, NORM_STD


# Experiment configuration. Distances are meters and speeds are m/s.
TM_PORT              = 8000
YOLO_CONF            = 0.4


# Requested scene composition; actual counts depend on spawn success.
TOTAL_PEDS           = 20
TOTAL_CARS           = 15
TOTAL_MOTORCYCLES    = 5
TOTAL_BICYCLES       = 5
TOTAL_BUSES          = 3
TOTAL_TRUCKS         = 5


# Pedestrian forward spawn offsets.
PED_SPREAD_MIN       = 10.0
PED_SPREAD_MAX       = 120.0

LOG_DIR              = "logs"
UNSTICK_INTERVAL     = 3.0
RUN_DURATION_S       = 100


# RTDeepIoT parameters: checkpoint, profiled stage times, utility model.
CHECKPOINT_PATH           = './rtdeepiot/checkpoints/best_road.pth'
WCET_PATH                 = './rtdeepiot/checkpoints/wcet_road.json'
UTILITY_HEURISTIC         = 'exp'
REWARD_QUANTIZATION_DELTA = 0.1     # best value in the paper's Fig. 12
CONFIDENCE_EMA_ALPHA      = 0.3
DEFAULT_SEED_CONFIDENCE   = 0.5
RTDEEPIOT_SCHEDULER       = 'RTDeepIoT-DP'


# Braking parameters (same rule and values as RETINA). Distances are meters.
BRAKE_DIST                = 20.0
CONF_THRESHOLD            = 0.60
# Classifier labels eligible to trigger prediction-based braking. Labels that
# are not in the classifier's class list are ignored (a warning is printed at
# startup). Edit this set to match the checkpoint's class names.
DANGER_BEHAVIORS          = {"MovTow", "Xing", "XingFmLft", "XingFmRht", "Stop"}


# Preferred CARLA blueprint IDs by actor category.
VEHICLE_BP_FILTERS = {
    "Car"    : ['vehicle.tesla.model3',
                'vehicle.audi.a2',
                'vehicle.bmw.grandtourer',
                'vehicle.chevrolet.impala',
                'vehicle.dodge.charger_2020',
                'vehicle.lincoln.mkz_2020',
                'vehicle.mercedes.coupe_2020',
                'vehicle.mini.cooper_s'],
    "Mobike" : ['vehicle.harley-davidson.low_rider',
                'vehicle.kawasaki.ninja',
                'vehicle.vespa.zx125',
                'vehicle.yamaha.yzf'],
    "Cyc"    : ['vehicle.bh.crossbike',
                'vehicle.diamondback.century',
                'vehicle.gazelle.omafiets'],
    "Bus"    : ['vehicle.mitsubishi.fusorosa',
                'vehicle.volkswagen.t2'],
    "LarVeh" : ['vehicle.carlamotors.carlacola',
                'vehicle.carlamotors.firetruck',
                'vehicle.ford.ambulance',
                'vehicle.mercedes.sprinter'],
}


# Map supported COCO class indices to the experiment actor vocabulary.
YOLO_CLASS_MAP = {
    0: "Ped",
    1: "Cyc",
    2: "Car",
    3: "Mobike",
    5: "Bus",
    7: "LarVeh",
}
YOLO_CLASSES = list(YOLO_CLASS_MAP.keys())


AGENT_COLORS = {
    "Ped"    : (0,   255, 127),
    "Car"    : (0,   128, 255),
    "Cyc"    : (255, 191,   0),
    "Mobike" : (255,   0, 200),
    "Bus"    : (0,   200, 255),
    "LarVeh" : (255,  80,   0),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE]   {DEVICE}")

os.makedirs(LOG_DIR, exist_ok=True)


# Shared state, updated by the camera callback and the main loop.
latest_frame: Optional[np.ndarray] = None

# actor_id -> running average of that actor's stage-1 confidence, used to seed
# the reward prediction for the next cycle.
actor_confidence_ema: Dict[int, float] = {}


# Load YOLOv5 from the local checkout and weights file.
print("[YOLO]     Loading YOLOv5n...")
yolo_model = torch.hub.load(
    './yolov5', 'custom', path='yolov5n.pt', source='local'
)
yolo_model.conf    = YOLO_CONF
yolo_model.classes = YOLO_CLASSES
print("[YOLO]     Ready")


# Load the multi-exit ResNet and the profiled cumulative stage times.
print("[RTDeepIoT] Loading multi-exit ResNet...")
rtdeepiot_model = build_model('road').to(DEVICE).eval()
_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
rtdeepiot_model.load_state_dict(_ckpt['model_state'])
print(f"[RTDeepIoT] Loaded checkpoint (epoch {_ckpt.get('epoch','?')}, "
      f"best_acc={_ckpt.get('best_acc','?')})")
print(f"[RTDeepIoT] Classifier classes: {list(ROAD_CLASS_NAMES)}")
_missing_danger = DANGER_BEHAVIORS - set(ROAD_CLASS_NAMES)
if _missing_danger == DANGER_BEHAVIORS:
    print("[WARN]     None of DANGER_BEHAVIORS are in the classifier's class list: "
          "prediction-based braking can never fire (proximity braking still works). "
          "Edit DANGER_BEHAVIORS to match the class names above.")
elif _missing_danger:
    print(f"[WARN]     DANGER_BEHAVIORS not in the classifier's class list "
          f"(can never trigger braking): {sorted(_missing_danger)}")
else:
    print(f"[RTDeepIoT] DANGER_BEHAVIORS {sorted(DANGER_BEHAVIORS)} all present in class list")

WCET_ROAD = load_wcet_profile(WCET_PATH)   # cumulative ms for stages 1, 2, 3
print(f"[RTDeepIoT] WCET (cumulative ms): "
      f"stage1={WCET_ROAD[0]:.3f}  stage2={WCET_ROAD[1]:.3f}  stage3={WCET_ROAD[2]:.3f}")


# Resize crops to the classifier input size and normalize.
classify_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])


def detect_agents_yolo(frame):
    """Return a frame copy and supported YOLO detections with pixel coordinates."""
    if frame is None:
        return None, []

    rgb        = frame[:, :, ::-1]
    results    = yolo_model(rgb.copy())
    annotated  = frame.copy()
    detections = []

    for *box, conf, cls in results.xyxy[0].tolist():
        cls_int = int(cls)
        if cls_int not in YOLO_CLASS_MAP:
            continue

        x1, y1, x2, y2 = map(int, box)
        detections.append({
            'box'        : (x1, y1, x2, y2),
            'confidence' : float(conf),
            'yolo_class' : cls_int,
            'agent_label': YOLO_CLASS_MAP[cls_int],
        })

    return annotated, detections


# Whether the model can run one stage at a time and resume from cached features.
# Expected interface (add it to multi_exit_resnet.py to enable true incremental
# execution): model.forward_stage(inp, stage_idx) -> (features, out_dict), where
# inp is the image tensor for stage_idx == 0 and the previous stage's features
# otherwise, and out_dict has 'predicted' and 'confidence' like the exit outputs.
STAGEWISE = hasattr(rtdeepiot_model, 'forward_stage')
print(f"[RTDeepIoT] Stage execution: "
      f"{'incremental (forward_stage)' if STAGEWISE else 'fallback (re-runs earlier stages; exposes no forward_stage)'}")

N_STAGES = len(WCET_ROAD)


def stage_time_ms(l: int) -> float:
    """Profiled execution time of stage l (1-based) alone, from cumulative WCETs."""
    return WCET_ROAD[l - 1] - (WCET_ROAD[l - 2] if l > 1 else 0.0)


class TaskState:
    """Runtime state of one actor's classification task within a cycle."""
    __slots__ = ('actor', 'atype', 'det', 'x', 'deadline_ms', 'rewards',
                 'assigned', 'done', 'feat', 'last_out')

    def __init__(self, actor, atype, det, x, deadline_ms):
        self.actor, self.atype, self.det, self.x = actor, atype, det, x
        self.deadline_ms = deadline_ms
        self.rewards = []        # rewards[d-1] = expected confidence at depth d
        self.assigned = 0        # planned depth
        self.done = 0            # stages executed
        self.feat = None         # cached features (incremental backend only)
        self.last_out = None     # latest exit output dict


def prepare_crop(frame_bgr: np.ndarray, box: tuple) -> Optional[torch.Tensor]:
    """Crop a detection and return the normalised input tensor, or None if empty."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    t = classify_transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return t.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def run_next_stage(st: TaskState) -> None:
    """Execute the next stage of a task and store its (label, confidence)."""
    k = st.done + 1
    if STAGEWISE:
        st.feat, out = rtdeepiot_model.forward_stage(
            st.x if st.feat is None else st.feat, k - 1)
    else:
        out = rtdeepiot_model(st.x, exit_after=k)[-1]
    st.last_out = {
        'label': ROAD_CLASS_NAMES[out['predicted'].item()],
        'confidence': out['confidence'].item(),
    }
    st.done = k


def stage_cost_ms(st: TaskState) -> float:
    """Time needed to run the task's next stage (cumulative if earlier stages re-run)."""
    k = st.done + 1
    return stage_time_ms(k) if STAGEWISE else WCET_ROAD[k - 1]


def future_rewards(conf: float, from_depth: int) -> list:
    """Rewards for depths from_depth..N_STAGES given the confidence observed at from_depth."""
    curve = predict_task_rewards(WCET_ROAD, seed_confidence=conf,
                                 heuristic=UTILITY_HEURISTIC)
    return [conf] + [curve[k] for k in range(1, N_STAGES - from_depth + 1)]


def plan_is_feasible(order: list, now_ms: float) -> bool:
    """Check that the planned remaining stages meet every deadline when run in EDF order."""
    t = now_ms
    for st in order:
        if st.assigned > st.done:
            t += sum(stage_time_ms(l) for l in range(st.done + 1, st.assigned + 1))
            if t > st.deadline_ms:
                return False
    return True


def greedy_reassign(cur: TaskState, order: list, now_ms: float) -> None:
    """Eq. (7): trade the current task's remaining planned stages for the other
    task's extra stages with the largest reward gain that fit in the freed time."""
    if cur.assigned <= cur.done:
        return
    freed = sum(stage_time_ms(l) for l in range(cur.done + 1, cur.assigned + 1))
    loss = cur.rewards[cur.assigned - 1] - cur.rewards[cur.done - 1]

    best, best_gain, best_depth = None, 0.0, 0
    for t in order:
        if t is cur:
            continue
        for l in range(t.assigned + 1, N_STAGES + 1):
            cost = sum(stage_time_ms(j) for j in range(t.assigned + 1, l + 1))
            if cost > freed:
                break
            gain = t.rewards[l - 1] - t.rewards[t.assigned - 1]
            if gain > best_gain:
                best, best_gain, best_depth = t, gain, l

    if best is None or best_gain <= loss:
        return
    old_cur, old_best = cur.assigned, best.assigned
    cur.assigned, best.assigned = cur.done, best_depth
    if not plan_is_feasible(order, now_ms):
        cur.assigned, best.assigned = old_cur, old_best


def run_rtdeepiot_cycle(visible_with_dets: list,
                        ttc_map: Dict[int, float],
                        frame: np.ndarray) -> tuple:
    """Schedule and execute the stage-wise classification tasks of one cycle.

    1. Admission: tasks whose mandatory stage cannot meet the deadline (TTC)
       in EDF order are dropped and get no result (deadline miss).
    2. Depths are planned once with the DP (Algorithm 1) from reward predictions
       seeded by each actor's confidence EMA.
    3. Stages run one at a time in EDF order. After every stage the utility
       estimate is updated from the observed confidence, the greedy rule
       (Eq. 7) may reassign depth, and a stage is only started if it can
       finish before the task's deadline.

    Returns (results, n_dropped); results maps actor_id -> {'label',
    'confidence', 'exit_after'} for tasks that completed at least one stage.
    """
    t0 = time.perf_counter()
    elapsed_ms = lambda: (time.perf_counter() - t0) * 1000.0

    states = []
    for actor, atype, det in visible_with_dets:
        x = prepare_crop(frame, det['box'])
        if x is None:
            continue
        states.append(TaskState(actor, atype, det, x,
                                ttc_map.get(actor.id, 999.0) * 1000.0))
    states.sort(key=lambda s: s.deadline_ms)          # EDF order

    # Admission control on the mandatory stage (no deadline floor).
    admitted, n_dropped, cum = [], 0, 0.0
    for st in states:
        if cum + WCET_ROAD[0] <= st.deadline_ms:
            cum += WCET_ROAD[0]
            admitted.append(st)
        else:
            n_dropped += 1
    states = admitted
    if not states:
        return {}, n_dropped

    # Plan depths with the DP.
    tasks = []
    for st in states:
        seed = actor_confidence_ema.get(st.actor.id, DEFAULT_SEED_CONFIDENCE)
        st.rewards = list(predict_task_rewards(
            WCET_ROAD, seed_confidence=seed, heuristic=UTILITY_HEURISTIC))
        tasks.append(Task(
            job_id=str(st.actor.id), deadline_ms=st.deadline_ms,
            omega=1, max_depth=N_STAGES,
            predicted_rewards=st.rewards, exec_times=WCET_ROAD,
        ))
    plan = dp_schedule(tasks, delta=REWARD_QUANTIZATION_DELTA)
    for st in states:
        st.assigned = int(plan.get(str(st.actor.id), 0))
    # A task the DP could not fit gets no stage at all.
    n_dropped += sum(1 for st in states if st.assigned < 1)
    states = [st for st in states if st.assigned >= 1]

    # Execute stage by stage in EDF order.
    for st in states:
        while st.done < st.assigned:
            if elapsed_ms() + stage_cost_ms(st) > st.deadline_ms:
                if st.done == 0:
                    n_dropped += 1                 # mandatory stage missed at runtime
                break
            run_next_stage(st)

            conf = st.last_out['confidence']
            st.rewards[st.done - 1:] = future_rewards(conf, st.done)
            if st.done == 1:
                prev = actor_confidence_ema.get(st.actor.id, conf)
                actor_confidence_ema[st.actor.id] = (
                    CONFIDENCE_EMA_ALPHA * conf
                    + (1 - CONFIDENCE_EMA_ALPHA) * prev)
            greedy_reassign(st, states, elapsed_ms())

    results = {}
    for st in states:
        if st.done >= 1:
            results[st.actor.id] = dict(st.last_out, exit_after=st.done)
    return results, n_dropped


def get_distance(actor1, actor2):
    """Return Euclidean distance between actor origins in meters."""
    l1 = actor1.get_location()
    l2 = actor2.get_location()
    return math.sqrt((l1.x-l2.x)**2 + (l1.y-l2.y)**2 + (l1.z-l2.z)**2)


def get_vehicle_speed_ms(vehicle):
    """Return the magnitude of actor velocity in meters per second."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def get_ped_speed(ped):
    """Return the magnitude of pedestrian velocity in meters per second."""
    v = ped.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def camera_callback(image):
    """Copy the latest CARLA BGRA image into the shared BGR frame buffer."""
    global latest_frame
    array        = np.frombuffer(image.raw_data, dtype=np.uint8)
    array        = array.reshape((image.height, image.width, 4))
    latest_frame = array[:, :, :3].copy()


def world_to_camera_pixel(world_loc, vehicle,
                           cam_w=800, cam_h=600, fov=90):
    """Project a world location using the ego yaw and fixed camera offsets.
    
    This approximation ignores pitch and roll. Return None for points behind
    the camera or outside the image; defaults match the episode RGB sensor."""
    vt      = vehicle.get_transform()
    yaw_rad = math.radians(vt.rotation.yaw)
    cam_x   = vt.location.x + math.cos(yaw_rad) * 2.5
    cam_y   = vt.location.y + math.sin(yaw_rad) * 2.5
    cam_z   = vt.location.z + 1.5
    dx = world_loc.x - cam_x
    dy = world_loc.y - cam_y
    dz = world_loc.z - cam_z
    fwd_x   =  math.cos(yaw_rad)
    fwd_y   =  math.sin(yaw_rad)
    right_x =  math.sin(yaw_rad)
    right_y = -math.cos(yaw_rad)
    forward = dx * fwd_x + dy * fwd_y
    right   = dx * right_x + dy * right_y
    up      = dz
    if forward <= 0.1:
        return None
    f = (cam_w / 2.0) / math.tan(math.radians(fov / 2.0))
    u = int(cam_w / 2 - (right / forward) * f)
    v = int(cam_h / 2 - (up    / forward) * f)
    if u < 0 or u >= cam_w or v < 0 or v >= cam_h:
        return None
    return (u, v)


def is_agent_in_detection(agent, vehicle, detections,
                           pixel_threshold=80):
    """Match an actor's projected origin to the first expanded detection box.
    
    Association uses CARLA actor positions, without class matching or a
    one-to-one assignment constraint."""
    if not detections:
        return False, None
    proj = world_to_camera_pixel(agent.get_location(), vehicle)
    if proj is None:
        return False, None
    u, v = proj
    for det in detections:
        x1, y1, x2, y2 = det['box']
        x1 -= pixel_threshold // 4
        y1 -= pixel_threshold // 4
        x2 += pixel_threshold // 4
        y2 += pixel_threshold // 4
        if x1 <= u <= x2 and y1 <= v <= y2:
            return True, det
    return False, None


def is_on_road(world, location):
    """Check whether a location lies on a driving lane without road projection."""
    carla_map = world.get_map()
    waypoint  = carla_map.get_waypoint(
        location, project_to_road=False,
        lane_type=carla.LaneType.Any
    )
    if waypoint is None:
        return False
    return waypoint.lane_type == carla.LaneType.Driving


def get_sidewalk_nav_location(world, attempts=50):
    """Sample a navigation location outside driving lanes, or return None."""
    for _ in range(attempts):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if not is_on_road(world, loc):
            return loc
    return None


def destroy_ped(entry):
    """Stop and destroy a pedestrian and controller, then remove cached state."""
    try:
        entry['ctrl'].stop()
        entry['ctrl'].destroy()
        pid = entry['ped'].id
        entry['ped'].destroy()
        actor_confidence_ema.pop(pid, None)
    except Exception:
        pass


def spawn_single_ped(world, bp_lib, vehicle,
                     fwd_dist, behaviour, ped_index):
    """Spawn a pedestrian near a forward target and configure its walker controller."""
    vt      = vehicle.get_transform()
    yaw_rad = math.radians(vt.rotation.yaw)
    target  = carla.Location(
        x = vt.location.x + math.cos(yaw_rad) * fwd_dist,
        y = vt.location.y + math.sin(yaw_rad) * fwd_dist,
        z = vt.location.z
    )


    # Choose the sampled non-driving navigation point closest to the forward target.
    best_loc  = None
    best_dist = float('inf')
    for _ in range(80):
        nav_loc = world.get_random_location_from_navigation()
        if nav_loc is None:
            continue
        if is_on_road(world, nav_loc):
            continue
        d = math.sqrt(
            (nav_loc.x-target.x)**2 + (nav_loc.y-target.y)**2
        )
        if d < best_dist:
            best_dist = d
            best_loc  = nav_loc

    if best_loc is None:
        return None

    spawn_loc = carla.Location(x=best_loc.x, y=best_loc.y,
                               z=best_loc.z + 0.5)
    if is_on_road(world, spawn_loc):
        return None

    ped_bp  = random.choice(bp_lib.filter('walker.pedestrian.*'))
    ctrl_bp = bp_lib.find('controller.ai.walker')

    ped = world.try_spawn_actor(ped_bp, carla.Transform(spawn_loc))
    if ped is None:
        spawn_loc.z += 0.5
        ped = world.try_spawn_actor(ped_bp, carla.Transform(spawn_loc))
    if ped is None:
        return None


    # Advance the world before starting the newly attached walker controller.
    ctrl = world.spawn_actor(ctrl_bp, carla.Transform(),
                             attach_to=ped)
    world.tick()
    world.tick()
    world.tick()
    ctrl.start()

    dest_1 = get_sidewalk_nav_location(world)
    dest_2 = get_sidewalk_nav_location(world)

    cross_dest    = None
    cross_started = False
    cross_done    = False

    if behaviour == 'standing':
        ctrl.set_max_speed(0.0)
        if dest_1:
            ctrl.go_to_location(dest_1)
    elif behaviour == 'walking':
        ctrl.set_max_speed(random.uniform(0.8, 1.2))
        if dest_1:
            ctrl.go_to_location(dest_1)
    elif behaviour == 'crossing':
        ctrl.set_max_speed(0.0)
        cross_dest = world.get_random_location_from_navigation()
    elif behaviour == 'crossed':
        ctrl.set_max_speed(0.5)
        if dest_1:
            ctrl.go_to_location(dest_1)
        cross_done = True

    print(f"[PED {ped_index:02d}]  behaviour={behaviour:10s} | "
          f"fwd={fwd_dist:.0f}m | id={ped.id}")

    return {
        'ped'          : ped,
        'ctrl'         : ctrl,
        'behaviour'    : behaviour,
        'cross_dest'   : cross_dest,
        'cross_started': cross_started,
        'cross_done'   : cross_done,
        'spawn_time'   : time.time(),
        'nav_dest_2'   : dest_2,
    }


def spawn_all_pedestrians(world, bp_lib, vehicle):
    """Spawn TOTAL_PEDS pedestrians with a 40% standing, 40% walking, 20% crossing mix."""
    pedestrians   = []
    fwd_distances = sorted([
        random.uniform(PED_SPREAD_MIN, PED_SPREAD_MAX)
        for _ in range(TOTAL_PEDS)
    ])
    n_crossing = TOTAL_PEDS // 5
    n_standing = (TOTAL_PEDS * 2) // 5
    n_walking  = TOTAL_PEDS - n_crossing - n_standing
    behaviours = (
        ['standing'] * n_standing + ['walking'] * n_walking +
        ['crossing'] * n_crossing
    )
    random.shuffle(behaviours)
    print(f"\n[SPAWN]    Spawning {TOTAL_PEDS} pedestrians")
    for i, (fwd, beh) in enumerate(zip(fwd_distances, behaviours)):
        entry = spawn_single_ped(world, bp_lib, vehicle,
                                 fwd, beh, i + 1)
        if entry is not None:
            pedestrians.append(entry)
    print(f"[SPAWN]    {len(pedestrians)} pedestrians spawned")
    return pedestrians


def spawn_all_vehicles(world, bp_lib, vehicle, tm):
    """Spawn traffic actors on available road points and enable autopilot."""
    spawned_vehicles = []
    carla_map        = world.get_map()


    all_spawn_pts  = carla_map.get_spawn_points()
    road_spawn_pts = []
    for sp in all_spawn_pts:
        wp = carla_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if wp is None:
            continue

        next_wps = wp.next(20.0)
        if not next_wps:
            continue

        if wp.is_junction:
            continue

        road_spawn_pts.append(sp)


    ego_loc = vehicle.get_location()
    road_spawn_pts = [
        sp for sp in road_spawn_pts
        if math.sqrt(
            (sp.location.x - ego_loc.x)**2 +
            (sp.location.y - ego_loc.y)**2
        ) > 15.0
    ]

    random.shuffle(road_spawn_pts)
    pt_idx = 0


    spawn_plan = [
        ("Car",    TOTAL_CARS),
        ("Mobike", TOTAL_MOTORCYCLES),
        ("Cyc",    TOTAL_BICYCLES),
        ("Bus",    TOTAL_BUSES),
        ("LarVeh", TOTAL_TRUCKS),
    ]

    print(f"\n[SPAWN]    Spawning road vehicles "
          f"({len(road_spawn_pts)} valid road points)...")

    for agent_type, count in spawn_plan:
        bp_names = VEHICLE_BP_FILTERS.get(agent_type, [])
        spawned  = 0

        for _ in range(count):
            if pt_idx >= len(road_spawn_pts):
                print(f"  [WARN]  No more road spawn points for {agent_type}")
                break

            bp_candidates = []
            for name in bp_names:
                try:
                    bp = bp_lib.find(name)
                    if bp is not None:
                        bp_candidates.append(bp)
                except Exception:
                    pass


            # Fall back to category filters when preferred blueprint IDs are unavailable.
            if not bp_candidates:
                fallback = {
                    "Cyc"    : 'vehicle.*bicycle*',
                    "Mobike" : 'vehicle.*moto*',
                    "Bus"    : 'vehicle.*bus*',
                    "LarVeh" : 'vehicle.*truck*',
                }.get(agent_type, 'vehicle.tesla.*')
                bp_candidates = list(bp_lib.filter(fallback))

            if not bp_candidates:
                continue

            bp = random.choice(bp_candidates)
            sp = road_spawn_pts[pt_idx]
            pt_idx += 1

            actor = world.try_spawn_actor(bp, sp)


            attempts = 0
            while actor is None and attempts < 3 and pt_idx < len(road_spawn_pts):
                sp     = road_spawn_pts[pt_idx]
                pt_idx += 1
                actor  = world.try_spawn_actor(bp, sp)
                attempts += 1

            if actor is not None:
                actor.set_autopilot(True, TM_PORT)


                tm.ignore_lights_percentage(actor, 0)
                tm.ignore_signs_percentage(actor, 0)
                tm.ignore_vehicles_percentage(actor, 0)
                tm.distance_to_leading_vehicle(actor, 3.0)
                tm.vehicle_percentage_speed_difference(actor, 10)
                tm.auto_lane_change(actor, True)
                tm.vehicle_percentage_speed_difference(
                    actor, random.uniform(0, 20)
                )


                actor.apply_control(
                    carla.VehicleControl(throttle=0.6, brake=0.0)
                )

                spawned_vehicles.append({
                    'actor'      : actor,
                    'agent_type' : agent_type,
                    'spawn_time' : time.time(),
                    'last_moved' : time.time(),
                })
                spawned += 1
                print(f"  [OK]  {agent_type:8s} id={actor.id}  "
                      f"bp={bp.id[:35]}")

        print(f"  {agent_type:8s}: {spawned}/{count} spawned")

    print(f"[SPAWN]    {len(spawned_vehicles)} road vehicles spawned")
    print(f"[SPAWN]    Waiting for TM to initialise all vehicles...")
    # Allow Traffic Manager initialization before entering the episode loop.
    time.sleep(2.0)
    print(f"[SPAWN]    All vehicles should be moving\n")
    return spawned_vehicles


def unstick_vehicles(road_vehicles, vehicle, tm, world):
    """Nudge idle traffic actors unless at a junction or a red/yellow signal."""
    now       = time.time()
    carla_map = world.get_map()

    for rv in road_vehicles:
        try:
            actor = rv['actor']
            spd   = get_vehicle_speed_ms(actor)

            if spd > 0.5:
                rv['last_moved'] = now
                rv['at_light']   = False
                continue

            wp = carla_map.get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            at_junction = (wp is not None and wp.is_junction)

            tl_state = None
            try:
                tl = actor.get_traffic_light()
                if tl is not None:
                    tl_state = tl.get_state()
            except Exception:
                pass

            at_red = (tl_state == carla.TrafficLightState.Red or
                      tl_state == carla.TrafficLightState.Yellow)

            if at_junction or at_red:
                rv['last_moved'] = now
                rv['at_light']   = True
                continue

            rv['at_light'] = False
            last_moved   = rv.get('last_moved', rv['spawn_time'])
            stopped_for  = now - last_moved

            if stopped_for > 8.0:
                actor.set_autopilot(False, TM_PORT)
                actor.apply_control(
                    carla.VehicleControl(throttle=0.7, brake=0.0)
                )
                time.sleep(0.15)
                actor.set_autopilot(True, TM_PORT)
                tm.ignore_lights_percentage(actor, 0)
                tm.ignore_signs_percentage(actor, 0)
                tm.ignore_vehicles_percentage(actor, 0)
                tm.distance_to_leading_vehicle(actor, 3.0)
                tm.auto_lane_change(actor, True)
                rv['last_moved'] = now
                print(f"  [UNSTICK] id={actor.id} {rv['agent_type']} "
                      f"stuck {stopped_for:.0f}s -- nudged")

        except Exception:
            pass


def respawn_single_ahead(world, bp_lib, vehicle,
                         ped_index, pedestrians):
    """Try to spawn a replacement pedestrian ahead and append it to the scene."""
    fwd_dist  = random.uniform(60.0, 100.0)
    behaviour = random.choice(
        ['standing', 'walking', 'crossing', 'crossing', 'standing']
    )
    entry = spawn_single_ped(world, bp_lib, vehicle,
                             fwd_dist, behaviour, ped_index)
    if entry is not None:
        pedestrians.append(entry)
        print(f"[RESPAWN]  New ped {fwd_dist:.0f}m ahead | {behaviour}")


def draw_hud(frame, speed, brake_active=False):
    """Speed text (green on black, top-left) plus an optional braking banner."""
    if frame is None:
        return None
    d = frame.copy()
    txt = f"Speed: {speed:.2f} m/s"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(d, (8, 8), (14 + tw, 22 + th), (0, 0, 0), -1)
    cv2.putText(d, txt, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    if brake_active:
        # Same style as RETINA_last: green text on black, bottom-centre.
        h, w = d.shape[:2]
        banner = "BRAKE PRESSED!"
        (bw, bh), bbase = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        bpad = 10
        bx = (w - bw) // 2 - bpad
        by = h - bh - 2 * bpad - bbase
        cv2.rectangle(d, (bx, by), (bx + bw + 2 * bpad, h - bpad),
                      (0, 0, 0), -1)
        cv2.putText(d, banner, (bx + bpad, h - bpad - bbase - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return d


def draw_confidence_on_frame(display, det, label, confidence, is_danger=False):
    """Draw the detection box with "<label><confidence>" (e.g. Ped0.71) above it.

    The confidence is the RTDeepIoT classifier confidence at the exit stage
    that was run. With confidence=None only the box is drawn."""
    if display is None or det is None:
        return display
    x1, y1, x2, y2 = det['box']
    col = (0, 0, 255) if is_danger else AGENT_COLORS["Ped"]
    cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
    if confidence is None:
        return display
    cv2.putText(display, f"{label}{confidence:.2f}",
                (x1, max(y1 - 6, 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return display


def run_episode(duration_sec: float = RUN_DURATION_S) -> bool:
    """Run an episode until timeout, collision, user quit, or interruption.

    Connect to CARLA on localhost:2000 and release spawned actors in the
    finally block. Return whether a collision was observed; False also
    includes early user exits and does not imply a completed episode."""

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world     = client.get_world()
    bp_lib    = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points()

    world.set_weather(carla.WeatherParameters.ClearNoon)
    print(f"[WORLD]    Map : {world.get_map().name}  (single run)")

    vehicle       = None
    pedestrians   = []
    road_vehicles = []
    camera        = None
    col_sensor    = None


    # Mutable flag shared with the collision callback; count at most one crash per run.
    crashed = [False]

    def on_collision(event):
        if crashed[0]:
            return
        crashed[0] = True
        print(f"\n  !! CRASH -- run ending "
              f"(hit {event.other_actor.type_id})\n")

    try:
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        v_spawn    = random.choice(spawn_pts)
        vehicle    = world.spawn_actor(vehicle_bp, v_spawn)
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0)
        )
        print(f"[VEHICLE]  Spawned at "
              f"({v_spawn.location.x:.2f},{v_spawn.location.y:.2f})")


        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '90')
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=2.5, z=1.5)),
            attach_to=vehicle
        )
        camera.listen(camera_callback)
        print("[CAMERA]   Attached")


        col_bp     = bp_lib.find('sensor.other.collision')
        col_sensor = world.spawn_actor(
            col_bp, carla.Transform(), attach_to=vehicle
        )
        col_sensor.listen(on_collision)


        tm = client.get_trafficmanager(TM_PORT)
        tm.set_synchronous_mode(False)
        # Seed Traffic Manager only; Python random scene sampling remains unseeded.
        tm.set_random_device_seed(42)

        tm.set_global_distance_to_leading_vehicle(3.0)
        tm.global_percentage_speed_difference(10.0)


        pedestrians      = spawn_all_pedestrians(world, bp_lib, vehicle)
        road_vehicles    = spawn_all_vehicles(world, bp_lib, vehicle, tm)


        time.sleep(1.0)
        vehicle.set_autopilot(True, TM_PORT)
        tm.ignore_lights_percentage(vehicle, 0)
        tm.ignore_signs_percentage(vehicle, 0)
        tm.ignore_vehicles_percentage(vehicle, 0)
        tm.distance_to_leading_vehicle(vehicle, 5.0)
        tm.auto_lane_change(vehicle, False)
        tm.vehicle_percentage_speed_difference(vehicle, 20)
        print("[VEHICLE]  Ego vehicle released")

        spectator = world.get_spectator()
        cv2.namedWindow("CARLA - RTDeepIoT", cv2.WINDOW_AUTOSIZE)

        print("\n" + "="*65)
        print(f"  SIMULATION RUNNING — urban scene")
        print(f"  Pedestrians      : {TOTAL_PEDS}")
        print(f"  Cars             : {TOTAL_CARS}")
        print(f"  Motorcycles      : {TOTAL_MOTORCYCLES}")
        print(f"  Cyclists         : {TOTAL_BICYCLES}")
        print(f"  Buses            : {TOTAL_BUSES}")
        print(f"  Large vehicles   : {TOTAL_TRUCKS}")
        print(f"  Duration cap     : {duration_sec:.0f}s (or until crash)")
        print(f"  Scheduler        : {RTDEEPIOT_SCHEDULER}")
        print(f"  WCET stage1/2/3  : {WCET_ROAD[0]:.2f}/{WCET_ROAD[1]:.2f}/{WCET_ROAD[2]:.2f} ms")
        print(f"  Utility heuristic: {UTILITY_HEURISTIC}")
        print("  Press Q to quit")
        print("="*65 + "\n")

        start_time         = time.time()
        ped_counter        = TOTAL_PEDS + 1
        last_unstick_check = time.time()
        was_braking        = False
        brake_hold         = False
        brake_hold_start_time = None
        # actor_id -> last predicted class label (used for the brake-release check).
        agent_predictions  = {}


        # Process the latest available image; the loop does not wait for a new camera frame.
        while True:
            now     = time.time()
            elapsed = now - start_time

            if elapsed >= duration_sec:
                print(f"\n[RUN] Duration reached "
                      f"({duration_sec:.0f}s) -- ending episode.")
                break

            if crashed[0]:
                print(f"\n[RUN] Crash detected -- "
                      f"ending episode early at {elapsed:.1f}s.")
                break


            # Recover idle traffic periodically while preserving traffic-light/junction stops.
            if now - last_unstick_check > UNSTICK_INTERVAL:
                last_unstick_check = now
                unstick_vehicles(road_vehicles, vehicle, tm, world)


            for entry in pedestrians:
                if (entry['behaviour'] == 'crossing' and
                        not entry['cross_started'] and
                        now - entry['spawn_time'] > 1.0):
                    entry['cross_started'] = True
                    dest = entry['cross_dest']
                    if dest is None:
                        dest = world.get_random_location_from_navigation()
                        entry['cross_dest'] = dest
                    if dest:
                        entry['ctrl'].go_to_location(dest)
                        entry['ctrl'].set_max_speed(
                            random.uniform(1.2, 1.6)
                        )


            for entry in pedestrians:
                if (entry['behaviour'] == 'crossing' and
                        entry['cross_started'] and
                        not entry['cross_done']):
                    spd        = get_ped_speed(entry['ped'])
                    cross_time = now - entry['spawn_time']
                    if (cross_time > 3.0 and spd < 0.1):
                        entry['cross_done'] = True
                        entry['ctrl'].set_max_speed(0.0)


            to_remove = []
            for entry in pedestrians:
                try:
                    d = get_distance(vehicle, entry['ped'])
                    if d > 100.0:
                        to_remove.append(entry)
                except Exception:
                    to_remove.append(entry)

            for entry in to_remove:
                destroy_ped(entry)
                pedestrians.remove(entry)
                respawn_single_ahead(world, bp_lib, vehicle,
                                     ped_counter, pedestrians)
                ped_counter += 1


            vt = vehicle.get_transform()
            spectator.set_transform(
                carla.Transform(
                    vt.location + carla.Location(x=-10, z=5),
                    carla.Rotation(pitch=-15, yaw=vt.rotation.yaw)
                )
            )

            speed_ms = get_vehicle_speed_ms(vehicle)

            frame         = latest_frame
            display_frame = None
            detections    = []


            # Perception: detect objects in the latest frame.
            if frame is not None:
                display_frame, detections = detect_agents_yolo(frame)


            # Associate each simulator actor with a detection using projected positions.
            visible_agents = []

            for entry in pedestrians:
                try:
                    found, det = is_agent_in_detection(
                        entry['ped'], vehicle, detections
                    )
                    if found:
                        visible_agents.append({
                            'actor'      : entry['ped'],
                            'agent_type' : 'Ped',
                            'det'        : det,
                            'cross_done' : entry['cross_done'],
                        })
                except Exception:
                    pass

            for rv in road_vehicles:
                try:
                    found, det = is_agent_in_detection(
                        rv['actor'], vehicle, detections
                    )
                    if found:
                        visible_agents.append({
                            'actor'      : rv['actor'],
                            'agent_type' : rv['agent_type'],
                            'det'        : det,
                            'cross_done' : False,
                        })
                except Exception:
                    pass


            # Safety analysis: heuristic TTC from simulator distance and speed magnitudes,
            # not a trajectory-intersection or full relative-velocity calculation.
            raw_spd = get_vehicle_speed_ms(vehicle)
            car_spd = raw_spd if raw_spd > 0.5 else 1.0
            ttc_map = {}
            for va in visible_agents:
                dist = get_distance(vehicle, va['actor'])
                try:
                    agent_spd = get_vehicle_speed_ms(va['actor'])
                except Exception:
                    try:
                        agent_spd = get_ped_speed(va['actor'])
                    except Exception:
                        agent_spd = 0.0

                ego_tf   = vehicle.get_transform()
                ego_fwd  = ego_tf.get_forward_vector()
                to_agent = va['actor'].get_location() - ego_tf.location
                dot      = (ego_fwd.x * to_agent.x
                            + ego_fwd.y * to_agent.y)
                in_front = dot > -0.3

                if in_front:
                    closing_spd = max(car_spd - agent_spd * 0.5, 0.5)
                    ttc         = dist / closing_spd
                else:
                    ttc = dist / car_spd

                ttc_map[va['actor'].id] = ttc


            brake_active  = False

            # Proximity braking (same rule as RETINA): any visible agent in the
            # ego lane within BRAKE_DIST, or closing with TTC < 2.5 s.
            proximity_brake = False
            ego_tf          = vehicle.get_transform()
            ego_fwd         = ego_tf.get_forward_vector()

            for va in visible_agents:
                try:
                    dist     = get_distance(vehicle, va['actor'])
                    to_agent = va['actor'].get_location() - ego_tf.location
                    if math.sqrt(to_agent.x**2 + to_agent.y**2) < 0.1:
                        continue

                    dot_fwd = ego_fwd.x * to_agent.x + ego_fwd.y * to_agent.y
                    dot_lat = abs(ego_fwd.y * to_agent.x - ego_fwd.x * to_agent.y)
                    if dot_fwd <= 0 or dot_lat >= 2.5:
                        continue

                    if dist < BRAKE_DIST:
                        proximity_brake = True
                        break

                    try:
                        agent_spd = get_vehicle_speed_ms(va['actor'])
                    except Exception:
                        try:
                            agent_spd = get_ped_speed(va['actor'])
                        except Exception:
                            agent_spd = 0.0

                    closing_spd = raw_spd - agent_spd
                    if closing_spd > 0.5 and dist / closing_spd < 2.5:
                        proximity_brake = True
                        print(f"  [BRAKE]  Closing too fast: "
                              f"ego={raw_spd:.1f} agent={agent_spd:.1f} "
                              f"ttc={dist / closing_spd:.1f}s")
                        break
                except Exception:
                    pass

            # Image-space proximity fallback: large, central detection box.
            if not proximity_brake and detections and frame is not None:
                frame_h, frame_w = frame.shape[:2]
                frame_area = frame_h * frame_w
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    bbox_area = (x2 - x1) * (y2 - y1)
                    center_x  = (x1 + x2) / 2
                    in_lane_x = frame_w * 0.25 < center_x < frame_w * 0.75
                    if in_lane_x and (
                        (y2 > frame_h * 0.30 and bbox_area > frame_area * 0.04) or
                        bbox_area > frame_area * 0.10
                    ):
                        proximity_brake = True
                        break

            if proximity_brake:
                brake_active = True
                print("  [BRAKE]  Object detected in front")


            # Scheduling and prediction: choose an exit depth per task, run it,
            # and draw the confidence overlay.
            if visible_agents and frame is not None:
                visible_with_dets = [
                    (va['actor'], va['agent_type'], va['det'])
                    for va in visible_agents
                ]
                results, n_dropped = run_rtdeepiot_cycle(
                    visible_with_dets, ttc_map, frame
                )
                if n_dropped:
                    print(f"  [SCHED]  {n_dropped} task(s) dropped "
                          f"(deadline missed)")

                for actor, atype, det in visible_with_dets:
                    result = results.get(actor.id)
                    if result is None:
                        display_frame = draw_confidence_on_frame(
                            display_frame, det, atype, None)
                        continue

                    agent_predictions[actor.id] = result['label']

                    # Prediction-based braking (same rule as RETINA): dangerous
                    # label, within BRAKE_DIST, confident, and not a pedestrian
                    # that has finished crossing.
                    va_cross_done = next(
                        (v['cross_done'] for v in visible_agents
                         if v['actor'].id == actor.id), False)
                    is_danger = (
                        result['label'] in DANGER_BEHAVIORS and
                        get_distance(vehicle, actor) < BRAKE_DIST and
                        result['confidence'] >= CONF_THRESHOLD and
                        not va_cross_done
                    )
                    if is_danger:
                        brake_active = True

                    display_frame = draw_confidence_on_frame(
                        display_frame, det, atype, result['confidence'],
                        is_danger)


            if brake_active and not was_braking:
                print("\n  *** VEHICLE BRAKING ***")
            if not brake_active and was_braking:
                print("\n  *** VEHICLE RESUMING ***")
            was_braking = brake_active

            # Latch the brake; release once no visible actor has a cached
            # dangerous label (distance/confidence are not rechecked here).
            if brake_active:
                if not brake_hold:
                    brake_hold_start_time = time.time()
                brake_hold = True
            elif brake_hold:
                still_danger = any(
                    agent_predictions.get(va['actor'].id, "") in DANGER_BEHAVIORS
                    and not va['cross_done']
                    for va in visible_agents
                )
                if not still_danger:
                    brake_hold            = False
                    brake_hold_start_time = None

            # Override the autopilot with full braking while the latch is on.
            if brake_hold:
                vehicle.set_autopilot(False, TM_PORT)
                vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                )
            else:
                vehicle.set_autopilot(True, TM_PORT)

            show_brake_banner = (
                brake_hold and brake_hold_start_time is not None and
                (time.time() - brake_hold_start_time) >= 1.0
            )

            if display_frame is not None:
                hud = draw_hud(display_frame, speed_ms, show_brake_banner)
                if hud is not None:
                    cv2.imshow("CARLA - RTDeepIoT", hud)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\n[RUN] Ended early by user.")
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping episode...")


    # Attempt sensor and actor cleanup on normal exit, interruption, or error.
    finally:
        print(f"[RUN] Cleaning up...  "
              f"(crashed: {'YES' if crashed[0] else 'no'})")
        cv2.destroyAllWindows()

        if camera:
            camera.stop()
            camera.destroy()

        if col_sensor:
            col_sensor.stop()
            col_sensor.destroy()

        for entry in pedestrians:
            destroy_ped(entry)

        for rv in road_vehicles:
            try:
                rv['actor'].set_autopilot(False)
                rv['actor'].destroy()
            except Exception:
                pass

        if vehicle:
            vehicle.set_autopilot(False)
            vehicle.destroy()

    return crashed[0]


def run_single():
    """Run one episode and append its collision flag to logs/RTDeepIoT/rtdeepiot_runs.csv."""
    crashed = run_episode(duration_sec=RUN_DURATION_S)

    summary_dir = os.path.join(LOG_DIR, "RTDeepIoT")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "rtdeepiot_runs.csv")
    is_new = not os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['timestamp', 'crashed'])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), int(crashed)])

    print(f"\n[RESULT]   {'CRASHED' if crashed else 'safe'}  "
          f"-> appended to {summary_path}")
    return crashed


if __name__ == '__main__':
    run_single()
