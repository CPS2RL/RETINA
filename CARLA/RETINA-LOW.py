
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
import json
from datetime import datetime
from torchvision import models
from torchvision import transforms
from collections import deque
from PIL import Image

sys.path.insert(0, './yolov5')


# CONFIGURATION CONSTANTS
# Networking / detection thresholds
TM_PORT              = 8000    # CARLA Traffic Manager port (controls autopilot)
YOLO_CONF            = 0.4     # YOLO minimum detection confidence to keep a box
N_FRAMES             = 16      # observation window length (frames)
FRAME_SIZE           = (224, 224)
BRAKE_DIST           = 20.0    # metres -- brake if a dangerous object is closer
CONF_THRESHOLD       = 0.60    # minimum behaviour-prediction confidence to trust

# Scene composition -- how many of each actor type CARLA spawns for this run
TOTAL_PEDS           = 20
TOTAL_CARS           = 15
TOTAL_MOTORCYCLES    = 5
TOTAL_BICYCLES       = 5
TOTAL_BUSES          = 3
TOTAL_TRUCKS         = 5

# How far (metres, forward of the ego) actors are scattered when spawned
PED_SPREAD_MIN       = 10.0
PED_SPREAD_MAX       = 120.0
VEH_SPREAD_MIN       = 20.0
VEH_SPREAD_MAX       = 150.0

RECORD_FPS           = 10          # output video frame rate
RECORD_DIR           = "recordings"
LOG_DIR              = "logs"
MODEL_DIR            = "model"     # directory containing the .pth checkpoints
DEADLINE_MS          = 500.0       # D -- per-cycle scheduling deadline

UNSTICK_INTERVAL     = 3.0     # seconds between checks for vehicles stuck idle

# Candidate CARLA blueprints for each spawned agent type (picked at random
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


def run_scheduler(job_number, ttc_list, deadline_ms=DEADLINE_MS):
    # Baseline scheduler: no accuracy optimization, always uses FIXED_MODEL_ID.
    # Only decides WHICH objects get served this cycle, by urgency (TTC).
    if job_number == 0:
        return [], []

    exec_time = MODEL_EXEC_TIMES[FIXED_MODEL_ID]

    # Serve most urgent (smallest TTC) objects first; admit greedily
    # until the fixed model's execution time no longer fits the budget.
    order = sorted(range(job_number), key=lambda j: ttc_list[j])

    schedule = []
    cumulative_time = 0.0
    for job_id in order:
        if cumulative_time + exec_time <= deadline_ms:
            schedule.append((job_id, FIXED_MODEL_ID))
            cumulative_time += exec_time
        else:
            schedule.append((job_id, None))   # deadline exhausted -> skip

    return schedule, [1] * job_number

# LABEL TAXONOMIES
# Maps YOLOv5's COCO class indices onto our agent-type vocabulary
YOLO_CLASS_MAP = {
    0: "Ped",
    1: "Cyc",
    2: "Car",
    3: "Mobike",
    5: "Bus",
    7: "LarVeh",
}
YOLO_CLASSES = list(YOLO_CLASS_MAP.keys())

# Agent-type / behaviour vocabularies predicted by the ShuffleNetGRU model
AGENT_CLASSES = [
    'Ped', 'Car', 'Cyc', 'Mobike',
    'MedVeh', 'LarVeh', 'Bus', 'EmVeh'
]
BEHAVIOR_CLASSES = [
    'MovAway', 'MovTow', 'Mov', 'Stop',
    'IncatLft', 'Ovtak', 'PushObj'
]

# Predicted behaviours that count as "dangerous" -- if a served object's
DANGER_BEHAVIORS = {"MovTow", "Xing", "Stop"}

# Bounding-box colour per agent type, for the on-screen HUD overlay.
AGENT_COLORS = {
    "Ped"    : (0,   255, 127),
    "Car"    : (0,   128, 255),
    "Cyc"    : (255, 191,   0),
    "Mobike" : (255,   0, 200),
    "Bus"    : (0,   200, 255),
    "LarVeh" : (255,  80,   0),
    "MedVeh" : (128, 128, 255),
    "EmVeh"  : (0,     0, 255),
}

# Standard ImageNet normalisation stats used to preprocess model inputs.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE]   Using {DEVICE}")

os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

# ── Module-level shared state, mutated by the camera callback / main loop ──
latest_frame        = None   # most recent camera frame (set by camera_callback)
agent_frame_buffers = {}     # actor_id -> deque of last N_FRAMES preprocessed crops
agent_predictions   = {}     # actor_id -> (last agent_type, last behaviour) cache,

video_writer = None
record_path  = None


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, gru_out):
        scores  = self.attn(gru_out)
        weights = torch.softmax(scores, dim=1)
        return (weights * gru_out).sum(dim=1)


class ShuffleNetGRU(nn.Module):
    def __init__(self,
                 num_agent_classes    = 8,
                 num_behavior_classes = 7,
                 gru_hidden           = 256,
                 dropout              = 0.5,
                 width_mult           = 1.0):
        super().__init__()
        import torchvision.models as tvm

        if width_mult >= 0.75:
            backbone   = tvm.shufflenet_v2_x1_0(pretrained=False)
            gru_hidden = 256
        elif width_mult >= 0.5:
            backbone   = tvm.shufflenet_v2_x0_5(pretrained=False)
            gru_hidden = 128
        else:
            backbone   = tvm.shufflenet_v2_x0_5(pretrained=False)
            gru_hidden = 64
        feat_dim      = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, gru_hidden),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size  = gru_hidden,
            hidden_size = gru_hidden,
            num_layers  = 1,
            batch_first = True,
        )

        self.temporal_attn = TemporalAttention(gru_hidden, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)

        self.agent_head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_agent_classes),
        )
        self.behavior_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_behavior_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)         # fold time into batch for the CNN backbone
        x = self.backbone(x)               # per-frame ShuffleNetV2 feature maps
        x = self.pool(x).flatten(1)        # global-average-pool -> (B*T, feat_dim)
        x = self.projection(x)             # feat_dim -> gru_hidden
        x = x.view(B, T, -1)               # unfold time back out -> (B, T, gru_hidden)
        gru_out, _ = self.gru(x)           # temporal modeling across the N_FRAMES window
        context    = self.temporal_attn(gru_out)  # learned weighted pooling over time
        context    = self.dropout(context)
        fused      = self.fusion(context)
        return self.agent_head(fused), self.behavior_head(fused)


AlexNetLSTM = ShuffleNetGRU   # legacy alias


# MODEL PORTFOLIO (accuracy/latency lookup table)
# Each entry is one candidate model configuration the scheduler can assign
MODEL_REGISTRY = [
    {'name':'ShuffleNetV2x1.0',      'display':'ShuffleNet-1.0',
     'pth':'1.0.pth',         'accuracy':0.925, 'exec_time_ms':233.555,
     'width_mult':1.0,  'int8':False},
    {'name':'ShuffleNetV2x0.75',     'display':'ShuffleNet-0.75',
     'pth':'0.75.pth',        'accuracy':0.8140, 'exec_time_ms':217.437,
     'width_mult':0.75, 'int8':False},
    {'name':'ShuffleNetV2x0.5',      'display':'ShuffleNet-0.5',
     'pth':'0.5.pth',         'accuracy':0.6630, 'exec_time_ms':204.683,
     'width_mult':0.5,  'int8':False},
    {'name':'ShuffleNetV2x0.25',     'display':'ShuffleNet-0.25',
     'pth':'0.25.pth',        'accuracy':0.4105, 'exec_time_ms':181.688,
     'width_mult':0.25, 'int8':False},
    {'name':'ShuffleNetV2x0.12',     'display':'ShuffleNet-0.12',
     'pth':'0.12.pth',        'accuracy':0.3470, 'exec_time_ms':170.436,
     'width_mult':0.12, 'int8':False},
    {'name':'ShuffleNetV2x1.0-INT8', 'display':'ShuffleNet-1.0-INT8',
     'pth':'1.0-INT8.pth',    'accuracy':0.7389, 'exec_time_ms':210.579,
     'width_mult':1.0,  'int8':True},
    {'name':'ShuffleNetV2x0.75-INT8','display':'ShuffleNet-0.75-INT8',
     'pth':'0.75-INT8.pth',   'accuracy':0.5473, 'exec_time_ms':199.528,
     'width_mult':0.75, 'int8':True},
    {'name':'ShuffleNetV2x0.5-INT8', 'display':'ShuffleNet-0.5-INT8',
     'pth':'0.5-INT8.pth',    'accuracy':0.4608, 'exec_time_ms':190.094,
     'width_mult':0.5,  'int8':True},
    {'name':'ShuffleNetV2x0.25-INT8','display':'ShuffleNet-0.25-INT8',
     'pth':'0.25-INT8.pth',   'accuracy':0.2963, 'exec_time_ms':165.381,
     'width_mult':0.25, 'int8':True},
    {'name':'ShuffleNetV2x0.12-INT8','display':'ShuffleNet-0.12-INT8',
     'pth':'0.12-INT8.pth',   'accuracy':0.2695, 'exec_time_ms':158.437,
     'width_mult':0.12, 'int8':True},
]

MODEL_NUMBER     = len(MODEL_REGISTRY)                       # q -- portfolio size
MODEL_ACCURACIES = [m['accuracy']      for m in MODEL_REGISTRY]  # A(m) for each m
MODEL_EXEC_TIMES = [m['exec_time_ms']  for m in MODEL_REGISTRY]  # C(m) for each m

# Baseline: always use the lowest-accuracy model in the portfolio
FIXED_MODEL_NAME = 'ShuffleNetV2x0.12-INT8'
FIXED_MODEL_ID   = next(
    i for i, m in enumerate(MODEL_REGISTRY) if m['name'] == FIXED_MODEL_NAME
)


def load_shufflenet_gru(pth_path, device, width_mult=1.0, int8=False):
    global AGENT_CLASSES, BEHAVIOR_CLASSES

    ckpt = torch.load(pth_path, map_location=device)
    if "model_state_dict" in ckpt:
        if "agent_classes"    in ckpt: AGENT_CLASSES    = ckpt["agent_classes"]
        if "behavior_classes" in ckpt: BEHAVIOR_CLASSES = ckpt["behavior_classes"]

    model = ShuffleNetGRU(
        num_agent_classes    = len(AGENT_CLASSES),
        num_behavior_classes = len(BEHAVIOR_CLASSES),
        width_mult           = width_mult,
    ).to(device)

    try:
        missing, unexpected = model.load_state_dict(
            ckpt.get("model_state_dict", ckpt), strict=False)
        if missing:
            print(f"  [WARN] Missing keys ({len(missing)}): {missing[:2]}")
        if unexpected:
            print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:2]}")
    except Exception as e:
        print(f"  [WARN] Load error: {e}")

    if int8:
        try:
            model = model.cpu()
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        except Exception as e:
            print(f"  [WARN] INT8 quantization failed: {e}")
            model = model.to(device)

    model.eval()
    return model


load_alexnet_lstm = load_shufflenet_gru   # legacy alias

# ── Load every model in the portfolio once at startup (not per-cycle) ──────
print("[MODEL]    Loading ShuffleNetV2-GRU-TemporalAttention models...")
base_model = load_shufflenet_gru(os.path.join(MODEL_DIR, '1.0.pth'), DEVICE, width_mult=1.0, int8=False)
print(f"[MODEL]    Agent classes   : {AGENT_CLASSES}")
print(f"[MODEL]    Behavior classes: {BEHAVIOR_CLASSES}")

loaded_models = {}   # model registry 'name' -> loaded ShuffleNetGRU instance
for reg in MODEL_REGISTRY:
    try:
        m = load_shufflenet_gru(
            os.path.join(MODEL_DIR, reg['pth']), DEVICE,
            width_mult=reg.get('width_mult', 1.0),
            int8=reg.get('int8', False))
        loaded_models[reg['name']] = m
        print(f"  [OK]   {reg['name']:30s} <- {reg['pth']}")
    except Exception as e:
        print(f"  [FAIL] {reg['name']:30s} : {e} — using base model")
        loaded_models[reg['name']] = base_model
print("[MODELS]   All models ready\n")

# ── YOLOv5n: lightweight, fast object detector used for PERCEPTION only ────
print("[YOLO]     Loading YOLOv5n...")
yolo_model = torch.hub.load(
    './yolov5', 'custom', path='yolov5n.pt', source='local'
)
yolo_model.conf    = YOLO_CONF
yolo_model.classes = YOLO_CLASSES
print("[YOLO]     Loaded on", next(yolo_model.parameters()).device)


# Preprocessing pipeline applied to every cropped detection before it's
inference_transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE[1], FRAME_SIZE[0])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_crop(frame_bgr, box, pad=10):
    h, w   = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0,  x1 - pad)
    y1 = max(0,  y1 - pad)
    x2 = min(w,  x2 + pad)
    y2 = min(h,  y2 + pad)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    return inference_transform(crop_pil)


def detect_agents_yolo(frame):
    if frame is None:
        return None, []

    rgb        = frame[:, :, ::-1]
    results    = yolo_model(rgb)
    annotated  = frame.copy()
    detections = []

    for *box, conf, cls in results.xyxy[0].tolist():
        cls_int = int(cls)
        if cls_int not in YOLO_CLASS_MAP:
            continue

        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        agent_label = YOLO_CLASS_MAP[cls_int]
        color       = AGENT_COLORS.get(agent_label, (200, 200, 200))

        detections.append({
            'box'        : (x1, y1, x2, y2),
            'center'     : (cx, cy),
            'confidence' : float(conf),
            'yolo_class' : cls_int,
            'agent_label': agent_label,
        })

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{agent_label} {conf:.2f}",
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return annotated, detections


def run_inference(model, frame_buffer):
    frames  = list(frame_buffer)
    stacked = torch.stack(frames, dim=0)
    model_device = next(model.parameters()).device
    x_t     = stacked.unsqueeze(0).to(model_device)

    with torch.no_grad():
        agent_logits, beh_logits = model(x_t)

        agent_probs = torch.softmax(agent_logits, dim=1)[0]
        beh_probs   = torch.softmax(beh_logits,   dim=1)[0]

        agent_idx   = agent_probs.argmax().item()
        beh_idx     = beh_probs.argmax().item()

        agent_conf  = agent_probs[agent_idx].item()
        beh_conf    = beh_probs[beh_idx].item()

    agent_label = AGENT_CLASSES[agent_idx]    if agent_idx < len(AGENT_CLASSES)    else "Unknown"
    beh_label   = BEHAVIOR_CLASSES[beh_idx]   if beh_idx   < len(BEHAVIOR_CLASSES) else "Unknown"

    return agent_label, beh_label, agent_conf, beh_conf


def get_distance(actor1, actor2):
    l1 = actor1.get_location()
    l2 = actor2.get_location()
    return math.sqrt((l1.x-l2.x)**2 + (l1.y-l2.y)**2 + (l1.z-l2.z)**2)


def get_vehicle_speed_ms(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def get_ped_speed(ped):
    v = ped.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def camera_callback(image):
    global latest_frame
    array        = np.frombuffer(image.raw_data, dtype=np.uint8)
    array        = array.reshape((image.height, image.width, 4))
    latest_frame = array[:, :, :3].copy()


def world_to_camera_pixel(world_loc, vehicle,
                           cam_w=800, cam_h=600, fov=90):
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
    carla_map = world.get_map()
    waypoint  = carla_map.get_waypoint(
        location, project_to_road=False,
        lane_type=carla.LaneType.Any
    )
    if waypoint is None:
        return False
    return waypoint.lane_type == carla.LaneType.Driving


def get_sidewalk_nav_location(world, attempts=50):
    for _ in range(attempts):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if not is_on_road(world, loc):
            return loc
    return None


def destroy_ped(entry):
    try:
        entry['ctrl'].stop()
        entry['ctrl'].destroy()
        pid = entry['ped'].id
        entry['ped'].destroy()
        agent_frame_buffers.pop(pid, None)
        agent_predictions.pop(pid, None)
    except Exception:
        pass


def spawn_single_ped(world, bp_lib, vehicle,
                     fwd_dist, behaviour, ped_index):
    vt      = vehicle.get_transform()
    yaw_rad = math.radians(vt.rotation.yaw)
    target  = carla.Location(
        x = vt.location.x + math.cos(yaw_rad) * fwd_dist,
        y = vt.location.y + math.sin(yaw_rad) * fwd_dist,
        z = vt.location.z
    )

    # Search for a sidewalk nav-point close to the desired forward offset
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
                               z=best_loc.z + 0.5)   # small z-offset to avoid ground clipping
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

    # AI walker controllers need a few world.tick()s after spawn before
    ctrl = world.spawn_actor(ctrl_bp, carla.Transform(),
                             attach_to=ped)
    world.tick()
    world.tick()
    world.tick()
    ctrl.start()

    dest_1 = get_sidewalk_nav_location(world)
    dest_2 = get_sidewalk_nav_location(world)   # currently unused, kept for parity/future use

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

    agent_frame_buffers[ped.id] = deque(maxlen=N_FRAMES)
    agent_predictions[ped.id]   = ("Ped", "buffering...")

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
    pedestrians   = []
    fwd_distances = sorted([
        random.uniform(PED_SPREAD_MIN, PED_SPREAD_MAX)
        for _ in range(TOTAL_PEDS)
    ])
    behaviours = (
        ['standing'] * 4 + ['walking'] * 4 +
        ['crossing'] * 2
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
    spawned_vehicles = []
    carla_map        = world.get_map()

    # Only keep spawn points on a drivable lane with road ahead
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

    # Exclude spawn points too close to the ego to avoid immediate overlap.
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

    # How many of each agent type to place (scene composition constants).
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

            # If none of our preferred named blueprints are available in
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

            # Spawn can fail (e.g. point occupied) -- retry a few more points.
            attempts = 0
            while actor is None and attempts < 3 and pt_idx < len(road_spawn_pts):
                sp     = road_spawn_pts[pt_idx]
                pt_idx += 1
                actor  = world.try_spawn_actor(bp, sp)
                attempts += 1

            if actor is not None:
                actor.set_autopilot(True, TM_PORT)

                # Traffic Manager safety/behaviour settings for this vehicle:
                tm.ignore_lights_percentage(actor, 0)
                tm.ignore_signs_percentage(actor, 0)
                tm.ignore_vehicles_percentage(actor, 0)
                tm.distance_to_leading_vehicle(actor, 3.0)
                tm.vehicle_percentage_speed_difference(actor, 10)
                tm.auto_lane_change(actor, True)
                tm.vehicle_percentage_speed_difference(
                    actor, random.uniform(0, 20)
                )

                # Give it an initial throttle nudge so it starts moving
                actor.apply_control(
                    carla.VehicleControl(throttle=0.6, brake=0.0)
                )

                agent_frame_buffers[actor.id] = deque(maxlen=N_FRAMES)
                agent_predictions[actor.id]   = (agent_type, "buffering...")
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
    time.sleep(2.0)   # give the Traffic Manager time to send the first
    print(f"[SPAWN]    All vehicles should be moving\n")
    return spawned_vehicles


def unstick_vehicles(road_vehicles, vehicle, tm, world):
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
    fwd_dist  = random.uniform(60.0, 100.0)
    behaviour = random.choice(
        ['standing', 'walking', 'crossing', 'crossing', 'standing']
    )
    entry = spawn_single_ped(world, bp_lib, vehicle,
                             fwd_dist, behaviour, ped_index)
    if entry is not None:
        pedestrians.append(entry)
        print(f"[RESPAWN]  New ped {fwd_dist:.0f}m ahead | {behaviour}")


def draw_hud(frame, speed_ms, n_yolo, n_agents,
             n_moving, n_still, elapsed,
             brake_active, vehicle_state,
             intent_lines):
    if frame is None:
        return None
    h, w    = frame.shape[:2]
    display = frame.copy()

    cv2.putText(display, f"Speed      : {speed_ms:.2f} m/s",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2)
    cv2.putText(display, f"State      : {vehicle_state}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if brake_active else (0, 255, 0), 2)
    cv2.putText(display, f"YOLO       : {n_yolo} detected",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 0), 2)
    cv2.putText(display, f"Agents     : {n_agents} total",
                (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 255), 2)
    cv2.putText(display, f"Moving     : {n_moving}",
                (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 180, 255), 2)
    cv2.putText(display, f"Still      : {n_still}",
                (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 255, 180), 2)
    cv2.putText(display, f"Time       : {elapsed:.1f}s",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 2)
    cv2.putText(display, f"Deadline   : {DEADLINE_MS:.0f}ms",
                (10, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 165, 0), 2)

    panel_x = w - 360
    cv2.rectangle(display, (panel_x - 8, 5), (w - 5, 18 + len(intent_lines) * 22),
                  (20, 20, 20), -1)
    cv2.rectangle(display, (panel_x - 8, 5), (w - 5, 18 + len(intent_lines) * 22),
                  (80, 80, 80), 1)

    cv2.putText(display, "PREDICTED BEHAVIOR (+1s)",
                (panel_x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    for i, line in enumerate(intent_lines):
        cv2.putText(display, line, (panel_x, 40 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 60), 1)

    if brake_active:
        cv2.rectangle(display, (0, h - 50), (w, h),
                      (0, 0, 150), -1)
        cv2.putText(display,
                    "!!! VEHICLE BRAKING -- DANGER DETECTED !!!",
                    (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)

    return display


def draw_prediction_on_frame(display, det, agent_pred,
                              beh_pred, beh_conf,
                              model_name=""):
    if det is None:
        return display

    x1, y1, x2, y2 = det['box']
    color = (0, 220, 60)

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    if beh_conf > 0:
        line1 = f"{agent_pred}->{beh_pred} {beh_conf:.0%}"
    else:
        line1 = f"{agent_pred}->{beh_pred}"

    short_model = model_name[:14] if model_name else ""
    line2 = f"[{short_model}]" if short_model else ""

    (lw1, lh1), _ = cv2.getTextSize(
        line1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )
    (lw2, lh2), _ = cv2.getTextSize(
        line2, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
    ) if line2 else ((0, 0), None)

    box_w  = max(lw1, lw2) + 8
    box_h  = lh1 + (lh2 + 4 if line2 else 0) + 8
    ty     = max(y1 - box_h - 2, 0)

    cv2.rectangle(display,
                  (x1, ty), (x1 + box_w, ty + box_h),
                  (10, 10, 10), -1)
    cv2.rectangle(display,
                  (x1, ty), (x1 + box_w, ty + box_h),
                  color, 1)

    cv2.putText(display, line1,
                (x1 + 3, ty + lh1 + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if line2:
        cv2.putText(display, line2,
                    (x1 + 3, ty + lh1 + lh2 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 180, 180), 1)

    return display


def main():
    global video_writer, record_path

    # ── Connect to the running CARLA server (must be started separately) ──
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world     = client.get_world()
    bp_lib    = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points()

    world.set_weather(carla.WeatherParameters.ClearNoon)
    print(f"[WORLD]    Map : {world.get_map().name}")

    vehicle     = None
    pedestrians = []
    road_vehicles = []
    camera      = None
    col_sensor  = None
    csv_file    = None
    detail_log_file = None

    try:
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        v_spawn    = random.choice(spawn_pts)
        vehicle    = world.spawn_actor(vehicle_bp, v_spawn)
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0)
        )
        print(f"[VEHICLE]  Spawned at "
              f"({v_spawn.location.x:.2f},{v_spawn.location.y:.2f})")

        # ── Forward-facing RGB camera (PERCEPTION sensor) ──────────────
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

        # ── Collision sensor -- just logs to console; ground-truth
        col_bp     = bp_lib.find('sensor.other.collision')
        col_sensor = world.spawn_actor(
            col_bp, carla.Transform(), attach_to=vehicle
        )
        col_sensor.listen(
            lambda e: print(
                f"\n  !! COLLISION with {e.other_actor.type_id} !!\n"
            )
        )

        # ── Traffic Manager: governs autopilot behaviour for every
        tm = client.get_trafficmanager(TM_PORT)
        tm.set_synchronous_mode(False)
        tm.set_random_device_seed(42)   # reproducible traffic behaviour

        tm.set_global_distance_to_leading_vehicle(3.0)
        tm.global_percentage_speed_difference(10.0)

        # ── Populate the scene (see spawn_all_pedestrians / spawn_all_vehicles) ──
        pedestrians      = spawn_all_pedestrians(world, bp_lib, vehicle)
        road_vehicles    = spawn_all_vehicles(world, bp_lib, vehicle, tm)

        # ── Release the ego onto autopilot. NOTE: ignore_vehicles_percentage=0
        time.sleep(1.0)
        vehicle.set_autopilot(True, TM_PORT)
        tm.ignore_lights_percentage(vehicle, 0)
        tm.ignore_signs_percentage(vehicle, 0)
        tm.ignore_vehicles_percentage(vehicle, 0)
        tm.distance_to_leading_vehicle(vehicle, 5.0)
        tm.auto_lane_change(vehicle, False)
        tm.vehicle_percentage_speed_difference(vehicle, 20)
        print("[VEHICLE]  Ego vehicle released")

        # ── Output video recording (for qualitative/visual review) ─────
        record_path  = os.path.join(
            RECORD_DIR,
            time.strftime("retina_low_%Y%m%d_%H%M%S.avi")
        )
        video_writer = cv2.VideoWriter(
            record_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            RECORD_FPS, (800, 600)
        )
        print(f"[RECORD]   Recording to {record_path}")

        # ── Logging: one CSV row + one JSON line per cycle ─────────────
        run_tag    = time.strftime("%Y%m%d_%H%M%S")
        log_path   = os.path.join(LOG_DIR, f"retina_low_{run_tag}.csv")
        detail_log_path = os.path.join(LOG_DIR, f"retina_low_{run_tag}.jsonl")

        csv_file   = open(log_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'timestamp', 'elapsed_s', 'n_visible_agents',
            'n_assigned', 'n_skipped',
            'avg_acc_all', 'total_weighted_accuracy',
            'deadline_ms', 'solver_ms',
            'vehicle_speed_ms', 'brake_active',
            'ttc_values', 'agent_types', 'behaviors', 'models_used'
        ])
        detail_log_file = open(detail_log_path, 'w')
        print(f"[LOG]      Summary CSV : {log_path}")
        print(f"[LOG]      Per-object  : {detail_log_path}")

        spectator = world.get_spectator()
        cv2.namedWindow(
            "CARLA - RETINA-Low (Fixed ShuffleNet-0.12-INT8) Baseline",
            cv2.WINDOW_AUTOSIZE
        )

        print("\n" + "="*65)
        print(f"  SIMULATION RUNNING — RETINA-Low fixed-model baseline")
        print(f"  Pedestrians      : {TOTAL_PEDS}")
        print(f"  Cars             : {TOTAL_CARS}")
        print(f"  Motorcycles      : {TOTAL_MOTORCYCLES}")
        print(f"  Cyclists         : {TOTAL_BICYCLES}")
        print(f"  Buses            : {TOTAL_BUSES}")
        print(f"  Large vehicles   : {TOTAL_TRUCKS}")
        print(f"  Deadline         : {DEADLINE_MS} ms per cycle")
        print(f"  Models           : {MODEL_NUMBER}")
        print(f"  Agent classes    : {AGENT_CLASSES}")
        print(f"  Behavior classes : {BEHAVIOR_CLASSES}")
        print(f"  Danger behaviors : {DANGER_BEHAVIORS}")
        print(f"  Fixed model      : {FIXED_MODEL_NAME}")
        print(f"  Scheduler        : TTC-priority, fixed model (no ILP)")
        print("  Press Q to quit")
        print("="*65 + "\n")

        start_time       = time.time()
        was_braking      = False
        brake_hold       = False   # latches the brake across cycles until
        ped_counter      = TOTAL_PEDS + 1
        detections       = []
        last_unstick_check = time.time()

        # MAIN LOOP -- runs once per available camera frame
        while True:
            now     = time.time()
            elapsed = now - start_time

            # Periodically nudge any vehicle that's genuinely stuck (not
            if now - last_unstick_check > UNSTICK_INTERVAL:
                last_unstick_check = now
                unstick_vehicles(road_vehicles, vehicle, tm, world)

            # ── Trigger crossing pedestrians ───────────────────────────
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

            # ── Detect when a crossing pedestrian has finished crossing ──
            for entry in pedestrians:
                if (entry['behaviour'] == 'crossing' and
                        entry['cross_started'] and
                        not entry['cross_done']):
                    spd        = get_ped_speed(entry['ped'])
                    cross_time = now - entry['spawn_time']
                    if (cross_time > 3.0 and spd < 0.1):
                        entry['cross_done'] = True
                        entry['ctrl'].set_max_speed(0.0)

            # ── Recycle pedestrians that have wandered too far away ────
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

            # Spectator camera trails behind/above the ego (cosmetic only)
            vt = vehicle.get_transform()
            spectator.set_transform(
                carla.Transform(
                    vt.location + carla.Location(x=-10, z=5),
                    carla.Rotation(pitch=-15, yaw=vt.rotation.yaw)
                )
            )

            speed_ms = get_vehicle_speed_ms(vehicle)

            n_yolo        = 0
            display_frame = None
            detections    = []

            # ── STAGE 1: PERCEPTION -- run YOLO on the latest frame, then
            if latest_frame is not None:
                annotated, detections = detect_agents_yolo(latest_frame)
                n_yolo        = len(detections)
                display_frame = annotated

                for entry in pedestrians:
                    try:
                        found, det = is_agent_in_detection(
                            entry['ped'], vehicle, detections
                        )
                        if found and det is not None:
                            crop_tensor = preprocess_crop(
                                latest_frame, det['box']
                            )
                            pid = entry['ped'].id
                            if pid not in agent_frame_buffers:
                                agent_frame_buffers[pid] = deque(
                                    maxlen=N_FRAMES
                                )
                            agent_frame_buffers[pid].append(crop_tensor)
                    except Exception:
                        pass

                for rv in road_vehicles:
                    try:
                        found, det = is_agent_in_detection(
                            rv['actor'], vehicle, detections
                        )
                        if found and det is not None:
                            crop_tensor = preprocess_crop(
                                latest_frame, det['box']
                            )
                            vid = rv['actor'].id
                            if vid not in agent_frame_buffers:
                                agent_frame_buffers[vid] = deque(
                                    maxlen=N_FRAMES
                                )
                            agent_frame_buffers[vid].append(crop_tensor)
                    except Exception:
                        pass

            brake_active  = False
            vehicle_state = "DRIVING"
            n_moving      = 0
            n_still       = 0
            intent_lines  = []

            # visible_agents = actors currently matched to a YOLO detection
            visible_agents = []

            for entry in pedestrians:
                try:
                    ped_spd = get_ped_speed(entry['ped'])
                    if ped_spd > 0.3:
                        n_moving += 1
                    else:
                        n_still  += 1
                    found, det = is_agent_in_detection(
                        entry['ped'], vehicle, detections
                    )
                    if found:
                        visible_agents.append({
                            'actor'      : entry['ped'],
                            'agent_type' : 'Ped',
                            'cross_done' : entry.get('cross_done', False),
                            'det'        : det,
                        })
                except Exception:
                    pass

            for rv in road_vehicles:
                try:
                    v_spd = get_vehicle_speed_ms(rv['actor'])
                    if v_spd > 0.5:
                        n_moving += 1
                    else:
                        n_still  += 1
                    found, det = is_agent_in_detection(
                        rv['actor'], vehicle, detections
                    )
                    if found:
                        visible_agents.append({
                            'actor'      : rv['actor'],
                            'agent_type' : rv['agent_type'],
                            'cross_done' : False,
                            'det'        : det,
                        })
                except Exception:
                    pass

            # ── STAGE 2: SAFETY ANALYSIS -- estimate time-to-collision
            raw_spd  = get_vehicle_speed_ms(vehicle)
            car_spd  = raw_spd if raw_spd > 0.5 else 1.0   # avoid div-by-zero when parked
            ttc_list = []
            for va in visible_agents:
                dist     = get_distance(vehicle, va['actor'])
                try:
                    agent_spd = get_vehicle_speed_ms(va['actor'])
                except Exception:
                    try:
                        agent_spd = get_ped_speed(va['actor'])
                    except Exception:
                        agent_spd = 0.0

                ego_tf    = vehicle.get_transform()
                ego_fwd   = ego_tf.get_forward_vector()
                to_agent  = va['actor'].get_location() - ego_tf.location
                dot       = (ego_fwd.x * to_agent.x
                             + ego_fwd.y * to_agent.y)
                in_front  = dot > -0.3

                if in_front:
                    closing_spd = max(car_spd - agent_spd * 0.5, 0.5)
                    ttc         = dist / closing_spd
                else:
                    ttc = dist / car_spd

                ttc_list.append(ttc)

            # Immediate proximity-brake pre-check (independent of scheduler)
            proximity_brake       = False
            proximity_brake_dist  = None
            ego_tf                = vehicle.get_transform()
            ego_fwd               = ego_tf.get_forward_vector()

            for va in visible_agents:
                try:
                    dist     = get_distance(vehicle, va['actor'])
                    to_agent = (va['actor'].get_location()
                                - ego_tf.location)

                    to_agent_len = math.sqrt(
                        to_agent.x**2 + to_agent.y**2
                    )
                    if to_agent_len < 0.1:
                        continue

                    dot_fwd = (ego_fwd.x * to_agent.x
                               + ego_fwd.y * to_agent.y)

                    ego_right_x =  ego_fwd.y
                    ego_right_y = -ego_fwd.x
                    dot_lat = abs(
                        ego_right_x * to_agent.x +
                        ego_right_y * to_agent.y
                    )

                    in_front      = dot_fwd > 0
                    lateral_offset = dot_lat
                    in_lane        = lateral_offset < 2.5

                    if not in_front or not in_lane:
                        continue

                    if dist < BRAKE_DIST:
                        proximity_brake      = True
                        proximity_brake_dist = dist
                        break

                    try:
                        agent_spd = get_vehicle_speed_ms(va['actor'])
                    except Exception:
                        try:
                            agent_spd = get_ped_speed(va['actor'])
                        except Exception:
                            agent_spd = 0.0

                    closing_spd = raw_spd - agent_spd
                    if closing_spd > 0.5:
                        ttc_closing = dist / closing_spd
                        if ttc_closing < 2.5:
                            proximity_brake      = True
                            proximity_brake_dist = dist
                            print(f"  [BRAKE]  Closing too fast: "
                                  f"ego={raw_spd:.1f} "
                                  f"agent={agent_spd:.1f} "
                                  f"closing={closing_spd:.1f}m/s "
                                  f"ttc={ttc_closing:.1f}s")
                            break

                except Exception:
                    pass

            if not proximity_brake and detections and latest_frame is not None:
                frame_h    = latest_frame.shape[0]
                frame_w    = latest_frame.shape[1]
                frame_area = frame_h * frame_w
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    bbox_area = (x2 - x1) * (y2 - y1)
                    center_x  = (x1 + x2) / 2

                    in_lane_x = (frame_w * 0.25 < center_x < frame_w * 0.75)

                    if in_lane_x and (
                        (y2 > frame_h * 0.30 and bbox_area > frame_area * 0.04) or
                        (bbox_area > frame_area * 0.10)
                    ):
                        proximity_brake      = True
                        proximity_brake_dist = 0.0
                        break

            if proximity_brake:
                brake_active  = True
                vehicle_state = "BRAKING"
                print(f"  [BRAKE]  Object detected in front")

            # ── STAGE 3: SCHEDULING -- the core RETINA contribution.
            if len(visible_agents) > 0:
                solver_start             = time.time()
                schedule, job_weights    = run_scheduler(
                    job_number  = len(visible_agents),
                    ttc_list    = ttc_list,
                    deadline_ms = DEADLINE_MS
                )
                solver_ms = (time.time() - solver_start) * 1000

                print(f"\n[{elapsed:6.1f}s] "
                      f"Speed={speed_ms:.2f}m/s | "
                      f"YOLO={n_yolo} | "
                      f"Vis={len(visible_agents)} | "
                      f"Solver={solver_ms:.1f}ms")
                print(f"  {'─'*60}")

                assigned_accs = []
                models_used   = []
                agent_types_log = []
                behaviors_log   = []
                n_assigned    = 0
                n_skipped_cnt = 0
                frame_timestamp = datetime.now().isoformat(timespec='milliseconds')
                per_object    = []

                # STAGE 4: PREDICTION -- run/reuse prediction per scheduled job
                for job_id, model_id in schedule:
                    va         = visible_agents[job_id]
                    actor      = va['actor']
                    aid        = actor.id
                    dist       = get_distance(vehicle, actor)
                    ttc        = ttc_list[job_id]
                    cross_done = va['cross_done']
                    buf_size   = len(agent_frame_buffers.get(aid, []))

                    # Case A: scheduler assigned a model AND we already
                    if model_id is not None and buf_size >= N_FRAMES:
                        model_name    = MODEL_REGISTRY[model_id]['name']
                        display_model = MODEL_REGISTRY[model_id]['display']
                        sel_model     = loaded_models.get(model_name, base_model)
                        agent_pred, beh_pred, agent_conf, beh_conf = \
                            run_inference(sel_model,
                                         agent_frame_buffers[aid])
                        agent_predictions[aid] = (agent_pred, beh_pred)
                        n_assigned   += 1
                        model_acc     = MODEL_ACCURACIES[model_id]
                        assigned_accs.append(model_acc)
                        models_used.append(
                            MODEL_REGISTRY[model_id]['display']
                        )
                        agent_types_log.append(agent_pred)
                        behaviors_log.append(beh_pred)

                    # Case B: scheduler assigned a model, but we haven't
                    elif model_id is not None:
                        prev = agent_predictions.get(
                            aid, (va['agent_type'], "buffering...")
                        )
                        agent_pred, beh_pred = prev
                        agent_conf = beh_conf = 0.0
                        model_name    = MODEL_REGISTRY[model_id]['name']
                        display_model = MODEL_REGISTRY[model_id]['display']
                        models_used.append('BUF')
                        agent_types_log.append(agent_pred)
                        behaviors_log.append(beh_pred)
                    else:
                        prev = agent_predictions.get(
                            aid, (va['agent_type'], "skipped")
                        )
                        agent_pred, beh_pred = prev[0], "skipped"
                        agent_conf = beh_conf = 0.0
                        model_name    = "SKIPPED"
                        display_model = "skipped"
                        n_skipped_cnt += 1
                        models_used.append('SKIP')
                        agent_types_log.append(agent_pred)
                        behaviors_log.append("skipped")

                    per_object.append({
                        'object_id'     : aid,
                        'agent_type'    : agent_pred,
                        'ttc_s'         : round(ttc, 2),
                        'distance_m'    : round(dist, 2),
                        'assigned'      : model_id is not None and buf_size >= N_FRAMES,
                        'model'         : display_model,
                        'behavior'      : beh_pred,
                        'behavior_conf' : round(beh_conf, 3),
                    })

                    if (beh_pred in DANGER_BEHAVIORS and
                            dist < BRAKE_DIST and
                            beh_conf >= CONF_THRESHOLD and
                            not cross_done and
                            model_id is not None):
                        brake_active  = True
                        vehicle_state = "BRAKING"

                    if display_frame is not None:
                        det = va.get('det', None)
                        display_frame = draw_prediction_on_frame(
                            display_frame, det,
                            agent_pred, beh_pred, beh_conf,
                            display_model
                        )

                    warn = "!" if (brake_active and
                                   beh_pred in DANGER_BEHAVIORS and
                                   dist < BRAKE_DIST) else " "

                    print(
                        f"  {warn} Job{job_id} "
                        f"id={aid:4d} | "
                        f"model={model_name:20s} | "
                        f"ttc={ttc:5.1f}s | "
                        f"AGENT={agent_pred:8s} "
                        f"-> IN 1s: {beh_pred}"
                        + (f" ({beh_conf:.0%})" if beh_conf > 0 else "")
                    )

                    intent_lines.append(
                        f"{agent_pred[:5]:5s}"
                        f"-> {beh_pred[:7]:7s}"
                        f" [{display_model[:12]}]"
                    )

                print(f"  {'─'*60}")
                print(f"  Moving={n_moving} Still={n_still} | "
                      f"State={vehicle_state} "
                      f"brake={brake_active}")

                avg_acc_all = (
                    sum(assigned_accs) / len(assigned_accs)
                    if assigned_accs else 0.0
                )
                weighted_acc = avg_acc_all

                csv_writer.writerow([
                    frame_timestamp,
                    f"{elapsed:.2f}",
                    len(visible_agents),
                    n_assigned,
                    n_skipped_cnt,
                    f"{avg_acc_all:.4f}",
                    f"{weighted_acc:.4f}",
                    DEADLINE_MS,
                    f"{solver_ms:.2f}",
                    f"{speed_ms:.2f}",
                    int(brake_active),
                    ';'.join([f"{t:.2f}" for t in ttc_list]),
                    ';'.join(agent_types_log),
                    ';'.join(behaviors_log),
                    '|'.join(models_used),
                ])
                csv_file.flush()

                detail_log_file.write(json.dumps({
                    'timestamp'    : frame_timestamp,
                    'elapsed_s'    : round(elapsed, 2),
                    'n_detected'   : len(visible_agents),
                    'n_assigned'   : n_assigned,
                    'n_skipped'    : n_skipped_cnt,
                    'vehicle_speed_ms': round(speed_ms, 2),
                    'brake_active' : brake_active,
                    'objects'      : per_object,
                }) + '\n')
                detail_log_file.flush()

            if brake_active and not was_braking:
                print(f"\n  *** VEHICLE BRAKING ***")
            if not brake_active and was_braking:
                print("\n  *** VEHICLE RESUMING ***")
            was_braking = brake_active

            # ── STAGE 5: CONTROL -- brake-hold latch.
            if brake_active:
                brake_hold = True
            elif brake_hold:
                still_danger = any(
                    agent_predictions.get(va['actor'].id,
                                         ("", ""))[1] in DANGER_BEHAVIORS
                    and not va['cross_done']
                    for va in visible_agents
                )
                if not still_danger:
                    brake_hold = False

            # Override the Traffic Manager's autopilot with a manual full
            if brake_hold:
                vehicle.set_autopilot(False, TM_PORT)
                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0, brake=1.0, steer=0.0
                    )
                )
                vehicle_state = "STOPPED"
            else:
                vehicle.set_autopilot(True, TM_PORT)

            if display_frame is not None:
                hud = draw_hud(
                    display_frame, speed_ms, n_yolo,
                    len(pedestrians) + len(road_vehicles),
                    n_moving, n_still,
                    elapsed, brake_active, vehicle_state,
                    intent_lines
                )
                if hud is not None:
                    cv2.imshow(
                        "CARLA - RETINA-Low (Fixed ShuffleNet-0.12-INT8) Baseline",
                        hud
                    )
                    if video_writer is not None:
                        video_writer.write(hud)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit.")
                break

            time.sleep(0.05)   # small yield so this isn't a pure busy-loop

    except KeyboardInterrupt:
        print("\nStopping...")

    # ── Cleanup: always run, even on Ctrl+C or an unhandled exception,
    finally:
        print("\nCleaning up...")
        cv2.destroyAllWindows()

        try:
            csv_file.close()
            print(f"  [OK] Summary log saved to {log_path}")
        except Exception:
            pass

        try:
            detail_log_file.close()
            print(f"  [OK] Per-object log saved to {detail_log_path}")
        except Exception:
            pass

        if video_writer is not None:
            video_writer.release()
            print(f"  [OK] Video saved to {record_path}")

        if camera:
            camera.stop()
            camera.destroy()
            print("  [OK] Camera destroyed")

        if col_sensor:
            col_sensor.stop()
            col_sensor.destroy()
            print("  [OK] Collision sensor destroyed")

        for entry in pedestrians:
            destroy_ped(entry)
        print(f"  [OK] {len(pedestrians)} peds destroyed")

        for rv in road_vehicles:
            try:
                rv['actor'].set_autopilot(False)
                rv['actor'].destroy()
            except Exception:
                pass
        print(f"  [OK] {len(road_vehicles)} road vehicles destroyed")

        if vehicle:
            vehicle.set_autopilot(False)
            vehicle.destroy()
            print("  [OK] Vehicle destroyed")

        print("Done.")


if __name__ == '__main__':
    main()
