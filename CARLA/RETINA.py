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
import gurobipy as gp
from gurobipy import GRB
from torchvision import models
from torchvision import transforms
from collections import deque
from PIL import Image

sys.path.insert(0, './yolov5')


# Experiment configuration. Distances are meters and speeds are m/s.
TM_PORT              = 8000
YOLO_CONF            = 0.4
# Number of buffered crops required before running prediction.
N_FRAMES             = 16
FRAME_SIZE           = (224, 224)
BRAKE_DIST           = 20.0
CONF_THRESHOLD       = 0.60


# Requested scene composition; actual counts depend on spawn success and behavior lists.
TOTAL_PEDS           = 20
TOTAL_CARS           = 15
TOTAL_MOTORCYCLES    = 5
TOTAL_BICYCLES       = 5
TOTAL_BUSES          = 3
TOTAL_TRUCKS         = 5


# Pedestrian forward spawn offsets. VEH_SPREAD_* are currently unused.
PED_SPREAD_MIN       = 10.0
PED_SPREAD_MAX       = 120.0
VEH_SPREAD_MIN       = 20.0
VEH_SPREAD_MAX       = 150.0

LOG_DIR              = "logs"
MODEL_DIR            = "model"
DEADLINE_MS          = 500.0


# Minimum lookup accuracy for ILP eligibility; 0.25 admits the full portfolio.
A_REQ                = 0.25


# Fixed budget reservation, not a solver time limit.
SCHEDULER_OVERHEAD_MS = 6.0
UNSTICK_INTERVAL     = 3.0


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


def run_scheduler(job_number, ttc_list, deadline_ms=DEADLINE_MS):
    """Select at most one model per job using TTC-weighted accuracy.
    
    TTC values are in seconds; the deadline and lookup costs are in
    milliseconds. Jobs with floor(TTC / deadline_seconds) <= 1 are mandatory
    for finite positive TTC. Return (job/model assignments, urgency weights),
    using None for deferred jobs. The budget covers lookup inference costs
    minus a fixed solver reservation, not measured end-to-end execution."""
    if job_number == 0:
        return [], []


    # Nonpositive or nonfinite TTC receives low weight and optional status.
    D_sec = deadline_ms / 1000.0
    eta, weight = [], []
    for ttc in ttc_list:
        if math.isfinite(ttc) and ttc > 0:
            eta.append(int(math.floor(ttc / D_sec)))
            weight.append(1.0 / ttc)
        else:
            eta.append(999)
            weight.append(0.01)
    urgent = [e <= 1 for e in eta]


    m_valid = M_VALID


    deadline_eff_ms = max(deadline_ms - SCHEDULER_OVERHEAD_MS, 0.0)

    try:
        model_g = gp.Model()
        model_g.setParam("OutputFlag", 0)
        model_g.setParam("Seed", 0)

        # Binary assignment variable for each job and eligible model.
        x = {}
        for job_id in range(job_number):
            for model_id in m_valid:
                x[job_id, model_id] = model_g.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_{job_id}_{model_id}"
                )
        model_g.update()


        # Maximize lookup accuracy weighted by inverse TTC.
        obj = gp.LinExpr()
        for job_id in range(job_number):
            for model_id in m_valid:
                obj += weight[job_id] * MODEL_ACCURACIES[model_id] * x[job_id, model_id]
        model_g.setObjective(obj, GRB.MAXIMIZE)


        # Mandatory jobs receive one model; optional jobs may be deferred.
        for job_id in range(job_number):
            s = gp.LinExpr()
            for model_id in m_valid:
                s += x[job_id, model_id]
            if urgent[job_id]:
                model_g.addConstr(s == 1, name=f"Mandatory_{job_id}")
            else:
                model_g.addConstr(s <= 1, name=f"Optional_{job_id}")


        # Bound the sum of nominal inference costs for sequential execution.
        total_time = gp.LinExpr()
        for job_id in range(job_number):
            for model_id in m_valid:
                total_time += MODEL_EXEC_TIMES[model_id] * x[job_id, model_id]
        model_g.addConstr(total_time <= deadline_eff_ms, name="Deadline")

        model_g.optimize()

        # Non-optimal solver fallback: admit jobs in TTC order with the fastest
        # model from the full registry. This path does not enforce A_REQ or mandatory service.
        if model_g.Status != GRB.OPTIMAL:
            fastest_id = int(np.argmin(MODEL_EXEC_TIMES))
            order      = sorted(range(job_number), key=lambda j: ttc_list[j])
            schedule, budget = [], deadline_eff_ms
            assign = {}
            for job_id in order:
                if MODEL_EXEC_TIMES[fastest_id] <= budget:
                    assign[job_id] = fastest_id
                    budget -= MODEL_EXEC_TIMES[fastest_id]
            for job_id in range(job_number):
                schedule.append((job_id, assign.get(job_id, None)))
            return schedule, weight

        schedule = []
        for job_id in range(job_number):
            selected = [
                mid for mid in m_valid
                if x[job_id, mid].X > 0.5
            ]
            schedule.append((job_id, selected[0] if selected else None))

        return schedule, weight

    except Exception as e:
        print(f"[SOLVER]   Error: {e}")
        # Exception fallback assigns model 0 to every job without a budget check.
        return [(i, 0) for i in range(job_number)], [1] * job_number


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


# Default output labels; checkpoint metadata may override these lists.
AGENT_CLASSES = [
    'Ped', 'Car', 'Cyc', 'Mobike',
    'MedVeh', 'LarVeh', 'Bus', 'EmVeh'
]


BEHAVIOR_CLASSES = [
    'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop',
    'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht',
    'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj'
]


# Fallback label order for seven-output checkpoints without behavior metadata.
LEGACY_BEHAVIOR_CLASSES_7 = [
    'MovAway', 'MovTow', 'Mov', 'Stop',
    'IncatLft', 'Ovtak', 'PushObj'
]


# Behavior labels eligible to trigger braking with distance/confidence checks.
DANGER_BEHAVIORS = {"MovTow", "Xing", "XingFmLft", "XingFmRht", "Stop"}


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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE]   Using {DEVICE}")

RUN_DURATION_S = 100

os.makedirs(LOG_DIR, exist_ok=True)


# Shared camera state and per-actor crop/prediction caches.
latest_frame        = None
agent_frame_buffers = {}
agent_predictions   = {}


# Learned softmax pooling over the GRU time dimension.
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


# Per-frame ShuffleNet features followed by a GRU, attention, and two heads.
class ShuffleNetGRU(nn.Module):
    def __init__(self,
                 num_agent_classes    = 8,
                 num_behavior_classes = 7,
                 gru_hidden           = 256,
                 dropout              = 0.5,
                 width_mult           = 1.0):
        super().__init__()
        import torchvision.models as tvm

        # Portfolio width labels select these backbone/hidden-size groups;
        # they do not directly scale every convolution by width_mult.
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
        # Fold time into the CNN batch, then restore the sequence for the GRU.
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.projection(x)
        x = x.view(B, T, -1)
        gru_out, _ = self.gru(x)
        context    = self.temporal_attn(gru_out)
        context    = self.dropout(context)
        fused      = self.fusion(context)
        return self.agent_head(fused), self.behavior_head(fused)


# Compatibility alias for earlier callers.
AlexNetLSTM = ShuffleNetGRU


# Fixed accuracy and nominal inference-latency lookup table.
# These values are configuration inputs, not measurements from the current run.
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

MODEL_NUMBER     = len(MODEL_REGISTRY)
MODEL_ACCURACIES = [m['accuracy']      for m in MODEL_REGISTRY]
MODEL_EXEC_TIMES = [m['exec_time_ms']  for m in MODEL_REGISTRY]


# Validate model eligibility once at startup.
M_VALID = [m for m in range(MODEL_NUMBER) if MODEL_ACCURACIES[m] >= A_REQ]
if not M_VALID:
    raise ValueError(
        f"A_REQ={A_REQ} excludes every model "
        f"(max accuracy in portfolio = {max(MODEL_ACCURACIES):.4f}). "
        f"Lower A_REQ.")
print(f"[A_REQ]    A_req={A_REQ:.4f} -> M_valid = "
      f"{len(M_VALID)}/{MODEL_NUMBER} models")
for _m in range(MODEL_NUMBER):
    _tag = "eligible" if _m in M_VALID else "EXCLUDED"
    print(f"           {MODEL_REGISTRY[_m]['display']:22s} "
          f"A={MODEL_ACCURACIES[_m]:.4f}  {_tag}")


def load_shufflenet_gru(pth_path, device, width_mult=1.0, int8=False):
    """Load checkpoint weights and labels, optionally applying CPU dynamic INT8.
    
    Checkpoint label metadata updates the global class lists. Linear and GRU
    layers are quantized; the convolutional backbone remains floating point.
    Weight-loading errors are reported but do not abort model construction."""
    global AGENT_CLASSES, BEHAVIOR_CLASSES

    ckpt = torch.load(pth_path, map_location=device)
    if "model_state_dict" in ckpt:
        if "agent_classes"    in ckpt: AGENT_CLASSES    = ckpt["agent_classes"]
        if "behavior_classes" in ckpt: BEHAVIOR_CLASSES = ckpt["behavior_classes"]
    has_own_labels = ("model_state_dict" in ckpt and
                      "behavior_classes" in ckpt)


    # Check available classifier dimensions against the label lists before loading.
    sd = ckpt.get("model_state_dict", ckpt)
    n_agent_ckpt = sd["agent_head.2.weight"].shape[0]    if "agent_head.2.weight"    in sd else None
    n_beh_ckpt   = sd["behavior_head.3.weight"].shape[0] if "behavior_head.3.weight" in sd else None

    if (n_beh_ckpt is not None and n_beh_ckpt != len(BEHAVIOR_CLASSES)
            and not has_own_labels
            and n_beh_ckpt == len(LEGACY_BEHAVIOR_CLASSES_7)):
        print(f"  [WARN] {os.path.basename(pth_path)}: 7 behaviour outputs and "
              f"no label list -> using legacy 7-class list (no 'Xing').")
        BEHAVIOR_CLASSES = list(LEGACY_BEHAVIOR_CLASSES_7)

    if n_agent_ckpt is not None and n_agent_ckpt != len(AGENT_CLASSES):
        raise ValueError(
            f"{pth_path}: checkpoint has {n_agent_ckpt} agent outputs but "
            f"AGENT_CLASSES has {len(AGENT_CLASSES)}. Save 'agent_classes' "
            f"in the checkpoint or fix the list.")
    if n_beh_ckpt is not None and n_beh_ckpt != len(BEHAVIOR_CLASSES):
        raise ValueError(
            f"{pth_path}: checkpoint has {n_beh_ckpt} behaviour outputs but "
            f"BEHAVIOR_CLASSES has {len(BEHAVIOR_CLASSES)}. Save "
            f"'behavior_classes' in the checkpoint (same order as training) "
            f"or fix the list.")

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


# Compatibility alias for earlier callers.
load_alexnet_lstm = load_shufflenet_gru


# Load the portfolio once at module initialization; failed entries use the base model.
print("[MODEL]    Loading ShuffleNetV2-GRU-TemporalAttention models...")
base_model = load_shufflenet_gru(os.path.join(MODEL_DIR, '1.0.pth'), DEVICE, width_mult=1.0, int8=False)
print(f"[MODEL]    Agent classes   : {AGENT_CLASSES}")
print(f"[MODEL]    Behavior classes ({len(BEHAVIOR_CLASSES)}): {BEHAVIOR_CLASSES}")
_missing_danger = DANGER_BEHAVIORS - set(BEHAVIOR_CLASSES)
if _missing_danger:
    print(f"[WARN]     DANGER_BEHAVIORS not in the model's label set "
          f"(can never trigger braking): {sorted(_missing_danger)}")
else:
    print(f"[MODEL]    DANGER_BEHAVIORS {sorted(DANGER_BEHAVIORS)} "
          f"all present in label set")

loaded_models = {}
for reg in MODEL_REGISTRY:
    try:
        m = load_shufflenet_gru(
            os.path.join(MODEL_DIR, reg['pth']), DEVICE,
            width_mult=reg.get('width_mult', 1.0),
            int8=reg.get('int8', False))
        loaded_models[reg['name']] = m
        assert len(BEHAVIOR_CLASSES) == base_model.behavior_head[-1].out_features, \
            f"{reg['pth']} changed the label set to a different size than the base model"
        print(f"  [OK]   {reg['name']:30s} <- {reg['pth']}")
    except Exception as e:
        print(f"  [FAIL] {reg['name']:30s} : {e} — using base model")
        loaded_models[reg['name']] = base_model
print("[MODELS]   All models ready\n")


# Load YOLOv5 from the local checkout and weights file.
print("[YOLO]     Loading YOLOv5n...")
yolo_model = torch.hub.load(
    './yolov5', 'custom', path='yolov5n.pt', source='local'
)
yolo_model.conf    = YOLO_CONF
yolo_model.classes = YOLO_CLASSES
print("[YOLO]     Loaded on", next(yolo_model.parameters()).device)


# Resize crops and normalize RGB channels with ImageNet statistics.
inference_transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE[1], FRAME_SIZE[0])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_crop(frame_bgr, box, pad=10):
    """Pad and clip an xyxy box, then return a normalized RGB crop tensor.
    
    Use the full frame if the clipped crop is empty."""
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
    """Return a frame copy and supported YOLO detections with pixel coordinates."""
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

        detections.append({
            'box'        : (x1, y1, x2, y2),
            'center'     : (cx, cy),
            'confidence' : float(conf),
            'yolo_class' : cls_int,
            'agent_label': agent_label,
        })


    return annotated, detections


def run_inference(model, frame_buffer):
    """Classify a buffered crop sequence on the selected model's device.
    
    Return agent label, behavior label, and their softmax confidences.
    The input sequence has shape (T, C, H, W) before adding a batch axis."""
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
        agent_frame_buffers.pop(pid, None)
        agent_predictions.pop(pid, None)
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
    """Spawn pedestrians using the fixed standing, walking, and crossing mix.
    
    The zip below limits attempts to the shorter of the distance and behavior
    lists; the current behavior list contains ten entries."""
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


def draw_hud(frame, speed_ms, brake_active=False):
    """Return a frame with speed text and an optional braking banner."""
    if frame is None:
        return None
    h, w    = frame.shape[:2]
    display = frame.copy()


    speed_text = f"Speed: {speed_ms:.2f} m/s"
    (tw, th), baseline = cv2.getTextSize(
        speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    pad_x, pad_y = 8, 6
    x0, y0 = 10, 15
    cv2.rectangle(
        display,
        (x0, y0),
        (x0 + tw + 2 * pad_x, y0 + th + 2 * pad_y + baseline),
        (0, 0, 0), -1
    )
    cv2.putText(
        display, speed_text,
        (x0 + pad_x, y0 + th + pad_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0), 2
    )


    if brake_active:
        banner  = "BRAKE PRESSED!"
        (bw, bh), bbase = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3
        )
        bpad    = 10
        bx      = (w - bw) // 2 - bpad
        by      = h - bh - 2 * bpad - bbase
        cv2.rectangle(
            display,
            (bx, by),
            (bx + bw + 2 * bpad, h - bpad),
            (0, 0, 0), -1
        )
        cv2.putText(
            display, banner,
            (bx + bpad, h - bpad - bbase - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 255, 0), 3
        )

    return display


def draw_prediction_on_frame(display, det,
                              agent_pred, beh_pred,
                              beh_conf, model_name,
                              is_danger=False):
    """Draw a detection box and model/behavior labels in place.
    
    Use a thicker box for dangerous predictions and omit labels for
    buffering or skipped jobs. All actor types use the same box color."""
    if display is None or det is None:
        return display
    x1, y1, x2, y2 = det['box']
    col       = AGENT_COLORS["Ped"]
    thickness = 3 if is_danger else 2
    cv2.rectangle(display, (x1, y1), (x2, y2), col, thickness)


    if beh_pred in ("buffering...", "buffering",
                    "skipped", "Skipped", ""):
        return display


    short = model_name.replace("ShuffleNetV2x", "ShuffleNet-")

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness2 = 2

    line1 = short
    line2 = beh_pred

    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness2)
    (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness2)
    pad   = 3
    box_w = max(w1, w2) + pad * 2
    box_h = h1 + h2 + pad * 3

    tx = x1
    ty = max(y1 - box_h - 4, 0)
    cv2.rectangle(display, (tx, ty),
                  (tx + box_w, ty + box_h), (0, 0, 0), -1)

    cv2.putText(display, line1, (tx + pad, ty + pad + h1),
                font, font_scale, col, thickness2)
    cv2.putText(display, line2, (tx + pad, ty + pad + h1 + h2 + pad),
                font, font_scale, col, thickness2)
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

    vehicle     = None
    pedestrians = []
    road_vehicles = []
    camera      = None
    col_sensor  = None


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
        cv2.namedWindow(
            "CARLA - ROAD Multi-Agent Behavior Prediction",
            cv2.WINDOW_AUTOSIZE
        )

        print("\n" + "="*65)
        print(f"  SIMULATION RUNNING — ROAD dataset urban scene")
        print(f"  Pedestrians      : {TOTAL_PEDS}")
        print(f"  Cars             : {TOTAL_CARS}")
        print(f"  Motorcycles      : {TOTAL_MOTORCYCLES}")
        print(f"  Cyclists         : {TOTAL_BICYCLES}")
        print(f"  Buses            : {TOTAL_BUSES}")
        print(f"  Large vehicles   : {TOTAL_TRUCKS}")
        print(f"  Duration cap     : {duration_sec:.0f}s (or until crash)")
        print(f"  Deadline         : {DEADLINE_MS} ms per cycle")
        print(f"  Models           : {MODEL_NUMBER}")
        print(f"  Agent classes    : {AGENT_CLASSES}")
        print(f"  Behavior classes : {BEHAVIOR_CLASSES}")
        print(f"  Danger behaviors : {DANGER_BEHAVIORS}")
        print(f"  Scheduler        : Gurobi ILP")
        print("  Press Q to quit")
        print("="*65 + "\n")

        start_time       = time.time()
        was_braking      = False
        brake_hold_start_time = None
        brake_hold       = False
        ped_counter      = TOTAL_PEDS + 1
        detections       = []
        last_unstick_check = time.time()


        # Actor ID -> deferred job. Fresh detections replace stale entries on re-admission.
        deferred_jobs    = {}


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

            n_yolo        = 0
            display_frame = None
            detections    = []


            # Perception: detect objects and append crops using simulator-based association.
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


            # Merge current detections with deferred live actors, without duplicate actor IDs.
            # Undetected deferred actors retain prior crops but lose their display boxes.
            current_map = {va['actor'].id: va for va in visible_agents}
            for aid_d, va_prev in list(deferred_jobs.items()):
                if aid_d in current_map:

                    current_map[aid_d]['deferred_count'] = \
                        va_prev.get('deferred_count', 0)
                    continue
                try:
                    alive = va_prev['actor'].is_alive
                except Exception:
                    alive = False
                if not alive:
                    deferred_jobs.pop(aid_d, None)
                    continue
                va_prev['det'] = None
                visible_agents.append(va_prev)
                current_map[aid_d] = va_prev


            # Safety analysis: heuristic TTC from simulator distance and speed magnitudes,
            # not a trajectory-intersection or full relative-velocity calculation.
            raw_spd  = get_vehicle_speed_ms(vehicle)
            car_spd  = raw_spd if raw_spd > 0.5 else 1.0
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


            # Evaluate proximity braking independently of model assignment.
            # The control command is applied later, after sequential predictions finish.
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

            # Image-space proximity fallback using central box position and relative area.
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


            # Schedule visible/deferred jobs, including actors whose crop buffers are not full.
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
                per_object    = []


                # Execute assigned predictions sequentially in job order; no inference preemption.
                for job_id, model_id in schedule:
                    va         = visible_agents[job_id]
                    actor      = va['actor']
                    aid        = actor.id
                    dist       = get_distance(vehicle, actor)
                    ttc        = ttc_list[job_id]
                    cross_done = va['cross_done']
                    buf_size   = len(agent_frame_buffers.get(aid, []))


                    # Run inference only when an assigned job has a full observation window.
                    if model_id is not None and buf_size >= N_FRAMES:
                        model_name    = MODEL_REGISTRY[model_id]['name']
                        display_model = MODEL_REGISTRY[model_id]['display']
                        sel_model     = loaded_models.get(model_name, base_model)
                        agent_pred, beh_pred, agent_conf, beh_conf = \
                            run_inference(sel_model,
                                         agent_frame_buffers[aid])
                        agent_predictions[aid] = (agent_pred, beh_pred)
                        deferred_jobs.pop(aid, None)
                        va['deferred_count'] = 0
                        n_assigned   += 1
                        model_acc     = MODEL_ACCURACIES[model_id]
                        assigned_accs.append(model_acc)
                        models_used.append(
                            MODEL_REGISTRY[model_id]['display']
                        )
                        agent_types_log.append(agent_pred)
                        behaviors_log.append(beh_pred)


                    # A buffering job clears deferral but does not perform inference this cycle.
                    elif model_id is not None:
                        prev = agent_predictions.get(
                            aid, (va['agent_type'], "buffering...")
                        )
                        agent_pred, beh_pred = prev
                        agent_conf = beh_conf = 0.0
                        model_name    = MODEL_REGISTRY[model_id]['name']
                        display_model = MODEL_REGISTRY[model_id]['display']
                        deferred_jobs.pop(aid, None)
                        va['deferred_count'] = 0
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
                        va['deferred_count'] = va.get('deferred_count', 0) + 1
                        # Retain skipped jobs for re-admission in the next cycle.
                        deferred_jobs[aid] = va
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
                        'deferred_cycles': va.get('deferred_count', 0),
                    })

                    is_danger = (beh_pred in DANGER_BEHAVIORS and
                                 dist < BRAKE_DIST and
                                 beh_conf >= CONF_THRESHOLD and
                                 not cross_done and
                                 model_id is not None)
                    if is_danger:
                        brake_active  = True
                        vehicle_state = "BRAKING"

                    if display_frame is not None:
                        det = va.get('det', None)
                        display_frame = draw_prediction_on_frame(
                            display_frame, det,
                            agent_pred, beh_pred, beh_conf,
                            display_model, is_danger
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

            if brake_active and not was_braking:
                print(f"\n  *** VEHICLE BRAKING ***")
            if not brake_active and was_braking:
                print("\n  *** VEHICLE RESUMING ***")
            was_braking = brake_active


            if brake_active:
                if not brake_hold:
                    brake_hold_start_time = time.time()
                brake_hold = True
            elif brake_hold:
                # Hold braking while a pooled actor has a cached dangerous behavior,
                # without rechecking distance or confidence in this release condition.
                still_danger = any(
                    agent_predictions.get(va['actor'].id,
                                         ("", ""))[1] in DANGER_BEHAVIORS
                    and not va['cross_done']
                    for va in visible_agents
                )
                if not still_danger:
                    brake_hold            = False
                    brake_hold_start_time = None


            # Override autopilot with full braking while the latch is active.
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


            # Show the braking banner after the latch has been held for one second.
            show_brake_banner = (
                brake_hold and
                brake_hold_start_time is not None and
                (time.time() - brake_hold_start_time) >= 1.0
            )
            if display_frame is not None:
                hud = draw_hud(display_frame, speed_ms, show_brake_banner)
                if hud is not None:
                    cv2.imshow(
                        "CARLA - ROAD Multi-Agent Behavior Prediction",
                        hud
                    )

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
    """Run one episode and append its collision flag to logs/RETINA/retina_runs.csv."""
    crashed = run_episode(duration_sec=RUN_DURATION_S)

    summary_dir = os.path.join(LOG_DIR, "RETINA")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "retina_runs.csv")
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
