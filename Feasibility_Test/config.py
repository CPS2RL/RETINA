# config.py
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent

SCHEDULER_MODE="RETINA_LOW"
#SCHEDULER_MODE="RETINA_HIGH"
#SCHEDULER_MODE="RETINA"
#SCHEDULER_MODE="CAMOT"
#SCHEDULER_MODE="RTDeepIoT"

SUPPORTED_MODES={"RETINA_LOW","RETINA_HIGH","RETINA","CAMOT","RTDeepIoT"}

# Common hardware experiment deadline.
DEADLINE_MS=2500.0
CAMOT_TASK_DEADLINE_MS=2500.0
SCHEDULER_OVERHEAD_MS=10.0
A_REQ=0.20

# Output.
RUN_DIR=BASE_DIR/SCHEDULER_MODE
IMAGE_DIR=RUN_DIR/"captured_images"
ANNOTATED_DIR=RUN_DIR/"annotated_images"
LOG_DIR=RUN_DIR/"logs"
for d in (IMAGE_DIR,ANNOTATED_DIR,LOG_DIR): d.mkdir(parents=True,exist_ok=True)
CSV_LOG_PATH=LOG_DIR/f"{SCHEDULER_MODE}.csv"

# Camera.
TARGET_FPS=20
FRAME_WIDTH,FRAME_HEIGHT=1024,768
OUTPUT_IMAGE_WIDTH,OUTPUT_IMAGE_HEIGHT=320,320
JPEG_QUALITY=100

# YOLO.
YOLO_MODEL_PATH=BASE_DIR/"best_yolov8n.pt"
YOLO_INTERVAL=3.0
YOLO_IMG_SIZE=320
YOLO_CONF=0.40
YOLO_IOU=0.30
YOLO_MAX_DET=6
TARGET_LABELS={"person","car"}

# TTC from bounding-box height.
TTC_CONSTANT=100.0
ROBOT_VELOCITY_MM_S=70.0
ROBOT_SPEED_MPS=ROBOT_VELOCITY_MM_S/1000.0

# Hardware tracking.
TRACK_IOU_THRESHOLD=0.30
TRACK_MAX_MISSES=10

# Robot safety.
BRAKE_TTC_SEC=2.5
SAFE_RELEASE_TTC_SEC=2.0
CONF_THRESHOLD=0.60

# Behavior model.
NUM_FRAMES=16
BEHAVIOR_CLASSES=[
    "MovAway","Mov","MovTow","Brake","Stop","IncatLft","IncatRht","HazLit",
    "TurLft","TurRht","Wait2X","Ovtak","XingFmLft","XingFmRht","PushObj","Xing"
]
DANGER_BEHAVIORS={"MovTow","Xing","Stop"}


MODEL_REGISTRY=[
    {"name":"ShuffleNetV2x1.0","display":"ShuffleNet-1.0","path":BASE_DIR/"models/1.0.pth","accuracy":0.9250,"exec_time_ms":1127.921,"int8":False},
    {"name":"ShuffleNetV2x0.75","display":"ShuffleNet-0.75","path":BASE_DIR/"models/0.75.pth","accuracy":0.8140,"exec_time_ms":1139.681,"int8":False},
    {"name":"ShuffleNetV2x0.5","display":"ShuffleNet-0.50","path":BASE_DIR/"models/0.5.pth","accuracy":0.6630,"exec_time_ms":633.542,"int8":False},
    {"name":"ShuffleNetV2x0.25","display":"ShuffleNet-0.25","path":BASE_DIR/"models/0.25.pth","accuracy":0.4105,"exec_time_ms":639.454,"int8":False},
    {"name":"ShuffleNetV2x0.12","display":"ShuffleNet-0.12","path":BASE_DIR/"models/0.12.pth","accuracy":0.3470,"exec_time_ms":614.317,"int8":False},

    {"name":"ShuffleNetV2x1.0-INT8","display":"ShuffleNet-1.0-INT8","path":BASE_DIR/"models/1.0-INT8.pth","accuracy":0.7389,"exec_time_ms":1127.921,"int8":True},
    {"name":"ShuffleNetV2x0.75-INT8","display":"ShuffleNet-0.75-INT8","path":BASE_DIR/"models/0.75-INT8.pth","accuracy":0.5473,"exec_time_ms":1173.012,"int8":True},
    {"name":"ShuffleNetV2x0.5-INT8","display":"ShuffleNet-0.50-INT8","path":BASE_DIR/"models/0.5-INT8.pth","accuracy":0.4608,"exec_time_ms":608.753,"int8":True},
    {"name":"ShuffleNetV2x0.25-INT8","display":"ShuffleNet-0.25-INT8","path":BASE_DIR/"models/0.25-INT8.pth","accuracy":0.2963,"exec_time_ms":635.938,"int8":True},
    {"name":"ShuffleNetV2x0.12-INT8","display":"ShuffleNet-0.12-INT8","path":BASE_DIR/"models/0.12-INT8.pth","accuracy":0.2695,"exec_time_ms":615.625,"int8":True},
]

RETINA_LOW_MODEL="ShuffleNetV2x0.12-INT8"
RETINA_HIGH_MODEL="ShuffleNetV2x1.0"

# CA-MOT.
CAMOT_SCHEDULER="EDF-Slack"
CAMOT_TTC_CRITICAL_SEC=2.0
CAMOT_DET_SIZE={"L":160,"M":256,"H":416}
CAMOT_ASSOC_MAX_FEAT={"L":1,"M":2,"H":3}

CAMOT_WCET_DET={"L":154.9,"M":154.9,"H":536.3}
CAMOT_WCET_ASSOC={"L":0.101,"M":347.482,"H":662.670,}
CAMOT_TASK_PERIOD_MS=DEADLINE_MS
CAMOT_TASK_DEADLINE_MS=DEADLINE_MS

# RTDeepIoT 

RTDEEPIOT_CHECKPOINT_PATH=BASE_DIR/"rtdeepiot/checkpoints/best_road_behavior.pth"
RTDEEPIOT_WCET_PATH=BASE_DIR/"rtdeepiot/checkpoints/wcet_road_behavior.json"

RTDEEPIOT_UTILITY_HEURISTIC="exp"
RTDEEPIOT_REWARD_QUANTIZATION_DELTA=0.01
RTDEEPIOT_CONFIDENCE_EMA_ALPHA=0.30
RTDEEPIOT_DEFAULT_SEED_CONFIDENCE=0.50
RTDEEPIOT_MAX_DEPTH=3


RTDEEPIOT_BUDGET_MS=2500.0
# Robot.
SERIAL_PORT="/dev/ttyAMA0"
BAUDRATE=115200
FORWARD_LEFT,FORWARD_RIGHT=0.113,0.1