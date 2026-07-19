# config.py

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent



DEADLINE_MS = 1800.0

# SCHEDULER_MODE = "greedy_low_cost"
# SCHEDULER_MODE = "greedy_high_cost"
SCHEDULER_MODE = "RETINA"
# SCHEDULER_MODE = "MOT"



RUN_DIR = BASE_DIR / SCHEDULER_MODE

IMAGE_DIR = RUN_DIR / "captured_images"
CROP_DIR = RUN_DIR / "cropped_images"
ANNOTATED_DIR = RUN_DIR / "annotated_images"
LOG_DIR = RUN_DIR / "logs"

for d in [
    IMAGE_DIR,
    CROP_DIR,
    ANNOTATED_DIR,
    LOG_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )



TARGET_FPS = 20
FRAME_WIDTH = 1024
FRAME_HEIGHT = 768
JPEG_QUALITY = 100

OUTPUT_IMAGE_WIDTH = 320
OUTPUT_IMAGE_HEIGHT = 320



YOLO_MODEL_PATH = "best_yolov8n.pt"
YOLO_INTERVAL = 3.0
YOLO_IMG_SIZE = 320
YOLO_CONF = 0.2
YOLO_IOU = 0.30
YOLO_MAX_DET = 20
TARGET_LABELS = {
    "person",
    "car",
}


TTC_CONSTANT = 100.0


ROBOT_VELOCITY_MM_S = 70.0
ROBOT_SPEED_MPS = ROBOT_VELOCITY_MM_S / 1000.0



NUM_FRAMES = 16


MODEL_REGISTRY = [
    {
        "name": "ShuffleNet-0.12-int8",
        "path": BASE_DIR / "models/0.12-INT8.pth",
        "accuracy": 0.2695,
        "exec_time_ms": 615.625,
    },
    {
        "name": "ShuffleNet-0.25-int8",
        "path": BASE_DIR / "models/0.25-INT8.pth",
        "accuracy": 0.2963,
        "exec_time_ms": 635.938,
    },
    {
        "name": "ShuffleNet-0.50-int8",
        "path": BASE_DIR / "models/0.5-INT8.pth",
        "accuracy": 0.4608,
        "exec_time_ms": 608.753,
    },
    {
        "name": "ShuffleNet-0.75-int8",
        "path": BASE_DIR / "models/0.75-INT8.pth",
        "accuracy": 0.5473,
        "exec_time_ms": 1173.012,
    },
    {
        "name": "ShuffleNet-1.0-int8",
        "path": BASE_DIR / "models/1.0-INT8.pth",
        "accuracy": 0.7389,
        "exec_time_ms": 1127.921,
    },
    {
        "name": "ShuffleNet-0.12",
        "path": BASE_DIR / "models/0.12.pth",
        "accuracy": 0.3470,
        "exec_time_ms": 614.317,
    },
    {
        "name": "ShuffleNet-0.25",
        "path": BASE_DIR / "models/0.25.pth",
        "accuracy": 0.4105,
        "exec_time_ms": 639.454,
    },
    {
        "name": "ShuffleNet-0.50",
        "path": BASE_DIR / "models/0.5.pth",
        "accuracy": 0.6630,
        "exec_time_ms": 633.542,
    },
    {
        "name": "ShuffleNet-0.75",
        "path": BASE_DIR / "models/0.75.pth",
        "accuracy": 0.8140,
        "exec_time_ms": 1139.681,
    },
    {
        "name": "ShuffleNet-1.0",
        "path": BASE_DIR / "models/1.0.pth",
        "accuracy": 0.925,
        "exec_time_ms": 1127.921,
    },
]


DANGER_BEHAVIORS = {
    "Wait2X",
    "Xing",
    "XingFmRht",
    "XingFmLft",
}



CSV_LOG_PATH = (
    LOG_DIR
    / f"{SCHEDULER_MODE}.csv"
)



SERIAL_PORT = "/dev/ttyAMA0"
BAUDRATE = 115200

FORWARD_LEFT = 0.116
FORWARD_RIGHT = 0.1
