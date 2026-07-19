# behavior_stage.py

from pathlib import Path
import time

import cv2
import torch
import torch.nn as nn
import torchvision.models as tvm

from config import (
    IMAGE_DIR,
    NUM_FRAMES,
)

BEHAVIOR_CLASSES = [

    "MovAway","MovTow","Mov","Brake","Stop","IncatLft","IncatRht","HazLit",
    "TurLft","TurRht","Wait2X","Ovtak","XingFmLft","XingFmRht","PushObj","Xing"
]

class TemporalAttention(nn.Module):

    def __init__(
        self,
        hidden=256,
    ):

        super().__init__()

        self.attn = nn.Sequential(

            nn.Linear(
                hidden,
                hidden // 2,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden // 2,
                1,
            ),
        )

    def forward(self, x):

        weights = torch.softmax(

            self.attn(x),

            dim=1,
        )

        return (
            weights * x
        ).sum(dim=1)

class BehaviorModel(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = (
            tvm.shufflenet_v2_x1_0(
                weights=None
            )
        )

        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection = nn.Sequential(

            nn.Linear(1024, 256),

            nn.ReLU(),
        )

        self.gru = nn.GRU(
            256,
            256,
            batch_first=True,
        )

        self.temporal_attn = (
            TemporalAttention(256)
        )

        self.fusion = nn.Sequential(

            nn.Linear(512, 256),

            nn.ReLU(),
        )

        self.behavior_head = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Linear(128, 16),
        )

    def forward(self, x):

        B, T, C, H, W = x.shape

        x = x.view(
            B * T,
            C,
            H,
            W,
        )

        x = self.backbone(x)

        x = self.pool(x).flatten(1)

        x = self.projection(x)

        x = x.view(B, T, -1)

        x, _ = self.gru(x)

        attention = (
            self.temporal_attn(x)
        )

        last = x[:, -1]

        x = torch.cat(
            [attention, last],
            dim=1,
        )

        x = self.fusion(x)

        return self.behavior_head(x)



def load_behavior_model(
    model_path: str,
):

    model = BehaviorModel()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    state = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model_dict = model.state_dict()

    filtered = {

        k: v

        for k, v in state.items()

        if (
            k in model_dict
            and model_dict[k].shape
            == v.shape
        )
    }

    model_dict.update(filtered)

    model.load_state_dict(
        model_dict
    )


    model = (
        torch.quantization
        .quantize_dynamic(

            model,

            {nn.Linear, nn.GRU},

            dtype=torch.qint8,
        )
    )

    model.eval()

    return model



def get_previous_sequence(
    image_path: str,
    num_frames: int,
):

    current_path = Path(
        image_path
    )

    images = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    try:

        current_index = (
            images.index(current_path)
        )

    except ValueError:

        return []

    start_index = max(
        0,
        current_index
        - num_frames
        + 1,
    )

    sequence = images[
        start_index:
        current_index + 1
    ]

    return sequence

def fixed_size_crop_from_box(
    frame,
    box,
    crop_size=320,
):

    h, w = frame.shape[:2]

    crop_size = min(
        crop_size,
        w,
        h,
    )

    x1, y1, x2, y2 = map(
        int,
        box,
    )

    center_x = int(
        (x1 + x2) / 2
    )

    center_y = int(
        (y1 + y2) / 2
    )

    half = crop_size // 2

    crop_x1 = center_x - half
    crop_y1 = center_y - half

    crop_x1 = max(
        0,
        min(
            crop_x1,
            w - crop_size,
        ),
    )

    crop_y1 = max(
        0,
        min(
            crop_y1,
            h - crop_size,
        ),
    )

    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    return crop

def preprocess(crop):


    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )

    crop = (
        torch.from_numpy(crop)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return crop



def run_behavior_model(
    model,
    image_path: str,
    obj: dict,
):


    sequence_paths = (
        get_previous_sequence(
            image_path,
            NUM_FRAMES,
        )
    )

    if len(sequence_paths) < NUM_FRAMES:

        print(
            "Not enough previous frames"
        )

        return None

    x1, y1, x2, y2 = map(
        int,
        obj["box"]
    )

    frames = []

    # =====================================================
    # LOAD FRAME SEQUENCE
    # =====================================================

    for path in sequence_paths:

        frame = cv2.imread(
            str(path)
        )

        if frame is None:

            continue

        crop = fixed_size_crop_from_box(
        frame,
        obj["box"],
        crop_size=90,
        )

        if crop.size == 0:

            continue

        crop = preprocess(
            crop
        )

        frames.append(crop)

    # =====================================================
    # VALIDATE SEQUENCE
    # =====================================================

    if len(frames) != NUM_FRAMES:

        print(
            "Invalid frame sequence"
        )

        return None

    # Shape:
    # [T, C, H, W]

    sequence = torch.stack(
        frames
    )

    # Shape:
    # [1, T, C, H, W]

    sequence = (
        sequence.unsqueeze(0)
    )

    # =====================================================
    # INFERENCE
    # =====================================================

    with torch.no_grad():

        start = (
            time.perf_counter()
        )

        output = model(sequence)

        end = (
            time.perf_counter()
        )

    runtime_ms = (
        end - start
    ) * 1000.0

    probabilities = torch.softmax(
        output,
        dim=1,
    )[0]

    prediction_index = (
        output.argmax(1).item()
    )

    behavior = (
        BEHAVIOR_CLASSES[
            prediction_index
        ]
    )

    return {

        "job_uid":
            obj["job_uid"],

        "label":
            obj["label"],

        "behavior":
            behavior,

        "runtime_ms":
            round(
                runtime_ms,
                2,
            ),

        "sequence_length":
            NUM_FRAMES,

        "probabilities": {

            cls:
                round(
                    probabilities[i].item(),
                    6,
                )

            for i, cls
            in enumerate(
                BEHAVIOR_CLASSES
            )
        },
    }