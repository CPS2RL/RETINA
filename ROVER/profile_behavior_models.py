import re
import time
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


MODEL_PATHS = [
    ("0.12 FP32", 0.12, "./models/0.12.pth", False),
    ("0.12 INT8", 0.12, "./models/0.12-INT8.pth", True),

    ("0.25 FP32", 0.25, "./models/0.25.pth", False),
    ("0.25 INT8", 0.25, "./models/0.25-INT8.pth", True),

    ("0.50 FP32", 0.50, "./models/0.5.pth", False),
    ("0.50 INT8", 0.50, "./models/0.5-INT8.pth", True),

    ("0.75 FP32", 0.75, "./models/0.75.pth", False),
    ("0.75 INT8", 0.75, "./models/0.75-INT8.pth", True),

    ("1.00 FP32", 1.00, "./models/1.0.pth", False),
    ("1.00 INT8", 1.00, "./models/1.0-INT8.pth", True),
]


IMAGE_DIR = Path("./test_images")
SEQUENCE_LENGTH = 16
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class ShuffleNetGRU(nn.Module):
    def __init__(
        self,
        width_mult,
        num_agent_classes=8,
        num_behavior_classes=7,
        dropout=0.5,
    ):
        super().__init__()

        if width_mult >= 0.75:
            backbone = models.shufflenet_v2_x1_0(weights=None)
            hidden_size = 256
        elif width_mult >= 0.5:
            backbone = models.shufflenet_v2_x0_5(weights=None)
            hidden_size = 128
        else:
            backbone = models.shufflenet_v2_x0_5(weights=None)
            hidden_size = 64

        feature_size = backbone.fc.in_features

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.temporal_attn = TemporalAttention(hidden_size, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)

        self.agent_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_agent_classes),
        )

        self.behavior_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_behavior_classes),
        )

    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.shape

        x = x.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width,
        )

        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.projection(x)
        x = x.reshape(batch_size, sequence_length, -1)

        x, _ = self.gru(x)
        x = self.temporal_attn(x)
        x = self.fusion(self.dropout(x))

        return self.agent_head(x), self.behavior_head(x)


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_input():
    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg"),
        key=natural_sort_key,
    )

    if len(image_paths) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected at least {SEQUENCE_LENGTH} JPG images, "
            f"found {len(image_paths)}."
        )

    frames = []

    for path in image_paths[:SEQUENCE_LENGTH]:
        with Image.open(path) as image:
            frames.append(IMAGE_TRANSFORM(image.convert("RGB")))

    return torch.stack(frames).unsqueeze(0)


def clean_state_dict(state):
    return {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state.items()
    }


def load_model(model_path, width_mult, int8=False):
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    state = clean_state_dict(checkpoint)

    model = ShuffleNetGRU(width_mult=width_mult)

    model_state = model.state_dict()

    compatible_state = {
        key: value
        for key, value in state.items()
        if (
            key in model_state
            and isinstance(value, torch.Tensor)
            and value.shape == model_state[key].shape
        )
    }


    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    model.eval()
    model.cpu()



    if int8:
        try:
            model = torch.ao.quantization.quantize_dynamic(
                model,
                qconfig_spec={nn.Linear, nn.GRU},
                dtype=torch.qint8,
                inplace=False,
            )
            model.eval()

        except Exception as exc:
            print(f"  [WARN] INT8 quantization failed: {exc}")
            print("  [WARN] Continuing with the FP32 model.")
            model = model.cpu().eval()

    return model


@torch.inference_mode()
def measure_p95(model, input_tensor):
    for _ in range(WARMUP_RUNS):
        model(input_tensor)

    latencies = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter_ns()
        model(input_tensor)
        latencies.append(
            (time.perf_counter_ns() - start) / 1_000_000
        )

    return torch.quantile(
        torch.tensor(latencies),
        0.95,
    ).item()


def main():
    input_tensor = load_input().cpu()

    for label, width_mult, model_path, int8 in MODEL_PATHS:
        model = load_model(
            model_path=model_path,
            width_mult=width_mult,
            int8=int8,
        )

        p95_ms = measure_p95(model, input_tensor)
        print(f"{label},{p95_ms:.3f}")

if __name__ == "__main__":
    main()