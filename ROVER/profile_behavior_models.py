# profile_behavior_models.py

import time
import csv
import torch
import cv2
from pathlib import Path

from config import (
    IMAGE_DIR,
    NUM_FRAMES,
    MODEL_REGISTRY,
    LOG_DIR,
)

from behavior_stage import load_behavior_model


def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")


    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return image


def load_image_sequence():

    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    if len(image_paths) < NUM_FRAMES:
        raise RuntimeError(
            f"Need at least {NUM_FRAMES} images in {IMAGE_DIR}, "
            f"but found {len(image_paths)}"
        )

    sequence_paths = image_paths[-NUM_FRAMES:]

    frames = []

    for path in sequence_paths:
        frame = preprocess_image(path)
        frames.append(frame)

    sequence = torch.stack(frames)

    sequence = sequence.unsqueeze(0)

    return sequence, sequence_paths


def profile_model(
    model,
    model_name,
    sequence,
    warmup_runs=5,
    test_runs=5,
):

    model.eval()

    with torch.no_grad():

        for _ in range(warmup_runs):
            _ = model(sequence)

    runtimes = []

    with torch.no_grad():

        for _ in range(test_runs):

            start = time.perf_counter()

            _ = model(sequence)

            end = time.perf_counter()

            runtime_ms = (end - start) * 1000.0

            runtimes.append(runtime_ms)

    avg_ms = sum(runtimes) / len(runtimes)
    min_ms = min(runtimes)
    max_ms = max(runtimes)

    return {
        "model": model_name,
        "runs": test_runs,
        "avg_ms": round(avg_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
    }


def main():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading image sequence from:")
    print(IMAGE_DIR)

    sequence, sequence_paths = load_image_sequence()

    print("\nImages used for profiling:")

    for p in sequence_paths:
        print(p.name)

    print("\nInput tensor shape:")
    print(sequence.shape)

    results = []

    for item in MODEL_REGISTRY:

        model_name = item["name"]
        model_path = item["path"]

        print("\n====================================")
        print(f"Profiling Model: {model_name}")
        print("====================================")

        model = load_behavior_model(
            str(model_path)
        )

        result = profile_model(
            model=model,
            model_name=model_name,
            sequence=sequence,
            warmup_runs=5,
            test_runs=5,
        )

        results.append(result)

        print(f"Average Runtime : {result['avg_ms']} ms")
        print(f"Min Runtime     : {result['min_ms']} ms")
        print(f"Max Runtime     : {result['max_ms']} ms")

    csv_path = LOG_DIR / "behavior_model_profile.csv"

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "runs",
                "avg_ms",
                "min_ms",
                "max_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n====================================")
    print("FINAL PROFILING RESULTS")
    print("====================================")

    for r in results:
        print(
            f"{r['model']:<25} "
            f"Avg: {r['avg_ms']:>8.2f} ms | "
            f"Min: {r['min_ms']:>8.2f} ms | "
            f"Max: {r['max_ms']:>8.2f} ms"
        )

    print("\nSaved CSV:")
    print(csv_path)


if __name__ == "__main__":
    main()