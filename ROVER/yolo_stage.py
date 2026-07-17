# yolo_stage.py

from datetime import datetime
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

from config import (
    ANNOTATED_DIR,
    YOLO_IMG_SIZE,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_MAX_DET,
    TARGET_LABELS,
)


def wait_until_file_ready(
    image_path: Path,
    checks: int = 5,
    delay: float = 0.05,
) -> bool:

    previous_size = -1

    for _ in range(checks):

        if not image_path.exists():
            return False

        if image_path.name.endswith(".tmp.jpg"):
            return False

        current_size = image_path.stat().st_size

        if (
            current_size > 0
            and current_size == previous_size
        ):

            image = cv2.imread(
                str(image_path)
            )

            if image is not None:
                return True

        previous_size = current_size

        time.sleep(delay)

    return False


def get_latest_image(
    image_dir: Path,
) -> Path | None:

    images = [
        p
        for p in image_dir.glob("*.jpg")
        if not p.name.endswith(".tmp.jpg")
    ]

    if not images:
        return None

    return max(
        images,
        key=lambda p: p.stat().st_mtime_ns,
    )


def detect_objects(
    model: YOLO,
    image_path: Path,
) -> dict:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Skipping unreadable image: {image_path}"
        )

        return {
            "image_path": str(image_path),
            "annotated_image_path": "",
            "object_count": 0,
            "detections": [],
            "image_width": 0,
            "image_height": 0,
            "yolo_called_at": "",
            "yolo_returned_at": "",
            "inference_time_sec": 0.0,
            "image_read_error": True,
        }

    image_height, image_width = image.shape[:2]

    ANNOTATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotated_image = image.copy()

    yolo_called_at = datetime.now()

    inference_start = time.perf_counter()

    try:

        result = model(
            image,
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            agnostic_nms=True,
            max_det=YOLO_MAX_DET,
            verbose=False,
        )[0]

    except Exception as e:

        print(
            f"YOLO failed on image {image_path.name}: {e}"
        )

        return {
            "image_path": str(image_path),
            "annotated_image_path": "",
            "object_count": 0,
            "detections": [],
            "image_width": image_width,
            "image_height": image_height,
            "yolo_called_at": yolo_called_at.isoformat(
                timespec="milliseconds"
            ),
            "yolo_returned_at": datetime.now().isoformat(
                timespec="milliseconds"
            ),
            "inference_time_sec": 0.0,
            "image_read_error": True,
        }

    inference_end = time.perf_counter()

    yolo_returned_at = datetime.now()

    inference_time_sec = (
        inference_end - inference_start
    )

    detections = []

    print("\n====================================")
    print(f"Processing Image: {image_path.name}")
    print("====================================")

    if result.boxes is not None:

        for box in result.boxes:

            label = model.names[
                int(box.cls[0])
            ]

            print(
                f"Detected label: {label}"
            )

            if label not in TARGET_LABELS:
                continue

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0],
            )

            x1 = max(
                0,
                min(
                    x1,
                    image_width - 1,
                ),
            )

            y1 = max(
                0,
                min(
                    y1,
                    image_height - 1,
                ),
            )

            x2 = max(
                0,
                min(
                    x2,
                    image_width - 1,
                ),
            )

            y2 = max(
                0,
                min(
                    y2,
                    image_height - 1,
                ),
            )

            if x2 <= x1 or y2 <= y1:
                continue

            box_xyxy = [
                x1,
                y1,
                x2,
                y2,
            ]

            detection = {
                "label": label,
                "confidence": confidence,
                "box": box_xyxy,
                "image_width": image_width,
                "image_height": image_height,
            }

            detections.append(
                detection
            )

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1,
            )

    annotated_path = (
        ANNOTATED_DIR
        / f"annotated_{image_path.name}"
    )

    cv2.imwrite(
        str(annotated_path),
        annotated_image,
    )

    return {
        "image_path": str(image_path),
        "annotated_image_path": str(annotated_path),
        "object_count": len(detections),
        "detections": detections,
        "image_width": image_width,
        "image_height": image_height,
        "yolo_called_at": yolo_called_at.isoformat(
            timespec="milliseconds"
        ),
        "yolo_returned_at": yolo_returned_at.isoformat(
            timespec="milliseconds"
        ),
        "inference_time_sec": round(
            inference_time_sec,
            4,
        ),
        "image_read_error": False,
    }