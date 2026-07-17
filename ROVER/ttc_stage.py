# ttc_stage.py

import math

from config import (
    TTC_CONSTANT,
    YOLO_INTERVAL,
    ROBOT_VELOCITY_MM_S,
)


def add_ttc_to_objects(
    yolo_result: dict,
) -> list[dict]:

    scheduler_objects = []

    for obj in yolo_result.get(
        "detections",
        [],
    ):

        x1, y1, x2, y2 = obj["box"]

        box_height_px = max(
            1.0,
            float(y2 - y1),
        )

        box_height_ratio = (
            box_height_px
            / max(
                1.0,
                float(
                    obj["image_height"]
                ),
            )
        )

        ttc = (
            TTC_CONSTANT
            / box_height_px
        )

        estimated_distance_mm = (
            ttc
            * ROBOT_VELOCITY_MM_S
        )

        cycles_to_collision = math.ceil(
            ttc / YOLO_INTERVAL
        )

        obj_with_ttc = {
            **obj,

            "box_height_px": round(
                box_height_px,
                2,
            ),

            "box_height_ratio": round(
                box_height_ratio,
                4,
            ),

            "ttc": round(
                ttc,
                3,
            ),

            "initial_ttc": round(
                ttc,
                3,
            ),

            "distance_mm": round(
                estimated_distance_mm,
                2,
            ),

            "cycles_to_collision": cycles_to_collision,
        }

        scheduler_objects.append(
            obj_with_ttc
        )

    return scheduler_objects