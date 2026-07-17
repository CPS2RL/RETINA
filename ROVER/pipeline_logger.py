# scheduler_logger.py

import csv
from pathlib import Path


class SchedulerCSVLogger:

    def __init__(
        self,
        csv_path: Path,
    ):

        self.csv_path = csv_path

        self.csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_exists = (
            self.csv_path.exists()
        )

        self.file = open(
            self.csv_path,
            "a",
            newline="",
        )

        self.writer = csv.writer(
            self.file
        )

        if not file_exists:

            self.writer.writerow([
                "cycle",
                "image",
                "yolo_runtime_sec",
                "object_count",
                "job_uid",
                "class",
                "ttc_sec",
                "selected_model",
                "model_exec_ms",
                "behavior",
                "behavior_runtime_ms",
                "robot_command",
            ])

            self.file.flush()

    def log_job(
        self,
        cycle,
        image_name,
        yolo_runtime_sec,
        object_count,
        job_uid,
        cls,
        ttc_sec,
        selected_model,
        model_exec_ms,
        behavior,
        behavior_runtime_ms,
        robot_command,
    ):

        self.writer.writerow([
            cycle,
            image_name,
            yolo_runtime_sec,
            object_count,
            job_uid,
            cls,
            round(ttc_sec, 3),
            selected_model,
            round(model_exec_ms, 2),
            behavior,
            round(
                behavior_runtime_ms,
                2,
            ),
            robot_command,
        ])

        self.file.flush()

    def close(self):

        self.file.close()