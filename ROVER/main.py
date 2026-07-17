# main.py

import time
from pathlib import Path

from pipeline_logger import SchedulerCSVLogger
from ultralytics import YOLO
import cv2

from config import (
    IMAGE_DIR,
    YOLO_MODEL_PATH,
    YOLO_INTERVAL,
    MODEL_REGISTRY,
    DEADLINE_MS,
    SERIAL_PORT,
    BAUDRATE,
    DANGER_BEHAVIORS,
    SCHEDULER_MODE,
    CSV_LOG_PATH,
    CROP_DIR,
)

from capture_camera import start_camera_capture

from yolo_stage import (
    get_latest_image,
    wait_until_file_ready,
    detect_objects,
)

from ttc_stage import add_ttc_to_objects

from scheduler_stage import (
    ModelSpec,
    SchedulerMemory,
    run_scheduler,
)

from behavior_stage import (
    load_behavior_model,
    run_behavior_model,
)

from robot_stage import (
    MotorController,
    choose_next_command,
)


def build_model_specs():

    specs = []

    for item in MODEL_REGISTRY:

        specs.append(
            ModelSpec(
                name=item["name"],
                model_path=str(item["path"]),
                accuracy=item["accuracy"],
                exec_time_ms=item["exec_time_ms"],
            )
        )

    return specs

def save_job_crop(
    image_path,
    obj,
):

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        print(f"Could not read image: {image_path}")
        return None

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        obj["box"],
    )

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        print(f"Invalid box for {obj['job_uid']}: {obj['box']}")
        return None

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    label_lines = [
        f"{obj['job_uid']}",
        f"{obj.get('label', '')}",
        f"TTC={obj.get('ttc', '')}",
    ]

    line_height = 16
    start_y = y1 + 16

    for i, text in enumerate(label_lines):

        y_text = start_y + i * line_height

        cv2.putText(
            frame,
            text,
            (x1 + 3, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    bbox_name = f"{obj['job_uid']}.jpg"

    bbox_path = CROP_DIR / bbox_name

    cv2.imwrite(
        str(bbox_path),
        frame,
    )

    obj["bbox_image_path"] = str(
        bbox_path
    )

    return bbox_path
    
def main():

    print("\nStarting Camera Capture...")

    camera_thread, camera_stop_event = start_camera_capture()

    time.sleep(1.0)

    print("\nLoading YOLO...")

    yolo_model = YOLO(YOLO_MODEL_PATH)

    csv_logger = SchedulerCSVLogger(CSV_LOG_PATH)

    model_specs = build_model_specs()

    scheduler_memory = SchedulerMemory()

    behavior_models = {}

    for spec in model_specs:


        behavior_models[spec.name] = load_behavior_model(
            spec.model_path
        )


    motor = MotorController(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
    )

    motor.start()

    cycle_id = 0
    last_image = None

    total_missed_deadlines = 0
    total_served_jobs = 0

    try:

        print("\nRunning Warm-Up Cycle...")

        while True:

            time.sleep(YOLO_INTERVAL)

            image_path = get_latest_image(IMAGE_DIR)

            if image_path is None:
                print("\nWarm-Up: No images found.")
                continue

            if not wait_until_file_ready(image_path):
                print(
                    f"\nWarm-Up: Image not ready: "
                    f"{image_path.name}"
                )
                continue



            yolo_result = detect_objects(
                yolo_model,
                image_path,
            )


            objects = add_ttc_to_objects(yolo_result)

            for idx, obj in enumerate(objects):

                obj["job_uid"] = f"warmup_{idx}"

                obj["label"] = obj.get(
                    "label",
                    "unknown",
                )

                obj["ttc"] = obj.get(
                    "ttc",
                    9999.0,
                )

            if len(objects) > 0:

                obj = objects[0]
                spec = model_specs[0]



                behavior_result = run_behavior_model(
                    behavior_models[spec.name],
                    str(image_path),
                    obj,
                )



            print("\nWarm-Up Complete.")

            last_image = image_path

            break

        while True:

            time.sleep(YOLO_INTERVAL)

            image_path = get_latest_image(IMAGE_DIR)

            if image_path is None:
                print("\nNo images found.")
                continue

            if image_path == last_image:
                print("\nNo new image available.")
                continue

            if not wait_until_file_ready(image_path):
                print(
                    f"\nImage not ready: "
                    f"{image_path.name}"
                )
                continue

            print("\n================================")
            print(f"Cycle {cycle_id}")

            yolo_result = detect_objects(
                yolo_model,
                image_path,
            )

            print(
                f"\nYOLO Runtime : "
                f"{yolo_result['inference_time_sec']} sec"
            )

            print(
                f"Objects Found : "
                f"{yolo_result['object_count']}"
            )

            objects = add_ttc_to_objects(yolo_result)

            new_jobs_arrived = len(objects)

            old_jobs_buffered = scheduler_memory.count_pending_jobs()

            scheduler_memory.update_ttc(YOLO_INTERVAL)

            missed_jobs = (
                scheduler_memory
                .remove_missed_deadline_jobs(
                    threshold_sec=1.0
                )
            )

            missed_deadlines_this_cycle = len(missed_jobs)

            total_missed_deadlines += missed_deadlines_this_cycle

            if missed_jobs:

                print("\nMissed Deadline Jobs:")

                for uid in missed_jobs:
                    print(uid)

            pending_objects = (
                scheduler_memory
                .add_current_cycle_jobs(
                    cycle_id,
                    objects,
                )
            )
            
            for obj in objects:

              crop_path = save_job_crop(
              image_path,
              obj,
              )

              if crop_path is not None:
                print(
                  f"Saved crop for {obj['job_uid']}: {crop_path.name}"
                  )

            # remaining_deadline_ms = (
            #     DEADLINE_MS
            #     - (
            #         yolo_result[
            #             "inference_time_sec"
            #         ]
            #         * 1000.0
            #     )
            # )

            remaining_deadline_ms = (
                DEADLINE_MS - 600
            )

            print(
                f"\nRemaining Budget : "
                f"{remaining_deadline_ms:.2f} ms"
            )

            schedule = run_scheduler(
                model_specs,
                pending_objects,
                remaining_deadline_ms,
                scheduler_mode=SCHEDULER_MODE,
            )

            print("\nScheduler Decisions:")

            behavior_results = []
            served_job_uids = []

            for job_id, model_id in schedule:

                obj = pending_objects[job_id]

                if model_id is None:

                    print(
                        f"\nSKIP : "
                        f"{obj['job_uid']}"
                    )

                    continue

                spec = model_specs[model_id]

                print(
                    f"\nRUN : "
                    f"{obj['job_uid']}"
                )

                print(
                    f"Class : "
                    f"{obj['label']}"
                )

                print(
                    f"TTC   : "
                    f"{obj['ttc']} sec"
                )

                print(
                    f"Model : "
                    f"{spec.name}"
                )

                print(
                    f"Exec  : "
                    f"{spec.exec_time_ms} ms"
                )

                behavior_result = run_behavior_model(
                    behavior_models[spec.name],
                    str(image_path),
                    obj,
                )

                if behavior_result is None:

                    print("Behavior failed.")

                    continue

                served_job_uids.append(
                    obj["job_uid"]
                )

                behavior_results.append(
                    behavior_result
                )

                print(
                    f"Behavior : "
                    f"{behavior_result['behavior']}"
                )

                print(
                    f"Runtime  : "
                    f"{behavior_result['runtime_ms']} ms"
                )

                robot_command = "GO"

                if (
                    behavior_result["behavior"]
                    in DANGER_BEHAVIORS
                ):
                    robot_command = "STOP"

                print(
                    f"Robot Command = {robot_command}"
                )

                csv_logger.log_job(
                    cycle=cycle_id,
                    image_name=Path(image_path).name,
                    yolo_runtime_sec=yolo_result[
                        "inference_time_sec"
                    ],
                    object_count=yolo_result[
                        "object_count"
                    ],
                    job_uid=obj["job_uid"],
                    cls=obj["label"],
                    ttc_sec=obj["ttc"],
                    selected_model=spec.name,
                    model_exec_ms=spec.exec_time_ms,
                    behavior=behavior_result[
                        "behavior"
                    ],
                    behavior_runtime_ms=behavior_result[
                        "runtime_ms"
                    ],
                    robot_command=robot_command,
                )

            scheduler_memory.remove_served_jobs(
                served_job_uids
            )

            served_this_cycle = len(served_job_uids)

            total_served_jobs += served_this_cycle

            buffered_for_next = scheduler_memory.count_pending_jobs()

            left_cmd, right_cmd = choose_next_command(
                behavior_results,
                DANGER_BEHAVIORS,
            )

            motor.set_command(
                left_cmd,
                right_cmd,
            )

            print("\n================================")
            print("Cycle Summary")
            print("================================")
            print(f"New jobs arrived      : {new_jobs_arrived}")
            print(f"Old jobs buffered     : {old_jobs_buffered}")
            print(f"Jobs served           : {served_this_cycle}")
            print(f"Buffered for next     : {buffered_for_next}")
            print(f"Missed deadlines      : {missed_deadlines_this_cycle}")
            print(f"Total served jobs     : {total_served_jobs}")
            print(f"Total missed deadlines: {total_missed_deadlines}")

            last_image = image_path

            cycle_id += 1

    except KeyboardInterrupt:

        print("\nStopping system...")

        camera_stop_event.set()

        motor.shutdown()

        csv_logger.close()


if __name__ == "__main__":

    main()