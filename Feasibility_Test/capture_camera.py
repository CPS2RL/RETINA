# capture_camera.py

from datetime import datetime
import time
import threading

import cv2
from picamera2 import Picamera2

from config import (
    IMAGE_DIR,
    CROP_DIR,
    TARGET_FPS,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    JPEG_QUALITY,
    OUTPUT_IMAGE_WIDTH,
    OUTPUT_IMAGE_HEIGHT,
)


def wait_until_next_second() -> None:

    now = time.time()
    sleep_time = 1.0 - (now % 1.0)

    if sleep_time > 0:
        time.sleep(sleep_time)


def save_image_atomic(
    image_path,
    image,
) -> bool:

    tmp_path = image_path.with_suffix(
        ".tmp.jpg"
    )

    success = cv2.imwrite(
        str(tmp_path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )

    if not success:
        print(f"Failed to save temp image: {tmp_path}")
        return False

    tmp_path.replace(image_path)

    return True


def capture_loop(
    stop_event: threading.Event | None = None,
) -> None:

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    picam2 = Picamera2()

    camera_config = picam2.create_video_configuration(
        main={
            "size": (
                FRAME_WIDTH,
                FRAME_HEIGHT,
            ),
            "format": "RGB888",
        }
    )

    picam2.configure(camera_config)
    picam2.start()

    time.sleep(0.5)

    wait_until_next_second()

    frame_interval = 1.0 / TARGET_FPS
    next_frame_time = time.perf_counter()

    frame_id = 0

    try:

        while True:

            if (
                stop_event is not None
                and stop_event.is_set()
            ):
                break

            now = time.perf_counter()

            if now < next_frame_time:
                time.sleep(
                    next_frame_time - now
                )

            capture_time = datetime.now()

            frame = picam2.capture_array()

            if frame is None:

                frame_id += 1
                next_frame_time += frame_interval
                continue

            cropped = frame[
                119:FRAME_HEIGHT,
                157:806,
            ]

            resized = cv2.resize(
                cropped,
                (
                    OUTPUT_IMAGE_WIDTH,
                    OUTPUT_IMAGE_HEIGHT,
                ),
                interpolation=cv2.INTER_AREA,
            )

            second_stamp = capture_time.strftime(
                "%Y%m%d_%H%M%S"
            )

            frame_in_second = (
                frame_id % TARGET_FPS
            )

            image_path = (
                IMAGE_DIR
                / f"capture_{second_stamp}_f{frame_in_second:02d}.jpg"
            )

            success_resized = save_image_atomic(
                image_path,
                resized,
            )

            if not success_resized:
                print(
                    f"Failed to save image: {image_path}"
                )

            frame_id += 1
            next_frame_time += frame_interval

            if (
                time.perf_counter()
                > next_frame_time + frame_interval
            ):
                next_frame_time = time.perf_counter()

    except KeyboardInterrupt:

        print("Capture stopped.")

    finally:

        picam2.stop()


def start_camera_capture() -> tuple[
    threading.Thread,
    threading.Event,
]:

    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(stop_event,),
        daemon=True,
    )

    capture_thread.start()

    return capture_thread, stop_event


if __name__ == "__main__":

    capture_loop()