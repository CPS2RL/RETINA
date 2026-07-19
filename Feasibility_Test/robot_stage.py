import time
import threading
import serial

from config import (
    FORWARD_LEFT,
    FORWARD_RIGHT,
)



class MotorController:

    def __init__(
        self,
        port: str,
        baudrate: int,
        send_interval: float = 0.05,
    ):

        self.ser = serial.Serial(

            port,

            baudrate=baudrate,

            timeout=1,

            dsrdtr=None,
        )

        self.ser.setRTS(False)

        self.ser.setDTR(False)

        time.sleep(0.1)

        self.send_interval = send_interval

        self.current_left = 0.0

        self.current_right = 0.0

        self.lock = threading.Lock()

        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )


    def start(self):

        self.thread.start()

    def set_command(
        self,
        left: float,
        right: float,
    ):

        with self.lock:

            self.current_left = left

            self.current_right = right


    def _send_now(
        self,
        left: float,
        right: float,
    ):

        command = (
            f'{{"T":1,'
            f'"L":{left},'
            f'"R":{right}}}\n'
        )

        self.ser.write(
            command.encode()
        )

        self.ser.flush()


    def _loop(self):

        while not self.stop_event.is_set():

            with self.lock:

                left = self.current_left

                right = self.current_right

            self._send_now(
                left,
                right,
            )

            time.sleep(
                self.send_interval
            )


    def shutdown(self):

        self.set_command(
            0.0,
            0.0,
        )

        time.sleep(0.1)

        self.stop_event.set()

        self.thread.join(timeout=1.0)

        self._send_now(
            0.0,
            0.0,
        )

        self.ser.close()


def choose_next_command(
    behavior_results,
    danger_behaviors,
):

    for result in behavior_results:

        behavior = result.get(
            "behavior",
            ""
        )

        if behavior in danger_behaviors:

            print(
                f"Robot Command = STOP "
                f"because {behavior}"
            )

            return (
                0.0,
                0.0,
            )

    print(
        "Robot Command = MOVE FORWARD"
    )

    return (
        FORWARD_LEFT,
        FORWARD_RIGHT,
    )