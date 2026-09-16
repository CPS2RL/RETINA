# robot_stage.py
import threading,time,serial

from config import (
    FORWARD_LEFT,FORWARD_RIGHT,BRAKE_TTC_SEC,SAFE_RELEASE_TTC_SEC,
    DANGER_BEHAVIORS,CONF_THRESHOLD
)


class MotorController:
    def __init__(self,port,baudrate,send_interval=0.05):
        self.ser=serial.Serial(port,baudrate=baudrate,timeout=1,dsrdtr=None)
        self.ser.setRTS(False)
        self.ser.setDTR(False)
        time.sleep(0.1)
        self.interval=send_interval
        self.left=0.0
        self.right=0.0
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self._loop,daemon=True)

    def start(self): self.thread.start()

    def set_command(self,left,right):
        with self.lock: self.left,self.right=left,right

    def _send(self,left,right):
        self.ser.write(f'{{"T":1,"L":{left},"R":{right}}}\n'.encode())
        self.ser.flush()

    def _loop(self):
        while not self.stop.is_set():
            with self.lock: left,right=self.left,self.right
            self._send(left,right)
            time.sleep(self.interval)

    def shutdown(self):
        self.set_command(0.0,0.0)
        time.sleep(0.1)
        self.stop.set()
        self.thread.join(timeout=1)
        try: self._send(0.0,0.0)
        finally: self.ser.close()


def choose_ttc_command(objects,brake_hold,brake_ttc=BRAKE_TTC_SEC,release_ttc=SAFE_RELEASE_TTC_SEC):
    risk=False

    for obj in objects:
        ttc=float(obj.get("ttc",999.0))
        x1,y1,x2,y2=map(float,obj.get("box",[0,0,0,0]))
        w=max(1.0,float(obj.get("image_width",1)))
        h=max(1.0,float(obj.get("image_height",1)))
        cx=(x1+x2)/2
        area=max(0,x2-x1)*max(0,y2-y1)
        frame_area=w*h
        in_lane=w*0.25<cx<w*0.75

        if (in_lane and ttc<brake_ttc) or (in_lane and ((y2>h*0.30 and area>frame_area*0.04) or area>frame_area*0.10)):
            risk=True
            break

    if risk: 
        brake_hold=True
    elif brake_hold and not any(float(o.get("ttc",999.0))<release_ttc for o in objects): 
        brake_hold=False

    return (0.0,0.0,True,"STOPPED") if brake_hold else (FORWARD_LEFT,FORWARD_RIGHT,False,"DRIVING")


def choose_retina_command(objects,results,brake_hold):
    left,right,brake_hold,state=choose_ttc_command(objects,brake_hold)

    for result in results:
        if result.get("behavior") in DANGER_BEHAVIORS and float(result.get("behavior_confidence",0))>=CONF_THRESHOLD:
            return 0.0,0.0,True,"STOPPED"

    return left,right,brake_hold,state