# ttc_stage.py
import math
from config import TTC_CONSTANT,ROBOT_VELOCITY_MM_S,YOLO_INTERVAL

#Manually estimate the TTC
def estimate_ttc_from_box(box): 
    return TTC_CONSTANT/max(1.0,float(box[3]-box[1]))

# Add TTC to each detected objects
def add_ttc_to_objects(yolo_result):
    objects=[]
    for obj in yolo_result.get("detections",[]):
        h=max(1.0,float(obj["box"][3]-obj["box"][1]))
        ttc=estimate_ttc_from_box(obj["box"])
        objects.append({
            **obj,
            "box_height_px":round(h,2),
            "box_height_ratio":round(h/max(1.0,float(obj["image_height"])),4),
            "ttc":round(ttc,3),
            "initial_ttc":round(ttc,3),
            "distance_mm":round(ttc*ROBOT_VELOCITY_MM_S,2),
            "cycles_to_collision":math.ceil(ttc/YOLO_INTERVAL)
        })
    return sorted(objects,key=lambda o:(float(o["ttc"]),o["label"],(o["box"][0]+o["box"][2])/2))