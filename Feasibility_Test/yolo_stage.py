# yolo_stage.py
from datetime import datetime
from pathlib import Path
import time,cv2
from config import ANNOTATED_DIR,YOLO_IMG_SIZE,YOLO_CONF,YOLO_IOU,YOLO_MAX_DET,TARGET_LABELS

# Check whether the is camera files saved and also temporary files are skipped
def wait_until_file_ready(path,checks=5,delay=0.05):
    old=-1
    for _ in range(checks):
        # Check if files are ready
        if not path.exists() or path.name.endswith(".tmp.jpg"): 
            return False
        # Check if there is a file saved, but not saved properly which can lead to no data
        size=path.stat().st_size
        if size>0 and size==old and cv2.imread(str(path)) is not None: 
            return True
        old=size
        # Wait until there exists files
        time.sleep(delay)
    return False

#Get the latest image
def get_latest_image(directory):
    # Search the directory if there is a file named ".jpg".
    # Also ignore temporary files 
    # File largest modified time is the latest file
    files=[p for p in directory.glob("*.jpg") if not p.name.endswith(".tmp.jpg")]
    return max(files,key=lambda p:p.stat().st_mtime_ns) if files else None

# Run detection
def detect_objects(model,image_path):
    # Read image
    image=cv2.imread(str(image_path))
    if image is None:
        return {"image_path":str(image_path),"annotated_image_path":"","object_count":0,"detections":[],
                "image_width":0,"image_height":0,"inference_time_sec":0.0,"image_read_error":True}

    h,w=image.shape[:2]
    annotated=image.copy()
    called=datetime.now()
    start=time.perf_counter()

    # Run yolo to detect objects
    result=model(image,imgsz=YOLO_IMG_SIZE,conf=YOLO_CONF,iou=YOLO_IOU,agnostic_nms=True,
                 max_det=YOLO_MAX_DET,verbose=False)[0]
    runtime=time.perf_counter()-start; detections=[]

    if result.boxes is not None:
        for box in result.boxes:
            label=model.names[int(box.cls[0])]
            # Skip garbage objects
            if label not in TARGET_LABELS: 
                continue
            
            conf=float(box.conf[0])
            # Extract bounding box
            x1,y1,x2,y2=map(int,box.xyxy[0])
            x1=max(0,min(x1,w-1))
            y1=max(0,min(y1,h-1)); x2=max(0,min(x2,w))
            y2=max(0,min(y2,h))
            if x2<=x1 or y2<=y1: 
                continue

            detections.append({"label":label,"confidence":conf,"box":[x1,y1,x2,y2],"image_width":w,"image_height":h})
            # Uncomment to validate
            # cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),1)
            # cv2.putText(annotated,f"{label} {conf:.2f}",(x1,max(y1-5,10)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
    # Uncomment to validate: Create directory
    # ANNOTATED_DIR.mkdir(parents=True,exist_ok=True)
    # Uncomment to validate: Save Image for validation
    annotated_path=ANNOTATED_DIR/f"annotated_{image_path.name}"
    # cv2.imwrite(str(annotated_path),annotated)

    return {
        "image_path":str(image_path),"annotated_image_path":str(annotated_path),
        "object_count":len(detections),"detections":detections,
        "image_width":w,"image_height":h,
        "yolo_called_at":called.isoformat(timespec="milliseconds"),
        "yolo_returned_at":datetime.now().isoformat(timespec="milliseconds"),
        "inference_time_sec":round(runtime,4),"image_read_error":False
    }