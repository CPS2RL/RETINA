# capture_camera.py
import threading,time
import cv2
from picamera2 import Picamera2
from config import IMAGE_DIR,TARGET_FPS,FRAME_WIDTH,FRAME_HEIGHT,JPEG_QUALITY,OUTPUT_IMAGE_WIDTH,OUTPUT_IMAGE_HEIGHT

# Wait until next second to synchronize camera
def wait_until_next_second():
    #determine ms past previous second
    #determine ms to reach next second 
    delay=1.0-time.time()%1.0
    if delay>0: 
        time.sleep(delay)

# Save image
def save_image(path,image):
    # Create a temporary file
    tmp=path.with_suffix(".tmp.jpg")
    # return true if file was saved succesfully
    # or false if writing was failed
    if not cv2.imwrite(str(tmp),image,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY]): 
        return False
    # Replace the temporary filename with original file name
    # This prevents from processing a partially saved image later
    tmp.replace(path)
    return True

# Does not replace files while restarting the program
# So previous captured image sequence is determined to start new sequence
def get_next_frame_id():
    # Search the directory with file name capture_*.jpg
    files=list(IMAGE_DIR.glob("capture_*.jpg"))
    # If no file is present return start frame from number 0
    if not files: 
        return 0
    # Create a list to save the frame numbers
    ids=[]
    # If there exists image files in the directory
    for f in files:
        # Split the name to grab the numerical sequence and store in the list
        ids.append(int(f.stem.split("_")[-1]))
        # After having a list of all saved numerical numbers, next image will start after the max number
    return max(ids)+1

# Capture function
# Capture a image -> crop desired portion -> resize the image -> save sequentially
def capture_loop(stop_event=None):
    # Create the directory to save the images
    IMAGE_DIR.mkdir(parents=True,exist_ok=True)
    # Configure Camera
    cam=Picamera2()
    config=cam.create_video_configuration(main={"size":(FRAME_WIDTH,FRAME_HEIGHT),"format":"RGB888"},queue=False)
    cam.configure(config)
    cam.start()
    # Preporecessing before saving
    time.sleep(0.5)
    wait_until_next_second()

    interval=1.0/TARGET_FPS
    next_time=time.perf_counter()
    # Determine the numerical number to save the captured image
    frame_id=get_next_frame_id()

    try:
        # Indefinite loop until stop
        while not (stop_event and stop_event.is_set()):
            now=time.perf_counter()
            # Initialization delay
            if now<next_time: 
                time.sleep(next_time-now)

            # capture
            request=cam.capture_request(flush=True)

            try:
                # RGB image array
                frame=request.make_array("main")
            finally:
                request.release()

            # Crop and resize the image
            if frame is not None:
                # Crop image
                frame=frame[119:FRAME_HEIGHT,157:806]
                # Resize image
                frame=cv2.resize(frame,(OUTPUT_IMAGE_WIDTH,OUTPUT_IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)

                # filename
                path=IMAGE_DIR/f"capture_{frame_id:06d}.jpg"

                # Save image
                if not save_image(path,frame): 
                    print(f"Could not save {path}")
                else:
                    frame_id+=1

            next_time+=interval
            if time.perf_counter()>next_time+interval: 
                next_time=time.perf_counter()

    finally:
        cam.stop()

# Start capture
def start_camera_capture():
    stop=threading.Event()
    thread=threading.Thread(target=capture_loop,args=(stop,),daemon=True)
    thread.start()
    return thread,stop

if __name__=="__main__": 
    capture_loop()