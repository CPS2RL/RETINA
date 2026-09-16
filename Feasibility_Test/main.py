# main.py
import time
import cv2

from ultralytics import YOLO

from capture_camera import start_camera_capture
from config import (BAUDRATE,DEADLINE_MS,IMAGE_DIR,MODEL_REGISTRY,SCHEDULER_MODE,SERIAL_PORT,
                    YOLO_INTERVAL,YOLO_MODEL_PATH,CAMOT_TTC_CRITICAL_SEC)
from robot_stage import MotorController,choose_retina_command,choose_ttc_command
from ttc_stage import add_ttc_to_objects
from tracking_stage import HardwareTracker
from yolo_stage import detect_objects,get_latest_image,wait_until_file_ready

RETINA_MODES={"RETINA_LOW","RETINA_HIGH","RETINA"}


def build_model_specs():
    from scheduler_stage import ModelSpec
    return [
        ModelSpec(
            name=m["name"],display=m["display"],model_path=str(m["path"]),
            accuracy=m["accuracy"],exec_time_ms=m["exec_time_ms"],int8=m["int8"]
        )
        for m in MODEL_REGISTRY
    ]
# RETINA
def run_retina_cycle(cycle_id,image_path,yolo_model,tracker,memory,buffer,models,specs,motor,brake_hold,dt):
    from scheduler_stage import run_scheduler
    frame=cv2.imread(str(image_path))
    if frame is None: 
        return brake_hold,0,0
    yolo=detect_objects(yolo_model,image_path)
    objects=tracker.update(add_ttc_to_objects(yolo))
    visible_ids={int(o["track_id"]) for o in objects}
    for obj in objects: 
        buffer.update_from_history(image_path,obj["track_id"],obj["box"])
    buffer.remove_missing_tracks(tracker.active_track_ids())
    now=time.monotonic()
    memory.update_ttc(dt,visible_ids)
    memory.prune(tracker.active_track_ids())
    jobs=memory.add_current_cycle_jobs(cycle_id,objects,now)
    sched_start=time.monotonic()
    schedule,info=run_scheduler(specs,jobs,DEADLINE_MS,SCHEDULER_MODE)
    execution_start=sched_start
    admitted=processed=delayed=buffering=0
    behavior_ms=0.0
    selected_models=[]; used_models=[]; served=[]; results=[]; rows=[]
    missed_ttc=set(); missed_deadline=set()

    for job_id,model_id in schedule:
        obj=jobs[job_id]; uid=obj["job_uid"]
        urgent=bool(info["urgent"][job_id]); eta=int(info["eta"][job_id])
        due=float(obj.get("ttc_due_time",float("inf")))
        if model_id is None:
            delayed+=1; memory.mark_skipped(uid)
            if urgent: missed_deadline.add(uid)
            rows.append((obj,"DELAYED","",None,urgent,eta,"",False if urgent else ""))
            continue
        admitted+=1
        spec=specs[model_id]
        selected_models.append(spec.display)
        served.append(uid)
        if not buffer.ready(obj["track_id"]):
            buffering+=1
            completion=(time.monotonic()-execution_start)*1000.0
            if urgent: 
                missed_deadline.add(uid)
            rows.append((obj,"BUFFERING",spec,None,urgent,eta,completion,False if urgent else ""))
            continue
        result=buffer.infer(models[spec.name],obj["track_id"])
        if result is None:
            buffering+=1
            if urgent: missed_deadline.add(uid)
            rows.append((obj,"BUFFERING",spec,None,urgent,eta,"",False if urgent else ""))
            continue
        processed+=1
        used_models.append(spec.display)
        results.append(result)
        behavior_ms+=float(result["runtime_ms"])
        completion_time=time.monotonic()
        completion_ms=(completion_time-execution_start)*1000.0
        deadline_met=completion_ms<=DEADLINE_MS
        if not deadline_met: missed_deadline.add(uid)
        if completion_time>=due: 
            missed_ttc.add(uid)
            memory.mark_ttc_missed(uid)
        rows.append((obj,"PROCESSED",spec,result,urgent,eta,completion_ms,deadline_met))
    missed_ttc.update(memory.collect_ttc_misses(time.monotonic()))
    memory.remove_served(served)
    left,right,brake_hold,state=choose_retina_command(objects,results,brake_hold)
    motor.set_command(left,right)
    for obj,_,spec,result,urgent,eta,completion,deadline_met in rows:
        uid=obj["job_uid"]



# RTDeepIoT
def run_rtdeepiot_cycle(cycle_id,image_path,yolo_model,tracker,motor,model,device,wcet,brake_hold):
    from rtdeepiot_stage import (read_frame,run_rtdeepiot_schedule,classify_crop,job_deadline_ms)
    cycle_start=time.monotonic()
    frame=read_frame(image_path)
    if frame is None:
        return brake_hold,0
    # YOLO -> TTC -> persistent hardware tracks
    yolo=detect_objects(yolo_model,image_path)
    objects=tracker.update(add_ttc_to_objects(yolo))
    # RTDeepIoT scheduling
    rt_start=time.monotonic()
    schedule=run_rtdeepiot_schedule(model,device,wcet,frame,objects)
    final=schedule["final_assignment"]
    stage1=schedule["stage1_results"]
    admitted_ids=set(schedule["admitted_ids"])
    delayed_ids=set(schedule["delayed_ids"])
    # Execute final selected depths
    processed_results=[]
    final_extra_runtime_ms=0.0
    jobs_processed=0
    jobs_buffering=0
    missed_ttc=set()
    missed_deadline=set()
    selected_depths=[]
    used_depths=[]
    object_map={int(o["track_id"]):o for o in objects}

    for tid in admitted_ids:
        obj=object_map.get(tid)

        if obj is None:
            continue
        depth=int(final.get(tid,1))
        selected_depths.append(f"Depth-{depth}")

        # Depth 1 was already executed during scheduler phase
        if depth==1:
            result=stage1.get(tid)
            if result is None:
                jobs_buffering+=1
                continue
            completion_ms=float(result.get("completion_ms",(time.monotonic()-rt_start)*1000.0))
        # Depth 2 / 3 executes another cumulative inference
        else:
            result=classify_crop(model,device,frame,obj["box"],depth)

            if result is None:
                jobs_buffering+=1
                continue

            final_extra_runtime_ms+=float(result["runtime_ms"])

            completion_ms=(time.monotonic()-rt_start)*1000.0

        jobs_processed+=1

        used_depths.append(f"Depth-{depth}")
        deadline_ms=job_deadline_ms(obj,wcet)
        ttc_ms=float(obj["ttc"])*1000.0
        deadline_met=(completion_ms<=deadline_ms)
        ttc_miss=(completion_ms>ttc_ms)
        if not deadline_met:
            missed_deadline.add(tid)

        if ttc_miss:
            missed_ttc.add(tid)
        processed_results.append({"obj":obj,"depth":depth,"result":result,"completion_ms":completion_ms,"deadline_ms":deadline_ms,
                                  "deadline_met":deadline_met,"missed_ttc":ttc_miss})

    # Robot control
    left,right,brake_hold,state=choose_ttc_command(objects,brake_hold)

    motor.set_command(left,right)
    command=("STOP" if brake_hold else "GO")

    # Per-job log: PROCESSED
    processed_ids=set()

    for item in processed_results:
        obj=item["obj"]
        tid=int(obj["track_id"])
        processed_ids.add(tid)

        depth=item["depth"]
        result=item["result"]


    # Per-job log: DELAYED
    for obj in objects:
        tid=int(obj["track_id"])
        if tid not in delayed_ids:
            continue

    # Admitted job whose inference failed
    failed_ids=(admitted_ids-processed_ids)
    jobs_buffering+=len(failed_ids)
    # Runtime totals
    return brake_hold,jobs_processed
    
# CA-MOT
def run_camot_cycle(
    cycle_id,image_path,yolo_model,runtime,motor,brake_hold,dt):
    frame=cv2.imread(str(image_path))
    if frame is None:
        return brake_hold,0
    result=runtime.step(frame,yolo_model,dt)
    objects=result["objects"]
    # Robot control 
    left,right,brake_hold,state=choose_ttc_command(objects,brake_hold)
    motor.set_command(left,right)
    scheduler_choices=[]
    missed_ttc=0
    missed_deadline=0
    processed=0
    delayed=0
    # Per-track logging
    for obj in objects:
        status=obj.get("job_status","PROCESSED")
        ttc_missed=(float(obj["ttc"])<=0.0)
        is_urgent=(float(obj["ttc"])<CAMOT_TTC_CRITICAL_SEC)
        if ttc_missed:
            missed_ttc+=1
        # DELAYED JOB
        # No D/A computation was executed
        if status=="DELAYED":
            delayed+=1
            continue
        # PROCESSED JOB
        s=obj["detection_level"]
        f=obj["association_level"]
        processed+=1
        choice=f"D={s};A={f}"
        scheduler_choices.append(choice)
        deadline_met=bool(obj["deadline_met"])
        if not deadline_met:
            missed_deadline+=1

    return (brake_hold,result["jobs_processed"])


# Main
def main():
    print(f"\nStarting: {SCHEDULER_MODE}")

    camera_thread,camera_stop=start_camera_capture()
    time.sleep(1.0)

    yolo_model=YOLO(str(YOLO_MODEL_PATH))

    motor=None

    tracker=None; memory=None; buffer=None
    models={}; specs=[]
    camot=None
    rt_model=rt_device=rt_wcet=None

    try:
        # RETINA initialization
        if SCHEDULER_MODE in RETINA_MODES:
            from behavior_stage import BehaviorBuffer,load_behavior_model
            from scheduler_stage import SchedulerMemory

            tracker=HardwareTracker()
            memory=SchedulerMemory()
            buffer=BehaviorBuffer()

            specs=build_model_specs()

            for spec in specs:
                models[spec.name]=load_behavior_model(spec.model_path,spec.int8)

        # CA-MOT initialization
        elif SCHEDULER_MODE=="CAMOT":
            from camot_stage import CAMOTRuntime
            camot=CAMOTRuntime()

        # RTDeepIoT initialization
        elif SCHEDULER_MODE=="RTDeepIoT":
            from rtdeepiot_stage import load_rtdeepiot_runtime,reset_confidence_ema

            tracker=HardwareTracker()
            reset_confidence_ema()

            rt_model,rt_device,rt_wcet=load_rtdeepiot_runtime()

        # Robot.
        motor=MotorController(SERIAL_PORT,BAUDRATE)
        motor.start()

        cycle=0
        last_image=None
        brake_hold=False
        previous_cycle_time=time.monotonic()

        while True:
            time.sleep(YOLO_INTERVAL)

            image_path=get_latest_image(IMAGE_DIR)

            if image_path is None or image_path==last_image: 
                continue
            if not wait_until_file_ready(image_path): 
                continue

            # Actual elapsed time used by missed CA-MOT tracks.
            now=time.monotonic()
            dt=now-previous_cycle_time
            previous_cycle_time=now


            if SCHEDULER_MODE in RETINA_MODES:
                brake_hold,_,_=run_retina_cycle(cycle,image_path,yolo_model,tracker,memory,buffer,models,specs,motor,brake_hold,dt)

            elif SCHEDULER_MODE=="CAMOT":
                brake_hold,_=run_camot_cycle(cycle,image_path,yolo_model,camot,motor,brake_hold,dt)

            else:
                brake_hold,_=run_rtdeepiot_cycle(cycle,image_path,yolo_model,tracker,motor,rt_model,rt_device,rt_wcet,brake_hold)

            last_image=image_path
            cycle+=1

    except KeyboardInterrupt:
        print("\nStopping experiment...")

    finally:
        camera_stop.set()
        camera_thread.join(timeout=2)

        if motor:
            try: motor.shutdown()
            except Exception as e: print(f"Motor shutdown warning: {e}")



if __name__=="__main__": main()