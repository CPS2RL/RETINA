# **RETINA**

*Artifact and code for "Time-Aware Intent Prediction for Autonomous Vehicles using Adaptive Scheduling"*

**How to use the code?**

We used carla to simulate the scheduling algorithm in a city environment to determine safety and temporal gurantee of the system (`CARLA`). To proof the feasibility of the system on real hardware, we tested on a robot hardware (`ROVER`).

The repository is divided into two parts: `ROVER` and `CARLA`.

## **CARLA**
This CARLA folder has the code used to test RETINA against fixed-model versions and CA-MOT. All four scripts use the same CARLA scene, camera setup, and pedestrian/vehicle spawn settings, so the results can be fairly compared.

The model/ folder holds the ten ShuffleNetV2-GRU-TemporalAttention model files used by the three RETINA scripts. All four scripts need yolov5n.pt to detect objects. You need to download the pretrained YOLOv5n from the official Ultralytics repo.
### Dependencies

Install the required packages before running any script:

```bash
pip install carla torch torchvision opencv-python numpy gurobipy
```

Note: `gurobipy` is only needed for `RETINA.py` (it solves the per-cycle scheduling problem with Gurobi). `RETINA-HIGH.py`, `RETINA-LOW.py`, and `CAMOT.py` do not use it. Gurobi also requires a license; a free academic license is available from gurobi.com.

## **ROVER**

In the hardware experiment, we have trained the YOLO model offline and put it in the ROVER folder. It is recommended to profile the prediction models on the hardware before running. Profiling can be done using the `profile_behavior_models.py`. Result should be saved in file `config.py` (line 71,134). In file `config.py` line (11,16) consists the scheduling algorithm, uncomment the desried one and comment others before running. Use the `main.py` to do the experiment.

<!-- !Install the necessary python library listed in `{TBD}`. -->
