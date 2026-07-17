# **RETINA**

*Artifact and code for "Time-Aware Intent Prediction for Autonomous Vehicles using Adaptive Scheduling"*

**How to use the code?**

We used carla to simulate the scheduling algorithm in a city environment to determine safety and temporal gurantee of the system (`CARLA`). To proof the feasibility of the system on real hardware, we tested on a robot hardware (`ROVER`).

The repository is divided into two parts: `ROVER` and `CARLA`.

## **CARLA**
{TBD}


## **ROVER**

In the hardware experiment, we have trained the YOLO model offline and put it in the ROVER folder. It is recommended to profile the prediction models on the hardware before running. Profiling can be done using the `profile_behavior_models.py`. Result should be saved in file `config.py` (line 71,134). In file `config.py` line (11,16) consists the scheduling algorithm, uncomment the desried one and comment others before running. Use the `main.py` to do the experiment.

<!-- !Install the necessary python library listed in `{TBD}`. -->
