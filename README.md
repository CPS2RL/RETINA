# **RETINA**

*Artifact and code for "Time-Aware Intent Prediction for Autonomous Vehicles using Adaptive Scheduling"*

## Table of Contents

- [How to use the code?](#how-to-use-the-code?)
- [CARLA](#carla)
- [Installing Carla Package](#installing-carla-package)
- [Running the Experiments](#running-the-experiments)
    - [Scheduler Behavior Under Deadline](#experiment:_scheduler_behavior_under_deadline)
    - [Per-Object Model Selection Over Time](#experiment:_per-object_model_selection_over_time)
    - [Baseline Comparisons](#experiment:_baseline_comparisons)
    - [Performance Comparisons](#experiment:_performance_comparisons)
    - [Solver Overhead](#experiment:_solver_overhead)
    - [Urgent and Non-urgent Jobs Percentage](#experiment:_urgent_and_non-urgent_jobs_percentage)
    - [Safe Jobs](#experiment:_Safe_Jobs)
- [Reproducing the Results](#reproducing-the-results)
- [Hardware Feasibility Test](#hardware-feasibility-test)


## **How to use the code?**

We used carla to simulate the scheduling algorithm in a city environment to determine safety and temporal gurantee of the system (`CARLA`). To proof the feasibility of the system on real hardware, we tested on a robot hardware (`Feasibility_Test`).

## **CARLA**
This CARLA folder has the code used to test RETINA against fixed-model versions and CA-MOT. All four scripts use the same CARLA scene, camera setup, and pedestrian/vehicle spawn settings, so the results can be fairly compared.

The model/ folder holds the ten ShuffleNetV2-GRU-TemporalAttention model files used by the three RETINA scripts. All four scripts need yolov5n.pt to detect objects. You need to download the pretrained YOLOv5n from the official Ultralytics repo.

**Dependencies**

Install the required packages before running any script:

```bash
pip install carla torch torchvision opencv-python numpy gurobipy
```

---

## **Installing Carla Package**

The minimum system requirements for installing CARLA and the official  installation guide can be found
 [here](https://carla.readthedocs.io/en/latest/start_quickstart/). 
 However, while installing on Windows 11, we found that the following additional components were required:

- DirectX Runtime [Download Link](https://www.microsoft.com/en-us/download/details.aspx?id=35) 
-  Microsoft Visual C++ Redistributable [Download Link](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version). 

 
**Download and extract a CARLA package**
Carla Version 0.9.15 can be downloaded from [here](https://github.com/carla-simulator/carla/releases/tag/0.9.15/).

**Launch the CARLA server**
> .\CarlaUE4.exe

**Install the Python client library**

Create a Conda environment with Python 3.7
>conda create -n carla python=3.7 \
conda activate carla \
cd path_to_carla_package\PythonAPI\carla\dist   \
python -m pip install carla-0.9.15-cp37m-manylinux_2_27_x86_64.whl  \
python -m pip install carla \
cd path_to_carla_package\PythonAPI\examples\ \
python -m pip install -r requirements.txt 

**Run a Python client script**

>cd path_to_RETINA\CARLA
python .\RETINA.py

---

## **Running the Experiments**

**Experiment: Scheduler Behavior Under Deadline**

{TBD}

**Experiment: Per-Object Model Selection Over Time**

{TBD}

**Experiment: Baseline Comparisons**

>cd path_to_RETINA\CARLA    \
python .\RETINA.py  \
python .\CAMOT  \
python .\RETINA-HIGH    \
python .\RETINA-LOW

**Experiment: Performance Comparisons**

{TBD}

**Experiment: Solver Overhead**

In this experiment we measure the time to the problem in RETINA. To have an idea of the solve time we have meassured the average solver overhead by varying number of jobs, model configurations and deadline. Additionally, the P95th solve time and well as the max solve time can be measured. Uncomment line (130,140) in `solver_overhead.py` to do so.

>cd RETINA/Exp_ Solver Overhead/    \
python3 solver_overhead.py

Two kind of csv files will be save. One contains solve time for all iteration, and for convenience a summary csv file is also saved.

Output: RETINA/Exp_Solver_Overhead/solver_overhead/

**Experiment: Urgent and Non-urgent Jobs Percentage**

In this experiment, we measure the the urgent and non-urgent jobs (%) while varying the deadline. 

>cd RETINA/Exp_Urgent_vs_Non_urgent/    \
python3 urgent_non_urgent.py

Output files are saved in the same directory.

**Experiment: Safe Jobs**

In this experiment, we measure the number of jobs that are safe served by different algorithms while varying the deadline. 

>cd RETINA/Exp_Safe_vs_Unsafe/    \
python3 safe.py

Output files are saved in the same directory.

---

## **Reproducing the Results**

**Experiment: Scheduler Behavior Under Deadline**

{TBD}

**Experiment: Per-Object Model Selection Over Time**

{TBD}

**Experiment: Safe Navigation**

{TBD}

**Experiment: Performance Comparisons**

{TBD}

Running all the experiments may take a significant amount of time. For convenience, we provide the output files in each experiment directory. These output files can be used to regenerate results reported in the paper.

**Experiment: Solver Overhead**

In this we have measured the solver overhead by varying number of jobs, model configurations and deadline. The number of jobs varies from 1 to 30 and model configurations is varied as 5, 10 and 15.

To reproduce the results corresponding to Fig. 12(a)-(b), run:

>cd RETINA/Exp_ Solver Overhead/    \
python3 plot_overhead_D150.py   \
python3 plot_overhead_D200.py   

The generated figures will be saved in:

Output: RETINA/Exp_Solver_Overhead/

**Experiment: Urgent and Non-urgent Jobs Percentage**

To reproduce the results corresponding to Fig. 13(a)-(b), run:

>cd RETINA/Exp_Urgent_vs_Non_urgent/    \
python3 plot_percentage_D150.py   \
python3 plot_percentage_D190.py   

The generated figures will be saved in:

Output: RETINA/Exp_Urgent_vs_Non_urgent/

**Experiment: Safe Jobs**

To reproduce the results corresponding to Fig. 14(a)-(b), run:

>cd RETINA/Exp_Safe_vs_Unsafe/    \
python3 plot_safe_D200.py   \
python3 plot_safe_D250.py   

The generated figures will be saved in:

Output: RETINA/Exp_Safe_vs_Unsafe/

---

## **Hardware Feasibility Test**

To perform the hardware evaluation, we used the rover from [here](https://www.waveshare.com/product/ugv01.htm). On top of it, we mounted a camera sensor from [here](https://www.waveshare.com/product/robotics/robot-arm-control/pan-tilt-control/2-axis-pan-tilt-camera-module.htm). A testbed is designed by placing toy road tapes, cars and humanoid figures. Load the code from `RETINA/Feasibility_Test` to the robot and perform the hardware feasiblity test.

In the hardware experiment, we have trained the YOLO model offline and put it in the ROVER folder. It is recommended to profile the prediction models on the hardware before running. Profiling can be done using the `profile_behavior_models.py`. Result should be saved in file `config.py` (line 71,134). In file `config.py` line (11,16) consists the scheduling algorithm, uncomment the desried one and comment others before running. Use the `main.py` to do the rover simulation.

Install the necessary python library listed in `requirements.txt`.