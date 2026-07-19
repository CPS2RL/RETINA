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

## **ROVER**

In the hardware experiment, we have trained the YOLO model offline and put it in the ROVER folder. It is recommended to profile the prediction models on the hardware before running. Profiling can be done using the `profile_behavior_models.py`. Result should be saved in file `config.py` (line 71,134). In file `config.py` line (11,16) consists the scheduling algorithm, uncomment the desried one and comment others before running. Use the `main.py` to do the experiment.

Install the necessary python library listed in `requirements.txt`.

---

## **Running the Experiments**

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

---

## **Reproducing the Results**

Running all the experiments may take a significant amount of time. For convenience, we provide the output files in each experiment directory. These output files can be used to regenerate results reported in the paper.

**Experiment: Solver Overhead**

In this we have measured the solver overhead by varying number of jobs, model configurations and deadline. The number of jobs varies from 1 to 30 and model configurations is varied as 5, 10 and 15.

To reproduce the results corresponding to Fig. 12(a)-(b), run:

>python3 plot_overhead_D150.py   \
python3 plot_overhead_D200.py   

The generated figures will be saved in:

Output: RETINA/Exp_ Solver Overhead/

**Experiment: Urgent and Non-urgent Jobs Percentage**

To reproduce the results corresponding to Fig. 13(a)-(b), run:

>python3 plot_percentage_D150.py   \
python3 plot_percentage_D190.py   