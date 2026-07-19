import numpy as np

from gurobi_optimal import run_gurobi_experiment
from gurobi_noskip import run_gurobi_no_skip_experiment

from heuristics import (Low_cost_heuristic,High_cost_heuristic)


def program(jobNumber,TTC_all,job_weights_all,Deadline,modelNumber=20):
    TTC = TTC_all[:jobNumber]
    job_weights = job_weights_all[:jobNumber]

    n = np.floor(TTC / Deadline)
    mandatory_job_indices = np.where(n < 2)[0]

    accuracy = np.linspace(0.60, 0.90, modelNumber)
    executionTime = np.linspace(10, 15, modelNumber)

    model_choice = []
    model_choice_noskip = []

    used_jobs_low = []
    used_jobs_high = []


    optimal = run_gurobi_experiment(
        modelNumber=modelNumber,
        Deadline=Deadline,
        jobNumber=jobNumber,
        job_weights=job_weights,
        mandatory_job_indices=mandatory_job_indices,
        accuracy=accuracy,
        executionTime=executionTime,
        model_choice=model_choice
    )

    noskip = run_gurobi_no_skip_experiment(
        modelNumber=modelNumber,
        Deadline=Deadline,
        jobNumber=jobNumber,
        job_weights=job_weights,
        accuracy=accuracy,
        executionTime=executionTime,
        model_choice=model_choice_noskip
    )

    low = Low_cost_heuristic(
        used_jobs=used_jobs_low,
        job_weights=job_weights,
        accuracy=accuracy,
        executionTime=executionTime,
        Deadline=Deadline,
        mandatory_job_indices=mandatory_job_indices
    )

    high = High_cost_heuristic(
        used_jobs=used_jobs_high,
        job_weights=job_weights,
        accuracy=accuracy,
        executionTime=executionTime,
        Deadline=Deadline,
        mandatory_job_indices=mandatory_job_indices
    )



    accuracy_set = []
    accuracy_set_noskip = []

    for i in range(jobNumber):

        if model_choice[i] != -1:
            accuracy_set.append(accuracy[model_choice[i]])
        else:
            accuracy_set.append(0)

        if model_choice_noskip[i] != -1:
            accuracy_set_noskip.append(accuracy[model_choice_noskip[i]])
        else:
            accuracy_set_noskip.append(0)

    return (
        optimal,
        low,
        high,
        noskip,
        accuracy_set,
        accuracy_set_noskip,
        used_jobs_low,
        used_jobs_high,
        model_choice,
        model_choice_noskip,
        job_weights
    )
