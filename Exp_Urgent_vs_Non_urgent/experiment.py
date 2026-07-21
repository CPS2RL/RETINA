import numpy as np

from gurobi_optimal import run_gurobi_experiment

# Return the per-job accuracy values.
def program(jobNumber,TTC_all,job_weights_all,Deadline):
    TTC = TTC_all[:jobNumber]
    job_weights = job_weights_all[:jobNumber]

    n = np.floor(TTC / Deadline)
    mandatory_job_indices = np.where(n < 2)[0]

    modelNumber = 20

    accuracy = np.linspace(0.60, 0.90, modelNumber)
    executionTime = np.linspace(10, 15, modelNumber)

    model_choice = []


    # Run the optimization model.
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

    accuracy_set = []
    # Store the accuracy assigned to each job.
    for i in range(jobNumber):

        if model_choice[i] != -1:
            accuracy_set.append(accuracy[model_choice[i]])
        else:
            accuracy_set.append(0)
    # Return the per-job accuracy values.
    return accuracy_set
