import numpy as np
import gurobipy as gp

from gurobipy import GRB


def run_gurobi_experiment(
    modelNumber,
    Deadline,
    jobNumber,
    accuracy,
    executionTime,
    job_weights,
    mandatory_job_indices,
    model_choice
):

    model = gp.Model()

    model.setParam("OutputFlag", 0)
    model.setParam("Seed", 0)

    x = {}

    for job_id in range(jobNumber):

        for model_id in range(modelNumber):

            x[job_id, model_id] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x_{job_id}_{model_id}"
            )

    model.update()

    model.setObjective(

        gp.quicksum(

            job_weights[job_id]
            * accuracy[model_id]
            * x[job_id, model_id]

            for job_id in range(jobNumber)
            for model_id in range(modelNumber)

        ),

        GRB.MAXIMIZE
    )

    for job_id in range(jobNumber):

        model.addConstr(

            gp.quicksum(
                x[job_id, model_id]
                for model_id in range(modelNumber)
            ) <= 1
        )

    model.addConstr(

        gp.quicksum(

            executionTime[model_id]
            * x[job_id, model_id]

            for job_id in range(jobNumber)
            for model_id in range(modelNumber)

        ) <= Deadline
    )

    for mandatory_job in mandatory_job_indices:

        model.addConstr(

            gp.quicksum(
                x[mandatory_job, model_id]
                for model_id in range(modelNumber)
            ) == 1
        )

    model.optimize()

    if model.Status != GRB.OPTIMAL:

        model_choice.extend([-1] * jobNumber)

        return np.nan

    total_accuracy = 0.0

    for job_id in range(jobNumber):

        selected = [

            model_id

            for model_id in range(modelNumber)

            if x[job_id, model_id].X > 0.5
        ]

        if selected:

            chosen_model = selected[0]

            model_choice.append(chosen_model)

            total_accuracy += (
                job_weights[job_id]
                * accuracy[chosen_model]
            )

        else:

            model_choice.append(-1)

    return total_accuracy
