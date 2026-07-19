import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gurobi_optimal import run_gurobi_experiment


# =========================================================
# Synthetic workload generation
# =========================================================

np.random.seed(0)

max_job_number = 50

TTC_all = np.random.uniform(50, 400, max_job_number)
job_weights_all = np.random.uniform(0.5, 2.0, max_job_number)
position_uncertainty_all = np.random.uniform(0.0, 1.0, max_job_number)


# =========================================================
# Benchmark configuration
# =========================================================

Deadline_values = range(100, 301, 50)
modelNumber_values = range(5, 21, 5)
jobNumber_values = range(1, 31)

warmup_iterations = 5
measurement_iterations = 1000

output_dir = Path("solver_overhead")
output_dir.mkdir(parents=True, exist_ok=True)


# Each key is: (modelNumber, Deadline)
raw_results = defaultdict(list)
mean_results = defaultdict(list)


# =========================================================
# Benchmark loop
# =========================================================

for jobNumber in jobNumber_values:

    for Deadline in Deadline_values:

        for modelNumber in modelNumber_values:

            TTC = TTC_all[:jobNumber]
            job_weights = job_weights_all[:jobNumber]

            n = np.floor(TTC / Deadline)
            mandatory_job_indices = np.where(n < 2)[0]

            accuracy = np.linspace(0.60, 0.90, modelNumber)
            executionTime = np.linspace(10, 15, modelNumber)

            # =================================================
            # Warm-up runs
            # =================================================

            for warmup in range(1, warmup_iterations + 1):

                warmup_model_choice = []

                run_gurobi_experiment(
                    modelNumber=modelNumber,
                    Deadline=Deadline,
                    jobNumber=jobNumber,
                    accuracy=accuracy,
                    executionTime=executionTime,
                    job_weights=job_weights,
                    mandatory_job_indices=mandatory_job_indices,
                    model_choice=warmup_model_choice,
                )

            # =================================================
            # Measured runs
            # =================================================

            runtimes_ms = []
            objective = None

            for trial in range(1, measurement_iterations + 1):

                model_choice = []

                start_time = time.perf_counter()

                objective = run_gurobi_experiment(
                    modelNumber=modelNumber,
                    Deadline=Deadline,
                    jobNumber=jobNumber,
                    accuracy=accuracy,
                    executionTime=executionTime,
                    job_weights=job_weights,
                    mandatory_job_indices=mandatory_job_indices,
                    model_choice=model_choice,
                )

                runtime_ms = (
                    time.perf_counter() - start_time
                ) * 1000.0

                runtimes_ms.append(runtime_ms)

                raw_results[(modelNumber, Deadline)].append(
                    {
                        "jobNumber": jobNumber,
                        "trial": trial,
                        "runtime_ms": runtime_ms,
                    }
                )

            mean_runtime_ms = np.mean(runtimes_ms)
#            p95_runtime_ms = np.percentile(runtimes_ms, 95)
#            max_runtime_ms = np.max(runtimes_ms)
            
            mean_results[(modelNumber, Deadline)].append(
                {
                    "jobNumber": jobNumber,
                    "mean_runtime_ms": mean_runtime_ms,
                }
            )

#            mean_results[(modelNumber, Deadline)].append(
#                {
#                    "jobNumber": jobNumber,
#                    "mean_runtime_ms": mean_runtime_ms,
#                    "p95_runtime_ms": p95_runtime_ms,
#                    "max_runtime_ms": max_runtime_ms,
#                }
#            )

	



# =========================================================
# Store separate CSV files
# =========================================================

for modelNumber in modelNumber_values:

    for Deadline in Deadline_values:

        key = (modelNumber, Deadline)

        raw_df = pd.DataFrame(
            raw_results[key],
            columns=[
                "jobNumber",
                "trial",
                "runtime_ms",
            ],
        )

        mean_df = pd.DataFrame(
            mean_results[key],
            columns=[
                "jobNumber",
                "mean_runtime_ms",
            ],
#            columns=[
#                "jobNumber",
#                "mean_runtime_ms",
#                "p95_runtime_ms",
#                "max_runtime_ms",
#            ],
        )

        raw_file = (
            output_dir
            / f"raw_M{modelNumber}_D{Deadline}.csv"
        )

        mean_file = (
            output_dir
            / f"mean_M{modelNumber}_D{Deadline}.csv"
        )

        raw_df.to_csv(raw_file, index=False)
        mean_df.to_csv(mean_file, index=False)

    print(f"Saved")


