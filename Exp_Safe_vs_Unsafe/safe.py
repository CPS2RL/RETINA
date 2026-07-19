import numpy as np
import pandas as pd

from experiment import program


def run_experiment(run_id,Deadline):

    job_range = range(11, 24)

    rng = np.random.default_rng(run_id)

    max_jobs = max(job_range)

    modelNumber=5

    TTC_all = np.where(
        rng.random(max_jobs) < 0.2,
        Deadline + rng.random(max_jobs) * Deadline,
        2 * Deadline + rng.exponential(
            scale=2 * Deadline,
            size=max_jobs
        )
    )

    job_weights_all = rng.integers(
        low=1,
        high=10,
        size=max_jobs
    )



    csv_rows = []

    for jobNumber in job_range:


        result= program(
            jobNumber,
            TTC_all,
            job_weights_all,
            Deadline,
            modelNumber
        )

        total_weight = np.sum(result[10])

        optimal_weighted_accuracy = result[0] / total_weight
        low_weighted_accuracy = result[1] / total_weight
        high_weighted_accuracy = result[2] / total_weight
        noskip_weighted_accuracy = result[3] / total_weight

        optimal_safe_jobs = (optimal_weighted_accuracy * jobNumber)
        optimal_unsafe_jobs = (jobNumber - optimal_safe_jobs )

        low_safe_jobs = (low_weighted_accuracy * jobNumber)
        low_unsafe_jobs = (jobNumber - low_safe_jobs)

        high_safe_jobs = (high_weighted_accuracy * jobNumber)
        high_unsafe_jobs = (jobNumber - high_safe_jobs)

        noskip_safe_jobs = (noskip_weighted_accuracy * jobNumber)
        noskip_unsafe_jobs = (jobNumber - noskip_safe_jobs)



        csv_rows.append({
            "Run": run_id + 1,
            "Jobs": jobNumber,

            "Optimal_Safe": optimal_safe_jobs,
            "Optimal_Unsafe": optimal_unsafe_jobs,

            "Low_Safe": low_safe_jobs,
            "Low_Unsafe": low_unsafe_jobs,

            "High_Safe": high_safe_jobs,
            "High_Unsafe": high_unsafe_jobs,

            "CA_MOT_Safe": noskip_safe_jobs,
            "CA_MOT_Unsafe": noskip_unsafe_jobs,

        })

    return csv_rows


def main():

    all_rows = []

    Deadline=200

    for run_id in range(1000):

        print(f"\n========== RUN {run_id+1} ==========\n")

        rows = run_experiment(run_id, Deadline)

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    df.to_csv(
        f"Deadline_{Deadline}_safe_unsafe_results.csv",
        index=False
    )



if __name__ == "__main__":

    main()
