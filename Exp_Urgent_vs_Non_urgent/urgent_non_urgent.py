import numpy as np

from experiment import program

from pathlib import Path



def save_multiple_plot_data(x,y_dict,filename):

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as file:

        file.write(f"x = {np.asarray(x).tolist()}\n\n")

        for label, y_values in y_dict.items():

            file.write(
                f"{label} = {np.array(y_values).tolist()}\n\n"
            )



def main():


    rng = np.random.default_rng(3)

    iterations = 1

    # Ratio of jobs that belong to the mandatory-job group.
    mandatory_probs = np.arange(0.1, 1.0, 0.1)

    # Number of jobs tested in each experiment.
    job_numbers = np.arange(10, 30, 1)

    # Deadline values from 110 through 190 with a step size of 20.
    for Deadline in range(110, 201, 20):

        all_plot_data = {}

        # Run the experiment for each number of jobs.
        for job_number in job_numbers:

            served_mean_vals = []
            
            # Generate one integer weight between 1 and 9 for each job.
            job_weights_all = rng.integers(low=1,high=10,size=job_number)

            for mandatory_prob in mandatory_probs:

                served_iteration_results = []

                for _ in range(iterations):

                    # Generate the TTC value for every job.
                    TTC_all = np.where(
                        rng.random(job_number) < mandatory_prob,

                        Deadline
                        + rng.random(job_number)
                        * Deadline,

                        2 * Deadline
                        + rng.exponential(
                            scale=2 * Deadline,
                            size=job_number
                        )
                    )

                    # Run the scheduling algorithm.
                    accuracy_set  = program(
                        job_number,
                        TTC_all,
                        job_weights_all,
                        Deadline
                    )

                    served_jobs = np.sum(accuracy_set)

                    served_iteration_results.append(served_jobs)

                served_mean_vals.append(np.mean(served_iteration_results))

            all_plot_data[f"jobs_{job_number}"] = served_mean_vals



        save_multiple_plot_data(
            mandatory_probs,
            all_plot_data,
            f"Jobs_Served_Multiple_Job_Counts_{Deadline}.txt"
        )


if __name__ == "__main__":

    main()
