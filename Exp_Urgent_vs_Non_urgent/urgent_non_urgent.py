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

    iterations = 1000

    mandatory_probs = np.arange(0.1, 1.0, 0.1)

    job_numbers = np.arange(10, 30, 1)


    for Deadline in range(110, 201, 20):

        all_plot_data = {}


        for job_number in job_numbers:

            served_mean_vals = []

            job_weights_all = rng.integers(low=1,high=10,size=job_number)

            for mandatory_prob in mandatory_probs:

                served_iteration_results = []

                for _ in range(iterations):

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

                    program_results  = program(
                        job_number,
                        TTC_all,
                        job_weights_all,
                        Deadline
                    )

                    accuracy_set = program_results[4]

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
