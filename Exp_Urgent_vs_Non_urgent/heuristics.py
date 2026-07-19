import numpy as np


def Low_cost_heuristic(
    used_jobs,
    job_weights,
    accuracy,
    executionTime,
    Deadline,
    mandatory_job_indices
):

    nJ = len(job_weights)

    m_Low = int(np.argmin(executionTime))

    cost = executionTime[m_Low]

    remaining_budget = Deadline

    obj = 0.0

    for i in mandatory_job_indices:

        if cost > remaining_budget:
            return np.nan

        used_jobs.append(int(i))

        remaining_budget -= cost

        obj += job_weights[i] * accuracy[m_Low]

    for i in range(nJ):

        if i in used_jobs:
            continue

        if cost <= remaining_budget:

            remaining_budget -= cost

            obj += (
                job_weights[i]
                * accuracy[m_Low]
            )

            used_jobs.append(i)

        else:
            break

    used_jobs.sort()

    return obj


def High_cost_heuristic(
    used_jobs,
    job_weights,
    accuracy,
    executionTime,
    Deadline,
    mandatory_job_indices
):

    nJ = len(job_weights)

    m_High = int(np.argmax(executionTime))

    cost = executionTime[m_High]

    remaining_budget = Deadline

    obj = 0.0

    for i in mandatory_job_indices:

        if cost > remaining_budget:
            return np.nan

        used_jobs.append(int(i))

        remaining_budget -= cost

        obj += job_weights[i] * accuracy[m_High]

    for i in range(nJ):

        if i in used_jobs:
            continue

        if cost <= remaining_budget:

            remaining_budget -= cost

            obj += (
                job_weights[i]
                * accuracy[m_High]
            )

            used_jobs.append(i)

        else:
            break

    used_jobs.sort()

    return obj
