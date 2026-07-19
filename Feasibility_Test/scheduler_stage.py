# scheduler_stage.py

from dataclasses import dataclass, field

import numpy as np
import gurobipy as gp

from gurobipy import GRB



@dataclass
class ModelSpec:

    name: str

    model_path: str

    accuracy: float

    exec_time_ms: float



@dataclass
class PendingJob:

    job_uid: str

    first_cycle_id: int

    last_seen_cycle_id: int

    obj: dict = field(
        default_factory=dict
    )


class SchedulerMemory:

    def __init__(self):

        self.pending_jobs: dict[
            str,
            PendingJob,
        ] = {}


    def make_job_uid(
        self,
        cycle_id: int,
        object_index: int,
        obj: dict,
    ) -> str:

        return (
            f"cycle{cycle_id}_"
            f"Job{object_index}_"
            f"{obj['label']}"
        )


    def count_pending_jobs(self):

        return len(
            self.pending_jobs
        )


    def add_current_cycle_jobs(
        self,
        cycle_id: int,
        current_objects: list[dict],
    ) -> list[dict]:

        for object_index, obj in enumerate(
            current_objects,
            start=1,
        ):

            job_uid = self.make_job_uid(
                cycle_id,
                object_index,
                obj,
            )

            obj["job_uid"] = job_uid

            obj["first_cycle_id"] = cycle_id
            obj["last_seen_cycle_id"] = cycle_id

            self.pending_jobs[job_uid] = PendingJob(
                job_uid=job_uid,
                first_cycle_id=cycle_id,
                last_seen_cycle_id=cycle_id,
                obj=obj,
            )

        return [
            pending.obj
            for pending in self.pending_jobs.values()
        ]


    def update_ttc(
        self,
        delta_time_sec: float,
    ) -> None:

        for pending in self.pending_jobs.values():

            current_ttc = float(
                pending.obj.get(
                    "ttc",
                    0.0,
                )
            )

            new_ttc = (
                current_ttc
                - delta_time_sec
            )

            pending.obj["ttc"] = max(
                0.0,
                round(
                    new_ttc,
                    3,
                ),
            )


    def remove_missed_deadline_jobs(
        self,
        threshold_sec=1.0,
    ) -> list[str]:

        missed_jobs = []
        remove_keys = []

        for job_uid, pending in self.pending_jobs.items():

            ttc = float(
                pending.obj.get(
                    "ttc",
                    0.0,
                )
            )

            if ttc < threshold_sec:

                missed_jobs.append(
                    job_uid
                )

                remove_keys.append(
                    job_uid
                )

        for job_uid in remove_keys:

            del self.pending_jobs[job_uid]

        return missed_jobs


    def remove_served_jobs(
        self,
        served_job_uids,
    ):

        for uid in served_job_uids:

            if uid in self.pending_jobs:

                del self.pending_jobs[uid]


    def get_pending_objects(self):

        return [
            pending.obj
            for pending in self.pending_jobs.values()
        ]


def get_weight_from_ttc(
    ttc: float,
) -> float:

    if ttc <= 0:
        return 1000.0

    return 1.0 / ttc


def run_scheduler(
    model_specs,
    objects,
    deadline_ms,
    scheduler_mode,
):

    if len(objects) == 0:

        return []

    job_weights = []

    job_ttcs = []

    for obj in objects:

        ttc = float(
            obj["ttc"]
        )

        job_ttcs.append(ttc)

        job_weights.append(
            get_weight_from_ttc(ttc)
        )

    if scheduler_mode == "greedy_low_cost":

        return greedy_low_cost_scheduler(

            model_specs,

            objects,

            deadline_ms,

            job_ttcs,
        )

    if scheduler_mode == "greedy_high_cost":

        return greedy_high_cost_scheduler(

            model_specs,

            objects,

            deadline_ms,

            job_ttcs,
        )

    if scheduler_mode == "RETINA":

        return gurobi_RETINA(

            model_specs,

            objects,

            deadline_ms,

            job_weights,
        )


    if scheduler_mode == "MOT":

        return gurobi_MOT(

            model_specs,

            objects,

            deadline_ms,
        )

    return []



def greedy_low_cost_scheduler(
    model_specs,
    objects,
    deadline_ms,
    job_ttcs,
):

    print(
        "Low Cost Greedy Scheduler"
    )

    remaining_time = deadline_ms

    schedule = []

    jobs = list(
        range(len(objects))
    )

    jobs.sort(
        key=lambda i:
            job_ttcs[i]
    )

    model_order = sorted(

        range(len(model_specs)),

        key=lambda model_id:
            model_specs[
                model_id
            ].exec_time_ms,
    )

    for job_id in jobs:

        selected_model = None

        for model_id in model_order:

            exec_time = (
                model_specs[
                    model_id
                ].exec_time_ms
            )

            if exec_time <= remaining_time:

                selected_model = model_id

                remaining_time -= exec_time

                break

        schedule.append(
            (
                job_id,
                selected_model,
            )
        )

    return schedule


def greedy_high_cost_scheduler(
    model_specs,
    objects,
    deadline_ms,
    job_ttcs,
):

    print(
        "High Cost Greedy Scheduler"
    )

    remaining_time = deadline_ms

    schedule = []

    jobs = list(
        range(len(objects))
    )

    jobs.sort(
        key=lambda i:
            job_ttcs[i]
    )


    highest_model_id = max(

        range(len(model_specs)),

        key=lambda model_id:
            model_specs[
                model_id
            ].accuracy
    )

    highest_exec_time = (
        model_specs[
            highest_model_id
        ].exec_time_ms
    )

    for job_id in jobs:

        selected_model = None

        if (
            highest_exec_time
            <= remaining_time
        ):

            selected_model = (
                highest_model_id
            )

            remaining_time -= (
                highest_exec_time
            )

        schedule.append(
            (
                job_id,
                selected_model,
            )
        )

    return schedule


def gurobi_RETINA(
    model_specs,
    objects,
    deadline_ms,
    job_weights,
):

    print(
        "RETINA Scheduler"
    )

    job_number = len(objects)

    model_number = len(
        model_specs
    )

    accuracy = np.array([

        spec.accuracy

        for spec in model_specs
    ])

    execution_time = np.array([

        spec.exec_time_ms

        for spec in model_specs
    ])

    model = gp.Model()

    model.setParam(
        "OutputFlag",
        0,
    )

    x = {}

    for job_id in range(
        job_number
    ):

        for model_id in range(
            model_number
        ):

            x[
                job_id,
                model_id
            ] = model.addVar(
                vtype=GRB.BINARY
            )

    model.update()

    model.setObjective(

        gp.quicksum(

            job_weights[
                job_id
            ]
            * accuracy[
                model_id
            ]
            * x[
                job_id,
                model_id
            ]

            for job_id
            in range(job_number)

            for model_id
            in range(model_number)
        ),

        GRB.MAXIMIZE,
    )

    for job_id in range(
        job_number
    ):

        model.addConstr(

            gp.quicksum(

                x[
                    job_id,
                    model_id
                ]

                for model_id
                in range(
                    model_number
                )

            ) <= 1
        )

    model.addConstr(

        gp.quicksum(

            execution_time[
                model_id
            ]
            * x[
                job_id,
                model_id
            ]

            for job_id
            in range(job_number)

            for model_id
            in range(model_number)

        ) <= deadline_ms
    )

    model.optimize()

    schedule = []

    for job_id in range(
        job_number
    ):

        selected_model = None

        for model_id in range(
            model_number
        ):

            if (
                x[
                    job_id,
                    model_id
                ].X > 0.5
            ):

                selected_model = (
                    model_id
                )

                break

        schedule.append(
            (
                job_id,
                selected_model,
            )
        )

    return schedule


def gurobi_MOT(
    model_specs,
    objects,
    deadline_ms,
):

    print(
        "MOT Scheduler"
    )

    job_number = len(objects)

    model_number = len(
        model_specs
    )

    accuracy = np.array([

        spec.accuracy

        for spec in model_specs
    ])

    execution_time = np.array([

        spec.exec_time_ms

        for spec in model_specs
    ])

    model = gp.Model()

    model.setParam(
        "OutputFlag",
        0,
    )

    x = {}

    for job_id in range(
        job_number
    ):

        for model_id in range(
            model_number
        ):

            x[
                job_id,
                model_id
            ] = model.addVar(
                vtype=GRB.BINARY
            )

    model.update()


    model.setObjective(

        gp.quicksum(

            accuracy[
                model_id
            ]
            * x[
                job_id,
                model_id
            ]

            for job_id
            in range(job_number)

            for model_id
            in range(model_number)
        ),

        GRB.MAXIMIZE,
    )


    for job_id in range(
        job_number
    ):

        model.addConstr(

            gp.quicksum(

                x[
                    job_id,
                    model_id
                ]

                for model_id
                in range(
                    model_number
                )

            ) == 1
        )


    model.addConstr(

        gp.quicksum(

            execution_time[
                model_id
            ]
            * x[
                job_id,
                model_id
            ]

            for job_id
            in range(job_number)

            for model_id
            in range(model_number)

        ) <= deadline_ms
    )

    model.optimize()

    schedule = []

    if (
        model.Status
        != GRB.OPTIMAL
    ):

        for job_id in range(
            job_number
        ):

            schedule.append(
                (
                    job_id,
                    None,
                )
            )

        return schedule


    for job_id in range(
        job_number
    ):

        selected_model = None

        for model_id in range(
            model_number
        ):

            if (
                x[
                    job_id,
                    model_id
                ].X > 0.5
            ):

                selected_model = (
                    model_id
                )

                break

        schedule.append(
            (
                job_id,
                selected_model,
            )
        )

    return schedule

