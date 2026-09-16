"""
dp_scheduler.py
-----------------
RTDeepIoT's real-time depth-assignment scheduler (Algorithm 1, \u00a7II-C),
plus the greedy depth-reassignment heuristic (\u00a7II-E).

Deadline model (as agreed): d_i = TTC_i directly, no thresholding, no
mandatory/optional binning borrowed from anywhere else. Tasks are sorted
by deadline (EDF), exactly the multi-deadline setting Algorithm 1 was
designed for.

Mandatory-part model (structural to RTDeepIoT itself, \u00a7II-B): every
task must execute at least depth omega_i (here, omega_i = 1 for all
tasks -- stage 1 always runs). This is NOT derived from TTC or borrowed
from anywhere else; it's how the original paper defines "mandatory".

Reward model: reward at a given depth is the model's confidence
(softmax max-probability) at that stage's exit, matching the paper's
\u00a7II-D framing of "confidence in correctness" as the utility metric.

Two-phase scheduling per cycle:
  Phase A (plan): for every active task, predict R^1..R^3 using the
    running average of previously observed stage-1 confidence as the
    seed, and the Exp/Max/Linear heuristic for deeper stages. Run the
    DP to get an initial depth assignment for every task.
  Phase B (revise, \u00a7II-E): as stage 1 actually executes for each task
    and returns REAL confidence, run the greedy reassignment heuristic
    to check whether the task's remaining depth assignment is still
    optimal given the updated real number.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json

try:
    from .utility_heuristics import predict_future_utility
except ImportError:
    from utility_heuristics import predict_future_utility


# ─────────────────────────────────────────────
#  Task representation
# ─────────────────────────────────────────────

@dataclass
class Task:
    job_id: str
    deadline_ms: float          # d_i = TTC_i, in milliseconds
    omega: int = 1              # mandatory minimum depth (\u00a7II-B)
    max_depth: int = 3          # L_i
    assigned_depth: int = 0     # result of scheduling, filled in later
    predicted_rewards: List[float] = field(default_factory=list)  # R^1..R^L
    exec_times: List[float] = field(default_factory=list)         # P^1..P^L (cumulative ms)


# ─────────────────────────────────────────────
#  WCET profile loading
# ─────────────────────────────────────────────

def load_wcet_profile(wcet_json_path: str) -> List[float]:
    """
    Loads cumulative per-stage WCET (ms) from the JSON produced by
    profile_wcet.py, e.g.:
        {"stage_1_wcet_ms": 1.2559, "stage_2_wcet_ms": 2.3333,
         "stage_3_wcet_ms": 3.0712, ...}
    Returns [P^1, P^2, P^3] in order.
    """
    with open(wcet_json_path, 'r') as f:
        data = json.load(f)
    return [
        data['stage_1_wcet_ms'],
        data['stage_2_wcet_ms'],
        data['stage_3_wcet_ms'],
    ]


# ─────────────────────────────────────────────
#  Reward prediction (Phase A)
# ─────────────────────────────────────────────

def predict_task_rewards(exec_times: List[float], seed_confidence: float,
                         heuristic: str = 'exp') -> List[float]:
    """
    Predicts R^1..R^L for a task that hasn't executed any stage yet.
    R^1 is seeded from the running average of previously observed
    stage-1 confidence (since it's not yet known for THIS task).
    R^2, R^3 are predicted from R^1 using the chosen heuristic,
    chained stage-by-stage (R^2 predicted from R^1, R^3 from R^2).
    """
    L = len(exec_times)
    rewards = [seed_confidence]
    for l in range(1, L):
        r_next = predict_future_utility(
            R_current=rewards[-1],
            P_current=exec_times[l - 1],
            P_next=exec_times[l],
            method=heuristic,
        )
        rewards.append(r_next)
    return rewards


# ─────────────────────────────────────────────
#  DP scheduler (Algorithm 1)
# ─────────────────────────────────────────────

def dp_schedule(tasks: List[Task], delta: float = 0.01) -> Dict[str, int]:
    """
    Implements Algorithm 1 / Eqs (1)-(2): EDF-sorted, quantized-reward
    DP that assigns each task a depth l_i in {omega_i, ..., L_i} to
    maximize total reward subject to each task's own deadline
    (F_i^l <= d_i, where F_i^l is the finish time under EDF ordering).

    Args:
        tasks : list of Task objects, each with .deadline_ms,
                .omega, .max_depth, .predicted_rewards ([R^1..R^L]),
                .exec_times ([P^1..P^L], cumulative ms)
        delta : reward quantization step (paper uses ~0.1 on
                accuracy-scale rewards; since our rewards are raw
                confidence in [0,1], 0.01 gives finer resolution while
                keeping the DP table small for realistic task counts)

    Returns:
        dict[job_id] -> assigned depth (int, 0 if a task is entirely
        infeasible even at its mandatory depth -- should not normally
        happen since mandatory depth is by definition the cheapest
        option, but included defensively)
    """
    if not tasks:
        return {}

    # Sort by deadline (EDF), matching the paper's d_1 <= d_2 <= ... <= d_N
    sorted_tasks = sorted(tasks, key=lambda t: t.deadline_ms)
    N = len(sorted_tasks)

    # Max possible total reward (used to size the quantized reward axis)
    R_max_per_task = [max(t.predicted_rewards) for t in sorted_tasks]
    R_total_max = sum(R_max_per_task)
    n_buckets = max(1, int(R_total_max / delta) + 1)

    # DP tables: S[i][r] = best depth choice per task up to task i for
    # reaching quantized reward bucket r; P[i][r] = min cumulative
    # execution time to achieve that. Using dict-of-dicts sparse
    # representation since most buckets are unreachable for small N.
    # S[i] maps bucket -> list of depths chosen for tasks 1..i
    # P[i] maps bucket -> total execution time
    NEG_INF = float('inf')

    # Base case: task 1 (index 0)
    S: List[Dict[int, List[int]]] = [dict() for _ in range(N)]
    P: List[Dict[int, float]] = [dict() for _ in range(N)]

    t0 = sorted_tasks[0]
    for l in range(t0.omega, t0.max_depth + 1):
        exec_t = t0.exec_times[l - 1]
        if exec_t > t0.deadline_ms:
            continue  # infeasible depth for this task's own deadline
        reward = t0.predicted_rewards[l - 1]
        bucket = int(reward / delta)
        if bucket not in P[0] or exec_t < P[0][bucket]:
            P[0][bucket] = exec_t
            S[0][bucket] = [l]

    # Recursive fill (Eqs 1-2), single-deadline-per-task feasibility
    # checked as "cumulative finish time under EDF <= this task's own
    # deadline" -- consistent with the paper's F_i^l <= d_i check.
    for i in range(1, N):
        ti = sorted_tasks[i]
        prev_P = P[i - 1]
        prev_S = S[i - 1]

        for l in range(ti.omega, ti.max_depth + 1):
            exec_t = ti.exec_times[l - 1]
            reward = ti.predicted_rewards[l - 1]

            for prev_bucket, prev_time in prev_P.items():
                new_time = prev_time + exec_t
                # Feasibility: this task must finish (cumulatively)
                # by its own deadline under EDF ordering.
                if new_time > ti.deadline_ms:
                    continue
                new_bucket = prev_bucket + int(reward / delta)

                if new_bucket not in P[i] or new_time < P[i][new_bucket]:
                    P[i][new_bucket] = new_time
                    S[i][new_bucket] = prev_S[prev_bucket] + [l]

    # Backtrack: pick the best feasible bucket for the last task
    if not P[N - 1]:
        # No feasible schedule at all (shouldn't normally happen if
        # every task's mandatory depth alone fits its own deadline,
        # but guard against pathological inputs)
        return {t.job_id: 0 for t in sorted_tasks}

    best_bucket = max(P[N - 1].keys())
    depths = S[N - 1][best_bucket]

    return {t.job_id: d for t, d in zip(sorted_tasks, depths)}


# ─────────────────────────────────────────────
#  Greedy depth-reassignment (\u00a7II-E)
# ─────────────────────────────────────────────

def greedy_reassign(tasks: List[Task], current_task_id: str,
                    real_confidence: float, exec_times: List[float],
                    heuristic: str = 'exp') -> Dict[str, int]:
    """
    \u00a7II-E: after `current_task_id`'s stage 1 actually executes and
    returns REAL confidence (not the seeded estimate used in Phase A),
    check whether swapping its remaining depth assignment for a
    different task's remaining stages would improve total reward,
    given the time budget the current task's remaining stages would
    have consumed.

    This is a lightweight, greedy check -- NOT a full DP re-solve
    (which the paper explicitly avoids doing every stage completion,
    since the current task is always the shortest-deadline one under
    EDF and a full re-solve would need to recompute all subsequent
    tasks).

    Returns an updated depth-assignment dict (may be identical to the
    input if no beneficial swap is found).
    """
    current = next((t for t in tasks if t.job_id == current_task_id), None)
    if current is None or current.assigned_depth >= current.max_depth:
        return {t.job_id: t.assigned_depth for t in tasks}

    # Re-predict this task's remaining-stage rewards from the REAL
    # observed confidence (replacing the seeded estimate used in Phase A)
    updated_rewards = [real_confidence]
    for l in range(1, len(exec_times)):
        r_next = predict_future_utility(
            R_current=updated_rewards[-1],
            P_current=exec_times[l - 1],
            P_next=exec_times[l],
            method=heuristic,
        )
        updated_rewards.append(r_next)

    l1 = current.assigned_depth
    remaining_budget = (
        current.exec_times[current.assigned_depth - 1]
        if current.assigned_depth > 0 else 0.0
    )
    # Time the current task's ALREADY-assigned remaining stages would take
    if current.assigned_depth < current.max_depth:
        remaining_time_current = (
            current.exec_times[current.max_depth - 1]
            - current.exec_times[current.assigned_depth - 1]
        )
    else:
        remaining_time_current = 0.0

    current_remaining_gain = (
        updated_rewards[current.max_depth - 1] - updated_rewards[current.assigned_depth - 1]
        if current.max_depth > current.assigned_depth else 0.0
    )

    # Search other tasks for a better use of that time budget (Eq 7)
    best_alt_gain = -1.0
    best_alt_task = None
    best_alt_depth = None

    for other in tasks:
        if other.job_id == current_task_id:
            continue
        if other.assigned_depth >= other.max_depth:
            continue  # already at max depth, nothing to extend

        for l in range(other.assigned_depth + 1, other.max_depth + 1):
            extra_time = (
                other.exec_times[l - 1]
                - other.exec_times[other.assigned_depth - 1]
                if other.assigned_depth > 0 else other.exec_times[l - 1]
            )
            if extra_time > remaining_time_current:
                continue  # doesn't fit in the freed-up budget

            gain = other.predicted_rewards[l - 1] - (
                other.predicted_rewards[other.assigned_depth - 1]
                if other.assigned_depth > 0 else 0.0
            )
            if gain > best_alt_gain:
                best_alt_gain = gain
                best_alt_task = other
                best_alt_depth = l

    # Apply the swap only if it strictly beats keeping the current
    # task's original remaining-depth plan (Eq 7's decision rule)
    if best_alt_task is not None and best_alt_gain > current_remaining_gain:
        current.assigned_depth = l1  # current task does NOT get deeper
        best_alt_task.assigned_depth = best_alt_depth

    return {t.job_id: t.assigned_depth for t in tasks}


if __name__ == '__main__':
    # Small smoke test using your actual profiled WCET numbers
    exec_times = [1.2559, 2.3333, 3.0712]  # from wcet_road.json

    tasks = [
        Task(job_id='obj_1', deadline_ms=2000, exec_times=exec_times,
             predicted_rewards=predict_task_rewards(exec_times, seed_confidence=0.55)),
        Task(job_id='obj_2', deadline_ms=500,  exec_times=exec_times,
             predicted_rewards=predict_task_rewards(exec_times, seed_confidence=0.60)),
        Task(job_id='obj_3', deadline_ms=8000, exec_times=exec_times,
             predicted_rewards=predict_task_rewards(exec_times, seed_confidence=0.40)),
    ]

    assignment = dp_schedule(tasks)
    print("Initial DP depth assignment (by TTC-derived deadline):")
    for t in tasks:
        t.assigned_depth = assignment[t.job_id]
        print(f"  {t.job_id}: deadline={t.deadline_ms}ms -> depth={t.assigned_depth}  "
              f"predicted_rewards={[round(r,3) for r in t.predicted_rewards]}")

    # Simulate obj_2's stage 1 actually executing with a real confidence
    # very different from what was predicted
    updated = greedy_reassign(tasks, current_task_id='obj_2',
                              real_confidence=0.95, exec_times=exec_times)
    print("\nAfter obj_2's real stage-1 confidence=0.95 (greedy reassignment):")
    for t in tasks:
        print(f"  {t.job_id}: depth={updated[t.job_id]}")
