"""
utility_heuristics.py
----------------------
Section II-D of the RTDeepIoT paper: heuristics for predicting the
utility (confidence) of a future, not-yet-executed stage, given the
confidence already observed at the current stage.

These are needed because the scheduler must decide whether running a
task's NEXT stage is worthwhile *before* that stage actually executes
-- at that point the model's real confidence at the deeper stage is
unknown, so it has to be estimated.

All three heuristics take:
    R_current : float in [0,1] -- confidence at the current (already
                executed) stage
    P_current : float -- cumulative execution time (ms) to reach the
                current stage
    P_next    : float -- cumulative execution time (ms) to reach the
                next stage
Return:
    R_next    : float in [0,1] -- predicted confidence at the next stage
"""


def predict_max(R_current: float, P_current: float, P_next: float) -> float:
    """
    'Maximum increase' heuristic (\u00a7II-D):
    Assumes the next stage will push confidence all the way to 1.
    Optimistic -- favors tasks with low current confidence, since they
    look like they have the most to gain.
    """
    return 1.0


def predict_exp(R_current: float, P_current: float, P_next: float) -> float:
    """
    'Exponential increase' heuristic (\u00a7II-D):
    Assumes the next stage halves the remaining distance to 1.
    This was the best-performing heuristic in the original paper's
    evaluation (\u00a7IV-A), consistently closest to the oracle.
    """
    return R_current + 0.5 * (1.0 - R_current)


def predict_linear(R_current: float, P_current: float, P_next: float) -> float:
    """
    'Linear increase' heuristic (\u00a7II-D):
    Assumes confidence scales linearly with cumulative execution time.
    Guards against P_current == 0 (shouldn't happen once stage 1 has
    actually run, since it always has nonzero cost, but guarded
    defensively).
    """
    if P_current <= 0:
        return min(1.0, R_current)
    return min(1.0, R_current * (P_next / P_current))


HEURISTICS = {
    'max':    predict_max,
    'exp':    predict_exp,
    'linear': predict_linear,
}


def predict_future_utility(R_current: float, P_current: float,
                           P_next: float, method: str = 'exp') -> float:
    """
    Dispatch to the requested heuristic. 'exp' is the recommended default
    per the paper's own findings.
    """
    if method not in HEURISTICS:
        raise ValueError(
            f"Unknown heuristic '{method}'. Choose from {list(HEURISTICS)}."
        )
    return HEURISTICS[method](R_current, P_current, P_next)


if __name__ == '__main__':
    # Quick sanity check against the paper's own description
    R1, P1, P2 = 0.6, 1.26, 2.33
    print(f"Given R1={R1}, P1={P1}ms, P2={P2}ms:")
    print(f"  max:    R2 = {predict_max(R1, P1, P2):.3f}")
    print(f"  exp:    R2 = {predict_exp(R1, P1, P2):.3f}")
    print(f"  linear: R2 = {predict_linear(R1, P1, P2):.3f}")
