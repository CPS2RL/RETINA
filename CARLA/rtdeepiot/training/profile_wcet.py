"""
profile_wcet.py
---------------
Profiles Worst-Case Execution Time (WCET) of each neural network stage
on the GPU. This is a required step before running the RTDeepIoT scheduler.

The paper profiles each stage 10,000 times and takes the upper bound
of the 99% confidence interval as the WCET.

Run:
    python training/profile_wcet.py --dataset cifar10 \
        --checkpoint ./checkpoints/best_cifar10.pth

Output:
    wcet_cifar10.json  (or wcet_imagenet.json)
    with structure: {"stage_1_wcet_ms": X, "stage_2_wcet_ms": Y, "stage_3_wcet_ms": Z}
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multi_exit_resnet import build_model


# ─────────────────────────────────────────────
#  Stage-isolated forward passes
# ─────────────────────────────────────────────

def time_stage(model, stage_fn_sequence, dummy_input, n_runs=10000, warmup=200):
    """
    Time a partial forward pass (up to a given stage) on GPU.

    Args:
        model          : the MultiExit model
        stage_fn_sequence : callable that runs exactly up to stage k
        dummy_input    : (1, C, H, W) tensor on GPU
        n_runs         : number of timing runs
        warmup         : warm-up iterations (not timed)

    Returns:
        times_ms : np.array of per-run latencies in milliseconds
    """
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(warmup):
            _ = stage_fn_sequence(dummy_input)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _  = stage_fn_sequence(dummy_input)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return np.array(times_ms)


def wcet_from_times(times_ms, ci=0.99):
    """
    Upper bound of the ci-confidence interval.
    With ci=0.99 and 10,000 samples this is the 99th percentile.
    """
    return float(np.percentile(times_ms, ci * 100))


# ─────────────────────────────────────────────
#  Build isolated stage runners
# ─────────────────────────────────────────────

def get_stage_runners_cifar(model):
    """Returns 3 callables, each running exactly stages 1..k."""
    import torch.nn.functional as F

    def run_stage1(x):
        x = F.relu(model.bn0(model.conv0(x)))
        x = model.stage1(x)
        return model.head1(x)

    def run_stage2(x):
        x = F.relu(model.bn0(model.conv0(x)))
        x = model.stage1(x)
        x = model.stage2(x)
        return model.head2(x)

    def run_stage3(x):
        x = F.relu(model.bn0(model.conv0(x)))
        x = model.stage1(x)
        x = model.stage2(x)
        x = model.stage3(x)
        return model.head3(x)

    return [run_stage1, run_stage2, run_stage3]


def get_stage_runners_imagenet(model):
    def run_stage1(x):
        x = model.stem(x)
        x = model.stage1(x)
        return model.head1(x)

    def run_stage2(x):
        x = model.stem(x)
        x = model.stage1(x)
        x = model.stage2a(x)
        x = model.stage2b(x)
        return model.head2(x)

    def run_stage3(x):
        x = model.stem(x)
        x = model.stage1(x)
        x = model.stage2a(x)
        x = model.stage2b(x)
        x = model.stage3(x)
        return model.head3(x)

    return [run_stage1, run_stage2, run_stage3]


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def profile(args):
    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu {args.gpu} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) visible "
                f"(0..{torch.cuda.device_count()-1})."
            )
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
        print(f"Using device: {device}  ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU detected. WCET numbers from CPU are not "
              "representative for the scheduler. Connect a GPU.")

    # ── Load model ──
    model = build_model(args.dataset).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided — using randomly initialised weights.")

    # ── Dummy input ──
    if args.dataset in ('cifar10', 'road'):
        # ROAD crops are resized to 32x32 (same as CIFAR-10), and use
        # the identical ResNet-32 architecture/stage runners.
        dummy = torch.randn(1, 3, 32,  32,  device=device)
        runners = get_stage_runners_cifar(model)
    else:
        dummy = torch.randn(1, 3, 224, 224, device=device)
        runners = get_stage_runners_imagenet(model)

    # ── Profile each stage ──
    results = {}
    n_runs  = args.n_runs

    for i, runner in enumerate(runners, start=1):
        print(f"Profiling Stage {i}  ({n_runs} runs) ...", end=' ', flush=True)
        times = time_stage(model, runner, dummy, n_runs=n_runs, warmup=200)
        wcet  = wcet_from_times(times, ci=0.99)
        mean  = times.mean()
        std   = times.std()
        print(f"mean={mean:.3f}ms  std={std:.3f}ms  WCET(99%)={wcet:.3f}ms")

        results[f'stage_{i}_mean_ms']  = round(mean,  4)
        results[f'stage_{i}_std_ms']   = round(std,   4)
        results[f'stage_{i}_wcet_ms']  = round(wcet,  4)

    # ── Save ──
    out_path = os.path.join(
        os.path.dirname(args.checkpoint or '.'),
        f'wcet_{args.dataset}.json'
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWCET results saved -> {out_path}")
    print(json.dumps(results, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description='Profile stage WCET')
    p.add_argument('--dataset',    type=str, default='cifar10',
                   choices=['cifar10', 'imagenet', 'road'])
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to trained .pth checkpoint')
    p.add_argument('--n_runs',     type=int, default=10000,
                   help='Number of profiling runs per stage (paper uses 10,000)')
    p.add_argument('--gpu',        type=int, default=0,
                   help='CUDA device index to use, e.g. --gpu 6')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    profile(args)
