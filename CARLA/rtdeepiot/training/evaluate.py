"""
evaluate.py
-----------
Evaluates a trained Multi-Exit ResNet checkpoint.

Reports:
  - Per-stage Top-1 accuracy
  - Per-stage Expected Calibration Error (ECE)
  - Per-stage average confidence score
  - Confidence histogram data (useful for plotting)

Run:
    python training/evaluate.py --dataset cifar10 \
        --checkpoint ./checkpoints/best_cifar10.pth
"""

import argparse
import json
import os

import torch
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multi_exit_resnet import build_model
from data.dataloader          import get_dataloaders
from data.road_crop_dataset   import get_road_dataloaders, CLASS_NAMES as ROAD_CLASS_NAMES


@torch.no_grad()
def full_evaluate(model, loader, device, num_bins=10):
    model.eval()

    # Per-stage accumulators
    n_stages = 3
    correct      = [0] * n_stages
    total        = 0
    conf_buckets = [{'conf': [], 'correct': []} for _ in range(n_stages)]

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        stage_outputs = model(images)

        B = images.size(0)
        total += B

        for i, out in enumerate(stage_outputs):
            is_correct = (out['predicted'] == targets)
            correct[i]  += is_correct.sum().item()
            conf_buckets[i]['conf'].extend(out['confidence'].cpu().tolist())
            conf_buckets[i]['correct'].extend(is_correct.cpu().tolist())

    # ── Accuracy ──
    accs = [100.0 * correct[i] / total for i in range(n_stages)]

    # ── ECE ──
    eces = []
    for i in range(n_stages):
        conf_t    = torch.tensor(conf_buckets[i]['conf'])
        correct_t = torch.tensor(conf_buckets[i]['correct'], dtype=torch.float)
        bins      = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        N   = len(conf_t)
        for b in range(num_bins):
            lo, hi  = bins[b], bins[b + 1]
            in_bin  = (conf_t >= lo) & (conf_t < hi)
            if in_bin.sum() == 0:
                continue
            acc_b   = correct_t[in_bin].mean().item()
            conf_b  = conf_t[in_bin].mean().item()
            ece    += (in_bin.sum().item() / N) * abs(acc_b - conf_b)
        eces.append(ece)

    # ── Average confidence ──
    avg_confs = [np.mean(conf_buckets[i]['conf']) for i in range(n_stages)]

    return {
        'total_samples': total,
        'per_stage': [
            {
                'stage':      i + 1,
                'accuracy':   round(accs[i], 4),
                'ece':        round(eces[i], 4),
                'avg_conf':   round(float(avg_confs[i]), 4),
            }
            for i in range(n_stages)
        ]
    }


def evaluate(args):
    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu {args.gpu} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) visible "
                f"(0..{torch.cuda.device_count()-1})."
            )
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(f"Device: {device}"
          + (f"  ({torch.cuda.get_device_name(device)})" if device.type == 'cuda' else ''))

    # Load model
    model = build_model(args.dataset).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(best acc = {ckpt.get('best_acc', '?'):.2f}%)")

    # Data
    if args.dataset == 'road':
        if args.data_root is None:
            raise ValueError(
                "--data_root is required for --dataset road: point it at "
                "the manifest directory produced by build_road_manifest.py."
            )
        _, _, test_loader, _ = get_road_dataloaders(
            manifest_dir = args.data_root,
            batch_size   = 256,
            image_size   = 32,
        )
    else:
        _, test_loader = get_dataloaders(
            dataset    = args.dataset,
            data_root  = args.data_root,
            batch_size = 256,
        )

    # Run evaluation
    print("\nEvaluating ...")
    results = full_evaluate(model, test_loader, device)

    # Print results
    print(f"\nTotal test samples: {results['total_samples']:,}\n")
    print(f"{'Stage':<8} {'Accuracy':>10} {'ECE':>10} {'Avg Conf':>10}")
    print("-" * 42)
    for s in results['per_stage']:
        print(f"Stage {s['stage']:<3} {s['accuracy']:>9.2f}%  "
              f"{s['ece']:>9.4f}  {s['avg_conf']:>9.4f}")

    # Save
    out_path = os.path.join(
        os.path.dirname(args.checkpoint),
        f'eval_results_{args.dataset}.json'
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    type=str, required=True,
                   choices=['cifar10', 'imagenet', 'road'])
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_root',  type=str, default=None)
    p.add_argument('--gpu',        type=int, default=0,
                   help='CUDA device index to use, e.g. --gpu 6')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
