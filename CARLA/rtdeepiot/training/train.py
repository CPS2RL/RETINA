"""
train.py
--------
Main training script for the Multi-Exit ResNet.

What this script does
---------------------
1. Builds the multi-exit ResNet (CIFAR-10 or ImageNet)
2. Loads training and test data
3. Trains the full network end-to-end with joint multi-exit loss
4. Saves the best checkpoint (based on Stage-3 accuracy)
5. Logs per-stage accuracy + loss at every epoch

Run:
    # CIFAR-10
    python training/train.py --dataset cifar10 --epochs 200

    # ImageNet
    python training/train.py --dataset imagenet --epochs 90 \
        --data_root /path/to/imagenet
"""

import argparse
import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multi_exit_resnet import build_model
from training.loss            import MultiExitLoss
from data.dataloader          import get_dataloaders
from data.road_crop_dataset   import get_road_dataloaders


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def accuracy(predicted, targets):
    """Top-1 accuracy as a percentage."""
    return (predicted == targets).float().mean().item() * 100.0


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    stage_correct = [0, 0, 0]
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward through ALL 3 stages
        stage_outputs = model(images)

        # Compute weighted joint loss
        loss, per_stage_losses = criterion(stage_outputs, targets)

        loss.backward()
        # Gradient clipping to stabilise early training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate stats
        B = images.size(0)
        total_samples += B
        total_loss    += loss.item() * B

        for i, out in enumerate(stage_outputs):
            stage_correct[i] += (out['predicted'] == targets).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            s1_acc = 100.0 * stage_correct[0] / total_samples
            s3_acc = 100.0 * stage_correct[2] / total_samples
            print(f"  Epoch {epoch}  [{batch_idx+1}/{len(loader)}]  "
                  f"loss={total_loss/total_samples:.4f}  "
                  f"s1_acc={s1_acc:.1f}%  s3_acc={s3_acc:.1f}%")

    avg_loss = total_loss / total_samples
    stage_accs = [100.0 * c / total_samples for c in stage_correct]
    return avg_loss, stage_accs


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0.0
    stage_correct = [0, 0, 0]
    total_samples = 0

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        stage_outputs = model(images)
        loss, _ = criterion(stage_outputs, targets)

        B = images.size(0)
        total_samples += B
        total_loss    += loss.item() * B

        for i, out in enumerate(stage_outputs):
            stage_correct[i] += (out['predicted'] == targets).sum().item()

    avg_loss   = total_loss / total_samples
    stage_accs = [100.0 * c / total_samples for c in stage_correct]
    return avg_loss, stage_accs


# ─────────────────────────────────────────────
#  Confidence calibration evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_confidence_calibration(model, loader, device, num_bins=10):
    """
    Computes Expected Calibration Error (ECE) for each stage.
    ECE measures how well the confidence score matches actual accuracy.
    Lower is better. Well-calibrated: ECE < 0.05
    """
    model.eval()
    # Collect confidence and correctness per stage
    stage_data = [{'conf': [], 'correct': []} for _ in range(3)]

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        stage_outputs = model(images)
        for i, out in enumerate(stage_outputs):
            correct = (out['predicted'] == targets).cpu()
            conf    = out['confidence'].cpu()
            stage_data[i]['conf'].extend(conf.tolist())
            stage_data[i]['correct'].extend(correct.tolist())

    ece_per_stage = []
    for i, data in enumerate(stage_data):
        conf_t    = torch.tensor(data['conf'])
        correct_t = torch.tensor(data['correct'], dtype=torch.float)

        # Bin predictions by confidence
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        N   = len(conf_t)

        for b in range(num_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
            in_bin = (conf_t >= lo) & (conf_t < hi)
            if in_bin.sum() == 0:
                continue
            acc_bin  = correct_t[in_bin].mean().item()
            conf_bin = conf_t[in_bin].mean().item()
            ece     += (in_bin.sum().item() / N) * abs(acc_bin - conf_bin)

        ece_per_stage.append(ece)
        print(f"  Stage {i+1} ECE = {ece:.4f}")

    return ece_per_stage


# ─────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────

def train(args):
    # ── Device ──
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
    print(f"Using device: {device}"
          + (f"  ({torch.cuda.get_device_name(device)})" if device.type == 'cuda' else ''))

    # ── Data ──
    class_weights = None
    if args.dataset == 'road':
        # ROAD uses a manifest-based on-the-fly-crop loader, not
        # torchvision's dataset download flow. --data_root here should
        # point at the manifest directory produced by
        # data/build_road_manifest.py (contains train/val/test_manifest.csv
        # and class_weights.txt).
        if args.data_root is None:
            raise ValueError(
                "--data_root is required for --dataset road: point it at "
                "the manifest directory produced by build_road_manifest.py "
                "(e.g. ./data/road_manifests)."
            )
        train_loader, val_loader, test_loader, class_weights = get_road_dataloaders(
            manifest_dir = args.data_root,
            batch_size   = args.batch_size or 128,
            image_size   = 32,
            num_workers  = args.num_workers,
        )
        # train.py's evaluate() loop below uses "test_loader" as the
        # held-out split it reports/checkpoints against each epoch;
        # for ROAD we use the VAL split for that (standard practice --
        # the TEST split stays untouched until final evaluate.py run).
        test_loader = val_loader
        class_weights = class_weights.to(device)
    else:
        train_loader, test_loader = get_dataloaders(
            dataset     = args.dataset,
            data_root   = args.data_root,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
        )

    # ── Model ──
    model = build_model(args.dataset).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: MultiExitResNet ({args.dataset})  "
          f"trainable params: {total_params:,}")

    # ── Loss ──
    # Stage weights [0.3, 0.3, 0.4]: early exits slightly penalised,
    # final exit weighted most. Label smoothing = 0.1 aids calibration.
    # class_weights (ROAD only): inverse-frequency weighting to counter
    # the severe Ped/Mobike imbalance (~60x) in the ROAD dataset.
    criterion = MultiExitLoss(
        stage_weights   = [0.3, 0.3, 0.4],
        label_smoothing = 0.1,
        class_weights   = class_weights,
    ).to(device)

    # ── Optimiser ──
    # SGD with momentum matches the original ResNet paper.
    optimizer = optim.SGD(
        model.parameters(),
        lr           = args.lr,
        momentum     = 0.9,
        weight_decay = 5e-4,
        nesterov     = True,
    )

    # ── LR Scheduler ──
    # Cosine annealing smoothly reduces LR to near-zero by final epoch.
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Checkpoint directory ──
    os.makedirs(args.save_dir, exist_ok=True)

    best_acc   = 0.0
    best_epoch = 0
    log_rows   = []

    print(f"\nStarting training for {args.epochs} epochs ...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_accs = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)

        # Evaluate
        test_loss, test_accs = evaluate(
            model, test_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch:03d}/{args.epochs}  "
              f"lr={lr_now:.2e}  time={elapsed:.1f}s")
        print(f"  TRAIN  loss={train_loss:.4f}  "
              f"s1={train_accs[0]:.2f}%  s2={train_accs[1]:.2f}%  "
              f"s3={train_accs[2]:.2f}%")
        print(f"  TEST   loss={test_loss:.4f}  "
              f"s1={test_accs[0]:.2f}%  s2={test_accs[1]:.2f}%  "
              f"s3={test_accs[2]:.2f}%")

        # Log row
        log_rows.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_s1': train_accs[0], 'train_s2': train_accs[1], 'train_s3': train_accs[2],
            'test_loss':  test_loss,
            'test_s1':  test_accs[0],  'test_s2':  test_accs[1],  'test_s3':  test_accs[2],
        })

        # Save best checkpoint (based on stage-3 test accuracy)
        if test_accs[2] > best_acc:
            best_acc   = test_accs[2]
            best_epoch = epoch
            ckpt_path  = os.path.join(args.save_dir, f'best_{args.dataset}.pth')
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_acc':    best_acc,
                'args':        vars(args),
            }, ckpt_path)
            print(f"  [*] Best checkpoint saved -> {ckpt_path}  "
                  f"(Stage-3 acc = {best_acc:.2f}%)")

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            latest_path = os.path.join(args.save_dir, f'latest_{args.dataset}.pth')
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_acc':    best_acc,
                'args':        vars(args),
            }, latest_path)

    # ── Final calibration check ──
    print("\n=== Confidence Calibration (ECE) on Test Set ===")
    # Reload best checkpoint for calibration check
    ckpt = torch.load(os.path.join(args.save_dir, f'best_{args.dataset}.pth'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state'])
    evaluate_confidence_calibration(model, test_loader, device)

    # ── Save training log ──
    import csv
    log_path = os.path.join(args.save_dir, f'training_log_{args.dataset}.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved -> {log_path}")

    print(f"\nBest Stage-3 Test Accuracy: {best_acc:.2f}%  (epoch {best_epoch})")
    print("Training complete.")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train Multi-Exit ResNet')
    p.add_argument('--dataset',     type=str, default='cifar10',
                   choices=['cifar10', 'imagenet', 'road'],
                   help='Dataset to train on')
    p.add_argument('--data_root',   type=str, default=None,
                   help="Path to dataset root directory. For --dataset "
                        "road, this must be the manifest directory "
                        "produced by build_road_manifest.py "
                        "(e.g. ./data/road_manifests).")
    p.add_argument('--epochs',      type=int, default=200,
                   help='Number of training epochs (200 for CIFAR, 90 for ImageNet)')
    p.add_argument('--batch_size',  type=int, default=None,
                   help='Batch size (default: 128 CIFAR / 64 ImageNet)')
    p.add_argument('--lr',          type=float, default=0.1,
                   help='Initial learning rate')
    p.add_argument('--num_workers', type=int, default=4,
                   help='DataLoader worker count')
    p.add_argument('--save_dir',    type=str, default='./checkpoints',
                   help='Directory to save checkpoints and logs')
    p.add_argument('--gpu',         type=int, default=0,
                   help='CUDA device index to use, e.g. --gpu 6')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
