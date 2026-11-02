# RTDeepIoT — Training Phase

Implementation of the Multi-Exit ResNet from:
> "Scheduling Real-time Deep Learning Services as Imprecise Computations" (arXiv:2011.01112)

## Project Structure

```
rtdeepiot/
├── models/
│   └── multi_exit_resnet.py   # Single ResNet with 3 stage classifier heads
├── training/
│   ├── loss.py                # Multi-exit joint loss (weighted cross-entropy)
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Per-stage accuracy + ECE evaluation
│   └── profile_wcet.py        # GPU stage latency profiling (for scheduler)
├── data/
│   └── dataloader.py          # CIFAR-10 and ImageNet data loaders
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Step-by-Step Usage

### Step 1 — Train on CIFAR-10
```bash
python training/train.py --dataset cifar10 --epochs 200
# Checkpoint saved to: ./checkpoints/best_cifar10.pth
```

### Step 2 — Train on ImageNet (requires manual download)
```bash
# Download ImageNet from https://image-net.org and place at ./data/imagenet/
python training/train.py --dataset imagenet --epochs 90 \
    --data_root ./data/imagenet
```

### Step 3 — Evaluate the trained model
```bash
python training/evaluate.py --dataset cifar10 \
    --checkpoint ./checkpoints/best_cifar10.pth
```

### Step 4 — Profile WCET (needed by the scheduler)
```bash
python training/profile_wcet.py --dataset cifar10 \
    --checkpoint ./checkpoints/best_cifar10.pth
# Output: ./checkpoints/wcet_cifar10.json
```

## Expected Results (CIFAR-10)

| Stage | Accuracy | ECE    |
|-------|----------|--------|
| 1     | ~75%     | <0.05  |
| 2     | ~85%     | <0.05  |
| 3     | ~92%     | <0.05  |

## Key Design Decisions

- **One network, 3 exits** — not 3 separate models
- **Joint loss** = 0.3×CE(stage1) + 0.3×CE(stage2) + 0.4×CE(stage3)
- **Label smoothing = 0.1** improves confidence calibration (ECE)
- **Stage 1 = mandatory**, Stages 2 & 3 = optional (scheduler decides)
- **Confidence** = max softmax probability at each stage exit
