# Commands — Run in This Order

## 0. Unzip and enter the project
```bash
unzip rtdeepiot_training.zip
cd rtdeepiot
```

## 1. (Recommended) Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Train on CIFAR-10 (auto-downloads dataset, ~200 epochs)
```bash
python training/train.py --dataset cifar10 --epochs 200 --batch_size 128 --lr 0.1 --gpu 6
```
Output: `checkpoints/best_cifar10.pth`

## 4. Evaluate the trained CIFAR-10 model
```bash
python training/evaluate.py --dataset cifar10 --checkpoint checkpoints/best_cifar10.pth --gpu 6
```
Output: `checkpoints/eval_results_cifar10.json`

## 5. Profile stage latency (WCET) for the scheduler
```bash
python training/profile_wcet.py --dataset cifar10 --checkpoint checkpoints/best_cifar10.pth --n_runs 10000 --gpu 6
```
Output: `checkpoints/wcet_cifar10.json`

> **Using a specific GPU:** all three scripts accept `--gpu N` to select
> a CUDA device index (e.g. `--gpu 6` targets `cuda:6`). This requires
> that device to actually be visible to PyTorch — see the note below if
> `--gpu 6` errors out.
>
> Alternative: instead of `--gpu 6`, you can restrict visibility at the
> OS level and always use index 0 inside the script:
> ```bash
> CUDA_VISIBLE_DEVICES=6 python training/train.py --dataset cifar10 --epochs 200 --gpu 0
> ```
> Check available devices first with:
> ```bash
> nvidia-smi          # lists all physical GPUs and their indices
> python -c "import torch; print(torch.cuda.device_count())"
> ```

---

## Optional: ImageNet (only if you need it — much slower, needs manual download)

### 6. Download ImageNet manually first
Get it from https://image-net.org and arrange as:
```
data/imagenet/train/<synset folders>/...
data/imagenet/val/<synset folders>/...
```

### 7. Train on ImageNet (~90 epochs, needs a strong GPU, can take days)
```bash
python training/train.py --dataset imagenet --epochs 90 --data_root ./data/imagenet --batch_size 64 --lr 0.1
```
Output: `checkpoints/best_imagenet.pth`

### 8. Evaluate ImageNet model
```bash
python training/evaluate.py --dataset imagenet --checkpoint checkpoints/best_imagenet.pth --data_root ./data/imagenet
```

### 9. Profile ImageNet WCET
```bash
python training/profile_wcet.py --dataset imagenet --checkpoint checkpoints/best_imagenet.pth --n_runs 10000
```

---

## Quick sanity checks (optional, run anytime, no GPU wait)

Test that the model builds correctly:
```bash
python models/multi_exit_resnet.py
```

Test that the loss function works:
```bash
python training/loss.py
```
