"""
dataloader.py
-------------
PyTorch DataLoader setup for CIFAR-10 and ImageNet.

Usage:
    from data.dataloader import get_dataloaders
    train_loader, test_loader = get_dataloaders('cifar10', batch_size=128)
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# ─────────────────────────────────────────────
#  CIFAR-10
# ─────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def _cifar10_transforms():
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, test_transform


def get_cifar10_loaders(data_root='./data/cifar10', batch_size=128,
                        num_workers=4):
    train_tf, test_tf = _cifar10_transforms()

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f"[CIFAR-10] Train: {len(train_set):,}  Test: {len(test_set):,}")
    return train_loader, test_loader


# ─────────────────────────────────────────────
#  ImageNet
# ─────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _imagenet_transforms():
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, test_transform


def get_imagenet_loaders(data_root='./data/imagenet', batch_size=64,
                         num_workers=8):
    """
    NOTE: ImageNet must be downloaded manually.
    Expected directory structure:
        data_root/
            train/
                n01234567/   (synset folders)
                ...
            val/
                n01234567/
                ...
    """
    train_tf, test_tf = _imagenet_transforms()

    train_set = torchvision.datasets.ImageNet(
        root=data_root, split='train', transform=train_tf)
    val_set   = torchvision.datasets.ImageNet(
        root=data_root, split='val',   transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(
        val_set,   batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f"[ImageNet] Train: {len(train_set):,}  Val: {len(val_set):,}")
    return train_loader, val_loader


# ─────────────────────────────────────────────
#  Unified factory
# ─────────────────────────────────────────────

def get_dataloaders(dataset: str, data_root: str = None,
                    batch_size: int = None, num_workers: int = 4):
    """
    Args:
        dataset     : 'cifar10' or 'imagenet'
        data_root   : path to dataset root (auto-set if None)
        batch_size  : default 128 for cifar10, 64 for imagenet
        num_workers : DataLoader worker count
    Returns:
        (train_loader, test_loader)
    """
    if dataset == 'cifar10':
        root = data_root or './data/cifar10'
        bs   = batch_size or 128
        return get_cifar10_loaders(root, bs, num_workers)
    elif dataset == 'imagenet':
        root = data_root or './data/imagenet'
        bs   = batch_size or 64
        return get_imagenet_loaders(root, bs, num_workers)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")
