"""
road_crop_dataset.py
---------------------
PyTorch Dataset that crops ROAD bounding boxes on-the-fly from the
manifest CSVs produced by build_road_manifest.py.

Usage:
    from data.road_crop_dataset import get_road_dataloaders
    train_loader, val_loader, test_loader, class_weights = \
        get_road_dataloaders('./data/road_manifests', batch_size=128)
"""

import csv
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


CLASS_NAMES = ['Ped', 'Car', 'Cyc', 'LarVeh', 'Bus', 'Mobike']

# Reuse CIFAR-style normalization stats as a reasonable default;
# these are close enough to natural-image statistics for a first pass.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

# Padding added around each bounding box before cropping, as a
# fraction of box width/height -- gives the classifier a little
# context around tight boxes rather than only the exact annotation.
BOX_PADDING_FRAC = 0.10


class RoadCropDataset(Dataset):
    def __init__(self, manifest_path, image_size=32, train=False):
        self.rows = []
        with open(manifest_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append({
                    'image_path': row['image_path'],
                    'x1': int(row['x1']), 'y1': int(row['y1']),
                    'x2': int(row['x2']), 'y2': int(row['y2']),
                    'label_idx': int(row['label_idx']),
                })

        self.image_size = image_size
        self.train = train

        if train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(NORM_MEAN, NORM_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(NORM_MEAN, NORM_STD),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        try:
            img = Image.open(row['image_path']).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(
                f"Could not open image: {row['image_path']}. "
                f"Check that --img_root was set correctly when building "
                f"the manifest."
            ) from e

        W, H = img.size
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']

        # Add padding around the box, clamped to image bounds
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * BOX_PADDING_FRAC), int(bh * BOX_PADDING_FRAC)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x)
        y2 = min(H, y2 + pad_y)

        crop = img.crop((x1, y1, x2, y2))
        crop = self.transform(crop)

        label = row['label_idx']
        return crop, label


def load_class_weights(manifest_dir):
    """Loads class_weights.txt written by build_road_manifest.py."""
    weights_path = os.path.join(manifest_dir, 'class_weights.txt')
    weights = [1.0] * len(CLASS_NAMES)
    name_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    with open(weights_path, 'r') as f:
        for line in f:
            name, w = line.strip().split(',')
            weights[name_to_idx[name]] = float(w)

    return torch.tensor(weights, dtype=torch.float32)


def get_road_dataloaders(manifest_dir, batch_size=128, image_size=32,
                         num_workers=4):
    """
    Args:
        manifest_dir : directory containing train/val/test_manifest.csv
                       and class_weights.txt (output of
                       build_road_manifest.py)
        batch_size   : batch size
        image_size   : crop resize target (32 for ResNet-32, matching
                       your CIFAR-10 setup)
        num_workers  : DataLoader worker count

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    train_set = RoadCropDataset(
        os.path.join(manifest_dir, 'train_manifest.csv'),
        image_size=image_size, train=True)
    val_set = RoadCropDataset(
        os.path.join(manifest_dir, 'val_manifest.csv'),
        image_size=image_size, train=False)
    test_set = RoadCropDataset(
        os.path.join(manifest_dir, 'test_manifest.csv'),
        image_size=image_size, train=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    class_weights = load_class_weights(manifest_dir)

    print(f"[ROAD] Train: {len(train_set):,}  Val: {len(val_set):,}  "
          f"Test: {len(test_set):,}")
    print(f"[ROAD] Class weights: "
          f"{dict(zip(CLASS_NAMES, class_weights.tolist()))}")

    return train_loader, val_loader, test_loader, class_weights
