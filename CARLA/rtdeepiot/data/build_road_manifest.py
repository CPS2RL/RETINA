"""
build_road_manifest.py
-----------------------
Parses road_annotations.csv into train/val/test manifest files for
RTDeepIoT's single-frame agent-type classifier.

Split strategy (frame-count mode, the default):
  Videos are sorted chronologically (their names are timestamps, e.g.
  "2014-06-25-16-45-34_stereo_centre_02", so alphabetical sort =
  chronological order).
    - TEST videos  = last 2 videos
    - VAL videos   = 3rd-and-4th-from-last videos
    - TRAIN videos = all remaining (earlier) videos
  From each pool, a fixed number of FRAMES (not object-boxes) is
  randomly sampled -- all object boxes within a sampled frame are
  kept together. Default: 1000 train frames, 200 val frames,
  200 test frames.

What it does:
  1. Drops non-agent rows (OthTL, TL, blank agent labels)
  2. Maps ROAD's 7 agent classes down to 6 CARLA classes:
       Ped, Car, Cyc, LarVeh, Bus, Mobike, EmVeh -> Car
  3. Splits by VIDEO first (test = last 2, val = 3rd/4th-last), then
     samples a fixed frame count within each split
  4. Writes 3 manifest CSVs with columns:
       image_path, x1, y1, x2, y2, label, label_idx
     Crops are NOT materialized to disk -- the dataloader crops
     on-the-fly at __getitem__ time from these coordinates.

Run (frame-count mode, default):
    python data/build_road_manifest.py \
        --csv /path/to/road_annotations.csv \
        --img_root /path/to/road/rgb-images \
        --out_dir ./data/road_manifests \
        --train_frames 1000 --val_frames 200 --test_frames 200

Run (fraction mode, full-dataset, for later scale-up):
    python data/build_road_manifest.py \
        --csv /path/to/road_annotations.csv \
        --img_root /path/to/road/rgb-images \
        --out_dir ./data/road_manifests \
        --split_mode fraction --val_frac 0.1 --test_frac 0.1
"""

import argparse
import csv
import os
import random
from collections import defaultdict


# ─────────────────────────────────────────────
#  Label mapping: ROAD's 7 agent classes -> 6 CARLA classes
# ─────────────────────────────────────────────

LABEL_MAP = {
    'Ped':    'Ped',
    'Car':    'Car',
    'Cyc':    'Cyc',
    'LarVeh': 'LarVeh',
    'Bus':    'Bus',
    'Mobike': 'Mobike',
    'EmVeh':  'Car',       # assumption: emergency vehicles are mostly
                            # car-sized (police cars, ambulances)
}

# Rows with these agent values (or blank) are dropped -- not real agents
DROP_AGENTS = {'OthTL', 'TL', ''}

CLASS_NAMES = ['Ped', 'Car', 'Cyc', 'LarVeh', 'Bus', 'Mobike']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ─────────────────────────────────────────────
#  Parsing
# ─────────────────────────────────────────────

def parse_annotations(csv_path):
    """
    Returns:
        rows_by_frame : dict[(video, frame)] -> list of row dicts
                         {image_path, x1, y1, x2, y2, label}
        videos_sorted : chronologically sorted list of unique video names
        class_counts  : dict[label] -> count (post-mapping, post-filter,
                         over the FULL dataset -- informational only)
    """
    rows_by_frame = defaultdict(list)
    class_counts  = defaultdict(int)
    videos_seen   = set()

    dropped_nonagent = 0
    dropped_badbox    = 0
    total_rows        = 0

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            agent = row['agent'].strip()
            video = row['video']
            videos_seen.add(video)

            if agent in DROP_AGENTS or agent not in LABEL_MAP:
                dropped_nonagent += 1
                continue

            label = LABEL_MAP[agent]

            try:
                x1, y1, x2, y2 = (int(float(row['x1'])), int(float(row['y1'])),
                                  int(float(row['x2'])), int(float(row['y2'])))
            except (ValueError, KeyError):
                dropped_badbox += 1
                continue

            if x2 <= x1 or y2 <= y1:
                dropped_badbox += 1
                continue

            rgb_file   = row['rgb_file']
            frame_num  = row['frame']
            image_path = os.path.join(video, rgb_file)
            key        = (video, frame_num)

            rows_by_frame[key].append({
                'image_path': image_path,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'label': label,
            })
            class_counts[label] += 1

    videos_sorted = sorted(videos_seen)  # alphabetical = chronological

    print(f"Total CSV rows:          {total_rows:,}")
    print(f"Dropped (non-agent):     {dropped_nonagent:,}")
    print(f"Dropped (bad box):       {dropped_badbox:,}")
    print(f"Kept (valid agent rows): {sum(class_counts.values()):,}")
    print(f"Unique videos:           {len(videos_sorted)}")
    print(f"Unique (video,frame):    {len(rows_by_frame):,}")
    print("\nFull-dataset class distribution (before any sampling):")
    for c in CLASS_NAMES:
        print(f"  {c:8s}: {class_counts.get(c, 0):>8,}")

    return rows_by_frame, videos_sorted, class_counts


# ─────────────────────────────────────────────
#  Split strategy 1 (default): fixed frame counts,
#  test = last 2 videos, val = 3rd/4th-from-last videos
# ─────────────────────────────────────────────

def split_by_frame_count(rows_by_frame, videos_sorted,
                         train_frames, val_frames, test_frames, seed=42):
    if len(videos_sorted) < 5:
        raise ValueError(
            f"Need at least 5 unique videos for this split strategy "
            f"(2 test + 2 val + >=1 train), found {len(videos_sorted)}."
        )

    test_videos  = set(videos_sorted[-2:])
    val_videos   = set(videos_sorted[-4:-2])
    train_videos = set(videos_sorted[:-4])

    print(f"\nTEST videos  (last 2):        {sorted(test_videos)}")
    print(f"VAL videos   (3rd/4th-last):  {sorted(val_videos)}")
    print(f"TRAIN videos (all earlier):   {len(train_videos)} videos")

    def frames_in(video_set):
        return [k for k in rows_by_frame.keys() if k[0] in video_set]

    train_pool = frames_in(train_videos)
    val_pool   = frames_in(val_videos)
    test_pool  = frames_in(test_videos)

    rng = random.Random(seed)

    def sample_frames(pool, n_requested, split_name):
        if len(pool) < n_requested:
            print(f"  WARNING: {split_name} pool has only {len(pool)} "
                  f"frames, fewer than requested {n_requested}. "
                  f"Using all available.")
            return pool
        return rng.sample(pool, n_requested)

    train_frame_keys = sample_frames(train_pool, train_frames, 'TRAIN')
    val_frame_keys   = sample_frames(val_pool,   val_frames,   'VAL')
    test_frame_keys  = sample_frames(test_pool,  test_frames,  'TEST')

    def collect(frame_keys):
        out = []
        for k in frame_keys:
            out.extend(rows_by_frame[k])
        return out

    train_rows = collect(train_frame_keys)
    val_rows   = collect(val_frame_keys)
    test_rows  = collect(test_frame_keys)

    print(f"\nSampled frames -> train: {len(train_frame_keys)}  "
          f"val: {len(val_frame_keys)}  test: {len(test_frame_keys)}")
    print(f"Resulting object-box rows -> train: {len(train_rows):,}  "
          f"val: {len(val_rows):,}  test: {len(test_rows):,}")

    return train_rows, val_rows, test_rows


# ─────────────────────────────────────────────
#  Split strategy 2 (optional): fraction-based, full dataset,
#  video-level split -- kept for later scale-up
# ─────────────────────────────────────────────

def split_by_fraction(rows_by_frame, videos_sorted, val_frac, test_frac, seed=42):
    rng = random.Random(seed)
    videos = list(videos_sorted)
    rng.shuffle(videos)

    n = len(videos)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_videos = set(videos[:n_train])
    val_videos   = set(videos[n_train:n_train + n_val])
    test_videos  = set(videos[n_train + n_val:])

    print(f"\nVideo split -> train: {len(train_videos)}  "
          f"val: {len(val_videos)}  test: {len(test_videos)}")

    def collect(video_set):
        out = []
        for k, rows in rows_by_frame.items():
            if k[0] in video_set:
                out.extend(rows)
        return out

    return collect(train_videos), collect(val_videos), collect(test_videos)


# ─────────────────────────────────────────────
#  Writing manifests
# ─────────────────────────────────────────────

def write_manifest(rows, img_root, out_path):
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'x1', 'y1', 'x2', 'y2', 'label', 'label_idx'])
        for r in rows:
            full_path = os.path.join(img_root, r['image_path'])
            writer.writerow([
                full_path, r['x1'], r['y1'], r['x2'], r['y2'],
                r['label'], CLASS_TO_IDX[r['label']],
            ])
    print(f"  Wrote {len(rows):,} rows -> {out_path}")


def print_split_class_balance(name, rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r['label']] += 1
    print(f"\n{name} class distribution:")
    for c in CLASS_NAMES:
        print(f"  {c:8s}: {counts.get(c, 0):>8,}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main(args):
    print(f"Parsing {args.csv} ...\n")
    rows_by_frame, videos_sorted, class_counts = parse_annotations(args.csv)

    if args.split_mode == 'frames':
        train_rows, val_rows, test_rows = split_by_frame_count(
            rows_by_frame, videos_sorted,
            args.train_frames, args.val_frames, args.test_frames,
            seed=args.seed)
    else:
        train_rows, val_rows, test_rows = split_by_fraction(
            rows_by_frame, videos_sorted,
            args.val_frac, args.test_frac, seed=args.seed)

    print_split_class_balance('TRAIN', train_rows)
    print_split_class_balance('VAL',   val_rows)
    print_split_class_balance('TEST',  test_rows)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nWriting manifests to {args.out_dir} ...")
    write_manifest(train_rows, args.img_root, os.path.join(args.out_dir, 'train_manifest.csv'))
    write_manifest(val_rows,   args.img_root, os.path.join(args.out_dir, 'val_manifest.csv'))
    write_manifest(test_rows,  args.img_root, os.path.join(args.out_dir, 'test_manifest.csv'))

    # Class weights computed from the TRAIN split actually used
    # (not the full dataset), since that's what the loss function sees.
    train_counts = defaultdict(int)
    for r in train_rows:
        train_counts[r['label']] += 1

    total = sum(train_counts.get(c, 0) for c in CLASS_NAMES)
    weights = []
    for c in CLASS_NAMES:
        cnt = train_counts.get(c, 1)  # avoid div-by-zero for absent classes
        weights.append(total / (len(CLASS_NAMES) * max(cnt, 1)))

    weights_path = os.path.join(args.out_dir, 'class_weights.txt')
    with open(weights_path, 'w') as f:
        for c, w in zip(CLASS_NAMES, weights):
            f.write(f"{c},{w:.6f}\n")
    print(f"\nClass weights (inverse frequency, computed on TRAIN split) "
          f"saved -> {weights_path}")
    for c, w in zip(CLASS_NAMES, weights):
        n = train_counts.get(c, 0)
        flag = "  <-- ZERO SAMPLES, check train pool!" if n == 0 else ""
        print(f"  {c:8s}: weight={w:.3f}  (n={n}){flag}")

    print("\nDone.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',       type=str, required=True,
                   help='Path to road_annotations.csv')
    p.add_argument('--img_root',  type=str, required=True,
                   help='Path to road/rgb-images directory')
    p.add_argument('--out_dir',   type=str, default='./data/road_manifests',
                   help='Where to write train/val/test manifest CSVs')
    p.add_argument('--split_mode', type=str, default='frames',
                   choices=['frames', 'fraction'],
                   help="'frames': fixed frame counts, test=last 2 videos, "
                        "val=3rd/4th-last videos (default, for quick "
                        "prototyping). 'fraction': full-dataset fractional "
                        "video split (for later scale-up).")
    # frame-count mode args
    p.add_argument('--train_frames', type=int, default=10000)
    p.add_argument('--val_frames',   type=int, default=2000)
    p.add_argument('--test_frames',  type=int, default=2000)
    # fraction mode args
    p.add_argument('--val_frac',  type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--seed',      type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
