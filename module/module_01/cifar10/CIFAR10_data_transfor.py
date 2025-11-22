"""CIFAR-10 batch-to-image export utility.

This script converts the original CIFAR-10 python batch files into per-class
PNG images and produces a `manifest.csv` describing each exported image.

Key Features:
 1. Reads CIFAR-10 batches (data_batch_1..5 + test_batch).
 2. Reconstructs images from flattened arrays and saves them by class name.
 3. Generates a CSV manifest containing: filepath, label (name), label_id, split.
 4. Optional per-class export cap for quick sampling / debugging.

Optimizations / Improvements Added:
    - Parameterized `main` to allow programmatic use.
    - CLI via argparse instead of hard-coded paths.
    - Safer file operations and early validations.
    - Slight loop simplification (zipping data & labels).

Usage Example:
    python cifar_data_transfor.py \
            --data-dir /path/to/cifar-10-batches-py \
            --out-dir ./cifar10_images \
            --limit-per-class 500

If `--limit-per-class` is omitted, all images are exported.

Note: CIFAR-10 official python batch format contains keys (b"data", b"labels").
            Each image flattened: 3072 = 32*32*3 stored as R(0..1023) G(1024..2047) B(2048..3071).
"""

import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys
import numpy as np
from PIL import Image


# Default Config (only used if not provided via CLI) 
DEFAULT_DATA_DIR = "./datas/cifar-10-batches-py"   # Default input directory (please adjust as needed)
DEFAULT_OUTPUT_DIR = "./datas/cifar10_images"      # Default output directory
DEFAULT_LIMIT = None  # None => export all

def unpickle(file: str) -> Dict:
    """
    Load CIFAR-10 batch pickle file.

    The CIFAR python version stores keys as bytes; we keep them as-is and
    decode label names later.
    """
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")

def ensure_dirs(class_names: List[str], out_root: Path) -> None:
    """
    Create per-class subdirectories under the output root if missing.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    for c in class_names:
        (out_root / c).mkdir(parents=True, exist_ok=True)

def export_split(batch_file: str) -> str:
    """
    Infer split name (train/test) from the batch filename.
    """
    return "test" if "test_batch" in batch_file else "train"

def save_image(img_flat: np.ndarray, save_path: Path) -> None:
    """Reconstruct and save a single CIFAR-10 image.

    CIFAR-10 layout: 3072-length array: 1024 R, 1024 G, 1024 B.
    We reshape to (3, 32, 32) then transpose to (32, 32, 3).
    """
    img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
    Image.fromarray(img).save(save_path)

def main(data_dir: Path, out_dir: Path, num_per_class: Optional[int] = None) -> None:
    """
    Export CIFAR-10 batches to per-class PNG images and a manifest CSV.

    Args:
        data_dir : Path
            Path to CIFAR-10 python batches (contains data_batch_* and test_batch).
        out_dir : Path
            Destination root directory for exported images.
        num_per_class : Optional[int]
            If provided, maximum number of images to export per class.
    """
    # Validate input directory and meta file
    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {data_dir}")

    meta_path = data_dir / "batches.meta" # the meta file with class names
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing CIFAR meta file: {meta_path}")

    # Read class label names (decode from meta bytes)
    meta = unpickle(str(meta_path))
    class_names = [c.decode("utf-8") for c in meta[b"label_names"]]

    # Prepare output directory tree
    ensure_dirs(class_names, out_dir)

    # Batch file names as per CIFAR-10 python format
    batch_files: List[str] = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]

    # Counter for number of exported images per class
    counters = {c: 0 for c in class_names}

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        # Write CSV header: filepath, label, label_id, split
        writer.writerow(["filepath", "label", "label_id", "split"])

        for bf in batch_files:
            bf_path = data_dir / bf
            if not bf_path.exists():
                print(f"[WARN] Skip missing batch: {bf_path}")
                continue

            print(f"[INFO] Processing {bf_path.name} ...")
            batch_dict = unpickle(str(bf_path))
            data = batch_dict[b"data"]      # shape: (N, 3072)
            labels = batch_dict[b"labels"]  # list[int]
            split = export_split(bf) # Determine split from filename, e.g., "train" or "test"

            # Iterate over images & labels together for clarity
            for img_flat, label_id in zip(data, labels):
                cls_name = class_names[label_id]

                # Respect per-class limit if set
                if num_per_class is not None and counters[cls_name] >= num_per_class:
                    continue

                filename = f"{cls_name}_{counters[cls_name]:05d}.png"
                save_path = out_dir / cls_name / filename

                save_image(img_flat, save_path)
                counters[cls_name] += 1
                writer.writerow([str(save_path), cls_name, label_id, split])

    print("\n=== Export Finished ===")
    for c in class_names:
        print(f"{c:>10s}: {counters[c]} images -> {out_dir / c}")
    print(f"Manifest File (manifest.csv): {manifest_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export CIFAR-10 python batches to per-class PNG images and a manifest CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="Path to CIFAR-10 python batches directory (contains data_batch_1..5, test_batch)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory to store exported images and manifest.csv."
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=DEFAULT_LIMIT,
        help="Optional maximum number of images to export per class (default: all)."
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.data_dir, args.out_dir, args.limit_per_class)

