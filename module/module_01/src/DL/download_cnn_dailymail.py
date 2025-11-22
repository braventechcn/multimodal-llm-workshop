"""
Download and export the CNN/DailyMail summarization dataset to a CSV file.

Information:
    - Supports gzip compression automatically if filename ends with .gz.
    - CLI arguments (output file, splits, config version, optional row limits).
    - Resilient: handles KeyboardInterrupt gracefully, flushes partial results.
    - Writes a unified header across splits: id, article, highlights, split.
    - Saves by default into repository path: core/datas/module_01/cnn_dailymail.

Usage example:
    python download_cnn_dailymail.py \
        --out-file core/datas/module_01/cnn_dailymail/cnn_dailymail_full.csv.gz \
        --splits train validation test \
        --config 3.0.0

Optional limit (only export first N(`--row-limit`) rows):
    python download_cnn_dailymail.py --row-limit 1000

Environment proxy (if needed for HF downloads):
    export HTTP_PROXY=http://127.0.0.1:7890
    export HTTPS_PROXY=http://127.0.0.1:7890
or
    export HF_ENDPOINT=https://hf-mirror.com
"""

from pathlib import Path
import csv
from datasets import load_dataset
from tqdm import tqdm
import gzip
import argparse
from typing import Iterable, List, Optional, Tuple
import sys

# ======== 默认配置（可被CLI覆盖）========
DEFAULT_SPLITS: List[str] = ["train", "validation", "test"]
DEFAULT_CONFIG = "3.0.0"
# 默认输出目录（相对仓库根目录）
DEFAULT_OUT_DIR = Path("core/datas/module_01/cnn_dailymail")
DEFAULT_OUT_FILE = DEFAULT_OUT_DIR / "cnn_dailymail_full.csv.gz"
# ======================================

def open_outfile(path: Path) -> Tuple[Iterable, csv.writer]:
    """Return file handle + csv writer; auto gzip if suffix is .gz.

    如果文件名以 .gz 结尾则使用 gzip 文本写入；否则普通 csv。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        f = gzip.open(path, "wt", newline="", encoding="utf-8")
    else:
        f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    return f, writer


def write_split(
    writer: csv.writer,
    split: str,
    config_name: str,
    row_limit: Optional[int] = None,
) -> int:
    """Load one split and write rows to CSV; returns number of written rows.

    Parameters
    ----------
    writer : csv.writer
        CSV writer already positioned after header.
    split : str
        Dataset split to load (train / validation / test).
    config_name : str
        Version/config (e.g. '3.0.0').
    row_limit : Optional[int]
        If provided, only write the first N rows of this split.
    """
    print(f"\n[STEP] 加载 split = {split} ...")
    ds = load_dataset("cnn_dailymail", config_name, split=split)
    n = len(ds)
    print(f"[INFO] {split}: {n} 条")

    rows_written = 0
    # 迭代写入（带进度条）。row_limit 控制早停。
    for ex in tqdm(ds, total=min(n, row_limit) if row_limit else n, desc=f"Writing {split}", unit="row"):
        _id = ex.get("id", "")
        article = ex.get("article", "")
        highlights = ex.get("highlights", "")
        writer.writerow([_id, article, highlights, split])
        rows_written += 1
        if row_limit is not None and rows_written >= row_limit:
            break
    return rows_written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for flexible usage."""
    parser = argparse.ArgumentParser(
        description="Download CNN/DailyMail dataset and export unified CSV (optionally gzip)."
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=DEFAULT_OUT_FILE,
        help="Output CSV(.gz) path; parent directories auto-created."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Which splits to export (default: train validation test)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Dataset config/version (default: 3.0.0)."
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional max rows per split (for quick sampling)."
    )
    return parser.parse_args(argv)

def main(
    out_file: Path,
    splits: List[str],
    config_name: str,
    row_limit: Optional[int] = None,
) -> None:
    """Main orchestration: open output, iterate splits, and write rows."""
    print(f"[INFO] 目标文件 (output): {out_file}")
    print(f"[INFO] 数据配置 (dataset config): cnn_dailymail / {config_name} / {splits}")
    if row_limit:
        print(f"[INFO] 行数限制 (per-split row limit): {row_limit}")

    fout, writer = open_outfile(out_file)
    header = ["id", "article", "highlights", "split"]
    writer.writerow(header)

    total_rows = 0
    try:
        for split in splits:
            written = write_split(writer, split, config_name, row_limit=row_limit)
            total_rows += written
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断 (Interrupted by user)，已写入的内容保留。")
    finally:
        fout.flush()
        fout.close()

    print(f"\n[DONE] 写入完成 (finished): {out_file}")
    print(f"[STATS] 总行数 (total rows): {total_rows}（含所选 splits 合并）")

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(
        out_file=args.out_file,
        splits=args.splits,
        config_name=args.config,
        row_limit=args.row_limit,
    )



