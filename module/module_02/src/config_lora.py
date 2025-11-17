from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Dict, List

def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
    return cur.anchor  # 文件系统根

def get_config(argv: List[str] | None = None) -> Dict:
    """
    生成训练所需CONFIG（优先级：命令行 > 环境变量 > 合理默认）。
    不暴露绝对路径：默认使用相对项目根的目录；本地模型不存在则回退到HF模型ID。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-id", type=str, help="HF模型ID，例如 Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-model-path", type=str, help="本地模型目录（可相对）")
    parser.add_argument("--data-dir", type=str, help="数据目录(含train.jsonl,test.jsonl)")
    parser.add_argument("--output-dir", type=str, help="输出目录(保存LoRA权重)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--labels", type=str, help="逗号分隔标签，如: a,b,c")
    args, _ = parser.parse_known_args(argv)

    project_root = _find_project_root(Path(__file__).parent)

    # 模型源：优先显式路径，其次本地 models/<model_id>，否则用HF模型ID
    model_id = args.model_id or os.getenv("BASE_MODEL_ID", "Qwen2.5-1.5B-Instruct")
    if args.base_model_path:
        base_model_source = args.base_model_path
    elif os.getenv("BASE_MODEL_PATH"):
        base_model_source = os.getenv("BASE_MODEL_PATH")
    else:
        local_dir = project_root / "models" / model_id
        base_model_source = str(local_dir if local_dir.exists() else model_id)

    # 数据目录
    data_dir = Path(
        args.data_dir
        or os.getenv("DATA_DIR", str(project_root / "module" / "module_02" / "bank_intent_data"))
    )
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"

    # 输出目录
    output_dir = Path(
        args.output_dir
        or os.getenv("LORA_OUTPUT_DIR", str(project_root / "outputs" / "bank_lora"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 标签
    labels_env = args.labels or os.getenv("LABELS", "fraud_risk,refund,balance,card_lost,other")
    labels = [s.strip() for s in labels_env.split(",") if s.strip()]

    cfg = {
        "base_model_path": base_model_source,
        "train_data_path": str(train_path),
        "test_data_path": str(test_path),
        "lora_save_path": str(output_dir),
        "labels": labels,
        "max_seq_len": args.max_seq_len or int(os.getenv("MAX_SEQ_LEN", 64)),
        "batch_size": args.batch_size or int(os.getenv("BATCH_SIZE", 1)),
        "epochs": args.epochs or int(os.getenv("EPOCHS", 1)),
        "lr": args.lr or float(os.getenv("LR", 2e-5)),
    }
    return cfg