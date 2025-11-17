"""Unified LoRA fine-tuning configuration for Qwen2.5 1.5B.

This module exposes a single helper `get_config` that returns a dictionary of
all training / evaluation hyperparameters and important paths. Each parameter
is documented below so the main script can stay clean. Paths are resolved
relative to the repository root so the training script can be executed from
ANY current working directory, e.g.:

	python /absolute/path/to/module/module_02/src/qwen2.5_1.5b_lora.py
or (while inside the repo root):
	python module/module_02/src/qwen2.5_1.5b_lora.py

Overrides: you can override any key by passing keyword arguments to
`get_config(...)` or by setting environment variables with prefix `LORA_`.
Environment variable names are UPPERCASE of the key, e.g. `export LORA_EPOCHS=3`.

Returned CONFIG keys (with explanations):
	base_model_path: Local path or HF model ID for the base pretrained model.
	train_data_path: Path to training dataset (.jsonl) used for gradient updates.
	val_data_path:   Path to validation dataset for metric monitoring & checkpoint selection.
	test_data_path:  Path to held-out test dataset used ONLY for final evaluation.
	lora_save_path:  Directory to store LoRA adapter checkpoints and evaluation artifacts.
	labels:          Ordered list of class label names for sequence classification.
	max_seq_len:     Maximum sequence length after tokenization (truncates longer texts).
	train_batch_size:Per-device batch size for training steps.
	eval_batch_size: Per-device batch size for validation evaluation during training.
	test_batch_size: Per-device batch size for final test evaluation.
	batch_size:      Generic unified batch size (legacy convenience field; kept for compatibility).
	epochs:          Total number of training epochs.
	gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step.
	lr:              Base learning rate for the AdamW optimizer (applied to LoRA + head parameters).
	seed:            Random seed for reproducibility (PyTorch, NumPy, etc.).

Design choices:
 - Paths are constructed from the repository root (auto-detected) to avoid hard-coded absolute paths.
 - `get_config` applies precedence: kwargs > environment variables > defaults.
 - A companion `CONFIG_DESCRIPTIONS` dictionary is exported for external inspection / documentation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

# Map each config key to a human-readable description (could be consumed by CLI or docs generator)
CONFIG_DESCRIPTIONS: Dict[str, str] = {
	"base_model_path": "Local path or model ID of base pretrained Qwen model.",
	"train_data_path": "Training dataset path (.jsonl).",
	"val_data_path": "Validation dataset path for checkpoint selection (never used for final test metrics).",
	"test_data_path": "Held-out test dataset path, used only after training finishes.",
	"lora_save_path": "Directory root to store LoRA checkpoints, metrics, and evaluation outputs.",
	"labels": "Ordered list of intent / class labels for sequence classification head.",
	"max_seq_len": "Maximum tokenized sequence length (longer inputs truncated).",
	"train_batch_size": "Per-device batch size for training steps.",
	"eval_batch_size": "Per-device batch size for validation evaluation during training.",
	"test_batch_size": "Per-device batch size for test evaluation.",
	"batch_size": "Legacy unified batch size field; kept for compatibility with existing code paths.",
	"epochs": "Total number of training epochs.",
	"gradient_accumulation_steps": "Steps to accumulate gradients before performing an optimizer step.",
	"lr": "Learning rate for AdamW optimizer applied to trainable parameters (LoRA + classification head).",
	"seed": "Global random seed for reproducibility (PyTorch / NumPy / Transformers).",
}


def _detect_repo_root() -> Path:
	"""Detect repository root assuming this file sits at: <root>/module/module_02/src/config_lora.py.

	We climb three parents from current file path:
		config_lora.py -> src -> module_02 -> module -> root
	If a `.git` directory is found higher up, we prefer that as a safety net.
	"""
	here = Path(__file__).resolve()
	# Expected root via structural assumption
	structural_root = here.parents[3]
	for parent in here.parents:
		if (parent / ".git").exists():  # Use git root if available
			return parent
	return structural_root


def _env_override(key: str) -> Any:
	"""Fetch environment variable override for a config key using prefix LORA_."""
	env_key = f"LORA_{key.upper()}"
	if env_key in os.environ:
		return os.environ[env_key]
	return None


def get_config(**overrides: Any) -> Dict[str, Any]:
	"""Return configuration dictionary with dynamic path resolution and override support.

	Precedence per key: explicit kwargs overrides > environment variable (LORA_<KEY>) > default.
	Type casting for simple numeric fields is performed if overrides are strings.
	"""
	root = _detect_repo_root()

	defaults: Dict[str, Any] = {
		"base_model_path": str(root / "core/models/Qwen2.5-1.5B-Instruct"),
		"train_data_path": str(root / "core/datas/module_02/bank_intent_data/train.jsonl"),
		"val_data_path": str(root / "core/datas/module_02/bank_intent_data/val.jsonl"),
		"test_data_path": str(root / "core/datas/module_02/bank_intent_data/test.jsonl"),
		"lora_save_path": str(root / "module/module_02/outputs/bank_lora_model"),
		"labels": ["fraud_risk", "refund", "balance", "card_lost", "other"],
		"max_seq_len": 64,
		"train_batch_size": 8,
		"eval_batch_size": 8,
		"test_batch_size": 8,
		"batch_size": 8,  # compatibility field
		"epochs": 1,
		"gradient_accumulation_steps": 1,
		"lr": 2e-5,
		"seed": 42,
	}

	cfg: Dict[str, Any] = {}
	for key, default_val in defaults.items():
		# Environment variable override
		env_val = _env_override(key)
		value = default_val if env_val is None else env_val
		# Kwarg override has highest precedence
		if key in overrides:
			value = overrides[key]
		# Basic casting for numeric fields if env provided as string
		if isinstance(default_val, (int, float)) and isinstance(value, str):
			try:
				# Keep float if original default is float, else int
				value = float(value) if isinstance(default_val, float) else int(value)
			except ValueError:
				pass  # leave as string if casting fails
		cfg[key] = value

	# Ensure save directory exists (do not create dataset/model directories silently)
	os.makedirs(cfg["lora_save_path"], exist_ok=True)
	return cfg


__all__ = ["get_config", "CONFIG_DESCRIPTIONS"]
