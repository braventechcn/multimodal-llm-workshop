#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

####################################################################################
# BERT Embedding Transformer
####################################################################################

class BertEncoder(BaseEstimator, TransformerMixin):
	"""Sklearn-compatible transformer that converts texts to BERT CLS embeddings.

	Implementation notes:
	- Loads model/tokenizer lazily on first transform to keep import overhead minimal.
	- Uses CLS hidden state (can be swapped to mean pooling if desired).
	- Batch processes input for reasonable speed.
	- Returns a 2D numpy array (n_samples, hidden_size).
	"""

	def __init__(
		self,
		model_name: str = "bert-base-chinese",
		max_length: int = 128,
		batch_size: int = 16,
		device: Optional[str] = None,
		use_mean_pool: bool = False,
		verbose: bool = False,
	) -> None:
		self.model_name = model_name
		self.max_length = max_length
		self.batch_size = batch_size
		self.device = device
		self.use_mean_pool = use_mean_pool
		self.verbose = verbose
		self._tokenizer = None
		self._model = None

	def _lazy_init(self):
		if self._model is not None:
			return
		try:
			from transformers import AutoTokenizer, AutoModel  # type: ignore
			import torch  # type: ignore
		except Exception as e:
			raise ImportError(
				"BERT mode requires 'transformers' and 'torch' packages. Install them via pip."
			) from e
		if self.device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		if self.verbose:
			print(f"Initializing BERT model '{self.model_name}' on device={self.device}")
		self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		self._model = AutoModel.from_pretrained(self.model_name)
		self._model.to(self.device)
		self._model.eval()

	def fit(self, X: List[str], y: Optional[List[int]] = None):  # noqa: D401
		# No fitting needed; embeddings are from a frozen pretrained model.
		return self

	def transform(self, X: List[str]) -> np.ndarray:
		"""
		Convert raw texts to embedding matrix.

		Accepts list/tuple/numpy array of texts. Ensures each element is str before
		calling the tokenizer. This avoids transformers complaining about numpy array
		input types.
		"""
		# 1) 惰性初始化：仅在第一次调用时加载 tokenizer 和 model，避免未使用时的启动开销
		self._lazy_init()
		import torch  # type: ignore

		# Normalize input to a python list of strings
		# 2) 兼容三种常见集合类型：np.ndarray / list / tuple，统一转为 list
		if isinstance(X, np.ndarray):
			X_list = X.tolist()
		elif isinstance(X, (list, tuple)):
			X_list = list(X)
		else:
			raise TypeError("BertEncoder.transform expects a list/tuple/numpy.ndarray of texts")

		# Coerce all items to str (robustness) and strip
		# 3) 将所有元素安全转换为字符串并去除首尾空白；过滤 None 和空字符串，保证后续 tokenizer 输入有效
		X_list = [str(t).strip() for t in X_list if t is not None and str(t).strip() != ""]
		if len(X_list) == 0:
			# 返回形状 (0, hidden_size) 的空矩阵，遵守 sklearn transformer 输出约定
			return np.empty((0, self._model.config.hidden_size), dtype=np.float32)  # type: ignore

		embeddings: List[np.ndarray] = []
		# 4) 小批次遍历：降低显存峰值，提高吞吐；batch_size 可在初始化时调整
		for i in range(0, len(X_list), self.batch_size):
			batch_texts = X_list[i : i + self.batch_size]
			enc = self._tokenizer(
				batch_texts,
				padding=True,
				truncation=True,
				max_length=self.max_length,
				return_tensors="pt",
			)
			# 5) 将所有输入张量迁移到目标设备（CPU/GPU）；避免跨设备计算报错
			enc = {k: v.to(self.device) for k, v in enc.items()}
			with torch.no_grad():
				# 6) 关闭梯度，减少显存与计算开销；仅做前向推理
				outputs = self._model(**enc)
				last_hidden = outputs.last_hidden_state
				if self.use_mean_pool:
					# 7) 均值池化：利用 attention_mask 过滤 padding，再按有效 token 求平均
					mask = enc["attention_mask"].unsqueeze(-1)
					summed = (last_hidden * mask).sum(dim=1)
					counts = mask.sum(dim=1).clamp(min=1)
					pooled = summed / counts
				else:
					# 8) CLS 池化：直接取序列第一个位置向量，常用于句级表示
					pooled = last_hidden[:, 0]
			# 9) 转回 CPU，转 numpy 并统一为 float32 减少内存占用
			embeddings.append(pooled.cpu().numpy().astype(np.float32))

		# 10) vstack 合并所有 batch 的向量，得到 (n_samples, hidden_size) 输出供下游分类器使用
		return np.vstack(embeddings)



def _default_data_path() -> str:
	"""
	Return the default dataset path within this repository.
	Uses a relative path from the project root. Adjust if project structure changes.
	"""
	return os.path.abspath(
		os.path.join(
			os.path.dirname(__file__),
			"../../../..",  # up to repo root
			"core/datas/module_01/NLP-text_classification/training.csv",
		)
	)


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
	"""	
	Load dataset from a CSV or JSONL file and return (X_text, y_label).

	Auto-detects format by file extension:
	  - .csv  : expects two columns (label, text) header optional.
	  - .jsonl: each line is a JSON object with keys 'label' and 'text'.
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Dataset file not found: {path}")

	ext = os.path.splitext(path)[1].lower() # get file extension

	if ext == ".jsonl":
		data = []
		with open(path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					data.append(json.loads(line))
				except json.JSONDecodeError as e:
					raise ValueError(f"Invalid JSONL line: {e}") from e
		df = pd.DataFrame(data)
		if "label" not in df or "text" not in df:
			raise ValueError("JSONL must contain 'label' and 'text' fields")
	else:  # treat as CSV by default
		try:
			df = pd.read_csv(path, header=None, names=["label", "text"], encoding="utf-8")
		except Exception:
			df = pd.read_csv(path, encoding="utf-8")
			if df.shape[1] < 2:
				raise ValueError("Expected at least 2 columns in the CSV (label, text)")
			df = df.iloc[:, :2]
			df.columns = ["label", "text"]
		if "label" not in df or "text" not in df:
			raise ValueError("CSV must contain 'label' and 'text' columns (first two columns)")

	def _safe_int(x):
		try:
			return int(x)
		except Exception:
			return np.nan

	df["label"] = df["label"].apply(_safe_int)
	df = df.dropna(subset=["label", "text"]).copy()
	df["label"] = df["label"].astype(int)
	df["text"] = df["text"].astype(str).str.strip()
	df = df[df["text"].str.len() > 0]

	X = df["text"].to_numpy()
	y = df["label"].to_numpy()

	return X, y


def build_pipeline(
	use_char_ngrams: bool = False,
	ngram_min: int = 1,
	ngram_max: int = 2,
	max_features: int | None = 30000,
	use_bert: bool = False,
	bert_model_name: str = "bert-base-chinese",
	bert_max_length: int = 128,
	bert_batch_size: int = 16,
	bert_mean_pool: bool = False,
	bert_verbose: bool = False,
	classifier: str = "svm",
	C: float = 1.0,  # SVM C
	logreg_C: float = 1.0,
	logreg_max_iter: int = 1000,
	logreg_solver: str = "lbfgs",
) -> Pipeline:
	"""Create a scikit-learn text classification pipeline.

	Modes:
	  - TF-IDF (word via jieba or character n-gram) + classifier
	  - BERT embeddings (CLS or mean pooled) + classifier

	Args:
		use_char_ngrams: If True and not using BERT, use char n-gram TF-IDF.
		ngram_min/ngram_max: n-gram range for TF-IDF (ignored in BERT mode).
		max_features: Cap vocabulary size for TF-IDF.
		C: Regularization strength for LinearSVC (if classifier='svm').
		classifier: 'svm' or 'logreg'.
		use_bert: If True, use BERT embeddings instead of TF-IDF.
		bert_model_name: Pretrained model name.
		bert_max_length: Max sequence length for tokenization.
		bert_batch_size: Batch size during embedding extraction.
		bert_mean_pool: Use mean pooling over tokens instead of CLS.
		bert_verbose: Print initialization info.
	"""

	clf: BaseEstimator
	if classifier == "svm":
		clf = LinearSVC(C=C)
	elif classifier == "logreg":
		# LogisticRegression for multi-class; expose key hyperparameters.
		clf = LogisticRegression(
			C=logreg_C,
			max_iter=logreg_max_iter,
			solver=logreg_solver,
			multinomial="auto", #  'multi_class' was deprecated in version 1.5 and will be removed in 1.8.
		)
	else:
		raise ValueError("classifier must be one of {'svm','logreg'}")

	if use_bert:
		embed = BertEncoder(
			model_name=bert_model_name,
			max_length=bert_max_length,
			batch_size=bert_batch_size,
			use_mean_pool=bert_mean_pool,
			verbose=bert_verbose,
		)
		return Pipeline([
			("embed", embed),
			("clf", clf),
		])

	# TF-IDF path
	if use_char_ngrams:
		vectorizer = TfidfVectorizer(
			analyzer="char",
			ngram_range=(ngram_min, ngram_max),
			max_features=max_features,
		)
	else:
		try:
			import jieba  # type: ignore
			try:
				jieba.setLogLevel(logging.ERROR)
			except Exception:
				pass
		except Exception as e:
			raise ImportError(
				"jieba is required for word-level tokenization. Install or use --use-char-ngrams or --use-bert."
			) from e

		def jieba_tokenize(text: str):
			return list(jieba.lcut(text))

		vectorizer = TfidfVectorizer(
			tokenizer=jieba_tokenize,
			token_pattern=None,
			ngram_range=(ngram_min, ngram_max),
			max_features=max_features,
		)

	return Pipeline([
		("tfidf", vectorizer),
		("clf", clf),
	])


# Hold-out evaluation
def evaluate_with_holdout(
	pipe: Pipeline, 
	X: np.ndarray, 
	y: np.ndarray, 
	test_size: float, 
	random_state: int
) -> Tuple[float, str, np.ndarray]:
	"""
	Train on a stratified hold-out split and return accuracy, report, and confusion matrix.
	"""
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		stratify=y, # using stratified split by label to maintain balanced classes distribution
		random_state=random_state,
	)
	pipe.fit(X_train, y_train)
	y_pred = pipe.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, digits=4)
	cm = confusion_matrix(y_test, y_pred)
	return acc, report, cm


# Cross-validation evaluation
def evaluate_with_cv(
	pipe: Pipeline, 
	X: np.ndarray, 
	y: np.ndarray, 
	cv: int, 
	random_state: int
) -> Tuple[float, float, np.ndarray]:
	"""
 	Run stratified K-fold cross validation and return mean, std, and per-fold scores.
	"""
	skf = StratifiedKFold(
			n_splits=cv, 
			shuffle=True, 
			random_state=random_state
		)
	scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
	return float(scores.mean()), float(scores.std()), scores


def ensure_dir(path: str) -> None:
	"""Create directory if it doesn't exist."""
	os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments and return namespace."""
	parser = argparse.ArgumentParser(description="Train and validate a 10-class text classifier")
	# Data & split
	parser.add_argument("--data-path", type=str, default=_default_data_path(), help="Path to training CSV")
	parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test size fraction (default: 0.2)")
	parser.add_argument("--cv", type=int, default=5, help="Number of CV folds (default: 5)")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting and CV")
	# Feature extraction (TF-IDF)
	parser.add_argument("--use-char-ngrams", action="store_true", help="Use character n-grams instead of jieba (TF-IDF mode)")
	parser.add_argument("--ngram-min", type=int, default=1, help="Min n in n-gram range (default: 1)")
	parser.add_argument("--ngram-max", type=int, default=2, help="Max n in n-gram range (default: 2)")
	parser.add_argument("--max-features", type=int, default=30000, help="Max features for TF-IDF (default: 30000)")
	# BERT feature extraction
	parser.add_argument("--use-bert", action="store_true", help="Use BERT embeddings instead of TF-IDF")
	parser.add_argument("--bert-model-name", type=str, default="bert-base-chinese", help="HuggingFace model name (default: bert-base-chinese)")
	parser.add_argument("--bert-max-length", type=int, default=128, help="Max sequence length for BERT (default: 128)")
	parser.add_argument("--bert-batch-size", type=int, default=16, help="Batch size for BERT embedding extraction (default: 16)")
	parser.add_argument("--bert-mean-pool", action="store_true", help="Use mean pooling (default CLS token)")
	# Classifier selection
	parser.add_argument("--classifier", type=str, choices=["svm", "logreg"], default="svm", help="Classifier type: svm | logreg (default: svm)")
	parser.add_argument("--C", type=float, default=1.0, help="LinearSVC regularization parameter C (ignored if logreg)")
	parser.add_argument("--logreg-C", type=float, default=1.0, help="LogisticRegression inverse regularization strength C (only if classifier=logreg)")
	parser.add_argument("--logreg-max-iter", type=int, default=1000, help="LogisticRegression max iterations (only if classifier=logreg)")
	parser.add_argument("--logreg-solver", type=str, default="lbfgs", choices=["lbfgs","liblinear","saga","newton-cg"], help="LogisticRegression solver (only if classifier=logreg)")
	# Persistence
	parser.add_argument("--save-model", action="store_true", help="Persist trained pipeline after training")
	parser.add_argument(
		"--output-dir",
		type=str,
		default=os.path.abspath(
			os.path.join(
				os.path.dirname(__file__),
				"../../outputs/checkpoints",
			)
		),
		help="Directory to save the trained model",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	print("Loading dataset ...")
	X, y = load_dataset(args.data_path)
	print(f"Loaded {len(X)} samples across {len(set(y))} unique labels.")

	# Typical two-stage process of machine learning:
	# - Feature extraction + classifier
	# Build pipeline
	# Two options: 'TF-IDF + classifier' OR 'BERT embeddings + classifier'
	pipe = build_pipeline(
		use_char_ngrams=args.use_char_ngrams,
		ngram_min=args.ngram_min,
		ngram_max=args.ngram_max,
		max_features=args.max_features,
		use_bert=args.use_bert,
		bert_model_name=args.bert_model_name,
		bert_max_length=args.bert_max_length,
		bert_batch_size=args.bert_batch_size,
		bert_mean_pool=args.bert_mean_pool,
		classifier=args.classifier,
		C=args.C,
		logreg_C=args.logreg_C,
		logreg_max_iter=args.logreg_max_iter,
		logreg_solver=args.logreg_solver,
	)

	mode_desc = (
		"BERT embeddings" if args.use_bert else ("Char n-gram TF-IDF" if args.use_char_ngrams else "Jieba word TF-IDF")
	)
	print(f"Feature mode: {mode_desc}; Classifier: {args.classifier}")

	# Hold-out evaluation (80/20 by default)
	print("\n=== Hold-out evaluation (train/validation split) ===")
	acc_holdout, report, cm = evaluate_with_holdout(
		pipe, X, y, test_size=args.test_size, random_state=args.random_state
	)
	print(f"Hold-out accuracy: {acc_holdout:.4f}") # Accuracy = correct samples / total samples
	print("Classification report (hold-out):\n", report) # Precision, Recall, F1-score per class
	print("Confusion matrix (hold-out):\n", cm) # Rows: true labels, Columns: predicted labels

	# Cross-validation evaluation   
	print("\n=== Cross-validation evaluation ===")
	pipe_cv = build_pipeline(
		use_char_ngrams=args.use_char_ngrams,
		ngram_min=args.ngram_min,
		ngram_max=args.ngram_max,
		max_features=args.max_features,
		C=args.C,
		classifier=args.classifier,
		use_bert=args.use_bert,
		bert_model_name=args.bert_model_name,
		bert_max_length=args.bert_max_length,
		bert_batch_size=args.bert_batch_size,
		bert_mean_pool=args.bert_mean_pool,
		logreg_C=args.logreg_C,
		logreg_max_iter=args.logreg_max_iter,
		logreg_solver=args.logreg_solver,
	)
	mean_acc, std_acc, scores = evaluate_with_cv(
		pipe_cv, X, y, cv=args.cv, random_state=args.random_state
	)
	print(f"CV accuracy (mean ± std over {args.cv} folds): {mean_acc:.4f} ± {std_acc:.4f}")
	print("Per-fold scores:", np.round(scores, 4))

	meets_threshold = mean_acc >= 0.80
	status = "PASS" if meets_threshold else "FAIL"
	print(f"\nAcceptance check (>= 0.80 mean CV accuracy): {status}")

	# Optionally persist the trained hold-out model
	if args.save_model:
		ensure_dir(args.output_dir)
		# Refit on full data for final model
		pipe_final = build_pipeline(
			use_char_ngrams=args.use_char_ngrams,
			ngram_min=args.ngram_min,
			ngram_max=args.ngram_max,
			max_features=args.max_features,
			C=args.C,
			classifier=args.classifier,
			use_bert=args.use_bert,
			bert_model_name=args.bert_model_name,
			bert_max_length=args.bert_max_length,
			bert_batch_size=args.bert_batch_size,
			bert_mean_pool=args.bert_mean_pool,
			logreg_C=args.logreg_C,
			logreg_max_iter=args.logreg_max_iter,
			logreg_solver=args.logreg_solver,
		)
		pipe_final.fit(X, y)
		# Adjust filename suffix based on mode
		mode_tag = "bert" if args.use_bert else ("char_tfidf" if args.use_char_ngrams else "jieba_tfidf")
		clf_tag = args.classifier
		base_name = f"text_classifier_{mode_tag}_{clf_tag}.joblib"
		model_path = os.path.join(args.output_dir, base_name)
		joblib.dump(pipe_final, model_path)
		print(f"Model saved to: {model_path}")

	# Exit code: 0 if pass, 1 otherwise (useful for CI)
	sys.exit(0 if meets_threshold else 1)


if __name__ == "__main__":
	main()