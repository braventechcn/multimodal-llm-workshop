#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text classification for 10 company types using TF-IDF + Linear SVM.

Dataset: CSV with two columns
 - column[0]: integer label in [0..9]
 - column[1]: text description (likely Chinese)

This script performs:
 - Robust CSV loading and basic cleaning
 - Stratified 80/20 train/hold-out split for a quick sanity check
 - K-fold cross-validation (stratified) for reliable validation
 - Accuracy metrics and optional model persistence

Notes:
 - For Chinese text, default tokenization uses jieba (word-level). Optionally
   character n-grams can be used, which sometimes perform better without a
   tokenizer.
 - Accuracy must be >= 0.80 to meet acceptance criteria.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import joblib
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


def _default_data_path() -> str:
	"""Return the default dataset path within this repository.

	Uses a relative path from the project root. Adjust if project structure changes.
	"""
	return os.path.abspath(
		os.path.join(
			os.path.dirname(__file__),
			"../../../..",  # up to repo root
			"core/data/module_01/NLP-text_classification/training.csv",
		)
	)


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
	"""Load the dataset from CSV and return (X_text, y_label).

	This function is tolerant of a header row; it will try to coerce the first
	column to integers and treat the second column as text.
	"""
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	# Try reading without header first for maximum compatibility
	try:
		df = pd.read_csv(csv_path, header=None, names=["label", "text"], encoding="utf-8")
	except Exception:
		# Fallback: let pandas infer header
		df = pd.read_csv(csv_path, encoding="utf-8")
		# Try to locate appropriate columns
		if df.shape[1] < 2:
			raise ValueError("Expected at least 2 columns in the CSV (label, text)")
		df = df.iloc[:, :2]
		df.columns = ["label", "text"]

	# Basic sanitation
	if "label" not in df or "text" not in df:
		raise ValueError("CSV must contain 'label' and 'text' columns (first two columns)")

	# Coerce labels to integers, drop rows that cannot be coerced
	def _safe_int(x):
		try:
			return int(x)
		except Exception:
			return np.nan

	df["label"] = df["label"].apply(_safe_int)
	df = df.dropna(subset=["label", "text"]).copy()
	df["label"] = df["label"].astype(int)
	df["text"] = df["text"].astype(str).str.strip()

	# Filter out any empty texts
	df = df[df["text"].str.len() > 0]

	X = df["text"].to_numpy()
	y = df["label"].to_numpy()

	# Optional: sanity check for label range
	unique_labels = sorted(pd.unique(y))
	if not all(isinstance(v, (int, np.integer)) for v in unique_labels):
		raise ValueError("Labels must be integers (0..9)")
	if len(unique_labels) > 10:
		print(
			f"Warning: found {len(unique_labels)} unique labels, expected 10 (0..9)",
			file=sys.stderr,
		)

	return X, y


def build_pipeline(
	use_char_ngrams: bool = False,
	ngram_min: int = 1,
	ngram_max: int = 2,
	max_features: int | None = 30000,
	C: float = 1.0,
) -> Pipeline:
	"""Create a scikit-learn text classification pipeline.

	- If use_char_ngrams is True, character n-grams are used.
	- Otherwise, jieba tokenizer is used at word level.
	- Classifier: Linear Support Vector Machine (LinearSVC).
	"""
	if use_char_ngrams:
		vectorizer = TfidfVectorizer(
			analyzer="char",
			ngram_range=(ngram_min, ngram_max),
			max_features=max_features,
		)
	else:
		# Lazy import to avoid requiring jieba when using char-ngrams
		try:
			import jieba
		except Exception as e:
			raise ImportError(
				"jieba is required for word-level tokenization. Install or use --use-char-ngrams."
			) from e

		def jieba_tokenize(text: str):
			# Accurate mode segmentation
			return list(jieba.lcut(text))

		vectorizer = TfidfVectorizer(
			tokenizer=jieba_tokenize,  # custom tokenizer for Chinese
			token_pattern=None,  # must be None when using a custom tokenizer
			ngram_range=(ngram_min, ngram_max),
			max_features=max_features,
		)

	clf = LinearSVC(C=C)
	pipe = Pipeline([
		("tfidf", vectorizer),
		("clf", clf),
	])
	return pipe


def evaluate_with_holdout(
	pipe: Pipeline, X: np.ndarray, y: np.ndarray, test_size: float, random_state: int
) -> Tuple[float, str, np.ndarray]:
	"""Train on a stratified hold-out split and return accuracy, report, and confusion matrix."""
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		stratify=y,
		random_state=random_state,
	)
	pipe.fit(X_train, y_train)
	y_pred = pipe.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, digits=4)
	cm = confusion_matrix(y_test, y_pred)
	return acc, report, cm


def evaluate_with_cv(
	pipe: Pipeline, X: np.ndarray, y: np.ndarray, cv: int, random_state: int
) -> Tuple[float, float, np.ndarray]:
	"""Run stratified K-fold cross validation and return mean, std, and per-fold scores."""
	skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
	scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
	return float(scores.mean()), float(scores.std()), scores


def ensure_dir(path: str) -> None:
	"""Create directory if it doesn't exist."""
	os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train and validate a 10-class text classifier")
	parser.add_argument("--data-path", type=str, default=_default_data_path(), help="Path to training CSV")
	parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test size fraction (default: 0.2)")
	parser.add_argument("--cv", type=int, default=5, help="Number of CV folds (default: 5)")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting and CV")
	parser.add_argument("--use-char-ngrams", action="store_true", help="Use character n-grams instead of jieba")
	parser.add_argument("--ngram-min", type=int, default=1, help="Min n in n-gram range (default: 1)")
	parser.add_argument("--ngram-max", type=int, default=2, help="Max n in n-gram range (default: 2)")
	parser.add_argument("--max-features", type=int, default=30000, help="Max features for TF-IDF (default: 30000)")
	parser.add_argument("--C", type=float, default=1.0, help="LinearSVC regularization parameter C (default: 1.0)")
	parser.add_argument(
		"--save-model",
		action="store_true",
		help="Persist trained pipeline (vectorizer + classifier) after hold-out training",
	)
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

	# Build pipeline
	pipe = build_pipeline(
		use_char_ngrams=args.use_char_ngrams,
		ngram_min=args.ngram_min,
		ngram_max=args.ngram_max,
		max_features=args.max_features,
		C=args.C,
	)

	# Hold-out evaluation (80/20 by default)
	print("\n=== Hold-out evaluation (train/validation split) ===")
	acc_holdout, report, cm = evaluate_with_holdout(
		pipe, X, y, test_size=args.test_size, random_state=args.random_state
	)
	print(f"Hold-out accuracy: {acc_holdout:.4f}")
	print("Classification report (hold-out):\n", report)
	print("Confusion matrix (hold-out):\n", cm)

	# Cross-validation evaluation
	print("\n=== Cross-validation evaluation ===")
	pipe_cv = build_pipeline(
		use_char_ngrams=args.use_char_ngrams,
		ngram_min=args.ngram_min,
		ngram_max=args.ngram_max,
		max_features=args.max_features,
		C=args.C,
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
		model_path = os.path.join(args.output_dir, "text_classifier_tfidf_svm.joblib")
		# Refit on full data for final model
		pipe_final = build_pipeline(
			use_char_ngrams=args.use_char_ngrams,
			ngram_min=args.ngram_min,
			ngram_max=args.ngram_max,
			max_features=args.max_features,
			C=args.C,
		)
		pipe_final.fit(X, y)
		joblib.dump(pipe_final, model_path)
		print(f"Model saved to: {model_path}")

	# Exit code: 0 if pass, 1 otherwise (useful for CI)
	sys.exit(0 if meets_threshold else 1)


if __name__ == "__main__":
	main()

