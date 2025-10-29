#!/usr/bin/env python3
"""
Standard Linear Regression pipeline on structured data.

This script demonstrates common ML components end-to-end:
- Config and reproducibility
- Data loading and target detection
- Train/validation/pred split handling
- Feature preprocessing for numeric/categorical columns
- Model training (LinearRegression) and evaluation (MAE/RMSE/R2)
- Cross-validation (KFold)
- Model persistence (save/load)
- Batch prediction and CSV export

Data file: core/data/module_01/ecommerce_sales_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "core/data/module_01/ecommerce_sales_dataset.csv"

RANDOM_STATE = 42

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "module/module_01/outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_PATH = CKPT_DIR / "linear_regression.joblib"
PRED_PATH = OUTPUT_DIR / "predictions_linear_regression.csv"

# Evaluation settings
CV_FOLDS = 5
# Split ratios (train = 1 - VAL_SIZE - TEST_SIZE)
VAL_SIZE = 0.2
TEST_SIZE = 0.1


# =========================
# Utilities
# =========================
def make_ohe() -> OneHotEncoder:
    """Return OneHotEncoder with version-compatible args.

    Newer sklearn uses `sparse_output`, older versions use `sparse`.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def detect_target_column(df: pd.DataFrame) -> str:
    """Pick target column: prefer `y`, otherwise use the last column."""
    return "y" if "y" in df.columns else str(df.columns[-1])


def split_datasets(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into labeled_df (for training/validation) and pred_df (for prediction).

    Rules:
    - If there are missing targets, those rows go to prediction set.
    - Otherwise, use the last 500 rows as prediction set.
    Returns (labeled_df, pred_df, original_df)
    """
    # Preserve original index for traceability
    df = df.reset_index(drop=False).rename(columns={"index": "row_index"})

    is_labeled = ~df[target_col].isna()
    labeled_df = df[is_labeled].copy()
    pred_df = df[~is_labeled].copy()

    # Note: No longer forcing last 500 as prediction set; we use proper train/val/test splits

    return labeled_df, pred_df, df


def split_train_val_test(
    labeled_df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split labeled data into train/val/test by ratios.

    - val_size and test_size are fractions in (0, 1)
    - train fraction = 1 - val_size - test_size
    """
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("val_size and test_size must be in (0,1) and sum to < 1")

    # First split out the test set
    rest_df, test_df = train_test_split(
        labeled_df, test_size=test_size, shuffle=shuffle, random_state=random_state
    )

    # Then split the remaining into train/val, preserving the global val ratio
    effective_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        rest_df, test_size=effective_val, shuffle=shuffle, random_state=random_state
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    drop_cols = {target_col, "row_index", "id", "ID", "Id"}
    return [c for c in df.columns if c not in drop_cols]


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    """Preprocessing + LinearRegression pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    reg = LinearRegression()
    return Pipeline(steps=[("prep", preprocessor), ("reg", reg)])


def evaluate_holdout(model: Pipeline, X_train, y_train, X_val, y_val) -> dict:
    """Fit on train and report metrics on validation set."""
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    mse = mean_squared_error(y_val, pred)
    rmse = float(np.sqrt(mse))
    return {
        "MAE": float(mean_absolute_error(y_val, pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_val, pred)),
    }


def evaluate_cv(model: Pipeline, X, y, folds: int = CV_FOLDS) -> float:
    """Return mean RMSE over K-Fold CV (lower is better)."""
    # Prefer built-in neg_root_mean_squared_error; fallback to neg_mean_squared_error
    use_rmse_direct = True
    try:
        scoring = "neg_root_mean_squared_error"
        _ = cross_val_score(model, X, y, cv=2, scoring=scoring)
    except Exception:
        scoring = "neg_mean_squared_error"
        use_rmse_direct = False

    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    if use_rmse_direct:
        return float(-scores.mean())
    else:
        # scores are negative MSE -> convert to RMSE
        return float(np.sqrt(-scores.mean()))


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)


def load_model(path: Path) -> Pipeline:
    return load(path)


def run_train_eval_predict(data_path: Path = DATA_PATH) -> None:
    # 1) Load data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)

    # 2) Detect target
    target_col = detect_target_column(df)

    # 3) Split labeled/pred sets
    labeled_df, pred_df, df_all = split_datasets(df, target_col)

    # 4) Train/Val/Test split by ratios
    train_df, val_df, test_df = split_train_val_test(
        labeled_df, val_size=VAL_SIZE, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )

    feature_cols = get_feature_columns(df_all, target_col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(float)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(float) if not val_df.empty else None
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(float) if not test_df.empty else None
    X_pred = pred_df[feature_cols]

    # 5) Feature typing
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # 6) Build model
    model = build_pipeline(numeric_cols, categorical_cols)

    # 7) Evaluate on validation
    if y_val is not None and len(val_df) > 0:
        metrics_val = evaluate_holdout(model, X_train, y_train, X_val, y_val)
        print("=== Validation metrics (lower RMSE/MAE is better) ===")
        for k, v in metrics_val.items():
            print(f"Val {k}: {v:.6f}")
    else:
        model.fit(X_train, y_train)

    # 8) Evaluate on test (using model trained only on train)
    if y_test is not None and len(test_df) > 0:
        # Re-train on train only to ensure fair test evaluation
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        mse_test = mean_squared_error(y_test, pred_test)
        rmse_test = float(np.sqrt(mse_test))
        mae_test = float(mean_absolute_error(y_test, pred_test))
        r2_test = float(r2_score(y_test, pred_test))
        print("=== Test metrics (lower RMSE/MAE is better) ===")
        print(f"Test MAE: {mae_test:.6f}")
        print(f"Test RMSE: {rmse_test:.6f}")
        print(f"Test R2: {r2_test:.6f}")

    # 9) Cross-Validation on TRAIN set (RMSE)
    rmse_cv = evaluate_cv(model, X_train, y_train, folds=CV_FOLDS)
    print(f"[CV {CV_FOLDS}-Fold on TRAIN] RMSE = {rmse_cv:.6f}")

    # 10) Refit on train+val and persist
    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = pd.concat([y_train, y_val], axis=0) if y_val is not None else y_train
    model.fit(X_trval, y_trval)
    save_model(model, CKPT_PATH)
    print(f"Model saved to: {CKPT_PATH}")

    # 11) Predict unlabeled and export
    if not X_pred.empty:
        y_pred = model.predict(X_pred)
        out = pred_df.copy()
        out["y_pred"] = y_pred
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(PRED_PATH, index=False)
        print(f"Predictions exported to: {PRED_PATH}")
    else:
        print("No prediction set detected; skipping export.")


# =========================
# CLI: Command-Line Interface
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standard Linear Regression pipeline")
    p.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to CSV data (default: core/data/module_01/ecommerce_sales_dataset.csv)")
    p.add_argument("--val-size", type=float, default=VAL_SIZE, help="Validation split ratio (default: 0.2)")
    p.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test split ratio (default: 0.1)")
    return p.parse_args()


def main():
    args = parse_args()
    run_train_eval_predict(Path(args.data))


if __name__ == "__main__":
    main()
