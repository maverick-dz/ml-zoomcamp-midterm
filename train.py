"""
train.py - trains the diabetes screening model and writes model.bin + artifacts.

    python train.py                    # full run (several minutes)
    python train.py --quick            # smoke test: tiny search, ~1-2 minutes
    python train.py --out-dir /tmp/x   # write model.bin / artifacts elsewhere

Pipeline
  1. Load the CSV and drop exact duplicate rows (9.5% of the raw file).
  2. Stratified split: 60% train / 20% validation / 20% test.
  3. Train 4 baseline families (logistic regression, decision tree, random
     forest, XGBoost) and score them on the validation set.
  4. Tune random forest and XGBoost with RandomizedSearchCV using stratified
     K-fold CV on the FULL training split (no subsampling).
  5. Select the model with the best validation ROC-AUC (done in code).
  6. On train+validation: get out-of-fold predictions, fit an isotonic
     calibrator and choose the decision threshold that reaches the target
     recall - so the test set is never used for any of these choices.
  7. Refit on train+validation, evaluate ONCE on the test set, save everything.

Models are trained WITHOUT class weighting; imbalance is handled by
calibrating the probabilities and moving the threshold instead. That makes the
served probability a real probability rather than a re-scaled score.
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from scoring import FEATURES  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
RANDOM_STATE = 42
TARGET = "diabetes_binary"
TARGET_RECALL = 0.80


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH, drop_duplicates: bool = True):
    """Read the CSV, lowercase columns, optionally drop exact duplicate rows."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    rows_raw = len(df)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    int_cols = [c for c in df.columns if c != "bmi"]
    df[int_cols] = df[int_cols].astype(int)
    info = {
        "rows_raw": int(rows_raw),
        "duplicates_removed": int(rows_raw - len(df)),
        "rows_used": int(len(df)),
        "positive_rate": float(df[TARGET].mean()),
    }
    return df, info


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """60/20/20 stratified split. Returns a dict of numpy arrays."""
    df_full, df_test = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df[TARGET])
    df_train, df_val = train_test_split(
        df_full, test_size=0.25, random_state=random_state,
        stratify=df_full[TARGET])

    def xy(d):
        return d[FEATURES].to_numpy(dtype=float), d[TARGET].to_numpy()

    out = {}
    for name, d in [("train", df_train), ("val", df_val),
                    ("test", df_test), ("full", df_full)]:
        out["X_" + name], out["y_" + name] = xy(d)
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def baseline_models() -> dict:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5, n_jobs=-1,
            random_state=RANDOM_STATE),
        "xgboost_default": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="auc",
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE),
    }


XGB_SPACE = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [1, 5, 10],
}
RF_SPACE = {
    "max_depth": [8, 10, 12, 15],
    "min_samples_leaf": [5, 10, 20, 50],
    "max_features": ["sqrt", 0.5],
}


def precision_at_recall(y, p, target=TARGET_RECALL) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec >= target
    return float(prec[ok].max()) if ok.any() else float("nan")


def score_on(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    return {
        "val_auc": roc_auc_score(y, p),
        "val_avg_precision": average_precision_score(y, p),
        "val_brier": brier_score_loss(y, p),
        "val_precision_at_recall80": precision_at_recall(y, p),
    }


def _plain(d: dict) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def tune(estimator, space, n_iter, cv, X, y):
    """RandomizedSearchCV (ROC-AUC) on the full training split."""
    search = RandomizedSearchCV(
        estimator, space, n_iter=n_iter, scoring="roc_auc",
        cv=StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    search.fit(X, y)
    i = search.best_index_
    cv_stats = {
        "cv_auc_mean": float(search.cv_results_["mean_test_score"][i]),
        "cv_auc_std": float(search.cv_results_["std_test_score"][i]),
        "cv_folds": cv,
        "search_iterations": n_iter,
    }
    return search.best_estimator_, _plain(search.best_params_), cv_stats


def use_all_cores(model):
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=-1)
    return model


def compare_models(data, n_iter_xgb=10, n_iter_rf=4, cv_xgb=5, cv_rf=3,
                   log=print):
    """Train baselines + tuned models. Returns (results_df, fitted, tuned_params)."""
    Xtr, ytr, Xv, yv = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    rows, fitted = [], {}

    for name, model in baseline_models().items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        rows.append({"model": name, **score_on(model, Xv, yv),
                     "fit_seconds": round(time.time() - t0, 1)})
        fitted[name] = model
        log(f"  {name:<20} val AUC={rows[-1]['val_auc']:.4f}  "
            f"({rows[-1]['fit_seconds']}s)")

    searches = [
        ("xgboost_tuned",
         xgb.XGBClassifier(eval_metric="auc", tree_method="hist", n_jobs=1,
                           random_state=RANDOM_STATE),
         XGB_SPACE, n_iter_xgb, cv_xgb),
        ("random_forest_tuned",
         RandomForestClassifier(n_estimators=100, max_samples=0.5, n_jobs=1,
                                random_state=RANDOM_STATE),
         RF_SPACE, n_iter_rf, cv_rf),
    ]
    tuned_params = {}
    for name, est, space, n_iter, cv in searches:
        t0 = time.time()
        best, params, cv_stats = tune(est, space, n_iter, cv, Xtr, ytr)
        rows.append({"model": name, **score_on(best, Xv, yv), **cv_stats,
                     "fit_seconds": round(time.time() - t0, 1)})
        fitted[name] = best
        tuned_params[name] = params
        log(f"  {name:<20} val AUC={rows[-1]['val_auc']:.4f}  "
            f"CV AUC={cv_stats['cv_auc_mean']:.4f}+/-{cv_stats['cv_auc_std']:.4f}  "
            f"({rows[-1]['fit_seconds']}s)  {params}")

    results = (pd.DataFrame(rows).sort_values("val_auc", ascending=False)
               .reset_index(drop=True))
    return results, fitted, tuned_params


# --------------------------------------------------------------------------
# Calibration + threshold
# --------------------------------------------------------------------------
def pick_threshold(y, p, target_recall=TARGET_RECALL) -> float:
    """Largest probability threshold whose recall is still >= target."""
    _prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok.max()]) if len(ok) else float(thr.min())


def fit_calibrator_and_threshold(model, X, y, target_recall=TARGET_RECALL,
                                 folds=5):
    """Out-of-fold predictions -> isotonic calibrator + decision threshold."""
    oof = cross_val_predict(
        use_all_cores(clone(model)), X, y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba")[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof, y)
    oof_cal = calibrator.predict(oof)
    return calibrator, pick_threshold(y, oof_cal, target_recall), oof, oof_cal


def metrics_at(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# --------------------------------------------------------------------------
# Figures (model-related; EDA figures live in the notebook)
# --------------------------------------------------------------------------
def make_figures(model, thr, data, p_raw, p_cal, fig_dir: Path, model_name: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    y = data["y_test"]

    # 05 confusion matrix
    cm = confusion_matrix(y, (p_cal >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set_yticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold={thr:.3f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_confusion_matrix.png", dpi=130)
    plt.close(fig)

    # 06 ROC + PR
    fpr, tpr, _ = roc_curve(y, p_cal)
    prec, rec, _ = precision_recall_curve(y, p_cal)
    m = metrics_at(y, p_cal, thr)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, p_cal):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[0].scatter([1 - m["specificity"]], [m["recall"]], c="red", zorder=5,
                    label="operating point")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC curve (test)")
    axes[0].legend()
    axes[1].plot(rec, prec, label=f"AP = {average_precision_score(y, p_cal):.3f}")
    axes[1].axhline(y.mean(), color="k", ls="--", alpha=.4, label="base rate")
    axes[1].scatter([m["recall"]], [m["precision"]], c="red", zorder=5,
                    label="operating point")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-recall curve (test)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "06_roc_pr_curves.png", dpi=130)
    plt.close(fig)

    # 07 permutation importance (model-agnostic, ROC-AUC drop on a test sample)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(y), size=min(8000, len(y)), replace=False)
    pi = permutation_importance(model, data["X_test"][idx], y[idx],
                                scoring="roc_auc", n_repeats=3,
                                random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    imp.tail(15).plot.barh(ax=ax, color="#2a6f97")
    ax.set(xlabel="Drop in ROC-AUC when the feature is shuffled",
           title=f"Permutation importance - {model_name}")
    fig.tight_layout()
    fig.savefig(fig_dir / "07_feature_importance.png", dpi=130)
    plt.close(fig)

    # 08 calibration (reliability) curve
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for label, p in [("raw model", p_raw), ("isotonic-calibrated", p_cal)]:
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4, label="perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
           title="Calibration on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "08_calibration.png", dpi=130)
    plt.close(fig)

    # 09 threshold trade-off
    ts = np.linspace(0.02, 0.8, 80)
    pr = [precision_score(y, p_cal >= t, zero_division=0) for t in ts]
    rc = [recall_score(y, p_cal >= t) for t in ts]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ts, rc, label="recall (sensitivity)")
    ax.plot(ts, pr, label="precision")
    ax.axvline(thr, color="red", ls="--", label=f"chosen threshold {thr:.3f}")
    ax.set(xlabel="Probability threshold", ylabel="Score",
           title="Precision / recall trade-off (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "09_threshold_tradeoff.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="tiny search + subsampled data (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=HERE,
                    help="where model.bin and artifacts/ are written")
    ap.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[1/7] Loading data and dropping duplicate rows...")
    df, info = load_data()
    if args.quick:
        df = df.sample(frac=0.3, random_state=RANDOM_STATE).reset_index(drop=True)
        info["rows_used"] = int(len(df))
    print(f"  {info}")

    print("[2/7] Splitting 60/20/20 (stratified)...")
    data = split_data(df)
    sizes = {k: int(len(data["y_" + k])) for k in ("train", "val", "test", "full")}
    print(f"  {sizes}")

    print("[3-4/7] Baselines + tuning (K-fold CV on the full train split)...")
    kw = dict(n_iter_xgb=2, n_iter_rf=1, cv_xgb=3, cv_rf=3) if args.quick else {}
    results, fitted, tuned_params = compare_models(data, **kw)
    results.to_csv(art_dir / "model_comparison.csv", index=False)
    print("\n" + results.round(4).to_string(index=False))

    print("\n[5/7] Selecting best model by validation ROC-AUC...")
    best_name = results.loc[0, "model"]
    print(f"  -> {best_name}")
    best_params = tuned_params.get(best_name, {})
    prototype = clone(fitted[best_name])

    print("[6/7] Out-of-fold calibration + threshold on train+val...")
    calibrator, thr, _oof_raw, oof_cal = fit_calibrator_and_threshold(
        prototype, data["X_full"], data["y_full"], args.target_recall)
    print(f"  threshold for recall>={args.target_recall:.2f}: {thr:.4f}")

    print("[7/7] Refitting on train+val and evaluating once on test...")
    final = use_all_cores(clone(prototype))
    final.fit(data["X_full"], data["y_full"])
    y_test = data["y_test"]
    p_raw = final.predict_proba(data["X_test"])[:, 1]
    p_cal = calibrator.predict(p_raw)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_cal)),
        "average_precision": float(average_precision_score(y_test, p_cal)),
        "brier_raw": float(brier_score_loss(y_test, p_raw)),
        "brier_calibrated": float(brier_score_loss(y_test, p_cal)),
        "at_threshold": metrics_at(y_test, p_cal, thr),
    }
    threshold_table = []
    for target in (0.70, 0.80, 0.90):
        t = pick_threshold(data["y_full"], oof_cal, target)
        threshold_table.append({"target_recall_on_train_val": target,
                                **metrics_at(y_test, p_cal, t)})
    pd.DataFrame(threshold_table).to_csv(art_dir / "threshold_table.csv",
                                         index=False)
    print(json.dumps(test_metrics, indent=2))

    make_figures(final, thr, data, p_raw, p_cal, art_dir / "figures", best_name)

    versions = {"python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "xgboost": xgb.__version__, "numpy": np.__version__}
    metrics = {
        "model": best_name,
        "best_params": best_params,
        "target_recall": args.target_recall,
        "threshold": thr,
        "data": {**info, "split": sizes},
        "test_metrics": test_metrics,
        "threshold_table": threshold_table,
        "all_model_results": results.round(5).to_dict(orient="records"),
        "versions": versions,
        "random_state": RANDOM_STATE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_run": bool(args.quick),
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    bundle = {
        "model": final, "calibrator": calibrator, "features": FEATURES,
        "threshold": thr, "model_name": best_name,
        "target_recall": args.target_recall, "versions": versions,
    }
    with open(out_dir / "model.bin", "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved {out_dir / 'model.bin'} and artifacts/ "
          f"in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
