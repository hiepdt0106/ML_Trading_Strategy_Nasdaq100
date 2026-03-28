"""
src/models/train.py
───────────────────
Walk-forward training — Cross-Sectional Ranking approach

Key design decisions:
─────────────────────────────────────────────────────────────────
1. PREPROCESSING:  cross-sectional rank per date (thay vì global scale)
   → XGB thấy "ticker A hôm nay rank #1 về momentum" thay vì raw value
   → Macro/regime features giữ nguyên (rank sẽ ra tie)

2. EVALUATION:     daily AUC + top-K mean return (cross-sectional metrics)

3. XGB TRAINING:   early stopping với date-block validation.
   Chia 15% ngày cuối train làm validation (theo date, không theo row,
   để tránh cắt giữa ngày). Early stopping vẫn theo logloss,
   nhưng chọn cấu hình XGB cuối cùng theo validation daily AUC.

4. GUARD:          global_auc trả NaN nếu fold chỉ có 1 class.
─────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import MACRO_FEATURE_NAMES, SKIP_RANK_FEATURES
from src.splits.walkforward import FoldSplit

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_models(scale_pos_weight: float = 1.0) -> dict:
    """
    LR / RF giữ nguyên như cũ.
    XGB dùng như base template; fit thực tế sẽ được chọn cấu hình cuối cùng
    bằng helper fit_xgb_select_by_daily_auc().
    """
    models = {
        "LR": LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=20,
            max_features=0.5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGB": XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            max_depth=5,
            learning_rate=0.01,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.5,
            colsample_bylevel=0.8,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — Cross-Sectional Rank 
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_raw(X: np.ndarray, medians: np.ndarray = None):
    """Inf → NaN → median impute. Trả về (X_clean, medians)."""
    X = np.where(np.isinf(X), np.nan, X)
    if medians is None:
        medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = medians[j]
    return X, medians


# Features KHÔNG được rank cross-sectional (xem config.py SKIP_RANK_FEATURES)
_SKIP_RANK = SKIP_RANK_FEATURES


def cross_sectional_rank(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Preprocessing hybrid:
    - Ticker-specific features → cross-sectional rank per date [0, 1]
    - Macro features → giữ nguyên
    - Features đã là rank (suffix không cần xử lý vì đã loại _rank ở feature engineering)
    """
    df = df.copy()
    for col in feature_cols:
        if col in _SKIP_RANK:
            continue
        df[col] = df.groupby(level="date")[col].rank(pct=True)
    return df


def _preprocess_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocessing 1 fold:
    1. Cross-sectional rank per date (cho RF, XGB)
    2. Clean + StandardScale (cho LR)

    Returns: X_train_rank, X_test_rank, X_train_scaled, X_test_scaled
    """
    # ── Cross-sectional rank (cho tree models) ──
    train_ranked = cross_sectional_rank(train_df, feature_cols)
    test_ranked = cross_sectional_rank(test_df, feature_cols)

    X_train_rank = train_ranked[feature_cols].values
    X_test_rank = test_ranked[feature_cols].values

    X_train_rank, med_r = _clean_raw(X_train_rank.copy())
    X_test_rank, _ = _clean_raw(X_test_rank.copy(), med_r)

    # ── StandardScale (cho LR) ──
    X_train_raw = train_df[feature_cols].values.copy()
    X_test_raw = test_df[feature_cols].values.copy()

    X_train_raw, med = _clean_raw(X_train_raw)
    X_test_raw, _ = _clean_raw(X_test_raw, med)

    # Clip ±5 std
    means = X_train_raw.mean(axis=0)
    stds = X_train_raw.std(axis=0)
    stds[stds == 0] = 1
    X_train_raw = np.clip(X_train_raw, means - 5 * stds, means + 5 * stds)
    X_test_raw = np.clip(X_test_raw, means - 5 * stds, means + 5 * stds)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_rank, X_test_rank, X_train_scaled, X_test_scaled


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION — Daily AUC + Top-K Return
# ═══════════════════════════════════════════════════════════════════════════════

def daily_auc(test_df: pd.DataFrame, y_prob: np.ndarray) -> float:
    """AUC trung bình per-date (cross-sectional)."""
    tmp = test_df[["tb_label"]].copy()
    tmp["prob"] = y_prob

    aucs = []
    for _, grp in tmp.groupby(level="date"):
        y = grp["tb_label"].values
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, grp["prob"].values))

    return np.mean(aucs) if aucs else 0.5


def top_k_return(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    k: int = 5,
) -> float:
    """Mean daily return của top-K cổ phiếu có y_prob cao nhất."""
    tmp = test_df[["tb_return"]].copy()
    tmp["prob"] = y_prob

    daily_rets = []
    for _, grp in tmp.groupby(level="date"):
        if len(grp) < k:
            continue
        top = grp.nlargest(k, "prob")
        daily_rets.append(top["tb_return"].mean())

    return np.mean(daily_rets) if daily_rets else 0.0


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC với guard khi vector label chỉ có 1 class."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _daily_auc_from_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    dates: pd.Index | np.ndarray,
) -> float:
    """Daily AUC chỉ dùng cho validation model selection của XGB."""
    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(np.asarray(dates)),
            "y": np.asarray(y_true),
            "score": np.asarray(y_score),
        }
    )

    aucs = []
    for _, grp in tmp.groupby("date", sort=True):
        if grp["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(grp["y"].values, grp["score"].values))

    return float(np.mean(aucs)) if aucs else 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# XGB MODEL SELECTION — logloss early stopping, choose by daily AUC
# ═══════════════════════════════════════════════════════════════════════════════

def fit_xgb_select_by_daily_auc(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_dates: pd.Index | np.ndarray,
    X_full_train: np.ndarray,
    y_full_train: np.ndarray,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> tuple[XGBClassifier, dict]:
    """
    Giữ early stopping bằng logloss để training ổn định, nhưng chọn cấu hình
    cuối cùng theo validation daily AUC (metric gần business objective hơn).
    Sau đó refit trên full train fold với số cây đã chọn.
    """
    base_params = {
        "objective": "binary:logistic",
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "random_state": random_state,
        "n_jobs": -1,
    }

    # Grid nhỏ, thực dụng: đủ đa dạng để giảm hiện tượng stop quá sớm,
    # nhưng không làm walk-forward quá nặng.
    param_grid = [
        {"min_child_weight": 3, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {"min_child_weight": 5, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {"min_child_weight": 5, "gamma": 0.1, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {"min_child_weight": 8, "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.5},
    ]

    best_model = None
    best_info = None

    for i, extra in enumerate(param_grid, start=1):
        trial_params = {**base_params, **extra}
        trial = XGBClassifier(**trial_params)
        trial.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_prob = trial.predict_proba(X_val)[:, 1]
        best_iteration = getattr(trial, "best_iteration", None)
        n_trees = trial_params["n_estimators"] if best_iteration is None else int(best_iteration) + 1

        info = {
            "trial": i,
            "params": extra,
            "n_trees": n_trees,
            "val_logloss": getattr(trial, "best_score", np.nan),
            "val_global_auc": _safe_auc(y_val, val_prob),
            "val_daily_auc": _daily_auc_from_arrays(y_val, val_prob, val_dates),
        }

        log.info(
            "    XGB trial %d: daily_auc=%.3f  global_auc=%.3f  logloss=%.5f  trees=%d  params=%s",
            i,
            info["val_daily_auc"],
            info["val_global_auc"] if not pd.isna(info["val_global_auc"]) else float("nan"),
            info["val_logloss"] if not pd.isna(info["val_logloss"]) else float("nan"),
            info["n_trees"],
            info["params"],
        )

        if best_info is None:
            best_model = trial
            best_info = info
            continue

        old_daily = best_info["val_daily_auc"]
        new_daily = info["val_daily_auc"]
        choose_new = False

        if new_daily > old_daily + 1e-6:
            choose_new = True
        elif np.isclose(new_daily, old_daily, atol=1e-6):
            old_auc = best_info["val_global_auc"]
            new_auc = info["val_global_auc"]
            old_auc_cmp = -np.inf if pd.isna(old_auc) else float(old_auc)
            new_auc_cmp = -np.inf if pd.isna(new_auc) else float(new_auc)
            if new_auc_cmp > old_auc_cmp + 1e-6:
                choose_new = True
            elif np.isclose(new_auc_cmp, old_auc_cmp, atol=1e-6):
                old_ll = np.inf if pd.isna(best_info["val_logloss"]) else float(best_info["val_logloss"])
                new_ll = np.inf if pd.isna(info["val_logloss"]) else float(info["val_logloss"])
                if new_ll < old_ll:
                    choose_new = True

        if choose_new:
            best_model = trial
            best_info = info

    assert best_info is not None
    assert best_model is not None

    min_trees = 20
    final_n_trees = max(int(best_info["n_trees"]), min_trees)

    final_params = {
        **base_params,
        **best_info["params"],
        "n_estimators": final_n_trees,
        "early_stopping_rounds": None,
    }
    final_model = XGBClassifier(**final_params)
    final_model.fit(X_full_train, y_full_train, verbose=False)

    best_info = {**best_info, "final_n_trees": final_n_trees}
    return final_model, best_info


# ═══════════════════════════════════════════════════════════════════════════════
# FOLD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    fold: int
    test_year: int
    model_name: str
    global_auc: float
    daily_auc: float
    top_k_ret: float
    train_size: int
    test_size: int


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD LOOP
# ═══════════════════════════════════════════════════════════════════════════════

_USE_RANK = {"RF", "XGB"}
_USE_SCALE = {"LR"}


def walk_forward_train(
    df: pd.DataFrame,
    splits: list[FoldSplit],
    feature_cols: list[str],
    target: str = "tb_label",
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward training — cross-sectional ranking approach.

    Returns: (results_df, pred_all_df)
    """
    all_results: list[FoldResult] = []
    all_preds: list[pd.DataFrame] = []

    for fold in splits:
        log.info(f"{'=' * 60}")
        log.info(f"Fold {fold.fold}: test {fold.test_year}")
        log.info(f"{'=' * 60}")

        train_df = df.loc[fold.train_idx].copy()
        test_df = df.loc[fold.test_idx].copy()

        # ── NaN guard: loại rows thiếu target ──
        train_df = train_df.dropna(subset=[target])
        test_df = test_df.dropna(subset=[target])
        if len(train_df) == 0 or len(test_df) == 0:
            log.warning(f"  Fold {fold.fold}: skip — empty after dropna(target)")
            continue

        y_train = train_df[target].values.astype(int)
        y_test = test_df[target].values.astype(int)

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        spw = n_neg / max(n_pos, 1)

        log.info(
            f"  Train: {fold.train_size:,} (pos={y_train.mean():.1%}) | "
            f"Test: {fold.test_size:,}"
        )

        X_tr_rank, X_te_rank, X_tr_scaled, X_te_scaled = _preprocess_fold(
            train_df,
            test_df,
            feature_cols,
        )

        models = get_models(scale_pos_weight=spw)

        for name, model in models.items():
            if name in _USE_RANK:
                X_tr, X_te = X_tr_rank, X_te_rank
            else:
                X_tr, X_te = X_tr_scaled, X_te_scaled

            if name == "XGB":
                # ── XGB: date-block validation + logloss early stopping ──
                train_dates_sorted = (
                    train_df.index.get_level_values("date").unique().sort_values()
                )
                n_dates = len(train_dates_sorted)

                # Guard mềm cho fold rất ngắn
                if n_dates < 8:
                    val_split_idx = max(1, int(n_dates * 0.8))
                else:
                    val_split_idx = int(n_dates * 0.85)
                val_split_idx = min(max(val_split_idx, 1), n_dates - 1)

                val_start_date = train_dates_sorted[val_split_idx]
                date_vals = train_df.index.get_level_values("date")
                val_mask = date_vals >= val_start_date

                X_fit = X_tr[~val_mask]
                X_val = X_tr[val_mask]
                y_fit = y_train[~val_mask]
                y_val = y_train[val_mask]
                val_dates = date_vals[val_mask]

                xgb_model, xgb_info = fit_xgb_select_by_daily_auc(
                    X_fit=X_fit,
                    y_fit=y_fit,
                    X_val=X_val,
                    y_val=y_val,
                    val_dates=val_dates,
                    X_full_train=X_tr,
                    y_full_train=y_train,
                    scale_pos_weight=spw,
                    random_state=42,
                )
                model = xgb_model

                log.info(
                    "    XGB selected: daily_auc=%.3f  global_auc=%.3f  "
                    "logloss=%.5f  trees=%d  params=%s",
                    xgb_info["val_daily_auc"],
                    xgb_info["val_global_auc"] if not pd.isna(xgb_info["val_global_auc"]) else float("nan"),
                    xgb_info["val_logloss"] if not pd.isna(xgb_info["val_logloss"]) else float("nan"),
                    xgb_info["final_n_trees"],
                    xgb_info["params"],
                )

                y_prob = model.predict_proba(X_te)[:, 1]
            else:
                model.fit(X_tr, y_train)
                y_prob = model.predict_proba(X_te)[:, 1]

            if len(np.unique(y_test)) < 2:
                g_auc = np.nan
            else:
                g_auc = roc_auc_score(y_test, y_prob)
            d_auc = daily_auc(test_df, y_prob)
            tk_ret = top_k_return(test_df, y_prob, k=top_k)

            result = FoldResult(
                fold=fold.fold,
                test_year=fold.test_year,
                model_name=name,
                global_auc=g_auc,
                daily_auc=d_auc,
                top_k_ret=tk_ret,
                train_size=len(X_tr),
                test_size=len(X_te),
            )
            all_results.append(result)

            pred_df = test_df[["adj_close"]].copy()
            pred_df["y_true"] = y_test
            pred_df["y_prob"] = y_prob
            pred_df["model"] = name
            pred_df["fold"] = fold.fold
            all_preds.append(pred_df)

            log.info(
                f"  {name:4s}: global_auc={g_auc:.3f}  "
                f"daily_auc={d_auc:.3f}  top{top_k}_ret={tk_ret:.4f}"
            )

    results_df = pd.DataFrame(
        [
            {
                "fold": r.fold,
                "test_year": r.test_year,
                "model": r.model_name,
                "global_auc": r.global_auc,
                "daily_auc": r.daily_auc,
                "top_k_ret": r.top_k_ret,
                "train_size": r.train_size,
                "test_size": r.test_size,
            }
            for r in all_results
        ]
    )
    pred_all_df = pd.concat(all_preds)

    log.info(f"\n✓ Done: {len(results_df)} fold×model results")
    return results_df, pred_all_df


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD COMPARE — ML Full vs ML Base
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_compare(
    df: pd.DataFrame,
    splits: list[FoldSplit],
    feature_cols_full: list[str],
    feature_cols_base: list[str],
    target: str = "tb_label",
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chạy walk-forward cho cả ML Full (có macro) và ML Base (không macro).
    Returns: results_full, pred_full, results_base, pred_base
    """
    log.info(f"ML Full: {len(feature_cols_full)} features")
    log.info(f"ML Base: {len(feature_cols_base)} features (no macro)")
    log.info("")

    log.info("━" * 60)
    log.info("▶ Training ML FULL (with macro features)")
    log.info("━" * 60)
    results_full, pred_full = walk_forward_train(
        df,
        splits,
        feature_cols_full,
        target=target,
        top_k=top_k,
    )

    log.info("")
    log.info("━" * 60)
    log.info("▶ Training ML BASE (no macro features)")
    log.info("━" * 60)
    results_base, pred_base = walk_forward_train(
        df,
        splits,
        feature_cols_base,
        target=target,
        top_k=top_k,
    )

    # Summary
    log.info("")
    log.info("━" * 60)
    log.info("▶ SUMMARY: ML Full vs ML Base")
    log.info("━" * 60)
    for model_name in results_full["model"].unique():
        full = results_full[results_full["model"] == model_name]
        base = results_base[results_base["model"] == model_name]
        log.info(
            f"  {model_name:4s}: "
            f"Full auc={full['daily_auc'].mean():.3f} topk={full['top_k_ret'].mean():.5f} | "
            f"Base auc={base['daily_auc'].mean():.3f} topk={base['top_k_ret'].mean():.5f} | "
            f"Δauc={full['daily_auc'].mean() - base['daily_auc'].mean():+.3f}"
        )

    return results_full, pred_full, results_base, pred_base


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_ensemble(
    pred_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Ensemble predictions: weighted average y_prob của LR, RF, XGB.
    Returns DataFrame với y_prob = weighted average, model='ENS'.
    """
    models = pred_df["model"].unique().tolist()
    if weights is None:
        weights = {m: 1.0 / len(models) for m in models}

    log.info(f"Ensemble: {models} → weights={weights}")

    pieces = []
    for fold_id in pred_df["fold"].unique():
        fold_data = pred_df[pred_df["fold"] == fold_id]

        model_probs = {}
        y_true_ref = None
        for model_name in models:
            model_sub = fold_data[fold_data["model"] == model_name]
            if len(model_sub) == 0:
                continue
            model_probs[model_name] = model_sub["y_prob"]
            if y_true_ref is None:
                y_true_ref = model_sub[["y_true", "adj_close"]].copy()

        if y_true_ref is None or len(model_probs) == 0:
            continue

        ens_prob = sum(model_probs[m] * weights.get(m, 0) for m in model_probs)

        ens_df = y_true_ref.copy()
        ens_df["y_prob"] = ens_prob
        ens_df["model"] = "ENS"
        ens_df["fold"] = fold_id
        pieces.append(ens_df)

    ensemble_df = pd.concat(pieces)
    log.info(f"Ensemble: {len(ensemble_df):,} predictions")
    return ensemble_df
