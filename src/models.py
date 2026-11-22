# AI-assisted: generated with Claude Code, reviewed and run by the author
"""Model builders, metrics and small training helpers used by the notebook."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42
FAMILY = {
    "Persistence": "Baseline", "Climatology": "Baseline", "Linear": "Baseline",
    "RandomForest": "Bagging", "BaggedTrees": "Bagging", "ExtraTrees": "Bagging",
    "HistGB": "Boosting", "XGBoost": "Boosting", "LightGBM": "Boosting",
    "CatBoost": "Boosting", "Stacking": "Stacking",
}


def is_clf(task: str) -> bool:
    return task == "rain_flag"


# ---------------------------------------------------------------- metrics
def regression_metrics(y, pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def classification_metrics(y, proba, threshold: float = 0.5) -> dict:
    hard = (np.asarray(proba) >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y, hard)),
        "F1": float(f1_score(y, hard)),
        "ROC_AUC": float(roc_auc_score(y, proba)),
        "PR_AUC": float(average_precision_score(y, proba)),
    }


def score(task: str, y, pred) -> dict:
    return classification_metrics(y, pred) if is_clf(task) else regression_metrics(y, pred)


# ---------------------------------------------------------------- helpers
def predict(model, X, task: str):
    """Probability of rain for the classifier, plain prediction otherwise."""
    if is_clf(task):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def timed_fit(model, X, y, **fit_kw):
    t0 = time.perf_counter()
    model.fit(X, y, **fit_kw)
    return model, time.perf_counter() - t0


def oof_predictions(make_model, X, y, task: str, n_splits: int = 5, fit_kw=None):
    """Out-of-fold predictions with KFold(shuffle=False) (contiguous time blocks)."""
    fit_kw = fit_kw or {}
    oof = np.zeros(len(X))
    kf = KFold(n_splits=n_splits, shuffle=False)
    for tr, va in kf.split(X):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr], **fit_kw)
        oof[va] = predict(m, X.iloc[va], task)
    return oof


# ---------------------------------------------------------------- builders
def linear_preprocessor(feature_cols):
    cat = ["city_code"]
    num = [c for c in feature_cols if c not in cat]
    return ColumnTransformer([
        ("city", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])


def make_linear(task, feature_cols):
    est = LogisticRegression(max_iter=2000, C=1.0) if is_clf(task) else Ridge(alpha=1.0)
    return Pipeline([("prep", linear_preprocessor(feature_cols)), ("model", est)])


def make_rf(task, **params):
    base = dict(n_estimators=200, min_samples_leaf=5, max_features=0.4,
                max_samples=0.5, n_jobs=-1, random_state=SEED)
    base.update(params)
    return RandomForestClassifier(**base) if is_clf(task) else RandomForestRegressor(**base)


def make_bagged_trees(task, n_estimators=50):
    kw = dict(n_estimators=n_estimators, max_samples=0.5, n_jobs=-1, random_state=SEED)
    if is_clf(task):
        return BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5), **kw)
    return BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=5), **kw)


def make_extratrees(task):
    kw = dict(n_estimators=200, min_samples_leaf=5, max_features=0.5,
              n_jobs=-1, random_state=SEED)
    return ExtraTreesClassifier(**kw) if is_clf(task) else ExtraTreesRegressor(**kw)


def make_hgb(task):
    kw = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
              early_stopping=False, random_state=SEED)
    return HistGradientBoostingClassifier(**kw) if is_clf(task) else HistGradientBoostingRegressor(**kw)


def make_xgb(task, n_estimators=2000, early_stopping_rounds=50):
    import xgboost as xgb
    kw = dict(n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              n_jobs=-1, random_state=SEED, early_stopping_rounds=early_stopping_rounds)
    if is_clf(task):
        return xgb.XGBClassifier(eval_metric="logloss", **kw)
    return xgb.XGBRegressor(eval_metric="rmse", **kw)


def make_lgbm(task, n_estimators=3000):
    import lightgbm as lgb
    kw = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=-1, random_state=SEED, verbose=-1)
    return lgb.LGBMClassifier(**kw) if is_clf(task) else lgb.LGBMRegressor(**kw)


def make_catboost(task, iterations=3000, **params):
    from catboost import CatBoostClassifier, CatBoostRegressor
    kw = dict(iterations=iterations, learning_rate=0.08, depth=6, thread_count=-1,
              random_seed=SEED, verbose=0, cat_features=["city"])
    kw.update(params)
    return CatBoostClassifier(**kw) if is_clf(task) else CatBoostRegressor(**kw)


def make_mlp():
    return MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
                        batch_size=512, max_iter=40, early_stopping=False,
                        random_state=SEED)


def catboost_frame(X: pd.DataFrame, city: pd.Series) -> pd.DataFrame:
    """CatBoost gets the real city name as a native categorical instead of a code."""
    return X.drop(columns=["city_code"]).assign(city=city.values)
