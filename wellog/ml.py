"""Predict missing logs (e.g. sonic) from the other curves.

A gradient-boosted tree model is trained on wells where the target curve
exists and used to fill gaps or synthesise the whole curve in wells that
lack it. Besides the point readings, each feature gets a short rolling
mean and a vertical gradient so the model sees a little depth context
(bed boundaries, trends) rather than single samples.

Evaluation is leave-one-well-out: holding out samples at random would leak
neighbouring, nearly identical samples into training and overstate skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .las_io import WellLog

DEFAULT_FEATURES = ["GR", "RHOB", "NPHI", "RT"]
LOG_SCALED = {"RT", "RXO"}  # resistivities are modelled in log10 space


def build_features(df: pd.DataFrame, curves: Sequence[str], window: int = 7) -> pd.DataFrame:
    """Point values plus rolling mean and gradient for each input curve."""
    feats = {}
    for c in curves:
        x = df[c].astype(float)
        if c in LOG_SCALED:
            x = np.log10(x.clip(lower=1e-3))
        feats[c] = x
        feats[f"{c}_mean"] = x.rolling(window, center=True, min_periods=1).mean()
        feats[f"{c}_grad"] = x.diff(window // 2).shift(-(window // 4))
    return pd.DataFrame(feats, index=df.index)


@dataclass
class LogPredictor:
    target: str = "DT"
    features: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    window: int = 7
    model_params: Dict = field(
        default_factory=lambda: {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0}
    )
    model: Optional[HistGradientBoostingRegressor] = None

    def _xy(self, wells: Sequence[WellLog]):
        xs, ys = [], []
        for w in wells:
            missing = [c for c in self.features if c not in w.data]
            if missing:
                raise ValueError(f"{w.name}: missing feature curves {missing}")
            if self.target not in w.data:
                continue
            X = build_features(w.data, self.features, self.window)
            y = w.data[self.target]
            ok = y.notna() & X[self.features].notna().all(axis=1)
            xs.append(X[ok])
            ys.append(y[ok])
        if not xs:
            raise ValueError(f"no training wells contain {self.target}")
        return pd.concat(xs), pd.concat(ys)

    def fit(self, wells: Sequence[WellLog]) -> "LogPredictor":
        X, y = self._xy(wells)
        self.model = HistGradientBoostingRegressor(random_state=0, **self.model_params)
        self.model.fit(X, y)
        return self

    def predict(self, well: WellLog) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        X = build_features(well.data, self.features, self.window)
        pred = pd.Series(self.model.predict(X), index=well.data.index, name=f"{self.target}_PRED")
        # don't extrapolate where the primary inputs are absent
        pred[X[self.features].isna().any(axis=1)] = np.nan
        return pred

    def fill(self, well: WellLog) -> pd.DataFrame:
        """Return <target>_PRED and <target>_FILLED (measured where available)."""
        pred = self.predict(well)
        measured = well.data[self.target] if self.target in well.data else pd.Series(np.nan, index=well.data.index)
        return pd.DataFrame(
            {pred.name: pred, f"{self.target}_FILLED": measured.fillna(pred)},
            index=well.data.index,
        )

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> "LogPredictor":
        return joblib.load(path)


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def leave_one_well_out(wells: Sequence[WellLog], target: str = "DT", features: Optional[List[str]] = None) -> pd.DataFrame:
    """Train on all wells but one, score on the held-out well, repeat.

    A multi-linear regression on the point features is scored alongside as a
    baseline so the benefit of the tree model is visible.
    """
    features = features or list(DEFAULT_FEATURES)
    usable = [w for w in wells if target in w.data and w.data[target].notna().any()]
    if len(usable) < 2:
        raise ValueError("need at least two wells with the target curve for cross-validation")

    rows = []
    for held in usable:
        train = [w for w in usable if w is not held]
        predictor = LogPredictor(target=target, features=features).fit(train)
        X_test, y_test = predictor._xy([held])
        gbm = _metrics(y_test, predictor.model.predict(X_test))

        X_train, y_train = predictor._xy(train)
        lin = LinearRegression().fit(X_train[features], y_train)
        base = _metrics(y_test, lin.predict(X_test[features]))

        rows.append(
            {
                "well": held.name,
                "n": len(y_test),
                "rmse": gbm["rmse"],
                "mae": gbm["mae"],
                "r2": gbm["r2"],
                "rmse_linear": base["rmse"],
                "r2_linear": base["r2"],
            }
        )
    return pd.DataFrame(rows).set_index("well")
