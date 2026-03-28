"""
src/regime/hmm.py
─────────────────
Hidden Markov Model cho Regime Detection

Sử dụng: Regime as Feature
  - P(high_vol) được đưa vào feature set qua add_regime_features()
  - Model tự học interaction giữa regime và các features khác

Anti-leakage:
  - HMM fit trên data đến t-1 (không dùng ngày hiện tại)
  - Refit theo frequency (mặc định mỗi quý)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    log.warning("hmmlearn not installed. pip install hmmlearn")


class RegimeHMM:
    """
    HMM-based regime detector.

    Fit 2-state Gaussian HMM trên:
      - VXN log returns (5d)
      - VIX-VXN spread z-score
      - VXN level z-score

    State assignment:
      State có mean VXN return CAO hơn = High-Vol state.
    """

    def __init__(self, n_states: int = 2, min_obs: int = 252):
        self.n_states = n_states
        self.min_obs = min_obs
        self.model: GaussianHMM | None = None
        self.high_state: int = 0

    def _build_features(
        self, vxn: pd.Series, vix: pd.Series
    ) -> pd.DataFrame:
        """Tạo feature matrix cho HMM."""
        vxn_ret5 = np.log(vxn / vxn.shift(5))

        spread = vxn - vix
        sp_mean = spread.rolling(63, min_periods=20).mean()
        sp_std = spread.rolling(63, min_periods=20).std().replace(0, np.nan)
        spread_z = (spread - sp_mean) / sp_std

        vxn_mean = vxn.rolling(63, min_periods=20).mean()
        vxn_std = vxn.rolling(63, min_periods=20).std().replace(0, np.nan)
        vxn_z = (vxn - vxn_mean) / vxn_std

        feat = pd.DataFrame({
            "vxn_ret5": vxn_ret5,
            "spread_z": spread_z,
            "vxn_z": vxn_z,
        }).dropna()

        return feat

    def fit(self, vxn: pd.Series, vix: pd.Series) -> "RegimeHMM":
        """Fit HMM trên historical VXN/VIX data."""
        if not HAS_HMMLEARN:
            log.warning("hmmlearn not available, skipping HMM fit")
            return self

        feat = self._build_features(vxn, vix)
        if len(feat) < self.min_obs:
            log.warning(f"HMM: only {len(feat)} obs, need {self.min_obs}. Skipping.")
            return self

        X = feat.values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                verbose=False,
            )
            self.model.fit(X)

        means = self.model.means_[:, 0]
        self.high_state = int(np.argmax(means))

        log.info(
            f"HMM fit: {len(feat)} obs, {self.n_states} states, "
            f"high_vol=state_{self.high_state} "
            f"(vxn_ret5 means: {means.round(4)})"
        )
        return self

    def predict_proba(self, vxn: pd.Series, vix: pd.Series) -> float:
        """P(high_volatility) tại thời điểm cuối cùng. Returns 0.5 nếu chưa fit."""
        if self.model is None:
            return 0.5

        feat = self._build_features(vxn, vix)
        if len(feat) < 10:
            return 0.5

        X = feat.values
        try:
            probs = self.model.predict_proba(X)
            return float(probs[-1, self.high_state])
        except Exception as e:
            log.warning(f"HMM predict_proba error: {e}")
            return 0.5