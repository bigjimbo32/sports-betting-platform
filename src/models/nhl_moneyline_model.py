"""NHL moneyline fair probability model based on weighted practical features."""

from __future__ import annotations

import numpy as np
import pandas as pd


class NHLMoneylineModel:
    """Simple transparent model combining ELO and form-based terms."""

    def __init__(self, home_ice_advantage: float) -> None:
        self.home_ice_advantage = home_ice_advantage

    @staticmethod
    def _sigmoid(x: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-x))

    def predict_home_win_probability(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict fair home and away win probabilities for each event."""

        if features.empty:
            return features.copy()

        df = features.copy()

        elo_term = (df["elo_diff_home_minus_away"] + self.home_ice_advantage) / 400.0
        form_term = 0.60 * df["form_diff"]
        goal_form_term = 0.08 * (df["goal_diff_form_home"] - df["goal_diff_form_away"])
        rest_term = 0.04 * df["rest_advantage_home"].clip(lower=-3.0, upper=3.0)

        logit = elo_term + form_term + goal_form_term + rest_term
        home_prob = self._sigmoid(logit).clip(lower=0.02, upper=0.98)

        df["model_prob_home"] = home_prob
        df["model_prob_away"] = 1.0 - home_prob
        return df
