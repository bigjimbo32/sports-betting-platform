"""NBA moneyline fair probability model."""

from __future__ import annotations

import numpy as np
import pandas as pd


class NBAMoneylineModel:
    """Transparent weighted model that outputs fair win probabilities."""

    def __init__(self, home_court_advantage: float) -> None:
        self.home_court_advantage = home_court_advantage

    @staticmethod
    def _sigmoid(x: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-x))

    def predict_home_win_probability(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict home/away win probabilities for upcoming NBA matchups."""

        if features.empty:
            return features.copy()

        df = features.copy()

        elo_term = (df["elo_diff_home_minus_away"] + self.home_court_advantage) / 400.0
        form_term = 0.75 * df["form_diff"]
        net_rating_term = 0.035 * (df["net_rating_form_home"] - df["net_rating_form_away"])
        pace_term = 0.003 * df["pace_diff"].clip(lower=-15.0, upper=15.0)
        rest_term = 0.05 * df["rest_advantage_home"].clip(lower=-3.0, upper=3.0)

        logit = elo_term + form_term + net_rating_term + pace_term + rest_term
        home_prob = self._sigmoid(logit).clip(lower=0.02, upper=0.98)

        df["model_prob_home"] = home_prob
        df["model_prob_away"] = 1.0 - home_prob
        return df
