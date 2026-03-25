"""Bet recommendation logic for +EV moneyline opportunities."""

from __future__ import annotations

import pandas as pd

from src.utils.odds import american_to_decimal, american_to_implied_probability, remove_vig_two_way


class BetRecommender:
    """Evaluate model probabilities against bookmaker odds and flag +EV bets."""

    def __init__(self, edge_threshold: float) -> None:
        self.edge_threshold = edge_threshold

    def evaluate(self, odds_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Join odds and model predictions then produce recommendations."""

        if odds_df.empty or predictions_df.empty:
            return pd.DataFrame()

        merged = odds_df.merge(
            predictions_df[
                [
                    "event_id",
                    "sport_key",
                    "commence_time",
                    "home_team",
                    "away_team",
                    "model_prob_home",
                    "model_prob_away",
                ]
            ],
            on=["event_id", "sport_key", "commence_time", "home_team", "away_team"],
            how="inner",
        )
        if merged.empty:
            return merged

        merged["american_odds"] = merged["american_odds"].astype(float)
        merged["decimal_odds"] = merged["american_odds"].map(american_to_decimal)
        merged["implied_prob_raw"] = merged["american_odds"].map(american_to_implied_probability)

        # Pair outcomes per bookmaker/event to compute no-vig probabilities.
        merged["implied_prob_novig"] = merged["implied_prob_raw"]
        for (event_id, bookmaker), group in merged.groupby(["event_id", "bookmaker"]):
            if len(group) != 2:
                continue
            p1, p2 = remove_vig_two_way(group.iloc[0]["implied_prob_raw"], group.iloc[1]["implied_prob_raw"])
            merged.loc[group.index[0], "implied_prob_novig"] = p1
            merged.loc[group.index[1], "implied_prob_novig"] = p2

        merged["model_probability"] = merged.apply(
            lambda r: r["model_prob_home"] if r["outcome_name"] == r["home_team"] else r["model_prob_away"],
            axis=1,
        )

        merged["edge_vs_raw"] = merged["model_probability"] - merged["implied_prob_raw"]
        merged["edge_vs_novig"] = merged["model_probability"] - merged["implied_prob_novig"]
        merged["recommended_bet"] = merged["edge_vs_novig"] >= self.edge_threshold

        output_cols = [
            "commence_time",
            "sport_key",
            "home_team",
            "away_team",
            "bookmaker",
            "market",
            "outcome_name",
            "american_odds",
            "decimal_odds",
            "implied_prob_raw",
            "implied_prob_novig",
            "model_probability",
            "edge_vs_raw",
            "edge_vs_novig",
            "recommended_bet",
        ]

        return merged[output_cols].sort_values(
            by=["recommended_bet", "edge_vs_novig"], ascending=[False, False]
        )
