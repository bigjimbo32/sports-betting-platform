"""Main entrypoint for Phase 1 NBA moneyline +EV engine."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import load_settings
from src.betting.recommender import BetRecommender
from src.collectors.nba_stats import NBAStatsCollector
from src.collectors.odds_api import OddsAPICollector
from src.features.nba_features import NBAFeatureBuilder
from src.models.nba_moneyline_model import NBAMoneylineModel
from src.utils.io import utc_now_str, write_csv
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def _save_results(
    run_ts: str,
    output_dir: str,
    odds_df: pd.DataFrame,
    features_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> None:
    odds_path = write_csv(odds_df, output_dir, "odds_snapshot", run_ts)
    features_path = write_csv(features_df, output_dir, "model_features", run_ts)
    preds_path = write_csv(predictions_df, output_dir, "model_predictions", run_ts)
    recs_path = write_csv(recommendations_df, output_dir, "bet_recommendations", run_ts)

    LOGGER.info("Saved odds snapshot: %s", odds_path)
    LOGGER.info("Saved model features: %s", features_path)
    LOGGER.info("Saved model predictions: %s", preds_path)
    LOGGER.info("Saved recommendations: %s", recs_path)


def _save_results_tracker(output_dir: str, run_ts: str, recommendations_df: pd.DataFrame) -> None:
    """Save a grading-ready record with placeholder outcome columns."""

    if recommendations_df.empty:
        return

    graded_cols = [
        "commence_time",
        "sport_key",
        "home_team",
        "away_team",
        "bookmaker",
        "market",
        "outcome_name",
        "american_odds",
        "model_probability",
        "implied_prob_novig",
        "edge_vs_novig",
        "recommended_bet",
    ]

    tracker = recommendations_df[graded_cols].copy()
    tracker["run_ts"] = run_ts
    tracker["result"] = pd.NA
    tracker["profit_units"] = pd.NA

    file_path = Path(output_dir) / "bet_results_tracker.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        tracker.to_csv(file_path, mode="a", header=False, index=False)
    else:
        tracker.to_csv(file_path, index=False)

    LOGGER.info("Updated grading tracker: %s", file_path)


def run() -> None:
    """Run one complete data -> model -> recommendation cycle."""

    settings = load_settings()
    configure_logging(settings.log_level)

    LOGGER.info("Starting sports betting pipeline for sport=%s", settings.sport_key)

    odds_collector = OddsAPICollector(settings.odds_api_key, timeout=settings.request_timeout_seconds)
    stats_collector = NBAStatsCollector(timeout=settings.request_timeout_seconds)

    events = odds_collector.fetch_h2h_odds(
        sport_key=settings.sport_key,
        regions=settings.regions,
        markets=settings.odds_markets,
        odds_format=settings.odds_format,
        date_format=settings.date_format,
        bookmakers=settings.bookmakers,
    )

    odds_df = odds_collector.normalize_h2h_events(events, sport_key=settings.sport_key)
    if odds_df.empty:
        LOGGER.warning("No odds rows returned. Exiting run.")
        return

    history_df = stats_collector.fetch_recent_history(settings.history_days)

    feature_builder = NBAFeatureBuilder(
        elo_base=settings.elo_base,
        elo_k_factor=settings.elo_k_factor,
        home_court_advantage=settings.home_court_advantage,
        recent_form_games=settings.recent_form_games,
    )
    team_states = feature_builder.build_team_states(history_df)
    features_df = feature_builder.features_for_matchups(odds_df, team_states)

    model = NBAMoneylineModel(home_court_advantage=settings.home_court_advantage)
    predictions_df = model.predict_home_win_probability(features_df)

    recommender = BetRecommender(edge_threshold=settings.edge_threshold)
    recommendations_df = recommender.evaluate(odds_df, predictions_df)

    run_ts = utc_now_str()
    _save_results(
        run_ts=run_ts,
        output_dir=settings.output_dir,
        odds_df=odds_df,
        features_df=features_df,
        predictions_df=predictions_df,
        recommendations_df=recommendations_df,
    )
    _save_results_tracker(settings.output_dir, run_ts, recommendations_df)

    n_recommended = int(recommendations_df["recommended_bet"].sum()) if not recommendations_df.empty else 0
    LOGGER.info(
        "Run complete. games=%s outcomes=%s recommended=%s threshold=%.4f",
        features_df.shape[0],
        recommendations_df.shape[0],
        n_recommended,
        settings.edge_threshold,
    )

    if not recommendations_df.empty:
        display_cols = [
            "commence_time",
            "home_team",
            "away_team",
            "bookmaker",
            "outcome_name",
            "american_odds",
            "model_probability",
            "implied_prob_novig",
            "edge_vs_novig",
            "recommended_bet",
        ]
        print(recommendations_df[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        raise
