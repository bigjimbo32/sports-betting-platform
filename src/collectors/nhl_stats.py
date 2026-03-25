"""Collector for free NHL game results and schedule data."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.utils.http import get_json

LOGGER = logging.getLogger(__name__)


class NHLStatsCollector:
    """Collect historical NHL game results via public NHL Stats API."""

    SCHEDULE_URL = "https://statsapi.web.nhl.com/api/v1/schedule"

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def fetch_games(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch NHL schedule and result data for date range."""

        params = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "expand": "schedule.linescore",
        }
        payload = get_json(self.SCHEDULE_URL, params=params, timeout=self.timeout)
        dates = payload.get("dates", []) if isinstance(payload, dict) else []

        rows: list[dict[str, Any]] = []
        for day in dates:
            for game in day.get("games", []):
                status = game.get("status", {}).get("detailedState")
                is_final = status in {"Final", "Game Over"}

                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})

                rows.append(
                    {
                        "game_id": game.get("gamePk"),
                        "game_date": game.get("gameDate"),
                        "status": status,
                        "is_final": is_final,
                        "home_team": home.get("team", {}).get("name"),
                        "away_team": away.get("team", {}).get("name"),
                        "home_score": home.get("score"),
                        "away_score": away.get("score"),
                    }
                )

        df = pd.DataFrame(rows)
        LOGGER.info(
            "Fetched %s NHL games between %s and %s",
            len(df),
            start_date.isoformat(),
            end_date.isoformat(),
        )
        return df

    def fetch_recent_history(self, history_days: int) -> pd.DataFrame:
        """Fetch historical window ending yesterday for model feature generation."""

        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=history_days)
        return self.fetch_games(start_date=start_date, end_date=end_date)
