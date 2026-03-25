"""Collector for free NBA game results from ESPN's public API."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.utils.http import get_json

LOGGER = logging.getLogger(__name__)


class NBAStatsCollector:
    """Collect historical NBA game results for feature generation."""

    SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def _fetch_day_games(self, day: date) -> list[dict[str, Any]]:
        payload = get_json(
            self.SCOREBOARD_URL,
            params={"dates": day.strftime("%Y%m%d"), "limit": 200},
            timeout=self.timeout,
        )
        events = payload.get("events", []) if isinstance(payload, dict) else []
        rows: list[dict[str, Any]] = []

        for event in events:
            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors", [])
            status_obj = competition.get("status", {}).get("type", {})
            is_final = bool(status_obj.get("completed", False))
            status_detail = status_obj.get("description") or status_obj.get("detail")

            home, away = None, None
            for team_entry in competitors:
                if team_entry.get("homeAway") == "home":
                    home = team_entry
                elif team_entry.get("homeAway") == "away":
                    away = team_entry

            if not home or not away:
                continue

            rows.append(
                {
                    "game_id": event.get("id"),
                    "game_date": event.get("date"),
                    "status": status_detail,
                    "is_final": is_final,
                    "home_team": home.get("team", {}).get("displayName"),
                    "away_team": away.get("team", {}).get("displayName"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                }
            )

        return rows

    def fetch_games(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch NBA games for an inclusive date range."""

        rows: list[dict[str, Any]] = []
        current = start_date
        while current <= end_date:
            rows.extend(self._fetch_day_games(current))
            current += timedelta(days=1)

        df = pd.DataFrame(rows)
        LOGGER.info(
            "Fetched %s NBA games between %s and %s",
            len(df),
            start_date.isoformat(),
            end_date.isoformat(),
        )
        return df

    def fetch_recent_history(self, history_days: int) -> pd.DataFrame:
        """Fetch historical window ending yesterday."""

        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=history_days)
        return self.fetch_games(start_date=start_date, end_date=end_date)
