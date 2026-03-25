"""Collector for The Odds API odds markets."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.utils.http import get_json

LOGGER = logging.getLogger(__name__)


class OddsAPICollector:
    """Collect and normalize moneyline odds from The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4/sports"

    def __init__(self, api_key: str, timeout: int = 20) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def fetch_h2h_odds(
        self,
        sport_key: str,
        regions: str,
        markets: str,
        odds_format: str,
        date_format: str,
        bookmakers: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch upcoming h2h odds events."""

        url = f"{self.BASE_URL}/{sport_key}/odds"
        params: dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        events = get_json(url, params=params, timeout=self.timeout)
        if not isinstance(events, list):
            raise ValueError("Unexpected odds payload: expected a list of events")

        LOGGER.info("Fetched %s upcoming odds events for %s", len(events), sport_key)
        return events

    @staticmethod
    def normalize_h2h_events(events: list[dict[str, Any]], sport_key: str) -> pd.DataFrame:
        """Normalize nested Odds API payload into row-wise market outcomes."""

        rows: list[dict[str, Any]] = []

        for event in events:
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            commence_time = event.get("commence_time")
            event_id = event.get("id")

            for book in event.get("bookmakers", []):
                bookmaker = book.get("key")
                last_update = book.get("last_update")

                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue

                    for outcome in market.get("outcomes", []):
                        rows.append(
                            {
                                "event_id": event_id,
                                "sport_key": sport_key,
                                "commence_time": commence_time,
                                "home_team": home_team,
                                "away_team": away_team,
                                "bookmaker": bookmaker,
                                "market": market.get("key"),
                                "outcome_name": outcome.get("name"),
                                "american_odds": outcome.get("price"),
                                "book_last_update": last_update,
                            }
                        )

        return pd.DataFrame(rows)
