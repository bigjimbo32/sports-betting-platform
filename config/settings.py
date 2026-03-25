"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the betting pipeline."""

    odds_api_key: str
    sport_key: str = "icehockey_nhl"
    regions: str = "us"
    bookmakers: str = "draftkings,fanduel,betmgm"
    odds_markets: str = "h2h"
    odds_format: str = "american"
    date_format: str = "iso"
    edge_threshold: float = 0.03
    history_days: int = 120
    elo_base: float = 1500.0
    elo_k_factor: float = 20.0
    home_ice_advantage: float = 45.0
    recent_form_games: int = 5
    request_timeout_seconds: int = 20
    log_level: str = "INFO"
    output_dir: str = "outputs"



def load_settings() -> Settings:
    """Load settings from environment variables."""

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ODDS_API_KEY is required. Add it to your environment or .env file.")

    return Settings(
        odds_api_key=api_key,
        sport_key=os.getenv("SPORT_KEY", "icehockey_nhl"),
        regions=os.getenv("REGIONS", "us"),
        bookmakers=os.getenv("BOOKMAKERS", "draftkings,fanduel,betmgm"),
        odds_markets=os.getenv("ODDS_MARKETS", "h2h"),
        odds_format=os.getenv("ODDS_FORMAT", "american"),
        date_format=os.getenv("DATE_FORMAT", "iso"),
        edge_threshold=float(os.getenv("EDGE_THRESHOLD", "0.03")),
        history_days=int(os.getenv("HISTORY_DAYS", "120")),
        elo_base=float(os.getenv("ELO_BASE", "1500")),
        elo_k_factor=float(os.getenv("ELO_K_FACTOR", "20")),
        home_ice_advantage=float(os.getenv("HOME_ICE_ADVANTAGE", "45")),
        recent_form_games=int(os.getenv("RECENT_FORM_GAMES", "5")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        output_dir=os.getenv("OUTPUT_DIR", "outputs"),
    )
