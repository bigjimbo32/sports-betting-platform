"""Feature engineering for NBA moneyline predictions."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


@dataclass
class TeamState:
    """Mutable team state used to generate features."""

    elo: float
    wins_recent: deque[int]
    points_for_recent: deque[float]
    points_against_recent: deque[float]
    pace_recent: deque[float]
    last_game_dt: datetime | None


class NBAFeatureBuilder:
    """Build pre-game matchup features from historical NBA games."""

    def __init__(
        self,
        elo_base: float,
        elo_k_factor: float,
        home_court_advantage: float,
        recent_form_games: int,
    ) -> None:
        self.elo_base = elo_base
        self.elo_k_factor = elo_k_factor
        self.home_court_advantage = home_court_advantage
        self.recent_form_games = recent_form_games

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

    @staticmethod
    def _safe_mean(values: deque[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _days_since(last_game_dt: datetime | None, now_dt: datetime) -> float:
        if last_game_dt is None:
            return 4.0
        return max((now_dt - last_game_dt).total_seconds() / 86400.0, 0.0)

    def _expected_home_win(self, home_elo: float, away_elo: float) -> float:
        diff = (home_elo + self.home_court_advantage) - away_elo
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def _update_elo(self, home_state: TeamState, away_state: TeamState, home_win: int, mov: float) -> None:
        expected_home = self._expected_home_win(home_state.elo, away_state.elo)
        mov_mult = ((abs(mov) + 3.0) ** 0.8) / (7.5 + 0.006 * abs(home_state.elo - away_state.elo))
        delta = self.elo_k_factor * mov_mult * (home_win - expected_home)
        home_state.elo += delta
        away_state.elo -= delta

    def build_team_states(self, history_games: pd.DataFrame) -> dict[str, TeamState]:
        """Create rolling team state from completed historical NBA games."""

        states: dict[str, TeamState] = defaultdict(
            lambda: TeamState(
                elo=self.elo_base,
                wins_recent=deque(maxlen=self.recent_form_games),
                points_for_recent=deque(maxlen=self.recent_form_games),
                points_against_recent=deque(maxlen=self.recent_form_games),
                pace_recent=deque(maxlen=self.recent_form_games),
                last_game_dt=None,
            )
        )

        if history_games.empty:
            return states

        games = history_games.copy()
        games = games[games["is_final"] == True]  # noqa: E712
        if games.empty:
            return states

        games["game_dt"] = games["game_date"].map(self._parse_dt)
        games = games.sort_values("game_dt")

        for _, row in games.iterrows():
            if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
                continue

            home_team = row["home_team"]
            away_team = row["away_team"]
            home_score = float(row["home_score"])
            away_score = float(row["away_score"])
            game_dt = row["game_dt"]

            home_state = states[home_team]
            away_state = states[away_team]

            home_win = int(home_score > away_score)
            away_win = 1 - home_win
            mov = home_score - away_score

            self._update_elo(home_state, away_state, home_win, mov)

            home_state.wins_recent.append(home_win)
            away_state.wins_recent.append(away_win)
            home_state.points_for_recent.append(home_score)
            home_state.points_against_recent.append(away_score)
            away_state.points_for_recent.append(away_score)
            away_state.points_against_recent.append(home_score)
            game_pace_proxy = home_score + away_score
            home_state.pace_recent.append(game_pace_proxy)
            away_state.pace_recent.append(game_pace_proxy)
            home_state.last_game_dt = game_dt
            away_state.last_game_dt = game_dt

        return states

    def features_for_matchups(self, events: pd.DataFrame, team_states: dict[str, TeamState]) -> pd.DataFrame:
        """Build model features for each upcoming event from odds feed."""

        rows: list[dict[str, float | str]] = []
        if events.empty:
            return pd.DataFrame(rows)

        for event_id, group in events.groupby("event_id", sort=False):
            first = group.iloc[0]
            home_team = first["home_team"]
            away_team = first["away_team"]
            commence_dt = self._parse_dt(first["commence_time"])

            home_state = team_states[home_team]
            away_state = team_states[away_team]

            rows.append(
                {
                    "event_id": event_id,
                    "sport_key": first["sport_key"],
                    "commence_time": first["commence_time"],
                    "home_team": home_team,
                    "away_team": away_team,
                    "elo_home": home_state.elo,
                    "elo_away": away_state.elo,
                    "elo_diff_home_minus_away": home_state.elo - away_state.elo,
                    "form_home": self._safe_mean(home_state.wins_recent),
                    "form_away": self._safe_mean(away_state.wins_recent),
                    "form_diff": self._safe_mean(home_state.wins_recent)
                    - self._safe_mean(away_state.wins_recent),
                    "points_for_home": self._safe_mean(home_state.points_for_recent),
                    "points_against_home": self._safe_mean(home_state.points_against_recent),
                    "points_for_away": self._safe_mean(away_state.points_for_recent),
                    "points_against_away": self._safe_mean(away_state.points_against_recent),
                    "net_rating_form_home": self._safe_mean(home_state.points_for_recent)
                    - self._safe_mean(home_state.points_against_recent),
                    "net_rating_form_away": self._safe_mean(away_state.points_for_recent)
                    - self._safe_mean(away_state.points_against_recent),
                    "pace_home": self._safe_mean(home_state.pace_recent),
                    "pace_away": self._safe_mean(away_state.pace_recent),
                    "pace_diff": self._safe_mean(home_state.pace_recent)
                    - self._safe_mean(away_state.pace_recent),
                    "rest_days_home": self._days_since(home_state.last_game_dt, commence_dt),
                    "rest_days_away": self._days_since(away_state.last_game_dt, commence_dt),
                    "rest_advantage_home": self._days_since(home_state.last_game_dt, commence_dt)
                    - self._days_since(away_state.last_game_dt, commence_dt),
                }
            )

        return pd.DataFrame(rows)
