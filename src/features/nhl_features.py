"""Feature engineering for NHL moneyline predictions."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


@dataclass
class TeamState:
    """Mutable team state used to generate model features."""

    elo: float
    wins_recent: deque[int]
    goals_for_recent: deque[int]
    goals_against_recent: deque[int]
    last_game_dt: datetime | None


class NHLFeatureBuilder:
    """Build pre-game matchup features from historical NHL games."""

    def __init__(
        self,
        elo_base: float,
        elo_k_factor: float,
        home_ice_advantage: float,
        recent_form_games: int,
    ) -> None:
        self.elo_base = elo_base
        self.elo_k_factor = elo_k_factor
        self.home_ice_advantage = home_ice_advantage
        self.recent_form_games = recent_form_games

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

    @staticmethod
    def _safe_mean(values: deque[int]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _days_since(last_game_dt: datetime | None, now_dt: datetime) -> float:
        if last_game_dt is None:
            return 7.0
        return max((now_dt - last_game_dt).total_seconds() / 86400.0, 0.0)

    def _expected_home_win(self, home_elo: float, away_elo: float) -> float:
        diff = (home_elo + self.home_ice_advantage) - away_elo
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def _update_elo(self, home_state: TeamState, away_state: TeamState, home_win: int) -> None:
        expected_home = self._expected_home_win(home_state.elo, away_state.elo)
        delta_home = self.elo_k_factor * (home_win - expected_home)
        home_state.elo += delta_home
        away_state.elo -= delta_home

    def build_team_states(self, history_games: pd.DataFrame) -> dict[str, TeamState]:
        """Build team rolling states from completed games history."""

        states: dict[str, TeamState] = defaultdict(
            lambda: TeamState(
                elo=self.elo_base,
                wins_recent=deque(maxlen=self.recent_form_games),
                goals_for_recent=deque(maxlen=self.recent_form_games),
                goals_against_recent=deque(maxlen=self.recent_form_games),
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
            home_team = row["home_team"]
            away_team = row["away_team"]
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
            game_dt = row["game_dt"]

            home_state = states[home_team]
            away_state = states[away_team]

            home_win = int(home_score > away_score)
            away_win = 1 - home_win

            self._update_elo(home_state, away_state, home_win)

            home_state.wins_recent.append(home_win)
            away_state.wins_recent.append(away_win)

            home_state.goals_for_recent.append(home_score)
            home_state.goals_against_recent.append(away_score)
            away_state.goals_for_recent.append(away_score)
            away_state.goals_against_recent.append(home_score)

            home_state.last_game_dt = game_dt
            away_state.last_game_dt = game_dt

        return states

    def features_for_matchups(self, events: pd.DataFrame, team_states: dict[str, TeamState]) -> pd.DataFrame:
        """Create model features for upcoming odds events."""

        rows = []
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
                    "goals_for_home": self._safe_mean(home_state.goals_for_recent),
                    "goals_against_home": self._safe_mean(home_state.goals_against_recent),
                    "goals_for_away": self._safe_mean(away_state.goals_for_recent),
                    "goals_against_away": self._safe_mean(away_state.goals_against_recent),
                    "goal_diff_form_home": self._safe_mean(home_state.goals_for_recent)
                    - self._safe_mean(home_state.goals_against_recent),
                    "goal_diff_form_away": self._safe_mean(away_state.goals_for_recent)
                    - self._safe_mean(away_state.goals_against_recent),
                    "rest_days_home": self._days_since(home_state.last_game_dt, commence_dt),
                    "rest_days_away": self._days_since(away_state.last_game_dt, commence_dt),
                    "rest_advantage_home": self._days_since(home_state.last_game_dt, commence_dt)
                    - self._days_since(away_state.last_game_dt, commence_dt),
                }
            )

        return pd.DataFrame(rows)
