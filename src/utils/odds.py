"""Odds conversion and no-vig normalization helpers."""

from __future__ import annotations


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""

    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    return (100.0 / abs(american_odds)) + 1.0


def american_to_implied_probability(american_odds: float) -> float:
    """Convert American odds to implied probability including vig."""

    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vig from a two-way market by proportional normalization."""

    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total
