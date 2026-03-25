"""HTTP helpers with error handling for API calls."""

from __future__ import annotations

import logging
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


class APIRequestError(RuntimeError):
    """Raised when external API requests fail."""


def get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 20) -> Any:
    """Run a GET request and return JSON, raising explicit errors on failure."""

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        LOGGER.exception("HTTP request failed. url=%s params=%s", url, params)
        raise APIRequestError(f"Request failed for {url}") from exc
