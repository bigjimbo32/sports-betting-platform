"""I/O helpers for writing timestamped CSV outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd



def utc_now_str() -> str:
    """Return UTC timestamp string for output filenames."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")



def write_csv(df: pd.DataFrame, output_dir: str, filename_prefix: str, run_ts: str) -> Path:
    """Write a DataFrame to CSV and return saved path."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{filename_prefix}_{run_ts}.csv"
    df.to_csv(file_path, index=False)
    return file_path
