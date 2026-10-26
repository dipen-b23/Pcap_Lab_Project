from __future__ import annotations

import pandas as pd


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"mode", "wall_ms"}
    if not required <= set(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    keys = [
        key
        for key in (
            "mode",
            "width",
            "height",
            "window",
            "threads",
            "processes",
            "cpu_percent",
        )
        if key in frame
    ]
    return (
        frame.groupby(keys, dropna=False)["wall_ms"]
        .agg(["median", "min", "max", "std"])
        .reset_index()
    )


def add_performance_columns(frame: pd.DataFrame, serial_ms: float) -> pd.DataFrame:
    result = frame.copy()
    result["speedup"] = serial_ms / result["wall_ms"]
    result["throughput_mpix_s"] = (
        result["width"] * result["height"] / (result["wall_ms"] * 1000.0)
    )
    result["efficiency"] = (
        result["speedup"] / result["threads"] if "threads" in result else float("nan")
    )
    return result
