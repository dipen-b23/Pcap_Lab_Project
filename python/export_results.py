"""Create quality JSON, combined benchmark CSV, and measured-result plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from metrics import quality

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
clean = np.asarray(Image.open(RESULTS / "clean.pgm"))
report = {}
for name in ("noisy", "serial", "openmp", "mpi", "cuda", "hybrid"):
    path = RESULTS / f"{name}.pgm"
    if path.is_file():
        report[name] = quality(clean, np.asarray(Image.open(path)))
(RESULTS / "quality_metrics.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

frames = [pd.read_csv(path) for path in sorted(RESULTS.glob("benchmark_*.csv"))]
if frames:
    frame = pd.concat(frames, ignore_index=True)
    serial = frame.loc[frame["mode"] == "serial", "median_ms"]
    baseline = float(serial.iloc[0]) if not serial.empty else float("nan")
    frame["speedup"] = baseline / frame["median_ms"]
    frame["efficiency"] = np.where(
        frame["mode"] == "openmp", frame["speedup"] / frame["threads"], np.nan
    )
    frame.to_csv(RESULTS / "benchmark_results.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    frame.plot.bar(
        x="mode",
        y="median_ms",
        legend=False,
        ax=axes[0],
        color=["#365486", "#dc6b19"][: len(frame)],
    )
    frame.plot.bar(
        x="mode",
        y="throughput_mpix_s",
        legend=False,
        ax=axes[1],
        color=["#365486", "#dc6b19"][: len(frame)],
    )
    axes[0].set_ylabel("Median time (ms)")
    axes[1].set_ylabel("Throughput (MPix/s)")
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_plot.png", dpi=160)
    plt.close(fig)
