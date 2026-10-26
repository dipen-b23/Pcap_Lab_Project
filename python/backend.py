"""Safe subprocess integration for the native denoising backends."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULTS = {
    "window": 9,
    "sigma_spatial": 3.0,
    "sigma_range": 25.0,
    "noise_sigma": 25.0,
    "seed": 42,
    "threads": 4,
    "cpu_percent": 25,
    "repetitions": 5,
}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    result = DEFAULTS | config
    window = int(result["window"])
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and at least 3")
    for key in ("sigma_spatial", "sigma_range"):
        if float(result[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(result["noise_sigma"]) < 0 or int(result["threads"]) < 1:
        raise ValueError("noise_sigma must be non-negative and threads positive")
    if not 0 <= int(result["cpu_percent"]) <= 100:
        raise ValueError("cpu_percent must be between 0 and 100")
    return result


def build_command(
    executable: Path,
    action: str,
    config: dict[str, Any],
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    clean_path: Path | None = None,
    noisy_path: Path | None = None,
    csv_path: Path | None = None,
) -> list[str]:
    c = validate_config(config)
    command = [
        str(executable),
        action,
        "--mode",
        str(c.get("mode", "serial")),
        "--window",
        str(c["window"]),
        "--sigma-spatial",
        str(c["sigma_spatial"]),
        "--sigma-range",
        str(c["sigma_range"]),
        "--noise-sigma",
        str(c["noise_sigma"]),
        "--seed",
        str(c["seed"]),
        "--threads",
        str(c["threads"]),
        "--cpu-percent",
        str(c["cpu_percent"]),
        "--repetitions",
        str(c["repetitions"]),
        "--width",
        str(c.get("width", 256)),
        "--height",
        str(c.get("height", 256)),
    ]
    for flag, path in (
        ("--input", input_path),
        ("--output", output_path),
        ("--clean-output", clean_path),
        ("--noisy-output", noisy_path),
        ("--csv", csv_path),
    ):
        if path is not None:
            command.extend((flag, str(path)))
    return command


def run_backend(command: list[str], timeout: int = 300) -> dict[str, Any]:
    if not Path(command[0]).is_file():
        raise FileNotFoundError(f"backend executable not found: {command[0]}")
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip() or f"backend exited {completed.returncode}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("backend returned no JSON")
    return json.loads(lines[-1])


def backend_info(executable: Path) -> dict[str, Any]:
    return run_backend([str(executable), "info"], timeout=15)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to export")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
