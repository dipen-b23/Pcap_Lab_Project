import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "python"))
from benchmark_analysis import add_performance_columns, aggregate
from metrics import quality


def test_aggregate_and_speedup():
    frame = pd.DataFrame(
        [
            {"mode": "serial", "wall_ms": 10, "width": 10, "height": 10, "threads": 1},
            {"mode": "serial", "wall_ms": 12, "width": 10, "height": 10, "threads": 1},
        ]
    )
    assert aggregate(frame).iloc[0]["median"] == 11
    assert add_performance_columns(frame, 20).iloc[0]["speedup"] == 2


def test_metrics():
    pytest.importorskip("skimage")
    image = np.full((8, 8), 10, dtype=np.uint8)
    result = quality(image, image)
    assert result["mse"] == 0 and np.isinf(result["psnr"]) and result["ssim"] == 1
