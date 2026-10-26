from __future__ import annotations

import math

import numpy as np


def quality(reference: np.ndarray, result: np.ndarray) -> dict[str, float]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(result, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("images must have equal, non-empty shapes")
    mse = float(np.mean((a - b) ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)
    try:
        from skimage.metrics import structural_similarity

        ssim = float(structural_similarity(a, b, data_range=255.0))
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for SSIM") from exc
    return {"mse": mse, "psnr": psnr, "ssim": ssim}
