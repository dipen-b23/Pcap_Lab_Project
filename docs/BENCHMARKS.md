# Benchmarks

The benchmark performs one unmeasured warm-up and a configurable number of measured runs (five by default). Median, min, max, population standard deviation, and throughput are emitted. Core times exclude image generation, disk I/O, validation, and plotting. Speedup uses serial median; CPU efficiency divides speedup by OpenMP threads.

Recorded Windows audit result from October 15, 2026, synthetic 256×256 image, window 9, sigmas 3/25, noise sigma 25, seed 42:

| Mode | Threads | Median ms | Min ms | Max ms | Stddev ms | MPix/s | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Serial | 1 effective | 340.0014 | 312.9519 | 421.8110 | 38.4977 | 0.192752 | 1.000 | N/A |
| OpenMP | 4 | 115.2431 | 113.2899 | 119.4041 | 2.1776 | 0.568676 | 2.950 | 0.738 |

The measured OpenMP result is faster; no claim is made for other dimensions, windows, MPI, CUDA, or hybrid. The requested 512–4096/window-5/9/15 matrix remains pending because a full run is expensive and optional backends were absent in the recorded environment. Generate combined results and plots with `python python/export_results.py` after producing the input images and benchmark CSV files. The outputs are `results/benchmark_results.csv` and `results/benchmark_plot.png`.
