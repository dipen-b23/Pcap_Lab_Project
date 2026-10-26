# Benchmarks

The benchmark performs one unmeasured warm-up and five measured runs. Median, min, max, population standard deviation, and throughput are emitted. Core times exclude image generation, disk I/O, validation, and plotting. Speedup uses serial median; CPU efficiency divides speedup by OpenMP threads.

Actual 2026-10-15 result, synthetic 256×256 image, window 9, sigmas 3/25, noise sigma 25, seed 42:

| Mode | Threads | Median ms | Min ms | Max ms | Stddev ms | MPix/s | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Serial | 1 effective | 340.0014 | 312.9519 | 421.8110 | 38.4977 | 0.192752 | 1.000 | N/A |
| OpenMP | 4 | 115.2431 | 113.2899 | 119.4041 | 2.1776 | 0.568676 | 2.950 | 0.738 |

The measured OpenMP result is faster; no claim is made for other dimensions, windows, MPI, CUDA, or hybrid. The requested 512–4096/window-5/9/15 matrix remains pending because a full run is expensive and optional backends are absent. CSV and plot: `results/benchmark_results.csv` and `results/benchmark_plot.png`.
