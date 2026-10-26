# Parallel Image Denoising Using Bilateral Filtering

## Abstract

This mini-project implements edge-preserving grayscale denoising in C11 across serial, OpenMP, MPI, CUDA, and heterogeneous OpenMP+CUDA designs. A capability-aware CLI, Python dashboard, deterministic noise pipeline, validation, quality metrics, tests, and benchmark exports support reproducible study. On the available Windows machine, serial and OpenMP were verified; missing MPI/CUDA dependencies are explicitly separated from successful execution.

## Problem And Objectives

Bilateral filtering is computationally costly because every output pixel evaluates spatial and intensity-range weights over a neighborhood. The objective is to preserve one mathematical model while exposing shared-memory, distributed-memory, GPU, and heterogeneous parallelism, then evaluate correctness, quality, and measured performance without fabricated results.

## Method

Pixels are float32 intensities. Out-of-range coordinates clamp to the closest edge. Gaussian noise is generated once with seeded xorshift64*/Box-Muller sampling. Serial is the canonical reference. OpenMP statically partitions rows. MPI scatters uneven row regions, exchanges requested halo rows, and gathers output. CUDA uses a 2D grid. Hybrid concurrently assigns disjoint top CPU and bottom GPU regions. See `ALGORITHMS.md` and `ARCHITECTURE.md`.

## Technology Stack

C11, GCC 6.3, OpenMP 4.5, CMake 4.4.4, Ninja 1.13.2, Python 3.12.4, Streamlit 1.66.0, NumPy, Pandas, Matplotlib, Pillow, scikit-image 0.26.0, CTest, and pytest 9.1.1. MPI/CUDA source exists but those toolchains were absent.

## Validation And Quality

OpenMP versus serial produced maximum error 0, RMSE 0, 100% tolerance agreement, and 100% exact saved-pixel agreement. For the actual 256×256 demonstration, noisy MSE/PSNR/SSIM were 596.897, 20.372 dB, and 0.254. Denoised serial and OpenMP metrics were 207.428, 24.962 dB, and 0.425. SSIM uses scikit-image with data range 255. Small differences between CLI float MSE and exported PGM MSE are due to documented 8-bit rounding.

## Performance And Discussion

Five-run medians were 340.001 ms serial and 115.243 ms OpenMP with four threads, yielding 2.950× speedup and 73.76% efficiency. This is a genuine measured gain, not a theoretical estimate. Runtime variability, old compiler age, scheduling, exponential cost, and the small test size limit generalization. Full tables are in `BENCHMARKS.md`.

## Testing

CTest passed 1/1 and pytest passed 7/7. A detected OpenMP race in neighborhood loop indices was fixed before measurements; rerun validation then achieved exact agreement. Dashboard startup was separately checked in headless mode.

## Limitations

MPI, CUDA, and hybrid are **NOT VERIFIED ON THIS MACHINE**. CUDA compilation additionally requires a Toolkit-supported MSVC host compiler on Windows. Native image I/O intentionally supports portable PNM; the dashboard handles common uploaded formats. The full 512–4096 benchmark matrix remains pending to avoid presenting partial or fabricated data as complete.

## Conclusion And Future Work

The verified OpenMP implementation preserves serial output and substantially reduces elapsed time on this machine. Future work is to execute MPI halo tests on multiple nodes, profile CUDA occupancy/transfers, tune hybrid partition share, complete the full benchmark matrix, and evaluate larger natural-image datasets such as externally supplied DIV2K.

## References

1. C. Tomasi and R. Manduchi, “Bilateral Filtering for Gray and Color Images,” Proceedings of ICCV, 1998.
2. OpenMP Architecture Review Board, OpenMP API Specification, https://www.openmp.org/specifications/.
3. MPI Forum, MPI Standard, https://www.mpi-forum.org/docs/.
4. NVIDIA, CUDA C++ Programming Guide, https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
5. scikit-image documentation, Structural Similarity Index, https://scikit-image.org/docs/stable/api/skimage.metrics.html.
