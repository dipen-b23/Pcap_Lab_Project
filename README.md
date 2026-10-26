# Parallel Image Denoising Using Bilateral Filtering

A parallel computing (PCAP) project implementing **bilateral image denoising** in C11 and CUDA, with five execution approaches:

- **Serial C11** — single-process CPU reference.
- **OpenMP** — shared-memory, multithreaded CPU execution.
- **MPI** — multi-process CPU execution with row partitioning and halo exchange.
- **CUDA** — NVIDIA GPU acceleration.
- **Hybrid OpenMP + CUDA** — concurrent CPU/GPU execution with a configurable workload split.

The optional MPI and CUDA components are enabled only when the required toolchains are available. Unavailable acceleration modes are not silently replaced with serial execution.

## Verified Environments and Status

### Ubuntu laptop — verified October 26–27, 2026

The following modes **built and executed successfully** on the project's Ubuntu 22.04 laptop:

| Mode | Built | Executed | Notes |
|---|---|---|---|
| Serial C11 | Yes | Yes | Baseline execution |
| OpenMP C11 | Yes | Yes | Tested with 4 threads |
| MPI C11 | Yes | Yes | Tested with 4 processes |
| CUDA | Yes | Yes | NVIDIA RTX 4060 Laptop GPU |
| Hybrid OpenMP + CUDA | Yes | Yes | Tested with CPU shares from 1% to 75% |

**Observed environment:** Ubuntu 22.04, GCC 11.4.0, Open MPI 4.1.2, CUDA Toolkit (`nvcc`) 11.5, NVIDIA driver 580.178.04, RTX 4060 Laptop GPU (8 GB), and CMake 3.22.1. `nvidia-smi` reported driver-supported CUDA version 13.0; this is **not** the installed CUDA Toolkit version.

The OpenMP, CUDA, and Hybrid validation commands all passed with 100% of 8-bit output pixels matching the serial reference. MPI executed successfully but was not tested using the same `validate` command (its CLI is separate).

**Historical Windows audit (October 15, 2026):** Serial and OpenMP were built and run. MPI and CUDA/Hybrid were **not** built or executed in that environment because their SDK/runtime or GPU prerequisites were absent. See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for that machine's original audit. Its historical status does not describe the later Ubuntu verification above.

## Requirements

- CMake 3.18+
- C11 compiler (GCC/Clang on Linux; a supported compiler on Windows)
- OpenMP-capable toolchain for OpenMP mode
- MPI implementation with C development headers and runtime for MPI mode
- NVIDIA GPU, working driver, CUDA Toolkit, and toolkit-compatible host compiler for CUDA/Hybrid modes
- Python 3 and dependencies in [`requirements.txt`](requirements.txt) for the dashboard and Python analysis

### Ubuntu dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build libopenmpi-dev openmpi-bin python3-venv
```

Install the CUDA Toolkit and NVIDIA driver appropriate for your GPU separately if not already installed. Verify:

```bash
gcc --version
mpicc --version
mpirun --version
nvcc --version
nvidia-smi
cmake --version
```

> **RTX 4060 + CUDA 11.5:** The verified Ubuntu build used `-DCMAKE_CUDA_ARCHITECTURES=86`, and CUDA execution succeeded on this laptop. CUDA 11.5 does not directly support the RTX 4060's native `sm_89` target; for native architecture compilation, use a newer supported CUDA Toolkit.

## Build on Ubuntu (all five implementations)

Run from the repository root:

```bash
cmake -S . -B build \
  -DENABLE_OPENMP=ON \
  -DENABLE_MPI=ON \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=ON

cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

The `86` architecture setting documents the configuration actually used for the Ubuntu/RTX 4060 + CUDA 11.5 verification. Choose an architecture supported by **your** toolkit and GPU on other machines.

Expected executables:

| Executable | Modes |
|---|---|
| `build/denoise` | Serial, OpenMP |
| `build/denoise_mpi` | MPI (separate CLI) |
| `build/denoise_cuda` | Serial, OpenMP, CUDA, Hybrid |

**Porting note:** The originally distributed ZIP required two local source/build fixes to compile CUDA on Ubuntu: move `dim3 block`/`grid` declarations in `src/cuda/bilateral_cuda.cu` before any `goto` that would bypass their initialization, and add the generated-header include directory to `denoise_cuda` in `CMakeLists.txt`:

```cmake
if(TARGET denoise_cuda)
    target_include_directories(denoise_cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
endif()
```

Those fixes were applied in the successful local Ubuntu build. **If you are using the original unmodified ZIP, apply them before expecting the full CUDA build to succeed.** This README update alone does not modify the source code or CMake configuration.

### Windows build (original, CPU-focused configuration)

```powershell
python -m pip install -r requirements.txt
cmake -S . -B build -G Ninja -DENABLE_OPENMP=ON -DENABLE_MPI=ON -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

MPI on Windows requires its development SDK/runtime. CUDA requires an NVIDIA GPU, compatible CUDA Toolkit, and supported host compiler. The original Windows audit did **not** verify these modes.

## Running the Implementations (Ubuntu)

Create the output directory:

```bash
mkdir -p results
```

Unless specified otherwise, the CLI generates a deterministic synthetic edge/gradient image with noise. Default filtering settings: **window 9**, **spatial sigma 3**, **range sigma 25**, **noise sigma 25**, and **seed 42**.

### 1. Serial

```bash
./build/denoise denoise --mode serial \
  --width 512 --height 512 \
  --output results/serial.pgm \
  --clean-output results/clean.pgm \
  --noisy-output results/noisy.pgm
```

### 2. OpenMP (four threads)

```bash
./build/denoise denoise --mode openmp --threads 4 \
  --width 512 --height 512 \
  --output results/openmp.pgm
```

### 3. MPI (four processes)

The MPI executable accepts positional arguments, **not** `denoise --mode mpi`. First create `results/clean.pgm` using the Serial example above, then run:

```bash
mpirun -np 4 ./build/denoise_mpi \
  results/clean.pgm results/mpi.pgm 9 3 25
```

MPI partitions the image into rows and exchanges halo rows between neighboring processes before gathering the filtered result. The three numbers are the window size, spatial sigma, and range sigma. The MPI executable's input/noise preparation and timing scope should be matched to those of the other executables for rigorous performance comparisons.

### 4. CUDA

```bash
./build/denoise_cuda denoise --mode cuda \
  --width 512 --height 512 \
  --output results/cuda.pgm
```

### 5. Hybrid (OpenMP + CUDA)

```bash
./build/denoise_cuda denoise --mode hybrid \
  --threads 4 --cpu-percent 1 \
  --width 512 --height 512 \
  --output results/hybrid.pgm
```

`--cpu-percent` specifies the percentage of image rows assigned to the CPU. **1% was the fastest hybrid split tested on the Ubuntu laptop at 512×512**, not a universal optimum.

### Native capability checks

```bash
./build/denoise info
./build/denoise_cuda info
```

In the verified Ubuntu build, `denoise_cuda info` reported Serial, OpenMP, CUDA, and Hybrid runtime support. It reported MPI as `compiled: true, runtime: false` because **MPI uses `denoise_mpi` as a separate executable**; MPI itself was verified via `mpirun`.

## Recorded Results (Single Runs)

These are **observed single-run measurements**, not statistically robust final benchmark averages. Times are in milliseconds.

| Mode | Configuration | 512×512 | 1024×1024 |
|---|---|---:|---:|
| Serial | Baseline | 1117.15 | 4544.98 |
| OpenMP | 4 threads | 222.34 | 909.45 |
| MPI | 4 processes | 286.43 | 1139.38 |
| CUDA | RTX 4060 | 144.48 | 141.54 |
| Hybrid | 1% CPU / 99% GPU, 4 threads | 152.66 | 182.92 |

At 1024×1024, a **separate** Serial run took 3286.23 ms, illustrating run-to-run variation. Avoid claiming definitive speedups from these single runs. CUDA wall time includes overhead not captured by its much smaller kernel-only time; compare equivalent wall-time boundaries for performance claims.

### Hybrid workload sweep (512×512)

| CPU share | GPU share | Wall time (ms) |
|---:|---:|---:|
| 1% | 99% | 152.66 |
| 2% | 98% | 155.35 |
| 5% | 95% | 187.90 |
| 10% | 90% | 237.69 |
| 25% | 75% | 379.55 |
| 50% | 50% | 614.20 |
| 75% | 25% | 855.98 |

### Validation results

```bash
./build/denoise validate --mode openmp --threads 4
./build/denoise_cuda validate --mode cuda
./build/denoise_cuda validate --mode hybrid --threads 4 --cpu-percent 25
```

All three returned `status: pass`, with **100% exact 8-bit pixel matches** to the serial reference. Small floating-point deviations (up to approximately `9.16e-05` for CUDA/Hybrid) were within the configured tolerance of `0.001`.

## Benchmarks

Run multiple repetitions before drawing final conclusions:

```bash
./build/denoise benchmark --mode serial \
  --width 1024 --height 1024 --repetitions 5 \
  --csv results/benchmark_serial.csv

./build/denoise benchmark --mode openmp --threads 4 \
  --width 1024 --height 1024 --repetitions 5 \
  --csv results/benchmark_openmp.csv

./build/denoise_cuda benchmark --mode cuda \
  --width 1024 --height 1024 --repetitions 5 \
  --csv results/benchmark_cuda.csv

./build/denoise_cuda benchmark --mode hybrid --threads 4 --cpu-percent 1 \
  --width 1024 --height 1024 --repetitions 5 \
  --csv results/benchmark_hybrid.csv
```

MPI has a different CLI and should be timed over repeated separate `mpirun` invocations. Use the same image, filter settings, noise settings, and comparable timing boundaries across modes. Distinguish GPU **kernel time** from **end-to-end wall time**.

## Streamlit Dashboard

The dashboard currently supports **Serial, OpenMP, CUDA, and Hybrid**. **MPI is not selectable in the dashboard** because it has a separate executable and invocation pattern. Run MPI from the terminal as shown above.

### Python setup and launch (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run python/dashboard.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

In the sidebar, set **Backend executable** to the full path of `build/denoise_cuda`, for example:

```text
/home/dipen_b23/Desktop/parallel-image-denoising/build/denoise_cuda
```

This enables the four dashboard modes when their runtime checks succeed. Selecting `build/denoise` instead usually shows only Serial and OpenMP.

The dashboard provides image upload (PNG/JPEG converted to grayscale PGM), synthetic image generation, execution/validation/benchmark controls, execution-time displays, plots, and results export.

**Dashboard limitation in the original ZIP:** Its hybrid CPU-share selector offers only **0, 25, 50, 75, 100%**. To allow 1% or any other share, change the `cpu_percent` widget in `python/dashboard.py` to:

```python
cpu_percent = st.sidebar.slider("Hybrid CPU share (%)", min_value=0, max_value=100, value=1, step=1)
```

## Tests and Analysis

```bash
ctest --test-dir build --output-on-failure
python -m pytest tests/python -q
python python/export_results.py
```

Before exporting analysis results, generate `results/clean.pgm` and the appropriate comparison outputs. Plots using benchmark CSVs require those files to have been generated first. Large image/filter sweeps are intentionally not run automatically.

## Repository Layout

| Path | Purpose |
|---|---|
| `include/denoise.h` | Image buffers, filter configuration, timing, validation, public C API |
| `src/serial/` | Serial bilateral filtering |
| `src/openmp/` | OpenMP multithreaded filtering |
| `src/mpi/` | MPI row partitioning and halo exchange |
| `src/cuda/` | CUDA GPU filtering |
| `src/hybrid/` | Concurrent OpenMP + CUDA filtering |
| `python/` | Dashboard, backend integration, analysis, exports |
| `tests/` | Native and Python tests |
| `docs/` | Architecture, methods, historical environment report, testing and benchmarks |
| `results/` | Generated outputs and benchmark data |

## Troubleshooting

- **`build_config.h: No such file or directory` while building `denoise_cuda`:** Add the generated-header include directory for `denoise_cuda` in `CMakeLists.txt` (see Build section), then rerun CMake and rebuild.
- **CUDA `transfer of control bypasses initialization`:** Declare/initialize `dim3 block` and `dim3 grid` before any error-handling `goto` that could skip them in `src/cuda/bilateral_cuda.cu`.
- **CUDA target skipped:** Check `nvcc --version`, CMake CUDA detection, and supported host compiler/toolkit combinations.
- **CUDA binary builds but does not run on GPU:** Check `nvidia-smi` and the target GPU architecture; `nvidia-smi` driver CUDA version is not the toolkit version.
- **MPI target skipped:** Install Open MPI (or another MPI implementation) with development headers, reconfigure, and rebuild.
- **Dashboard shows only Serial/OpenMP:** Set **Backend executable** to the absolute path of `build/denoise_cuda`.
- **MPI absent from dashboard:** This is expected in the current application; launch `build/denoise_mpi` with `mpirun`.
- **`cannot find -lpthread` (Windows/MinGW):** Install an appropriate pthread toolchain/development package or use a supported MSYS2/MinGW-w64 environment.
- **Stale build cache from another OS:** Delete the old build directory (`rm -rf build` on Linux) and configure again on the current machine.

## Reproducibility Notes

The Ubuntu verification results reflect successful local changes to CUDA source/CMake configuration that were necessary after unpacking the original ZIP. They are **not** evidence that the unmodified ZIP compiles CUDA without changes. Results may vary across CPU/GPU, toolchain versions, filter parameters, warm-up, and system load. For publication-quality numbers, report the full test configuration, repeat runs, and use medians or another stated statistic.
