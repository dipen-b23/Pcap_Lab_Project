# Parallel Image Denoising Using Bilateral Filtering

A C11-first academic implementation of bilateral image denoising with serial CPU, OpenMP, MPI, CUDA, and concurrent OpenMP+CUDA source implementations. Optional targets are capability-gated: an unavailable MPI runtime or CUDA device is reported, never replaced by serial work.

## Verified Status

| Mode | Source | Built | Executed here |
|---|---:|---:|---:|
| Serial C11 | Yes | Yes | Yes |
| OpenMP C11 | Yes | Yes | Yes, 1/2/4/8-thread-capable runtime |
| MPI C11 | Yes | No | No: MPI SDK/runtime absent |
| CUDA C | Yes | No | No: CUDA Toolkit/GPU absent |
| OpenMP + CUDA | Yes | No | No: CUDA Toolkit/GPU absent |

See `docs/ENVIRONMENT.md` for exact machine details and `results/` for measured outputs.

## Build

Windows PowerShell from this directory:

```powershell
python -m pip install -r requirements.txt
cmake -S . -B build -G Ninja -DENABLE_OPENMP=ON -DENABLE_MPI=ON -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

This machine's verified binary is in `build-verified/`. Enable CUDA only with an NVIDIA-supported host compiler and toolkit: `-DENABLE_CUDA=ON`. MinGW is not assumed compatible with CUDA; on Windows, use the MSVC version supported by the installed CUDA Toolkit. Install MS-MPI SDK and runtime to build `denoise_mpi`.

Linux uses the same CMake options or `scripts/build.sh` with GCC/Clang, an MPI development package, and optionally CUDA.

## CLI

```powershell
build-verified\denoise.exe info
build-verified\denoise.exe denoise --mode serial --width 256 --height 256 --output results\serial.pgm --clean-output results\clean.pgm --noisy-output results\noisy.pgm
build-verified\denoise.exe denoise --mode openmp --threads 4 --width 256 --height 256 --output results\openmp.pgm
build-verified\denoise.exe validate --mode openmp --threads 4 --width 256 --height 256
build-verified\denoise.exe benchmark --mode openmp --threads 4 --width 512 --height 512 --repetitions 5 --csv results\benchmark_openmp.csv
```

Accepted native inputs are PNM `P2`, `P5`, and `P6`. The dashboard accepts PNG/JPEG and converts consistently to grayscale PGM with Pillow. Without `--input`, the CLI creates a deterministic synthetic edge/gradient image. Defaults are window 9, spatial sigma 3, range sigma 25, noise sigma 25, and seed 42.

MPI, after building its optional target:

```powershell
mpiexec -n 4 build\denoise_mpi.exe input.pgm results\mpi.pgm 9 3 25
```

## Dashboard

```powershell
python -m streamlit run python\dashboard.py --server.headless true
```

The dashboard invokes the native executable with argument lists, displays only generated images, handles backend errors/timeouts, plots timings, and exports CSV/configuration/summary files. MPI is launched separately because MPI initialization must cover the whole process.

## Tests And Analysis

```powershell
ctest --test-dir build-verified --output-on-failure
python -m pytest tests\python -q
python python\export_results.py
```

Hardware-dependent modes are skipped at CMake target level when dependencies are absent. No DIV2K download is required. Large 4096×4096/window-15 sweeps are intentionally not launched automatically; use the benchmark scripts after checking memory and expected runtime.

## Layout

- `include/denoise.h`: owned image buffers, configuration, timing, validation, and stable C API.
- `src/serial`, `src/openmp`, `src/mpi`, `src/cuda`, `src/hybrid`: five execution modes.
- `python/`: safe backend integration, SSIM/analysis, exports, and Streamlit UI.
- `tests/`: native and Python tests using deterministic synthetic images.
- `docs/`: design, methods, environment, testing, benchmarks, and final report.
- `results/`: actual outputs from this machine.

## Troubleshooting

- `cannot find -lpthread`: install the MinGW pthread development package, or use a current MSYS2/MinGW-w64 toolchain.
- MPI target skipped: install an MPI implementation plus headers, then reconfigure CMake.
- CUDA target skipped: install CUDA Toolkit and select a compatible MSVC host compiler.
- Dashboard cannot find backend: enter the absolute executable path in the sidebar.
