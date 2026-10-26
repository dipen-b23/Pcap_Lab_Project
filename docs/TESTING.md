# Testing

Run from the repository root after building (see the [README](../README.md#build)):

```powershell
ctest --test-dir build --output-on-failure
python -m pytest tests/python -q
.\build\denoise.exe validate --mode openmp --threads 4 --width 256 --height 256
```

The October 15, 2026 Windows audit recorded CTest 1/1 passed and pytest 7/7 passed. That audit’s validation reported zero maximum/mean/RMSE error, 100% within `1e-3`, and 100% exact 8-bit agreement.

On Linux, replace `.\build\denoise.exe` with `./build/denoise`. Test counts may change as the suite grows. The native Python integration test currently discovers Windows `.exe` binaries; run CTest and CLI validation to check a Linux build.

C tests cover constant, one-pixel/tiny boundary behavior, impulse behavior, invalid windows/sigmas/dimensions, deterministic noise, MSE/PSNR, 8-bit validation, serial execution, OpenMP execution, and allocation cleanup. Python tests cover configuration, safe command construction, missing executables, JSON parsing, failures, aggregation, metrics, and CSV/JSON export.

MPI, CUDA, and hybrid were not verified in that audit. Testing them requires compatible toolchains and hardware. On an equipped machine, build the targets, compare float output to serial at tolerance `1e-3`, test MPI process counts 1/2/4 including height smaller than process count, and test hybrid shares 0/25/50/75/100.
