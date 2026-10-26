# Testing

Verified commands:

```powershell
ctest --test-dir build-verified --output-on-failure
python -m pytest tests\python -q
build-verified\denoise.exe validate --mode openmp --threads 4 --width 256 --height 256
```

Result: CTest 1/1 passed and pytest 7/7 passed. Validation reported zero maximum/mean/RMSE error, 100% within `1e-3`, and 100% exact 8-bit agreement.

C tests cover constant, one-pixel/tiny boundary behavior, impulse behavior, invalid windows/sigmas/dimensions, deterministic noise, MSE/PSNR, 8-bit validation, serial execution, OpenMP execution, and allocation cleanup. Python tests cover configuration, safe command construction, missing executables, JSON parsing, failures, aggregation, metrics, and CSV/JSON export.

MPI, CUDA, and hybrid tests are pending compatible toolchains/hardware and were not reported as passes. On an equipped machine, build the targets, compare float output to serial at tolerance `1e-3`, test MPI process counts 1/2/4 including height smaller than process count, and test hybrid shares 0/25/50/75/100.
