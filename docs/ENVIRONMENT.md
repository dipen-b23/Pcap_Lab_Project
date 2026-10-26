# Detected Environment

Audit date: 2026-10-15.

| Component | Detected value | Status |
|---|---|---|
| OS | Microsoft Windows 10.0.22621, x64 | Available |
| RAM | 16,831,942,656 bytes (15.68 GiB) | Available |
| PowerShell | 5.1.22621.963 | Available |
| C compiler | MinGW.org GCC 6.3.0 | Available; old |
| CMake | 4.4.4, installed with pip during setup | Available |
| Ninja | 1.13.2, installed with pip | Available |
| Python | 3.12.4 | Available |
| OpenMP | `_OPENMP=201511`; GCC/libgomp | Built and executed |
| MPI | No `mpicc`, `mpiexec`, or MS-MPI installation | Unavailable |
| NVIDIA GPU/tooling | No `nvidia-smi`, `nvcc`, or detected NVSMI install | Unavailable |
| Python analysis | NumPy, Pandas, Matplotlib, Pillow | Available |
| Added Python packages | pytest 9.1.1, Streamlit 1.66.0, scikit-image 0.26.0 | Available |

The original MinGW OpenMP installation lacked `libpthread.a`; `mingw-get install mingw32-pthreads-w32-dev` repaired linking. The old `mingw32-make` also failed in this space-containing workspace, so verified builds use Ninja.

Serial and OpenMP are **available and executable**. MPI, CUDA, and hybrid are **unavailable because dependencies/hardware are missing**, not failed executions. Their source was not compiled or executed here and is marked **NOT VERIFIED ON THIS MACHINE**.

For MPI on Windows, install both Microsoft MPI runtime and SDK, open a fresh shell, and verify `mpiexec` plus MPI headers/libraries before reconfiguring. For CUDA, install a Toolkit version supporting the installed NVIDIA driver and a compatible Visual Studio/MSVC host compiler. MinGW GCC 6.3 is not presented as a CUDA-compatible host toolchain.
