param(
    [switch]$Cuda
)

$ErrorActionPreference = 'Stop'
$cudaValue = if ($Cuda) { 'ON' } else { 'OFF' }
cmake -S . -B build -G Ninja `
    -DENABLE_OPENMP=ON `
    -DENABLE_MPI=ON `
    -DENABLE_CUDA=$cudaValue `
    -DBUILD_TESTING=ON
cmake --build build --parallel
