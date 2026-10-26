#!/usr/bin/env sh
set -eu
cmake -S . -B build -G Ninja \
    -DENABLE_OPENMP=ON \
    -DENABLE_MPI=ON \
    -DENABLE_CUDA="${ENABLE_CUDA:-OFF}" \
    -DBUILD_TESTING=ON
cmake --build build --parallel
