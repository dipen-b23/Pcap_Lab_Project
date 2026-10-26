#!/usr/bin/env sh
set -eu
mode="${1:-serial}"
size="${2:-512}"
threads="${3:-4}"
repetitions="${4:-5}"
./build/denoise benchmark \
    --mode "$mode" \
    --width "$size" \
    --height "$size" \
    --threads "$threads" \
    --repetitions "$repetitions" \
    --csv "results/benchmark_$mode.csv"
