#!/usr/bin/env sh
set -eu
mode="${1:-serial}"; size="${2:-256}"; threads="${3:-4}"
./build/denoise denoise --mode "$mode" --width "$size" --height "$size" --threads "$threads" --clean-output results/clean.pgm --noisy-output results/noisy.pgm --output "results/$mode.pgm"
