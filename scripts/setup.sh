#!/usr/bin/env sh
set -eu
python3 -m pip install -r requirements.txt
printf '%s\n' 'Install an MPI implementation and NVIDIA CUDA Toolkit separately for optional modes.'
