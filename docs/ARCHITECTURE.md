# Architecture

`Image` owns a contiguous `float` buffer in `[0,255]`; `image_free` releases it. Filter functions allocate independent output buffers and never mutate input. The CLI prepares one clean image and one seeded noisy buffer, then passes that same buffer to the selected implementation. File I/O and quality calculation are outside kernel timing.

The native layers are image/PNM I/O, deterministic noise, timing, metrics/validation, execution kernels, and CLI orchestration. CMake always builds serial, conditionally compiles OpenMP, creates MPI only when MPI C is found, and enables CUDA only on explicit request with `nvcc` available. Stub functions return `DENOISE_UNAVAILABLE`; they do not compute fallback output.

MPI rank 0 loads input. `Scatterv` distributes uneven contiguous row blocks. Ranks exchange requests for exactly the non-local rows in their filter radius using `Alltoallv`, return immutable row payloads, compute local output, and `Gatherv` to rank 0. Empty partitions are supported.

CUDA maps one output pixel to one 2D thread. Hybrid splits output rows by configured CPU percentage; two OpenMP sections overlap an OpenMP CPU-region calculation and a CUDA GPU-region kernel, writing disjoint output ranges.

Python uses safe `subprocess.run` argument lists and parses the final JSON line. Temporary per-run directories prevent collisions. Pillow converts uploaded formats to grayscale PGM. Pandas/Matplotlib provide analysis and scikit-image computes SSIM with `data_range=255`.
