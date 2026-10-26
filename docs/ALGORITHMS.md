# Algorithms

For center pixel `p` and clamped neighbor `q`, every mode computes

`w = exp(-distance_squared/(2*sigma_s^2)) * exp(-(I(p)-I(q))^2/(2*sigma_r^2))`

and `J(p) = sum(w*I(q))/sum(w)`. The square window is odd and at least 3. Serial/OpenMP precompute spatial terms. A non-positive or non-finite weight sum preserves the center pixel. Saving clamps and rounds with `floor(value + 0.5)`.

Noise uses xorshift64* with multiplier `2685821657736338717`. The upper 53 bits form open-interval uniform values and Box-Muller produces paired Gaussian samples. Noise is generated once and clipped before any mode runs.

OpenMP applies `parallel for schedule(static)` to rows. Every accumulator and neighborhood index is thread-local. MPI uses uneven contiguous blocks plus requested immutable halo rows. CUDA uses 16×16 blocks with bounds guards. Hybrid endpoints delegate to the exact OpenMP/CUDA mode; intermediate shares concurrently process disjoint row regions while reading the full immutable source needed for boundaries.
