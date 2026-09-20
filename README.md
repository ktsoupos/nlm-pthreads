# Parallel Non-Local Means Denoising

University final project for a parallel computing / HPC course, cross-listed with a
digital image processing course. Implements Non-Local Means (NLM) image denoising in
C, then optimizes it in layers — threading, vectorization, and an algorithmic
reformulation — measuring each layer independently.

**Stack:** C (C17), pthreads, AVX2 intrinsics. Python only for tooling (noise
generation, metrics, plots), never linked into the benchmarked binary.

## Status

Early setup. Only image loading (via vendored `stb_image`) is wired up so far; the
NLM variants themselves are not yet implemented.

## Build

```sh
make            # native build (-march=native)
make clean
```

Requires a C17 compiler, pthreads, and AVX2 support.

## Run

```sh
./nlm
```

Loads `test/images/pengbrew.png` and prints its dimensions and channel count.
