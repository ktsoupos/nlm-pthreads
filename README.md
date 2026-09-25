# Parallel Non-Local Means Denoising

University final project for a parallel computing / HPC course, cross-listed with a
digital image processing course. Implements Non-Local Means (NLM) image denoising in
C, then optimizes it in layers — vectorization, threading, and an algorithmic
reformulation — measuring each layer independently.

**Stack:** C (C17), pthreads, AVX2 intrinsics. Python only for tooling (noise
generation, metrics, reference implementations, plots), never linked into the
benchmarked binary.

See [CLAUDE.md](CLAUDE.md) for the algorithm details and design rationale, and
[report/report.tex](report/report.tex) for the full write-up.

## Status

All six variants are implemented and validated. Every variant reproduces the
sequential baseline **bit for bit** (`max|diff| = 0`), verified across patch radii
1–8 and thread counts 1–256.

| Variant | Description |
| --- | --- |
| `seq` | Sequential baseline, scalar patch distance |
| `seq_simd` | Sequential, AVX2 patch distance |
| `pthreads` | Row-block threads, scalar |
| `pthreads_simd` | Row-block threads, AVX2 |
| `integral` | Separable rolling box sums, single-threaded |
| `integral_thr` | Separable rolling box sums, row-block threads |

Peak measured speedup: **37.2×** (`integral_thr`, 16 threads, i7-10870H,
512×512 image).

## Build

```sh
make            # native build (-march=native)
make cluster    # portable build, no -march=native, for a different CPU
make clean
```

Requires a C17 compiler, pthreads, and a CPU with AVX2 + FMA.

## Run

```sh
./nlm --in data/noisy/camera_sigma25.png \
      --out data/out/denoised.png \
      --clean data/clean/camera.png \
      --variant integral_thr --threads 16
```

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--in PATH` | *(required)* | Noisy input PNG |
| `--out PATH` | — | Write the denoised image |
| `--clean PATH` | — | Clean reference; enables PSNR reporting |
| `--variant NAME` | `seq` | One of the six variants above |
| `--threads N` | `1` | Worker threads (ignored by `seq`/`seq_simd`/`integral`) |
| `--patch R` | `3` | Patch radius; patch side is `2R+1` |
| `--search R` | `10` | Search-window radius |
| `--sigma S` | `25` | Noise standard deviation |
| `--h H` | `0.4*sigma` | Filter strength |
| `--repeat N` | `1` | Repeat the timed region, one CSV row per run |
| `--csv PATH` | — | Append results as CSV |
| `--check` | off | Also run `seq` and report `max|diff|` against it |

Only the compute region is timed; file I/O is excluded.

## Python tooling

Needs a virtualenv (the scripts are never linked into the benchmarked binary):

```sh
python3 -m venv .venv
.venv/bin/pip install numpy pillow scikit-image matplotlib
```

| Script | Purpose |
| --- | --- |
| `scripts/make_noisy.py` | Add synthetic Gaussian noise: `--in --out --sigma [--seed]` |
| `scripts/reference_exact.py` | Exact float64 reference of the documented formula — the authoritative correctness check |
| `scripts/reference_nlm.py` | Cross-check against `skimage.restoration.denoise_nl_means` |
| `scripts/metrics.py` | PSNR and difference statistics between images |
| `scripts/sweep.sh` | Benchmark matrix → `bench/results/<machine>.csv` |
| `scripts/plots.py` | Generate report figures from the CSVs |

Typical end-to-end run:

```sh
.venv/bin/python3 scripts/make_noisy.py --in data/clean/camera.png \
    --out data/noisy/camera_sigma25.png --sigma 25
./nlm --in data/noisy/camera_sigma25.png --clean data/clean/camera.png \
    --variant integral_thr --threads 16 --check
./scripts/sweep.sh laptop
.venv/bin/python3 scripts/plots.py
```

## Validating correctness

`scripts/reference_exact.py` implements the documented weighting formula directly in
float64 and is the authoritative check — any disagreement is a bug in the C code:

```sh
.venv/bin/python3 scripts/reference_exact.py --in data/noisy/camera_sigma25.png \
    --out /tmp/ref.png
./nlm --in data/noisy/camera_sigma25.png --out /tmp/ours.png --variant seq
.venv/bin/python3 scripts/metrics.py /tmp/ref.png /tmp/ours.png
```

`reference_nlm.py` compares against scikit-image, but that is a *different*
algorithm: its classic mode applies a Gaussian weighting across the patch, while
this implementation (and skimage's `--fast-mode`) weight patch pixels uniformly.
Small differences there are expected and do not indicate bugs.

## Report

LaTeX source in `report/report.tex`, figures in `report/figures/` (regenerate with
`scripts/plots.py`). Building the PDF needs a TeX installation:

```sh
sudo apt install texlive-latex-base texlive-latex-recommended texlive-fonts-recommended
cd report && pdflatex report.tex && pdflatex report.tex
```

## Layout

```
include/        image.h  nlm.h  patch.h  timer.h
src/            main.c  image.c  nlm.c  patch.c  timer.c
external/stb/   stb_image.h  stb_image_write.h   (vendored, committed)
scripts/        Python tooling and the benchmark sweep
data/           clean/ (committed)  noisy/  out/  (both gitignored)
bench/results/  CSVs, one per machine
report/         report.tex  figures/
```
