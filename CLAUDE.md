# Parallel Non-Local Means Denoising

University final project for a parallel computing / HPC course, cross-listed with a
digital image processing course.

**Stack:** C (C17), pthreads, AVX2 intrinsics. Python only for tooling (noise
generation, metrics, plots) — never linked into the benchmarked binary.

## Goal

Implement NLM denoising, then optimize it in layers and measure each layer
independently. The report's argument is that naive parallelism is only part of the
story: memory layout, vectorization, and a better algorithm each contribute
separately, and they compose non-trivially.

Priority is a correct, validated sequential version first. Do not optimize anything
that has not been validated.

## Algorithm

For each output pixel `i`, average pixels `j` in a search window, weighted by the
similarity of the surrounding *patches*:

- `d²(i,j)` = mean squared difference between the patch at `i` and the patch at `j`
- `w(i,j) = exp(-max(d²(i,j) - 2σ², 0) / h²)`
- `Z(i) = Σ_j w(i,j)`, and `û(i) = (1/Z(i)) · Σ_j w(i,j) · v(j)`

Details that matter:

- Weights come from **patches**, but only the **center pixel** `v(j)` is averaged.
- The `-2σ²` term removes the known noise floor: for identical underlying patches
  `E[d²] = 2σ²`, not 0. Do not expect zero distances in debug output.
- Self-weight: set `w(i,i) = max_{j≠i} w(i,j)`, **not** 1. Otherwise the pixel's own
  noise dominates its own estimate and the output keeps visible speckle.
- `h` must scale with σ (`h ≈ 0.4σ` for 7×7 patches). A fixed `h` tuned at one noise
  level is wrong at another.

Typical parameters: `patch_radius = 3` (7×7), `search_radius = 10` (21×21).

Complexity `O(N · S² · P²)` ≈ 21,600 ops per output pixel. The working set per pixel
is ~3 KB (fits L1) with ~21,600 FLOPs on it — this workload is **compute-bound, not
memory-bound**, which is the whole reason it was chosen over plain convolution.

## Implementation variants

All variants share one signature and are selected at runtime, so the benchmark
harness is a single binary with an identical timing path for every configuration.

```c
typedef enum { NLM_SEQ, NLM_PTHREADS, NLM_PTHREADS_SIMD, NLM_INTEGRAL } nlm_variant;

void nlm_run(const image_t *in, image_t *out,
             const nlm_params *p, nlm_variant v);
```

1. `nlm_seq` — naive sequential baseline.
2. `nlm_pthreads` — row-block partitioning. Every output pixel reads only the
   *unmodified* input and writes to a disjoint output region, so there are **no locks
   in the hot loop**. Never denoise in place: overwriting the input corrupts patches
   that later pixels still need to read.
3. `nlm_pthreads_simd` — AVX2 in the patch-distance loop. Keep `patch_scalar.c` and
   `patch_avx2.c` behind one shared signature so this is a link-time swap, not a
   rewrite.
4. `nlm_integral` — integral-image reformulation (Darbon et al.): loop over
   displacements `t = j - i` rather than over pixels, so overlapping patch sums are
   computed once. Drops complexity to `O(N · S²)`, removing the patch factor.

## Repo layout

```
include/        image.h  nlm.h  patch.h  timer.h
src/            main.c  image.c  timer.c
                patch_scalar.c  patch_avx2.c
                nlm_seq.c  nlm_pthreads.c  nlm_integral.c
external/stb/   stb_image.h  stb_image_write.h   (vendored, committed)
scripts/        make_noisy.py  metrics.py  reference_nlm.py
                sweep.sh  sweep.slurm  plots.py
data/           clean/ (committed)  noisy/  out/   (both gitignored)
bench/results/  CSVs, one per machine
report/figures/
```

## Image I/O

- `stbi_load(path, &w, &h, &n, 1)` — the trailing `1` forces grayscale, so an
  accidentally-RGB input cannot silently break indexing. Note `n` reports what the
  *file* held, not what was returned.
- Always check the returned pointer for `NULL` **before** reading `w`/`h`/`n`; on
  failure stb leaves them untouched. Use `stbi_failure_reason()`.
- Convert to `float` in [0,255] immediately after load.
- Pad the buffer with a reflected border (`reflect_101`) of width
  `patch_radius + search_radius`, so the inner loops need no bounds checks at all —
  this also unblocks vectorization.
- `image_t.data` points at the top-left of the *padded* buffer; use `IMG_AT(img,y,x)`,
  which makes negative coordinates legal.
- On write, clamp to [0,255] before the uint8 cast. `stbi_write_png`'s last argument
  is the stride **in bytes** — pass a compacted buffer, not the padded one.
- **PNG only, never JPEG.** Lossy artifacts would corrupt both the `v = u + n` noise
  model and the PSNR reference.

## CLI

```
./nlm --in noisy.png --out denoised.png --variant pthreads --threads 8 \
      --patch 3 --search 10 --sigma 25 --csv bench/results/laptop.csv
```

Appends one CSV row per run: variant, threads, params, image size, seconds, PSNR.

## Correctness

- Round-trip: uint8 → float → uint8 with no processing must be byte-identical.
- Border: with `reflect_101`, row −1 equals row 1 and column −1 equals column 1.
- Validate against `skimage.restoration.denoise_nl_means`.
- **All variants must produce identical output** — parallelization and vectorization
  change speed, not results. Diff every variant against `nlm_seq`.
- PSNR against the clean reference must improve over the noisy input.

## Benchmarking

Primary target: **i7-10870H** — 8 physical cores / 16 threads, AVX2 (no AVX-512),
single NUMA node, L1d 32 KB/core, L2 256 KB/core, L3 16 MB shared, 2.2 GHz base /
5.0 GHz boost. Also a university HPC cluster (Slurm) for throttle-free numbers and
possible multi-socket NUMA effects.

- Thread sweep: 1, 2, 4, 6, 8, 10, 12, 14, 16. Expect the curve to bend at 8 — past
  that, hyperthreads share execution units. Annotate that boundary in plots.
- Laptop results will show thermal throttling under sustained multi-core load; treat
  the cluster as the source for reported numbers and say so.
- **Time only the compute.** File I/O stays outside the timed region.
- `perf stat -e cache-misses,cache-references,instructions,cycles` to back up claims
  with counters rather than just wall-clock.
- Parameter sweeps: `h`, patch size, search size — against both PSNR and runtime.

## Build

```makefile
CFLAGS  := -O3 -march=native -mavx2 -mfma -Wall -Wextra -Iexternal/stb
LDFLAGS := -lpthread -lm
```

`-lm` is required (stb calls `pow`; NLM calls `expf`). Link flags go **after** the
source files on the command line.

Keep a `make cluster` target **without** `-march=native`: the login node's CPU may
differ from the compute nodes', giving either illegal-instruction crashes or silently
suboptimal code.

## Conventions

- Keep the patch-distance computation isolated in its own function from day one.
- `aligned_alloc(64, ...)` for image buffers; size must be a multiple of the
  alignment. Note that `stride = width + 2*pad` means individual *rows* still may not
  be 64-byte aligned — use `_mm256_loadu_ps`, or round the stride up to a multiple of
  16 floats.
- Write benchmark results as CSV, never as printed tables; plotting reads the CSV.

## Open questions

- Color support (joint weights across RGB — **not** per-channel, which produces color
  speckle) is a stretch goal, not yet decided.
- Deadline not yet recorded; scope cuts should come off the bottom of the variant list
  (integral image first, then SIMD), never off correctness or the sequential baseline.
