#!/usr/bin/env bash
# Benchmark matrix -> bench/results/<machine>.csv. Plotting reads the CSV.
set -euo pipefail
cd "$(dirname "$0")/.."

MACHINE="${1:-laptop}"
CSV="bench/results/${MACHINE}.csv"
IMG=data/noisy/camera_sigma25.png
CLEAN=data/clean/camera.png
IMG_SMALL=data/noisy/camera256_sigma25.png
CLEAN_SMALL=data/clean/camera256.png
REPEAT=3

rm -f "$CSV"

# --- serial baselines -------------------------------------------------------
for V in seq seq_simd integral; do
  ./nlm --in "$IMG" --clean "$CLEAN" --variant "$V" --threads 1 --repeat $REPEAT --csv "$CSV"
done

# --- thread scaling ---------------------------------------------------------
for V in pthreads pthreads_simd integral_thr; do
  for T in 1 2 4 6 8 10 12 14 16; do
    ./nlm --in "$IMG" --clean "$CLEAN" --variant "$V" --threads "$T" --repeat $REPEAT --csv "$CSV"
  done
done

# --- patch-size scaling (smaller image: seq grows as P^2) -------------------
for P in 1 2 3 4 5 6 7; do
  for V in seq integral; do
    ./nlm --in "$IMG_SMALL" --clean "$CLEAN_SMALL" --variant "$V" --threads 1 \
          --patch "$P" --repeat 1 --csv "$CSV"
  done
done

echo "wrote $CSV"
