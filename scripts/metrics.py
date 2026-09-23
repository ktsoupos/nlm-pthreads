#!/usr/bin/env python3
"""Compare two images: PSNR plus absolute-difference statistics."""

import argparse

import numpy as np
from PIL import Image


def load(path):
    return np.array(Image.open(path).convert("L")).astype(np.float64)


def psnr(ref, x):
    mse = np.mean((ref - x) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="reference image (e.g. the clean original)")
    parser.add_argument("images", nargs="+", help="image(s) to compare against the reference")
    args = parser.parse_args()

    ref = load(args.reference)
    for path in args.images:
        img = load(path)
        diff = np.abs(ref - img)
        print(
            f"{path}: PSNR {psnr(ref, img):6.2f} dB | "
            f"mean|d| {diff.mean():6.3f} | median|d| {np.median(diff):5.1f} | "
            f"p99 {np.percentile(diff, 99):5.1f} | max {diff.max():5.1f}"
        )


if __name__ == "__main__":
    main()
