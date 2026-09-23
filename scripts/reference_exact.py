#!/usr/bin/env python3
"""Exact float64 reference implementation of the NLM formula in CLAUDE.md.

This is the authoritative correctness check for the C variants: unlike
reference_nlm.py (which cross-checks against scikit-image, a *different*
algorithm with its own patch weighting and approximations), this computes
precisely the documented formula, so any disagreement is a bug in the C code.

Vectorised over search-window shifts for speed, but mathematically identical
to nlm_seq's nested loops.
"""

import argparse

import numpy as np
from PIL import Image


def box_sum(a, r):
    """Sum over every (2r+1)x(2r+1) window, via a summed-area table."""
    k = 2 * r + 1
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]


def nlm(noisy, patch_radius, search_radius, sigma, h):
    pad = patch_radius + search_radius
    psize = 2 * patch_radius + 1
    height, width = noisy.shape

    # numpy's 'reflect' is reflect_101: the edge pixel is not duplicated,
    # matching image.c's border fill.
    padded = np.pad(noisy, pad, mode="reflect")

    r0, r1 = pad - patch_radius, pad + height - 1 + patch_radius
    c0, c1 = pad - patch_radius, pad + width - 1 + patch_radius
    base = padded[r0:r1 + 1, c0:c1 + 1]

    acc_z = np.zeros((height, width))
    acc_wsum = np.zeros((height, width))
    max_w = np.zeros((height, width))

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dy == 0 and dx == 0:
                continue  # self-weight is applied after the loop, as max over j != i
            shifted = padded[r0 + dy:r1 + 1 + dy, c0 + dx:c1 + 1 + dx]
            d2 = box_sum((base - shifted) ** 2, patch_radius)[:height, :width] / (psize * psize)
            w = np.exp(-np.maximum(d2 - 2.0 * sigma * sigma, 0.0) / (h * h))

            acc_z += w
            acc_wsum += w * padded[pad + dy:pad + dy + height, pad + dx:pad + dx + width]
            max_w = np.maximum(max_w, w)

    center = padded[pad:pad + height, pad:pad + width]
    total_z = acc_z + max_w
    total_wsum = acc_wsum + max_w * center

    # Every weight can underflow to zero; fall back to the pixel's own value.
    return np.where(total_z > 0, total_wsum / np.where(total_z > 0, total_z, 1.0), center)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="noisy input PNG")
    parser.add_argument("--out", dest="out_path", required=True, help="denoised output PNG")
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--search-radius", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--h", type=float, default=10.0)
    args = parser.parse_args()

    noisy = np.array(Image.open(args.in_path).convert("L")).astype(np.float64)
    out = nlm(noisy, args.patch_radius, args.search_radius, args.sigma, args.h)

    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L").save(args.out_path)
    print(
        f"{args.in_path} -> {args.out_path} "
        f"(exact float64 reference, patch={2*args.patch_radius+1}, "
        f"search={args.search_radius}, sigma={args.sigma}, h={args.h})"
    )


if __name__ == "__main__":
    main()
