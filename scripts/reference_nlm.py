#!/usr/bin/env python3
"""Run scikit-image's NLM (a trusted reference implementation) with matching
parameters, so nlm_seq's output can be validated against it."""

import argparse

import numpy as np
from PIL import Image
from skimage.restoration import denoise_nl_means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="noisy input PNG")
    parser.add_argument("--out", dest="out_path", required=True, help="denoised output PNG")
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--search-radius", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--h", type=float, default=10.0)
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="use skimage's fast (integral-image) variant, which weights all patch "
        "pixels uniformly -- this matches patch_dist2_scalar. The classic mode "
        "(default here) applies a Gaussian weighting across the patch instead.",
    )
    args = parser.parse_args()

    noisy = np.array(Image.open(args.in_path).convert("L")).astype(np.float64)

    denoised = denoise_nl_means(
        noisy,
        patch_size=2 * args.patch_radius + 1,
        patch_distance=args.search_radius,
        h=args.h,
        sigma=args.sigma,
        fast_mode=args.fast_mode,
        preserve_range=True,  # keep arithmetic in [0,255], same units nlm_seq uses
    )

    Image.fromarray(np.clip(denoised, 0, 255).astype(np.uint8), mode="L").save(args.out_path)
    print(
        f"{args.in_path} -> {args.out_path} "
        f"(skimage denoise_nl_means, patch={2*args.patch_radius+1}, "
        f"search={args.search_radius}, sigma={args.sigma}, h={args.h}, "
        f"fast_mode={args.fast_mode})"
    )


if __name__ == "__main__":
    main()
