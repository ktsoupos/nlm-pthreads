#!/usr/bin/env python3
"""Add synthetic additive Gaussian noise to a clean grayscale image: v = u + n."""

import argparse

import numpy as np
from PIL import Image


def add_gaussian_noise(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    noisy = clean.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="clean input PNG")
    parser.add_argument("--out", dest="out_path", required=True, help="noisy output PNG")
    parser.add_argument("--sigma", type=float, required=True, help="noise standard deviation")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for reproducible noise")
    args = parser.parse_args()

    clean = np.array(Image.open(args.in_path).convert("L"))
    noisy = add_gaussian_noise(clean, args.sigma, args.seed)
    Image.fromarray(noisy, mode="L").save(args.out_path)
    print(f"{args.in_path} -> {args.out_path} (sigma={args.sigma}, seed={args.seed})")


if __name__ == "__main__":
    main()
