#!/usr/bin/env python3
"""Generate the report's figures from the benchmark CSVs."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHYSICAL_CORES = 8

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LABELS = {
    "seq": "direct, scalar",
    "seq_simd": "direct, AVX2",
    "pthreads": "direct, scalar",
    "pthreads_simd": "direct, AVX2",
    "integral": "integral",
    "integral_thr": "integral",
}
STYLE = {
    "pthreads":      dict(color="#1f77b4", marker="o"),
    "pthreads_simd": dict(color="#d62728", marker="s"),
    "integral_thr":  dict(color="#2ca02c", marker="^"),
}


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in ("threads", "patch_radius", "search_radius", "width", "height", "run"):
                r[k] = int(r[k])
            for k in ("sigma", "h", "seconds"):
                r[k] = float(r[k])
            r["psnr"] = float(r["psnr"]) if r["psnr"] else None
            rows.append(r)
    return rows


def median_time(rows, **filt):
    sel = [r["seconds"] for r in rows
           if all(r[k] == v for k, v in filt.items())]
    return statistics.median(sel) if sel else None


def fig_thread_scaling(rows, outdir):
    big = [r for r in rows if r["width"] == 512 and r["patch_radius"] == 3]
    t_ref = median_time(big, variant="seq", threads=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    threads = sorted({r["threads"] for r in big if r["variant"] == "pthreads"})
    for v in ("pthreads", "pthreads_simd", "integral_thr"):
        sp = [t_ref / median_time(big, variant=v, threads=t) for t in threads]
        ax1.plot(threads, sp, label=LABELS[v], linewidth=1.3, markersize=4, **STYLE[v])
        base = median_time(big, variant=v, threads=1)
        eff = [(base / median_time(big, variant=v, threads=t)) / t for t in threads]
        ax2.plot(threads, eff, label=LABELS[v], linewidth=1.3, markersize=4, **STYLE[v])

    ax1.plot(threads, threads, "k--", linewidth=0.8, label="linear speedup")
    for ax in (ax1, ax2):
        ax.axvline(PHYSICAL_CORES, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("threads")
        ax.set_xticks([1, 4, 8, 12, 16])
    ax1.text(7.7, ax1.get_ylim()[1] * 0.55, "8 physical cores",
             fontsize=7, color="gray", rotation=90, va="center", ha="right")
    ax1.set_ylabel("speedup vs. sequential scalar")
    ax1.set_title("(a) Speedup")
    ax1.legend(loc="upper left")
    ax2.set_ylabel("parallel efficiency")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax2.set_title("(b) Efficiency, relative to own 1-thread time")
    ax2.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(outdir / "thread_scaling.pdf")
    plt.close(fig)


def fig_patch_scaling(rows, outdir):
    small = [r for r in rows if r["width"] == 256]
    radii = sorted({r["patch_radius"] for r in small})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for v, style in (("seq", dict(color="#1f77b4", marker="o")),
                     ("integral", dict(color="#2ca02c", marker="^"))):
        ts = [median_time(small, variant=v, patch_radius=p) for p in radii]
        ax.plot([2 * p + 1 for p in radii], ts, label=LABELS[v],
                linewidth=1.3, markersize=4, **style)
    ax.set_xlabel("patch side $P$ (pixels)")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Cost vs. patch size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "patch_scaling.pdf")
    plt.close(fig)


def fig_variant_bars(rows, outdir):
    big = [r for r in rows if r["width"] == 512 and r["patch_radius"] == 3]
    t_ref = median_time(big, variant="seq", threads=1)
    entries = [
        ("direct\nscalar\n1 thr", median_time(big, variant="seq", threads=1)),
        ("direct\nAVX2\n1 thr", median_time(big, variant="seq_simd", threads=1)),
        ("integral\n1 thr", median_time(big, variant="integral", threads=1)),
        ("direct\nscalar\n16 thr", median_time(big, variant="pthreads", threads=16)),
        ("direct\nAVX2\n16 thr", median_time(big, variant="pthreads_simd", threads=16)),
        ("integral\n16 thr", median_time(big, variant="integral_thr", threads=16)),
    ]
    labels = [e[0] for e in entries]
    speeds = [t_ref / e[1] for e in entries]
    colors = ["#aec7e8", "#ff9896", "#98df8a", "#1f77b4", "#d62728", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    bars = ax.bar(labels, speeds, color=colors, width=0.62)
    for b, s in zip(bars, speeds):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.5, f"{s:.1f}$\\times$",
                ha="center", fontsize=8)
    ax.set_ylabel("speedup vs. sequential scalar")
    ax.set_title("Contribution of each optimisation layer")
    ax.set_ylim(0, max(speeds) * 1.18)
    fig.tight_layout()
    fig.savefig(outdir / "variant_bars.pdf")
    plt.close(fig)


def fig_quality(qrows, noisy_psnr, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    hs = [(r["h"] / r["sigma"], r["psnr"]) for r in qrows if r["sigma"] == 25.0]
    hs = sorted(set(hs))
    ax1.plot([a for a, _ in hs], [b for _, b in hs],
             color="#1f77b4", marker="o", linewidth=1.3, markersize=4)
    best = max(hs, key=lambda t: t[1])
    ax1.axvline(best[0], color="gray", linestyle=":", linewidth=1.0)
    ax1.annotate(f"best $h={best[0]:.2f}\\sigma$\n{best[1]:.2f} dB",
                 xy=best, xytext=(best[0] + 0.12, best[1] - 0.55), fontsize=7)
    ax1.set_xlabel(r"$h/\sigma$")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(r"(a) Filter strength, $\sigma=25$")

    sig = sorted({r["sigma"] for r in qrows if abs(r["h"] / r["sigma"] - 0.6) < 1e-6})
    den = [max(r["psnr"] for r in qrows
               if r["sigma"] == s and abs(r["h"] / r["sigma"] - 0.6) < 1e-6) for s in sig]
    noi = [noisy_psnr[int(s)] for s in sig]
    x = range(len(sig))
    ax2.bar([i - 0.2 for i in x], noi, width=0.38, label="noisy input", color="#c7c7c7")
    ax2.bar([i + 0.2 for i in x], den, width=0.38, label="denoised", color="#2ca02c")
    for i, (a, b) in enumerate(zip(noi, den)):
        ax2.text(i + 0.2, b + 0.3, f"+{b - a:.1f}", ha="center", fontsize=7)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"$\\sigma={int(s)}$" for s in sig])
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_ylim(0, max(den) * 1.2)
    ax2.set_title(r"(b) Denoising gain, $h=0.6\sigma$")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "quality.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="bench/results/laptop.csv")
    ap.add_argument("--quality", default="bench/results/quality.csv")
    ap.add_argument("--outdir", default="report/figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load(args.bench)
    qrows = load(args.quality)
    noisy_psnr = {10: 28.23, 25: 20.59, 50: 15.19}   # measured with scripts/metrics.py

    fig_thread_scaling(rows, outdir)
    fig_patch_scaling(rows, outdir)
    fig_variant_bars(rows, outdir)
    fig_quality(qrows, noisy_psnr, outdir)
    print(f"figures written to {outdir}/")


if __name__ == "__main__":
    main()
