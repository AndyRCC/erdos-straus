#!/usr/bin/env python3
"""
Computational exploration of the 6 hard classes modulo 840:
    p <= N, p prime, p mod 840 in {1, 121, 169, 289, 361, 529}.

Uses exactly `search` and `sol_type` from `tools/verify.py`.
Saves:
  - data/hard_primes_1e7.csv
  - data/m_growth_1e7.png
Prints detailed statistics on the growth of m(p), its distribution, and a comparison with Dahan.
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import primerange

sys.path.insert(0, "tools")
from verify import search, is_solution, sol_type

HARD_CLASSES = (1, 121, 169, 289, 361, 529)
HARD_SET = set(HARD_CLASSES)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10_000_000)
    ap.add_argument("--csv", default="data/hard_primes_1e7.csv")
    ap.add_argument("--plot", default="data/m_growth_1e7.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    t0 = time.time()

    primes = [p for p in primerange(2, args.limit + 1) if (p % 840) in HARD_SET]
    print(f"Primes p <= {args.limit:,} in the 6 hard classes mod 840: {len(primes):,}")

    rows = []
    m_counts = Counter()
    type_counts = Counter()
    type_by_class = {r: Counter() for r in HARD_CLASSES}
    max_m = -1
    max_m_primes = []
    record_p = []
    record_m = []

    for p in primes:
        sol = search(p)
        if sol is None:
            raise RuntimeError(f"COUNTEREXAMPLE FOUND: p={p}")
        x, y, z = sol
        assert is_solution(p, x, y, z)
        m = 4 * x - p
        tp = sol_type(p, sol)
        r840 = p % 840
        rows.append((p, r840, x, y, z, m, tp))
        m_counts[m] += 1
        type_counts[tp] += 1
        type_by_class[r840][tp] += 1
        if m > max_m:
            max_m = m
            max_m_primes = [p]
            record_p.append(p)
            record_m.append(m)
        elif m == max_m:
            max_m_primes.append(p)

    elapsed = time.time() - t0
    print(f"Computation finished in {elapsed:.2f} s.")

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p", "r_mod_840", "x", "y", "z", "m", "type"])
        w.writerows(rows)
    print(f"CSV saved to {args.csv} ({len(rows):,} rows).")

    # --- Statistical summary ---
    ps = np.array([r[0] for r in rows], dtype=np.int64)
    ms = np.array([r[5] for r in rows], dtype=np.int64)
    k_vals = (ms + 1) // 4  # k = x - (p-1)/4 = (m+1)/4  (depth in steps of x)

    print("\n=== Maximum and growth of m(p) ===")
    print(f"Maximum of m(p) for p <= {args.limit:,}: max m = {max_m} (k = (m+1)/4 = {(max_m+1)//4})")
    print(f"Reached at ({len(max_m_primes)} prime(s)):")
    for mp in max_m_primes:
        row = next(r for r in rows if r[0] == mp)
        print(f"  p = {row[0]} (mod 840 = {row[1]}): x={row[2]}, y={row[3]}, z={row[4]}, m={row[5]}, type={row[6]}")

    print("\nTop 15 largest values of m(p):")
    top_rows = sorted(rows, key=lambda r: (-r[5], r[0]))[:15]
    for r in top_rows:
        print(f"  p={r[0]:>9d} (mod 840={r[1]:3d}): m={r[5]:4d} (k={(r[5]+1)//4:3d}), type={r[6]}, x={r[2]}")

    print("\nSequence of records of m(p):")
    for rp, rm in zip(record_p, record_m):
        print(f"  p={rp:>9d} (mod 840={rp%840:3d}): m={rm:4d} (k={(rm+1)//4:3d}), m/log(p)={rm/math.log(rp):.3f}, m/log^2(p)={rm/(math.log(rp)**2):.4f}")

    # Statistics by logarithmic decades
    print("\nStatistics by intervals of p:")
    bounds = [1, 10**4, 10**5, 10**6, 10**7]
    for a, b in zip(bounds[:-1], bounds[1:]):
        mask = (ps > a) & (ps <= b)
        if np.any(mask):
            sub_m = ms[mask]
            frac3 = 100.0 * np.mean(sub_m == 3)
            print(f"  ({a:>9,}, {b:>10,}]: n={mask.sum():>6,}, mean={sub_m.mean():6.2f}, median={np.median(sub_m):5.1f}, frac(m=3)={frac3:5.2f}%, p95={np.percentile(sub_m,95):6.1f}, p99={np.percentile(sub_m,99):6.1f}, max={sub_m.max():4d}")

    # Cumulative up to 10^k
    print("\nCumulative maximum and mean for p <= X:")
    for X in [10**3, 10**4, 10**5, 10**6, 2*10**6, 5*10**6, 10**7]:
        mask = ps <= X
        if np.any(mask):
            sub_m = ms[mask]
            idx_max = np.argmax(sub_m)
            print(f"  X={X:>10,}: count={mask.sum():>6,}, mean={sub_m.mean():6.2f}, max={sub_m.max():4d} (at p={ps[mask][idx_max]}), max/log(X)={sub_m.max()/math.log(X):.2f}, max/log^2(X)={sub_m.max()/(math.log(X)**2):.3f}")

    # --- Frequencies of m and missing values ---
    print("\n=== Distribution of m(p) and types ===")
    print(f"Types (of the solution returned by search(p)): {dict(type_counts)}")
    for r840 in HARD_CLASSES:
        print(f"  mod 840 = {r840:3d}: {dict(type_by_class[r840])}")

    print("\nDistribution of m mod 4:", Counter(int(v) % 4 for v in ms))
    print("Distribution of m mod 12:", Counter(int(v) % 12 for v in ms))
    print("Distribution of m mod 24:", Counter(int(v) % 24 for v in ms))
    print("Distribution of m mod 3:", Counter(int(v) % 3 for v in ms))
    print("Distribution of m mod 5:", Counter(int(v) % 5 for v in ms))
    print("Distribution of m mod 7:", Counter(int(v) % 7 for v in ms))

    print("\nTop 25 most frequent values of m:")
    total = len(rows)
    for mv, cnt in m_counts.most_common(25):
        print(f"  m={mv:4d} (k={(mv+1)//4:3d}, mod 12={mv%12:2d}, mod 24={mv%24:2d}): {cnt:6d} ({100*cnt/total:6.3f}%)")

    # Values m ≡ 3 (mod 4) that NEVER occur up to max_m
    all_m3mod4 = list(range(3, max_m + 1, 4))
    present_set = set(m_counts.keys())
    never_3mod4 = [v for v in all_m3mod4 if v not in present_set]
    print(f"\nValues m <= {max_m} with m ≡ 3 (mod 4): total={len(all_m3mod4)}, present={len(present_set)}, missing={len(never_3mod4)}")
    print("Missing with m <= 200 (among m ≡ 3 mod 4):", [v for v in never_3mod4 if v <= 200])
    # Check whether m = 3 never occurs, m ≡ 3 (mod 12), etc., or the prime factors of m
    m_mod3_eq0 = [v for v in present_set if v % 3 == 0]
    print(f"Values present with 3 | m? ({len(m_mod3_eq0)}): {sorted(m_mod3_eq0)[:20]}")
    # Check all the prime factors of the present vs the missing m
    # In particular, what happens with m = 4k-1 whose prime factors are all ≡ 1 (mod 4), or divisors of ...?
    # Note: for m = 4x-p, when m=3: 4x = p+3; since p ≡ 1 (mod 24), p+3 ≡ 4 (mod 24) => x ≡ 1 (mod 6).
    # But let us see whether some m with p ≡ 1 (mod 840) or similar has an obstruction!
    from sympy import factorint
    print("\nFirst 30 MISSING values with m ≡ 3 (mod 4) and their factorization:")
    for v in never_3mod4[:30]:
        print(f"  m={v:4d} = {factorint(v)}  (mod 12={v%12:2d}, mod 24={v%24:2d}, k={(v+1)//4})")

    print("\nFirst 25 PRESENT values, sorted by m:")
    for v in sorted(present_set)[:25]:
        print(f"  m={v:4d} = {factorint(v)}  count={m_counts[v]:5d}  (mod 12={v%12:2d}, mod 24={v%24:2d}, k={(v+1)//4})")

    # --- PNG plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)

    # Panel 1: m(p) vs p (log scale in p) + record curve + mean and 99th percentile
    ax = axes[0]
    ax.scatter(ps, ms, s=4, alpha=0.20, color="#1f77b4", rasterized=True, label="Primes $p$ in the 6 classes mod 840")
    ax.step(record_p, record_m, where="post", color="#d62728", lw=1.8, label="Cumulative record $\\max m(p)$")
    ax.scatter(record_p, record_m, color="#d62728", s=26, zorder=5)
    for rp, rm in zip(record_p, record_m):
        if rm >= 23:
            ax.annotate(f"p={rp:,}\nm={rm}", (rp, rm), xytext=(-6, 6), textcoords="offset points",
                        fontsize=7, color="#8b0000", ha="right" if rp > 5_000_000 else "left")

    x_grid = np.logspace(np.log10(max(800, ps.min())), np.log10(args.limit), 300)
    bins = np.logspace(np.log10(1000), np.log10(args.limit), 24)
    bx, bmean, bp99 = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        msk = (ps >= a) & (ps < b)
        if msk.sum() >= 5:
            bx.append(np.sqrt(a * b))
            bmean.append(ms[msk].mean())
            bp99.append(np.percentile(ms[msk], 99))
    ax.plot(bx, bmean, color="#2ca02c", lw=2.0, label="Mean per logarithmic bin (decreasing)")
    ax.plot(bx, bp99, color="#ff7f0e", lw=1.6, ls="--", label="99th percentile per bin")
    ax.plot(x_grid, 0.42 * (np.log(x_grid) ** 2), color="black", lw=1.1, ls=":", label=r"Reference $0.42\,(\log p)^2$")

    ax.set_xscale("log")
    ax.set_ylim(0, max_m + 18)
    ax.set_xlabel("Prime $p$ (log scale)")
    ax.set_ylabel(r"$m(p) = 4x_{\min}(p) - p$")
    ax.set_title(r"Growth of $m(p)$ for $p \leq 10^7$ (6 hard classes mod 840)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    # Panel 2: frequency of the values of m (log scale), distinguishing m ≡ 3 vs 7 (mod 8)
    ax2 = axes[1]
    top_m = sorted(m_counts.items(), key=lambda kv: kv[0])
    m_vals = [mv for mv, _ in top_m]
    c_vals = [cnt for _, cnt in top_m]
    colors = ["#1f77b4" if mv % 8 == 3 else "#ff7f0e" for mv in m_vals]
    ax2.bar(m_vals, c_vals, width=2.8, color=colors, alpha=0.88, edgecolor="black", linewidth=0.4)
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor="#1f77b4", edgecolor="black", label=r"$m \equiv 3\ (\mathrm{mod}\ 8)$ ($x_{\min}$ odd)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label=r"$m \equiv 7\ (\mathrm{mod}\ 8)$ ($x_{\min}$ even)"),
    ], fontsize=8.5, loc="upper right")
    ax2.set_yscale("log")
    ax2.set_xlim(0, max_m + 6)
    ax2.set_xlabel(r"$m = 4x_{\min}(p) - p$ (all values observed up to $10^7$)")
    ax2.set_ylabel("Frequency (log scale)")
    ax2.set_title(r"Distribution of $m(p)$ ($N = 20\,513$ primes; all $m \equiv 3\ (\mathrm{mod}\ 4)$)")
    ax2.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(args.plot)
    print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
