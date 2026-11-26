#!/usr/bin/env python3
"""
Exploration of the hard primes up to 10^8: m(p), the a = 1 slices (type I and type II), and
the number of prime shifts c = q tried (note, §4).
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import divisors, factorint, legendre_symbol, primerange

sys.path.insert(0, "tools")
from verify import is_solution, profile, types_at

HARD_CLASSES = (1, 121, 169, 289, 361, 529)
HARD_SET = set(HARD_CLASSES)
PRIMES_3MOD4 = [q for q in primerange(3, 50000) if q % 4 == 3]


def sieve_primes_mod(limit, mod, allowed_set):
    """Fast sieve of Eratosthenes with a numpy bytearray; returns the list of primes p <= limit with p % mod in allowed_set."""
    is_p = np.ones(limit + 1, dtype=np.bool_)
    is_p[:2] = False
    is_p[4::2] = False
    r = math.isqrt(limit)
    for i in range(3, r + 1, 2):
        if is_p[i]:
            is_p[i * i :: 2 * i] = False
    all_p = np.flatnonzero(is_p)
    mask = np.isin(all_p % mod, list(allowed_set))
    return all_p[mask].tolist()


def find_e_min_I1(p, fast_cap=4000):
    """Smallest e ≡ 3 (mod 4) with B | (ep+1)/4 and B ≡ -1 (mod e). Exhaustive up to (p+4)/3 if it goes beyond fast_cap."""
    emax = (p + 4) // 3
    for e in range(3, min(fast_cap, emax) + 1, 4):
        Me = (e * p + 1) // 4
        if e == 3:
            if any(q % 3 == 2 for q in factorint(Me)):
                return e
        else:
            if any(d % e == e - 1 for d in divisors(Me)):
                return e
    # Exhaustive search if it went beyond fast_cap
    for e in range(fast_cap + (3 - fast_cap % 4) % 4, emax + 1, 4):
        Me = (e * p + 1) // 4
        if any(d % e == e - 1 for d in divisors(Me)):
            return e
    return -1


def find_u_min_II1(p):
    """
    Smallest u >= 1 such that p + u has a divisor s ≡ -1 (mod 4u) (Dahan Alg. 2, p = 4*alpha*u*v - u - v).
    By symmetry in (u, v), if a solution exists then min(u, v) <= floor((1 + sqrt(4p + 1)) / 4).
    Hence the search up to umax is 100% EXHAUSTIVE. Returns -1 if there is none (wild prime).
    """
    umax = (1 + math.isqrt(4 * p + 1)) // 4
    for u in range(1, umax + 1):
        mod4u = 4 * u
        target = mod4u - 1
        if u == 1:
            # p + 1 has a divisor ≡ 3 (mod 4) <=> it has a prime factor ≡ 3 (mod 4)
            if any(q % 4 == 3 for q in factorint(p + 1)):
                return 1
        else:
            if any(s % mod4u == target for s in divisors(p + u)):
                return u
    return -1


def find_e_min_II_aET1(p, fast_cap=4000):
    """Smallest e ≡ 3 (mod 4) such that K_e = (p+e)/4 has a divisor b ≡ -1 (mod e) (Aigner Type II a_ET=1)."""
    emax = 3 * p - 1
    for e in range(3, min(fast_cap, emax) + 1, 4):
        Ke = (p + e) // 4
        if e == 3:
            if any(q % 3 == 2 for q in factorint(Ke)):
                return e
        else:
            if any(b % e == e - 1 for b in divisors(Ke)):
                return e
    for e in range(fast_cap + (3 - fast_cap % 4) % 4, emax + 1, 4):
        Ke = (p + e) // 4
        if any(b % e == e - 1 for b in divisors(Ke)):
            return e
    return -1


def find_k_q34(p):
    """Counts how many primes q ≡ 3 (mod 4) with (p/q) = -1 must be tried before finding a solution at c = q."""
    k = 0
    for q in PRIMES_3MOD4:
        if legendre_symbol(p, q) == -1:
            k += 1
            t1, t2 = types_at(p, q)
            if t1 or t2:
                return k, q
    return -1, -1


def worker_hard(chunk):
    out = []
    for p in chunk:
        m, (t1, t2), c1, c2 = profile(p, cap=100_000)
        e1 = find_e_min_I1(p)
        u2 = find_u_min_II1(p)
        e2a = find_e_min_II_aET1(p)
        kq, qhit = find_k_q34(p)
        out.append((p, p % 840, m, c1, c2, e1, u2, e2a, kq, qhit))
    return out


def worker_wild_check(chunk):
    wilds = []
    for p in chunk:
        if find_u_min_II1(p) == -1:
            wilds.append(p)
    return wilds


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100_000_000)
    ap.add_argument("--csv", default="data/hard_primes_1e8.csv")
    ap.add_argument("--plot", default="data/m_growth_1e8.png")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    t_start = time.time()

    # 1. Preliminary validation of u_min_II1 on ALL primes p <= 10^7, p ≡ 1 (mod 24)
    t_w0 = time.time()
    p_1mod24_1e7 = sieve_primes_mod(10_000_000, 24, {1})
    chunk_w = max(1, len(p_1mod24_1e7) // (args.workers * 8))
    chunks_w = [p_1mod24_1e7[i : i + chunk_w] for i in range(0, len(p_1mod24_1e7), chunk_w)]
    wilds_1e7 = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(worker_wild_check, chunks_w):
            wilds_1e7.extend(res)
    wilds_1e7.sort()
    expected_wild14 = [
        409, 577, 5569, 9601, 23929, 83449, 102001, 329617, 712321,
        1134241, 1724209, 1726201, 5212561, 8813281,
    ]
    print(f"=== Validation of u_min_II1 (Dahan Alg. 2) on ALL primes p <= 10^7, p ≡ 1 (mod 24) ({len(p_1mod24_1e7):,} primes) ===")
    print(f"  Time: {time.time() - t_w0:.2f} s")
    print(f"  Wild primes found ({len(wilds_1e7)}): {wilds_1e7}")
    print(f"  Matches EXACTLY Dahan's 14 wild primes from Prop. 3.8: {wilds_1e7 == expected_wild14}")
    assert wilds_1e7 == expected_wild14, "Discrepancy in the list of wild primes <= 10^7!"

    # 2. Sieve of the hard primes up to limit (10^8)
    t_s0 = time.time()
    primes = sieve_primes_mod(args.limit, 840, HARD_SET)
    print(f"\n=== Sieve of the hard primes mod 840 up to {args.limit:,} ===")
    print(f"  Total hard primes: {len(primes):,} (sieve in {time.time() - t_s0:.2f} s)")

    # 3. Parallel processing
    t_c0 = time.time()
    chunk_sz = max(1, len(primes) // (args.workers * 16))
    chunks = [primes[i : i + chunk_sz] for i in range(0, len(primes), chunk_sz)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(worker_hard, chunks):
            rows.extend(res)
    rows.sort(key=lambda r: r[0])
    t_calc = time.time() - t_c0
    print(f"  Full computation on {args.workers} cores: {t_calc:.2f} s (cumulative total: {time.time() - t_start:.2f} s)")

    # 4. Save the CSV
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p", "r_mod_840", "m", "min_c_I", "min_c_II", "e_min_I1", "u_min_II1", "e_min_II_aET1", "k_q34", "q_hit"])
        w.writerows(rows)
    print(f"  CSV saved to {args.csv} ({len(rows):,} rows).")

    # 5. Consistency with the CSV up to 10^7
    if os.path.exists("data/hard_primes_1e7.csv"):
        r1 = list(csv.DictReader(open("data/hard_primes_1e7.csv", encoding="utf-8")))
        r2_sub = [r for r in rows if r[0] <= 10_000_000]
        assert len(r1) == len(r2_sub)
        mismatches = sum(1 for a, b in zip(r1, r2_sub) if int(a["p"]) != b[0] or int(a["m"]) != b[2])
        print(f"  Cross-check with the data (<= 10^7, {len(r1):,} primes): discrepancies = {mismatches}")

    # --- Analysis extended to 10^8 ---
    ps = np.array([r[0] for r in rows], dtype=np.int64)
    ms = np.array([r[2] for r in rows], dtype=np.int64)
    c1s = np.array([r[3] for r in rows], dtype=np.int64)
    c2s = np.array([r[4] for r in rows], dtype=np.int64)
    e1s = np.array([r[5] for r in rows], dtype=np.int64)
    u2s = np.array([r[6] for r in rows], dtype=np.int64)
    e2as = np.array([r[7] for r in rows], dtype=np.int64)
    kqs = np.array([r[8] for r in rows], dtype=np.int64)

    print("\n=== Records of m(p) up to 10^8 and test of the heuristic 0.42 (log X)^2 ===")
    rec_p, rec_m = [], []
    cur = -1
    for p_v, m_v in zip(ps, ms):
        if m_v > cur:
            cur = int(m_v)
            rec_p.append(int(p_v))
            rec_m.append(int(m_v))
    print("Full sequence of records of m(p):")
    for rp, rm in zip(rec_p, rec_m):
        lp = math.log(rp)
        row = next(r for r in rows if r[0] == rp)
        print(f"  p={rp:>10d} (mod 840={rp%840:3d}): m={rm:4d} (cI={row[3]:4d}, cII={row[4]:4d}), m/log(p)={rm/lp:.3f}, m/log^2(p)={rm/(lp*lp):.4f}, 0.42*log^2(p)={0.42*lp*lp:.1f}")

    print("\nTop 15 largest values of m(p) for p <= 10^8:")
    for r in sorted(rows, key=lambda x: (-x[2], x[0]))[:15]:
        print(f"  p={r[0]:>10d} (mod 840={r[1]:3d}): m={r[2]:4d}, min_c_I={r[3]:4d}, min_c_II={r[4]:4d}, e_min_I1={r[5]:4d}, u_min_II1={r[6]:4d}, k_q34={r[8]}")

    print("\nCumulative maximum and mean of m(p) up to X:")
    for X in [10**4, 10**5, 10**6, 10**7, 3*10**7, 5*10**7, 10**8]:
        msk = ps <= X
        sub = ms[msk]
        im = int(np.argmax(sub))
        pmax = int(ps[msk][im])
        mmax = int(sub[im])
        lx = math.log(X)
        print(f"  X={X:>11,}: n={msk.sum():>7,}, mean={sub.mean():6.3f}, max={mmax:4d} (at p={pmax:>10d}), max/log^2(X)={mmax/(lx*lx):.4f}, 0.42*log^2(X)={0.42*lx*lx:.1f}")

    print("\nFrequencies of the largest m(p) and summary by decade (incl. T1c failure rate at c=3):")
    bounds = [10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
    for a, b in zip(bounds[:-1], bounds[1:]):
        msk = (ps > a) & (ps <= b)
        sub = ms[msk]
        fail3 = np.mean(sub > 3)
        xmid = math.sqrt(a * b)
        print(f"  ({a:>9,}, {b:>10,}]: n={msk.sum():>7,}, mean={sub.mean():6.3f}, p99={np.percentile(sub,99):5.1f}, max={sub.max():4d}, F_3={fail3:.4f}, F_3*sqrt(log x_mid)={fail3*math.sqrt(math.log(xmid)):.4f}")

    print("\n=== Slices with a = 1 (Type I a_ET=1, Type II Dahan a_D=1, Type II Aigner a_ET=1) ===")
    no_I1 = [r[0] for r in rows if r[5] == -1]
    no_II1_dahan = [r[0] for r in rows if r[6] == -1]
    no_II1_aigner = [r[0] for r in rows if r[7] == -1]
    no_both_I1_II1d = [r[0] for r in rows if r[5] == -1 and r[6] == -1]
    no_both_I1_II1a = [r[0] for r in rows if r[5] == -1 and r[7] == -1]
    print(f"  No solution in the Type I slice (a_ET = 1 = Dahan Thm 3.14 a_D=1): {len(no_I1)} -> {no_I1[:20]}")
    print(f"  No solution in the Type II slice Dahan Alg. 2 (a_D = c_ET = 1):    {len(no_II1_dahan)} -> {no_II1_dahan}")
    print(f"  No solution in the Type II Aigner slice (a_ET = 1, e|1+b,4b|p+e):  {len(no_II1_aigner)} -> {no_II1_aigner[:20]}")
    print(f"  No solution in EITHER (Type I a=1) or (Type II Dahan):     {len(no_both_I1_II1d)}")
    print(f"  No solution in EITHER (Type I a=1) or (Type II Aigner):    {len(no_both_I1_II1a)}")

    valid_e1 = e1s[e1s > 0]
    im_e1 = int(np.argmax(e1s))
    print(f"  e_min_I1: mean={valid_e1.mean():.3f}, median={np.median(valid_e1):.1f}, max={int(e1s[im_e1])} (at p={int(ps[im_e1])})")
    top_e1 = sorted(rows, key=lambda r: (-r[5], r[0]))[:10]
    print("  Top 10 largest e_min_I1:", [(r[0], r[5]) for r in top_e1])

    valid_u2 = u2s[u2s > 0]
    im_u2 = int(np.argmax(u2s))
    print(f"  u_min_II1 (Dahan Alg. 2, excl. wild): mean={valid_u2.mean():.3f}, median={np.median(valid_u2):.1f}, max={int(u2s[im_u2])} (at p={int(ps[im_u2])})")
    top_u2 = sorted([r for r in rows if r[6] > 0], key=lambda r: (-r[6], r[0]))[:10]
    print("  Top 10 largest u_min_II1 (finite):", [(r[0], r[6]) for r in top_u2])

    valid_e2a = e2as[e2as > 0]
    im_e2a = int(np.argmax(e2as))
    print(f"  e_min_II_aET1 (Aigner Type II a_ET=1): mean={valid_e2a.mean():.3f}, median={np.median(valid_e2a):.1f}, max={int(e2as[im_e2a])} (at p={int(ps[im_e2a])})")

    print("\n=== Number of primes q ≡ 3 (mod 4) with (p/q) = -1 needed ===")
    kq_cnt = Counter(int(v) for v in kqs)
    tot = len(rows)
    for k in sorted(kq_cnt):
        print(f"  k_q34 = {k:2d}: {kq_cnt[k]:7d} ({100*kq_cnt[k]/tot:6.3f}%)")
    max_k = int(kqs.max())
    worst_k = [r for r in rows if r[8] == max_k]
    print(f"  Maximum k_q34 for p <= {args.limit:,}: {max_k} (reached by {len(worst_k)} prime(s)):")
    for r in worst_k[:10]:
        print(f"    p={r[0]:>10d} (mod 840={r[1]:3d}): k_q34={r[8]}, q_hit={r[9]}, m(p)={r[2]}")

    # --- PNG plot up to 10^8 ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)

    ax = axes[0]
    # Light subsampling for the scatter plot if there are > 40k points + every m >= 35
    hi = ms >= 35
    rng = np.random.default_rng(42)
    lo_idx = np.flatnonzero(~hi)
    sub_idx = rng.choice(lo_idx, size=min(25000, len(lo_idx)), replace=False)
    plot_idx = np.concatenate([sub_idx, np.flatnonzero(hi)])
    ax.scatter(ps[plot_idx], ms[plot_idx], s=3, alpha=0.18, color="#1f77b4", rasterized=True, label="Hard primes mod 840 (sample + tail $m \\geq 35$)")
    ax.step(rec_p, rec_m, where="post", color="#d62728", lw=1.8, label="Cumulative record $\\max m(p)$")
    ax.scatter(rec_p, rec_m, color="#d62728", s=26, zorder=5)
    for rp, rm in zip(rec_p, rec_m):
        if rm >= 23:
            ax.annotate(f"p={rp:,}\nm={rm}", (rp, rm), xytext=(-6, 6), textcoords="offset points",
                        fontsize=7, color="#8b0000", ha="right" if rp > 15_000_000 else "left")
    x_grid = np.logspace(np.log10(1000), np.log10(args.limit), 300)
    ax.plot(x_grid, 0.42 * (np.log(x_grid) ** 2), color="black", lw=1.2, ls=":", label=r"Heuristic R1: $0.42\,(\log p)^2$")
    bins = np.logspace(np.log10(1000), np.log10(args.limit), 28)
    bx, bmean, bp99 = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        msk = (ps >= a) & (ps < b)
        if msk.sum() >= 5:
            bx.append(np.sqrt(a * b))
            bmean.append(ms[msk].mean())
            bp99.append(np.percentile(ms[msk], 99))
    ax.plot(bx, bmean, color="#2ca02c", lw=2.0, label="Mean per logarithmic bin")
    ax.plot(bx, bp99, color="#ff7f0e", lw=1.6, ls="--", label="99th percentile per bin")
    ax.set_xscale("log")
    ax.set_ylim(0, max(rec_m[-1], int(0.42 * math.log(args.limit) ** 2)) + 22)
    ax.set_xlabel("Prime $p$ (log scale)")
    ax.set_ylabel(r"$m(p) = 4x_{\min}(p) - p$")
    ax.set_title(r"Growth of $m(p)$ for $p \leq 10^8$ (6 hard classes mod 840)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")

    ax2 = axes[1]
    ks_sorted = sorted(kq_cnt)
    ax2.bar(ks_sorted, [kq_cnt[k] for k in ks_sorted], color="#2b6cb0", edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_xticks(ks_sorted)
    ax2.set_xlabel(r"Number of primes $q \equiv 3\ (\mathrm{mod}\ 4)$ with $(p/q)=-1$ tried ($k_{q34}$)")
    ax2.set_ylabel("Number of hard primes (log scale)")
    ax2.set_title(f"Distribution of $k_{{q34}}(p)$ for $p \\leq 10^8$ ($N = {tot:,}$, max = {max_k})")
    ax2.grid(True, axis="y", alpha=0.25)
    for k in ks_sorted:
        ax2.text(k, kq_cnt[k] * 1.15, f"{kq_cnt[k]:,}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(args.plot)
    print(f"\nPlot saved to {args.plot}")
    print(f"Total script time: {time.time() - t_start:.2f} s")


if __name__ == "__main__":
    main()
