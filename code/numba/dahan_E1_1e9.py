#!/usr/bin/env python3
"""
Exact replication, and extension up to 10^9, of Dahan's experiments (arXiv:2608.24035)
on E_1(N;J): Table 2 and the slopes against log log n (note, §2). Saves aggregated tables
to data/dahan_E1_1e9.csv and the plot to data/dahan_E1_1e9.png.
"""
import argparse
import csv
import math
import os
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from sympy import divisors, primerange

J_MAX = 20


@nb.njit(cache=True)
def _gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


@nb.njit(cache=True)
def _modinv(a, m):
    """Modular inverse of a mod m (assuming gcd(a, m) = 1, m > 1)."""
    t, newt = 0, 1
    r, newr = m, a % m
    while newr != 0:
        q = r // newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    if t < 0:
        t += m
    return t


@nb.njit(cache=True)
def sieve_small_primes(limit):
    is_p = np.ones(limit + 1, dtype=np.uint8)
    is_p[0] = 0
    is_p[1] = 0
    r = int(math.sqrt(limit))
    for i in range(2, r + 1):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = 0
    cnt = 0
    for i in range(2, limit + 1):
        if is_p[i]:
            cnt += 1
    out = np.empty(cnt, dtype=np.int64)
    idx = 0
    for i in range(2, limit + 1):
        if is_p[i]:
            out[idx] = i
            idx += 1
    return out


@nb.njit(cache=True)
def process_segment_E1(seg_idx, seg_k_len, small_primes, win_lo, win_hi, n_wins):
    """
    Processes the segment k in [seg_idx * seg_k_len, (seg_idx + 1) * seg_k_len),
    corresponding to n = 24*k + 1.
    Returns:
      counts[w, j] for j=0..J_MAX (j=0: total primes in window w; j>=1: primes in E_1(w; j))
      sum_log[w]   (sum of log(p) over window w)
      total_primes_lt_2e7 (auxiliary counter if applicable, not needed separately)
    """
    k0 = seg_idx * seg_k_len
    k1 = k0 + seg_k_len  # exclusive
    n_lo = 24 * k0 + 1
    n_hi = 24 * (k1 - 1) + 1  # inclusive

    # 1. Sieve of the primes n = 24*k + 1
    is_prime = np.ones(seg_k_len, dtype=np.uint8)
    if k0 == 0:
        is_prime[0] = 0  # n = 1 is not prime

    lim_p = int(math.sqrt(float(n_hi))) + 1
    for i in range(len(small_primes)):
        p = small_primes[i]
        if p > lim_p:
            break
        if p <= 3:
            continue
        # 24*k + 1 ≡ 0 (mod p)  <=>  k ≡ -inv(24, p) (mod p)
        inv24 = _modinv(24, p)
        r = (p - inv24) % p
        # First k >= k0 with k ≡ r (mod p)
        rem = k0 % p
        off = r - rem
        if off < 0:
            off += p
        # Avoid crossing out the prime p itself (n = p => 24*k + 1 = p)
        k_start = k0 + off
        if 24 * k_start + 1 == p:
            off += p
        for idx in range(off, seg_k_len, p):
            is_prime[idx] = 0

    # 2. For each candidate, record the smallest u in 1..J_MAX that solves it (255 = none <= J_MAX)
    min_u = np.full(seg_k_len, 255, dtype=np.uint8)

    for u in range(1, J_MAX + 1):
        max_sv = int(math.sqrt(float(n_hi + u))) + 1
        step_s = 4 * u

        # Phase 1: s = 4*alpha*u - 1 <= max_sv, with s | (n + u) and n + u >= s (v >= 1)
        s = step_s - 1
        while s <= max_sv:
            # Solve 24*k + 1 + u ≡ 0 (mod s)
            # Since s is odd, gcd(24, s) in {1, 3}
            if s % 3 == 0:
                if (1 + u) % 3 != 0:
                    s += step_s
                    continue
                s3 = s // 3
                # 8*k + (1+u)//3 ≡ 0 (mod s3)
                if s3 == 1:
                    off = 0
                    step_k = 1
                else:
                    inv8 = _modinv(8, s3)
                    rhs = (-(1 + u) // 3) % s3
                    r = (rhs * inv8) % s3
                    rem = k0 % s3
                    off = r - rem
                    if off < 0:
                        off += s3
                    step_k = s3
            else:
                inv24 = _modinv(24, s)
                rhs = (-(1 + u)) % s
                r = (rhs * inv24) % s
                rem = k0 % s
                off = r - rem
                if off < 0:
                    off += s
                step_k = s

            # Ensure n + u >= s => 24*(k0 + off) + 1 + u >= s
            while off < seg_k_len and (24 * (k0 + off) + 1 + u) < s:
                off += step_k
            for idx in range(off, seg_k_len, step_k):
                if is_prime[idx] and min_u[idx] == 255:
                    min_u[idx] = u
            s += step_s

        # Phase 2: v <= max_sv, with n ≡ -(u + v) (mod 4*u*v) and n >= 4*u*v - u - v (alpha >= 1)
        # Condition mod 4: -(u + v) ≡ 1 (mod 4) <=> v ≡ 3 - u (mod 4)
        v_start = (3 - u) % 4
        if v_start == 0:
            v_start = 4
        v = v_start
        while v <= max_sv:
            mod_m = 4 * u * v
            target = (-(u + v)) % mod_m
            # Compatibility with n ≡ 1 (mod 24): gcd(mod_m, 24) must divide (target - 1)
            g = _gcd(mod_m, 24)
            if (target - 1) % g == 0:
                # 24*k ≡ target - 1 (mod mod_m)  =>  (24/g)*k ≡ (target - 1)/g (mod mod_m/g)
                m_red = mod_m // g
                a_red = 24 // g
                b_red = ((target - 1) // g) % m_red
                if m_red == 1:
                    off = 0
                    step_k = 1
                else:
                    inv_a = _modinv(a_red, m_red)
                    r = (b_red * inv_a) % m_red
                    rem = k0 % m_red
                    off = r - rem
                    if off < 0:
                        off += m_red
                    step_k = m_red
                n_min_req = mod_m - u - v
                while off < seg_k_len and (24 * (k0 + off) + 1) < n_min_req:
                    off += step_k
                for idx in range(off, seg_k_len, step_k):
                    if is_prime[idx] and min_u[idx] == 255:
                        min_u[idx] = u
            v += 4

    # 3. Accumulate statistics per window
    counts = np.zeros((n_wins, J_MAX + 1), dtype=np.int64)
    sum_log = np.zeros(n_wins, dtype=np.float64)

    # Find which windows intersect [n_lo, n_hi]
    for w in range(n_wins):
        wlo = win_lo[w]
        whi = win_hi[w]
        if n_hi < wlo or n_lo >= whi:
            continue
        # Initial/final index k in this segment
        i_start = 0
        if wlo > n_lo:
            # smallest idx with 24*(k0 + idx) + 1 >= wlo
            i_start = (wlo - n_lo + 23) // 24
        i_end = seg_k_len
        if whi <= n_hi:
            # smallest idx with 24*(k0 + idx) + 1 >= whi
            i_end = (whi - n_lo + 23) // 24
        if i_start < 0:
            i_start = 0
        if i_end > seg_k_len:
            i_end = seg_k_len

        for idx in range(i_start, i_end):
            if is_prime[idx]:
                n_val = 24 * (k0 + idx) + 1
                mu = min_u[idx]
                counts[w, 0] += 1
                sum_log[w] += math.log(float(n_val))
                # In E_1(w; J) if mu > J (for J = 1..J_MAX)
                j_lim = J_MAX if mu > J_MAX else mu - 1
                for j in range(1, j_lim + 1):
                    counts[w, j] += 1

    return counts, sum_log


@nb.njit(parallel=True, cache=True)
def run_all_segments(n_segs, seg_k_len, small_primes, win_lo, win_hi):
    n_wins = len(win_lo)
    all_counts = np.zeros((n_segs, n_wins, J_MAX + 1), dtype=np.int64)
    all_sum_log = np.zeros((n_segs, n_wins), dtype=np.float64)
    for s in nb.prange(n_segs):
        c, sl = process_segment_E1(s, seg_k_len, small_primes, win_lo, win_hi, n_wins)
        all_counts[s] = c
        all_sum_log[s] = sl
    return all_counts.sum(axis=0), all_sum_log.sum(axis=0)


def ols_fit(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    sxy = np.sum((x - xm) * (y - ym))
    slope = sxy / sxx
    intercept = ym - slope * xm
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - ym) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    se_slope = math.sqrt(ss_res / ((len(x) - 2) * sxx)) if len(x) > 2 else 0.0
    return slope, intercept, r2, se_slope


def validate_against_bruteforce(limit_val=200_000):
    """Checks that there are 0 discrepancies between the inverted Numba sieve and direct divisors up to limit_val."""
    small_p = sieve_small_primes(int(math.isqrt(limit_val + 100)) + 10)
    k_max = (limit_val // 24) + 1
    win_lo = np.array([2], dtype=np.int64)
    win_hi = np.array([limit_val], dtype=np.int64)
    # Extract u_min per prime directly
    counts_sieve, _ = process_segment_E1(0, k_max, small_p, win_lo, win_hi, 1)

    counts_bf = np.zeros(J_MAX + 1, dtype=np.int64)
    for p in primerange(2, limit_val):
        if p % 24 != 1:
            continue
        counts_bf[0] += 1
        mu = 255
        for u in range(1, J_MAX + 1):
            mod4u = 4 * u
            target = mod4u - 1
            if any(s % mod4u == target for s in divisors(p + u)):
                mu = u
                break
        for j in range(1, J_MAX + 1):
            if mu > j:
                counts_bf[j] += 1
    ok = np.array_equal(counts_sieve[0], counts_bf)
    print(f"=== Validation of the inverted sieve against direct divisors (p < {limit_val:,}, p ≡ 1 mod 24) ===")
    print(f"  Primes: {counts_bf[0]}, Exact agreement for ALL J=0..{J_MAX}: {ok}")
    if not ok:
        print("  Sieve:", counts_sieve[0])
        print("  Brute:", counts_bf)
        raise RuntimeError("Validation of the inverted sieve failed!")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1_000_000_000)
    ap.add_argument("--csv", default="data/dahan_E1_1e9.csv")
    ap.add_argument("--plot", default="data/dahan_E1_1e9.png")
    args = ap.parse_args()

    t_all = time.time()
    print(f"Hardware: {platform.processor()} | {os.cpu_count()} logical cores | Numba {nb.__version__}")

    # 1. Exact validation up to 2*10^5
    validate_against_bruteforce(200_000)

    # 2. Define the windows:
    #    w0: [2, 10^3) (to check Dahan's total of 158,595 below 2*10^7)
    #    w1..w6: Dahan's 6 EXACT windows from Table 2 and §6(11)
    #    w7..w10: extension up to 10^9
    windows = [
        ("[2, 1e3)",        2,          1_000),
        ("[1e3, 1e4)",      1_000,      10_000),
        ("[1e4, 1e5)",      10_000,     100_000),
        ("[1e5, 1e6)",      100_000,    1_000_000),
        ("[1e6, 3e6)",      1_000_000,  3_000_000),
        ("[3e6, 1e7)",      3_000_000,  10_000_000),
        ("[1e7, 2e7)",      10_000_000, 20_000_000),
        ("[2e7, 5e7)",      20_000_000, 50_000_000),
        ("[5e7, 1e8)",      50_000_000, 100_000_000),
        ("[1e8, 3e8)",      100_000_000, 300_000_000),
        ("[3e8, 1e9)",      300_000_000, 1_000_000_000),
    ]
    windows = [w for w in windows if w[1] < args.limit]
    # Adjust the last bound if args.limit < 10^9
    if windows[-1][2] > args.limit:
        name, lo, _ = windows[-1]
        windows[-1] = (f"[{lo:.0e}, {args.limit:.0e})", lo, args.limit)

    win_lo = np.array([w[1] for w in windows], dtype=np.int64)
    win_hi = np.array([w[2] for w in windows], dtype=np.int64)

    # 3. Parallel segmented sieve up to args.limit
    t0 = time.time()
    small_p = sieve_small_primes(int(math.isqrt(args.limit + 100)) + 10)
    seg_k_len = 100_000  # 2,400,000 integers per segment
    total_k = (args.limit // 24) + 1
    n_segs = (total_k + seg_k_len - 1) // seg_k_len

    counts, sum_log = run_all_segments(n_segs, seg_k_len, small_p, win_lo, win_hi)
    dt = time.time() - t0
    print(f"\n=== Full segmented sieve up to N = {args.limit:,} ({n_segs} segments) ===")
    print(f"  Computation time (parallel Numba, {os.cpu_count()} cores): {dt:.2f} s")
    tot_lt_2e7 = int(counts[:7, 0].sum())
    tot_all = int(counts[:, 0].sum())
    print(f"  Primes p < 2*10^7, p ≡ 1 (mod 24): {tot_lt_2e7:,} (Dahan §6(11) reports 158,595)")
    print(f"  Primes p < {args.limit:,}, p ≡ 1 (mod 24): {tot_all:,}")

    # 4. Save the CSV aggregated by window and J
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["window", "lo", "hi", "log_geom_mid", "mean_log_p", "n_primes"] + [f"E1_J{j}" for j in range(1, J_MAX + 1)]
        w.writerow(header)
        for i, (name, lo, hi) in enumerate(windows):
            n_p = int(counts[i, 0])
            lgm = math.log(math.sqrt(lo * hi))
            mlp = sum_log[i] / n_p if n_p > 0 else 0.0
            w.writerow([name, lo, hi, f"{lgm:.6f}", f"{mlp:.6f}", n_p] + [int(counts[i, j]) for j in range(1, J_MAX + 1)])
    print(f"  Aggregated table saved to {args.csv}")

    # 5. Analysis: exact replication of Dahan (windows 1..6 = [1e3, 1e4) .. [1e7, 2e7))
    print("\n=== Exact replication of Dahan's Table 2 and slopes §4.1 / §6(11) (windows [1e3, 2e7)) ===")
    dahan_idx = list(range(1, 7))
    dahan_beta_pub = [1.134, 1.632, 1.905, 2.008, 2.200, 2.480]
    dahan_ratio_pub = [0.141, 0.157, 0.150, 0.140, 0.142, 0.151]

    betas_all = []
    r2s_all = []
    lgms_all = []
    mlps_all = []
    for i in range(1, len(windows)):
        name, lo, hi = windows[i]
        n_p = counts[i, 0]
        lgm = math.log(math.sqrt(lo * hi))
        mlp = sum_log[i] / n_p
        js = np.arange(1, J_MAX + 1, dtype=np.float64)
        fracs = counts[i, 1:] / float(n_p)
        # Exclude possible zeros (in [1e3, 1e4) there may be 0 or few for large J)
        msk = fracs > 0
        slope, _, r2, _ = ols_fit(np.log(js[msk]), np.log(fracs[msk]))
        beta = -slope
        betas_all.append(beta)
        r2s_all.append(r2)
        lgms_all.append(lgm)
        mlps_all.append(mlp)
        pub_str = ""
        if i <= 6:
            pub_str = f"  [Dahan: beta={dahan_beta_pub[i-1]:.3f}, beta/logn={dahan_ratio_pub[i-1]:.3f}]"
        print(f"  {name:14s}: n={n_p:9,d}, log(mid)={lgm:6.3f}, mean_log={mlp:6.3f}, beta={beta:6.3f}, beta/log(mid)={beta/lgm:.4f}, beta/mean_log={beta/mlp:.4f}, R2={r2:.4f}{pub_str}")

    # Regression of beta on log n (Dahan's 6 windows vs all 10 windows up to 10^9)
    sl6_gm, ic6_gm, r26_gm, se6_gm = ols_fit(lgms_all[:6], betas_all[:6])
    sl6_ml, ic6_ml, r26_ml, se6_ml = ols_fit(mlps_all[:6], betas_all[:6])
    print(f"\n  Fit beta vs log(n_mid) [Dahan's 6 windows]:  beta = {sl6_gm:.4f} (±{se6_gm:.4f}) log n + {ic6_gm:+.4f}, R2 = {r26_gm:.4f}  [Dahan: 0.1433 log n + 0.043, R2 = 0.967]")
    print(f"  Fit beta vs mean(log p) [Dahan's 6 windows]: beta = {sl6_ml:.4f} (±{se6_ml:.4f}) log n + {ic6_ml:+.4f}, R2 = {r26_ml:.4f}  [Dahan: 0.150 log n - 0.079]")

    sl10_gm, ic10_gm, r210_gm, se10_gm = ols_fit(lgms_all, betas_all)
    sl10_ml, ic10_ml, r210_ml, se10_ml = ols_fit(mlps_all, betas_all)
    sl_hi4_gm, ic_hi4_gm, r2_hi4_gm, _ = ols_fit(lgms_all[5:], betas_all[5:])
    print(f"  Fit beta vs log(n_mid) [10 windows ->1e9]: beta = {sl10_gm:.4f} (±{se10_gm:.4f}) log n + {ic10_gm:+.4f}, R2 = {r210_gm:.4f}")
    print(f"  Fit beta vs mean(log p) [10 windows ->1e9]:beta = {sl10_ml:.4f} (±{se10_ml:.4f}) log n + {ic10_ml:+.4f}, R2 = {r210_ml:.4f}")
    print(f"  High windows only [1e7..1e9] (5 windows):   beta = {sl_hi4_gm:.4f} log n + {ic_hi4_gm:+.4f}, R2 = {r2_hi4_gm:.4f}")

    # 6. Sieve-dimension slopes: log(|E_1(w; J)| / n_w) against log log n
    print("\n=== Slopes of log(|E_1(w; J)| / |N(w)|) against log log(n_mid) (J = 1..6) ===")
    dahan_slopes_pub = [-0.39, -0.50, -0.84, -1.24, -1.51, -1.59]
    ll_gm = np.log(np.array(lgms_all))
    ll_ml = np.log(np.array(mlps_all))
    print(f"  Range of log log(n_mid): Dahan's 6 windows = [{ll_gm[0]:.3f}, {ll_gm[5]:.3f}]; 10 windows ->1e9 = [{ll_gm[0]:.3f}, {ll_gm[-1]:.3f}]; high only [1e6..1e9] = [{ll_gm[3]:.3f}, {ll_gm[-1]:.3f}]")
    print(f"  {'J':>2s} | {'Theory -J/2':>9s} | {'Dahan pub':>9s} | {'6w (gm)':>8s} | {'6w (ml)':>8s} | {'10w ->1e9':>9s} | {'7w [1e6..1e9]':>13s} | {'4w [2e7..1e9]':>13s}")
    slopes_6w = []
    slopes_10w = []
    slopes_hi = []
    for j in range(1, 7):
        ys = np.array([math.log(counts[i, j] / float(counts[i, 0])) for i in range(1, len(windows))])
        s6g, _, _, _ = ols_fit(ll_gm[:6], ys[:6])
        s6m, _, _, _ = ols_fit(ll_ml[:6], ys[:6])
        s10g, _, _, _ = ols_fit(ll_gm, ys)
        s7g, _, _, _ = ols_fit(ll_gm[3:], ys[3:])
        s4g, _, _, _ = ols_fit(ll_gm[6:], ys[6:])
        slopes_6w.append(-s6g)
        slopes_10w.append(-s10g)
        slopes_hi.append(-s4g)
        print(f"  {j:2d} | {-j/2:9.2f} | {dahan_slopes_pub[j-1]:9.2f} | {s6g:8.3f} | {s6m:8.3f} | {s10g:9.3f} | {s7g:13.3f} | {s4g:13.3f}")

    # Also print the table of fractions f_W(J) for J = 1..6 in every window
    print("\n=== Fractions f_W(J) = |E_1(w; J)| / |N(w)| per window (J = 1..6 and J = 20) ===")
    print(f"  {'Window':14s} | {'n_primes':>10s} | {'J=1':>7s} | {'J=2':>7s} | {'J=3':>7s} | {'J=4':>7s} | {'J=5':>7s} | {'J=6':>7s} | {'J=20':>9s}")
    for i in range(1, len(windows)):
        name = windows[i][0]
        np_w = counts[i, 0]
        vals = [counts[i, j] / float(np_w) for j in (1, 2, 3, 4, 5, 6, 20)]
        print(f"  {name:14s} | {np_w:10,d} | {vals[0]:7.4f} | {vals[1]:7.4f} | {vals[2]:7.4f} | {vals[3]:7.4f} | {vals[4]:7.4f} | {vals[5]:7.4f} | {vals[6]:9.6f}")

    # Also check the local slope (finite difference) between consecutive windows for J=1..6
    print("\n=== Local slopes -Delta log(f_W(J)) / Delta log log(n_mid) between adjacent windows ===")
    print(f"  {'Transition':24s} | {'J=1':>6s} | {'J=2':>6s} | {'J=3':>6s} | {'J=4':>6s} | {'J=5':>6s} | {'J=6':>6s}")
    for idx in range(len(windows) - 2):
        w1 = windows[idx + 1][0]
        w2 = windows[idx + 2][0]
        dll = ll_gm[idx + 1] - ll_gm[idx]
        ds = []
        for j in range(1, 7):
            f1 = counts[idx + 1, j] / float(counts[idx + 1, 0])
            f2 = counts[idx + 2, j] / float(counts[idx + 2, 0])
            ds.append(-(math.log(f2) - math.log(f1)) / dll)
        print(f"  {w1:11s}->{w2:11s} | {ds[0]:6.2f} | {ds[1]:6.2f} | {ds[2]:6.2f} | {ds[3]:6.2f} | {ds[4]:6.2f} | {ds[5]:6.2f}")

    # 7. PNG plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)
    ax = axes[0]
    js_plot = np.arange(1, 7)
    ax.plot(js_plot, js_plot / 2.0, "k--o", lw=1.8, label="Theoretical prediction (Thm. 4.3): $J/2$")
    ax.plot(js_plot, [-x for x in dahan_slopes_pub], "s-", color="#d62728", lw=1.6, label=r"Dahan, 6 windows ($\leq 2\cdot 10^7$)")
    ax.plot(js_plot, slopes_10w, "^-", color="#1f77b4", lw=1.8, label=r"Extension, 10 windows ($\leq 10^9$)")
    ax.plot(js_plot, slopes_hi, "d-", color="#2ca02c", lw=1.8, label=r"High windows only $[2\cdot 10^7, 10^9)$")
    ax.set_xlabel("Depth $J$")
    ax.set_ylabel(r"Slope $-\mathrm{d}\log(|E_1(w;J)|/|\mathcal{N}(w)|) / \mathrm{d}\log\log n$")
    ax.set_title(r"Empirical sieve dimension of $E_1(N; J)$ up to $10^9$")
    ax.set_xticks(js_plot)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    ax2.plot(lgms_all[:6], [b / l for b, l in zip(betas_all[:6], lgms_all[:6])], "ro-", lw=1.8, label=r"Dahan's 6 windows ($\leq 2\cdot 10^7$)")
    ax2.plot(lgms_all[5:], [b / l for b, l in zip(betas_all[5:], lgms_all[5:])], "bs-", lw=1.8, label=r"Extension ($[10^7, 10^9)$)")
    ax2.axhline(0.1433, color="gray", ls=":", lw=1.2, label=r"$c_2 \approx 0.1433$ (Dahan, Table 2)")
    ax2.set_xlabel(r"$\log n_{\mathrm{mid}}$ of the window")
    ax2.set_ylabel(r"$\beta / \log n_{\mathrm{mid}}$")
    ax2.set_title(r"Stability of $\beta / \log n$ (second regime of eq. 16) up to $10^9$")
    ax2.set_ylim(0.11, 0.19)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig(args.plot)
    print(f"\nPlot saved to {args.plot}")
    print(f"Total script time: {time.time() - t_all:.2f} s")


if __name__ == "__main__":
    main()
