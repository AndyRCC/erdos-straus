#!/usr/bin/env python3
"""
Exact number of solutions f_m(p) (ordered triples) for every prime p <= 2·10^5 and
m = 4, 5, 6, 7, 8 (note, §7). Saves data/fm_primes_2e5.csv.
"""

import csv
import math
import os
import sys
import time
from math import gcd

import numpy as np
from numba import njit, prange

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bright_loughran import build_spf, all_positive_solutions  # noqa: E402


# ---------------------------------------------------------------------------
# Fast integer arithmetic in Numba (64 bits; for M <= 10^11 we use an __int128-safe mulmod)
# ---------------------------------------------------------------------------

@njit(inline="always")
def nb_gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(inline="always")
def mulmod(a, b, mod):
    # Safe for mod < 2^62 without overflowing uint64, using a 30-bit decomposition
    a %= mod
    b %= mod
    if a < (1 << 31) and b < (1 << 31):
        return (a * b) % mod
    res = 0
    while b > 0:
        if b & 1:
            res += a
            if res >= mod:
                res -= mod
        a <<= 1
        if a >= mod:
            a -= mod
        b >>= 1
    return res


@njit(inline="always")
def powmod(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            res = mulmod(res, base, mod)
        base = mulmod(base, base, mod)
        exp >>= 1
    return res


@njit
def is_prime_64(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
    # Deterministic for every n < 3.317 * 10^24 with bases <= 37 (here n < 10^12, so {2, 3, 5, 7, 11, 13} suffices)
    for a in (2, 3, 5, 7, 11, 13):
        if a >= n:
            continue
        x = powmod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        comp = True
        for _ in range(s - 1):
            x = mulmod(x, x, n)
            if x == n - 1:
                comp = False
                break
        if comp:
            return False
    return True


@njit
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    for c in range(1, 20):
        x = 2
        y = 2
        d = 1
        while d == 1:
            x = (mulmod(x, x, n) + c) % n
            y = (mulmod(y, y, n) + c) % n
            y = (mulmod(y, y, n) + c) % n
            diff = x - y if x >= y else y - x
            d = nb_gcd(diff, n)
        if d != n:
            return d
    # Fallback trial division (should not happen)
    d = 3
    while d * d <= n:
        if n % d == 0:
            return d
        d += 2
    return n


@njit
def factor_64(n, spf, spf_lim, primes_out, exps_out):
    """Factors n >= 1 into primes_out[:k], exps_out[:k]. Returns k."""
    k = 0
    if n <= 1:
        return 0
    # If it fits in the SPF table, instantaneous
    if n <= spf_lim:
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            primes_out[k] = p
            exps_out[k] = e
            k += 1
        return k

    # Extract small factors
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            primes_out[k] = p
            exps_out[k] = e
            k += 1
            if n <= spf_lim:
                break

    if n <= 1:
        return k
    if n <= spf_lim:
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            primes_out[k] = p
            exps_out[k] = e
            k += 1
        return k

    # Stack for the remaining factors via Pollard-rho
    stack = np.empty(64, dtype=np.int64)
    top = 0
    stack[top] = n
    top += 1

    raw = np.empty(64, dtype=np.int64)
    nraw = 0
    while top > 0:
        top -= 1
        cur = stack[top]
        if cur == 1:
            continue
        if cur <= spf_lim:
            while cur > 1:
                p = spf[cur]
                raw[nraw] = p
                nraw += 1
                cur //= p
        elif is_prime_64(cur):
            raw[nraw] = cur
            nraw += 1
        else:
            d = pollard_rho(cur)
            stack[top] = d
            top += 1
            stack[top] = cur // d
            top += 1

    # Group the prime factors of raw[:nraw] (insertion sort)
    for i in range(1, nraw):
        v = raw[i]
        j = i - 1
        while j >= 0 and raw[j] > v:
            raw[j + 1] = raw[j]
            j -= 1
        raw[j + 1] = v

    i = 0
    while i < nraw:
        p = raw[i]
        e = 1
        i += 1
        while i < nraw and raw[i] == p:
            e += 1
            i += 1
        primes_out[k] = p
        exps_out[k] = e
        k += 1
    return k


@njit
def build_spf_nb(limit):
    spf = np.arange(limit + 1, dtype=np.int32)
    lim_s = int(limit ** 0.5) + 1
    for i in range(2, lim_s):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


@njit
def sieve_primes_bool(limit):
    is_p = np.ones(limit + 1, dtype=np.bool_)
    is_p[0] = False
    is_p[1] = False
    lim_s = int(limit ** 0.5) + 1
    for i in range(2, lim_s):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
    return is_p


@njit(parallel=True)
def count_type2_all_primes(m, N, is_p, spf):
    """
    Counts 3 * f_{m, II}(p) (ordered triples) for every prime m < p <= N.
    Equation: m*a*b*d = p + e, c*e = a + b, 1 <= a <= b, gcd(a, b) = 1, a*b*d <= N // (m - 2).
    """
    X2 = N // (m - 2)
    a_max = int(X2 ** 0.5)
    # Partition by a in parallel, with per-block (lock-free) accumulators, one per thread
    n_threads = 8
    local_counts = np.zeros((n_threads, N + 1), dtype=np.int32)
    spf_lim = len(spf) - 1

    for tid in prange(n_threads):
        pr = np.empty(32, dtype=np.int64)
        ex = np.empty(32, dtype=np.int64)
        divs = np.empty(2048, dtype=np.int64)
        for a in range(1 + tid, a_max + 1, n_threads):
            b_max = X2 // a
            for b in range(a, b_max + 1):
                if nb_gcd(a, b) != 1:
                    continue
                s = a + b
                # Factor s and generate every divisor e | s
                nf = factor_64(s, spf, spf_lim, pr, ex)
                nd = 1
                divs[0] = 1
                for fi in range(nf):
                    pf = pr[fi]
                    ef = ex[fi]
                    old_nd = nd
                    pe = 1
                    for _ in range(ef):
                        pe *= pf
                        for idx in range(old_nd):
                            divs[nd] = divs[idx] * pe
                            nd += 1
                ab = a * b
                d_max = X2 // ab
                weight = 3 if a == b else 6
                for d in range(1, d_max + 1):
                    mabd = m * ab * d
                    for di in range(nd):
                        e = divs[di]
                        p = mabd - e
                        if m < p <= N and is_p[p]:
                            c = s // e
                            if a % p != 0 and b % p != 0 and c % p != 0 and d % p != 0:
                                local_counts[tid, p] += weight

    counts = np.zeros(N + 1, dtype=np.int32)
    for p in range(m + 1, N + 1):
        if is_p[p]:
            tot = 0
            for tid in range(n_threads):
                tot += local_counts[tid, p]
            counts[p] = tot
    return counts


@njit(parallel=True)
def count_type1_all_primes(m, N, is_p, spf):
    """
    Counts 3 * f_{m, I}(p) (ordered triples) for every prime m < p <= N.
    Equation: a*c*d <= 2N // (m - 1), f | (m*a^2*d + 1), e = (m*a^2*d + 1)/f,
    b = c*e - a >= a, gcd(a, c) = gcd(a, b) = 1, p = m*a*c*d - f.
    """
    X1 = (2 * N) // (m - 1)
    n_threads = 8
    local_counts = np.zeros((n_threads, N + 1), dtype=np.int32)
    spf_lim = len(spf) - 1

    for tid in prange(n_threads):
        pr = np.empty(32, dtype=np.int64)
        ex = np.empty(32, dtype=np.int64)
        divs = np.empty(4096, dtype=np.int64)
        # Cyclic distribution over a to balance the load (large a has few d but a larger M)
        for a in range(1 + tid, X1 + 1, n_threads):
            d_max = X1 // a
            for d in range(1, d_max + 1):
                ad = a * d
                c_max = X1 // ad
                f_upper = m * ad * c_max - (m + 1)
                if f_upper < 1:
                    continue
                M = m * a * ad + 1
                nf = factor_64(M, spf, spf_lim, pr, ex)
                # Generate the divisors f | M with f <= f_upper
                nd = 1
                divs[0] = 1
                for fi in range(nf):
                    pf = pr[fi]
                    ef = ex[fi]
                    old_nd = nd
                    pe = 1
                    for _ in range(ef):
                        pe *= pf
                        if pe > f_upper:
                            break
                        for idx in range(old_nd):
                            v = divs[idx] * pe
                            if v <= f_upper:
                                divs[nd] = v
                                nd += 1

                mad = m * ad
                two_a = 2 * a
                for di in range(nd):
                    f = divs[di]
                    e = M // f
                    # We need c*e >= 2*a and m*a*d*c - f > m and m*a*d*c - f <= N
                    c_lo1 = (two_a + e - 1) // e
                    c_lo2 = (f + m + 1 + mad - 1) // mad
                    c_lo = c_lo1 if c_lo1 > c_lo2 else c_lo2
                    if c_lo < 1:
                        c_lo = 1
                    c_hi1 = (N + f) // mad
                    c_hi = c_hi1 if c_hi1 < c_max else c_max
                    if c_lo > c_hi:
                        continue
                    for c in range(c_lo, c_hi + 1):
                        if nb_gcd(a, c) != 1:
                            continue
                        p = mad * c - f
                        if m < p <= N and is_p[p]:
                            b = c * e - a
                            if b >= a and nb_gcd(a, b) == 1:
                                if a % p != 0 and b % p != 0 and c % p != 0 and d % p != 0:
                                    local_counts[tid, p] += 3 if a == b else 6

    counts = np.zeros(N + 1, dtype=np.int32)
    for p in range(m + 1, N + 1):
        if is_p[p]:
            tot = 0
            for tid in range(n_threads):
                tot += local_counts[tid, p]
            counts[p] = tot
    return counts


def brute_ordered_fm(m, p, spf_py):
    """Counts ORDERED triples (x, y, z) in N^3 with m/p = 1/x + 1/y + 1/z by brute force."""
    sols = all_positive_solutions(m, p, spf_py)
    tot = 0
    for u1, u2, u3 in sols:
        if u1 == u2 == u3:
            tot += 1
        elif u1 == u2 or u2 == u3:
            tot += 3
        else:
            tot += 6
    return tot


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    t0 = time.time()

    N = 200_000
    VAL_N = 5_000
    MS = (4, 5, 6, 7, 8)

    spf_nb = build_spf_nb(1_000_000)
    is_p = sieve_primes_bool(N)
    primes = [int(p) for p in np.nonzero(is_p)[0]]
    print(f"Primes <= {N:,}: {len(primes):,} (SPF and sieve in {time.time() - t0:.2f} s)")

    # 1. Exact computation with Numba for N = 200,000 and m = 4..8
    spf_py_small = build_spf(200_000)
    results = {}
    for m in MS:
        tm = time.time()
        c1 = count_type1_all_primes(m, N, is_p, spf_nb)
        c2 = count_type2_all_primes(m, N, is_p, spf_nb)
        tot = c1 + c2
        # For p <= m (p in {2, 3, 5, 7}), use exact brute force
        for p in (2, 3, 5, 7):
            if p <= m:
                tot[p] = brute_ordered_fm(m, p, spf_py_small)
                c1[p] = -1
                c2[p] = -1
        results[m] = (tot, c1, c2)
        excs = [p for p in primes if tot[p] == 0]
        print(f"  m={m}: computed in {time.time() - tm:.2f} s | "
              f"sum f_m(p)={int(tot[primes].sum()):,} | prime exceptions <= {N:,}: {excs}")

    # 2. Exhaustive validation against brute force on ALL primes p <= 5,000
    t_val = time.time()
    # Largest B in all_positive_solutions for p <= 5000, m >= 4: 5000 * 3750 = 18.75M -> use an spf table of 5000*4000?
    # Better: factor_int from bright_loughran.py (spf=200k has a fallback) or spf=5M (in 1 s)
    spf_val = build_spf(5_000_000)
    val_primes = [p for p in primes if p <= VAL_N]
    disc = 0
    for m in MS:
        tot = results[m][0]
        for p in val_primes:
            bf = brute_ordered_fm(m, p, spf_val)
            if int(tot[p]) != bf:
                print(f"  DISCREPANCY m={m}, p={p}: numba={int(tot[p])}, bf={bf}")
                disc += 1
                if disc >= 10:
                    break
    # Also check the explicit values of Elsholtz-Tao p. 2: f_4(2)=3, f_4(3)=12, f_4(5)=12, f_4(7)=36
    et_check = {p: int(results[4][0][p]) for p in (2, 3, 5, 7)}
    print(f"\nValidation against brute force (all {len(val_primes)} primes <= {VAL_N:,}, m=4..8, "
          f"{len(val_primes)*len(MS)} pairs): discrepancies = {disc} ({time.time() - t_val:.2f} s)")
    print(f"Elsholtz-Tao control p. 2: f_4(2,3,5,7) = {et_check} (expected {{2: 3, 3: 12, 5: 12, 7: 36}})")
    assert disc == 0 and et_check == {2: 3, 3: 12, 5: 12, 7: 36}

    # 3. Save the CSV to data/fm_primes_2e5.csv
    out_csv = os.path.join(ROOT, "data", "fm_primes_2e5.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p", "f4", "f4_I", "f4_II", "f5", "f6", "f7", "f8"])
        for p in primes:
            w.writerow([
                p,
                int(results[4][0][p]), int(results[4][1][p]), int(results[4][2][p]),
                int(results[5][0][p]), int(results[6][0][p]),
                int(results[7][0][p]), int(results[8][0][p]),
            ])
    print(f"Saved: {out_csv} ({os.path.getsize(out_csv):,} bytes)")

    # 4. Quick statistical summary by m (primes p <= 200,000, p > m)
    print("\n=== STATISTICAL SUMMARY f_m(p) / (log p)^3 (primes 100 <= p <= 200,000) ===")
    p_arr = np.array([p for p in primes if p >= 100], dtype=np.float64)
    l3 = np.log(p_arr) ** 3
    p_idx = p_arr.astype(np.int64)
    for m in MS:
        vals = results[m][0][p_idx].astype(np.float64)
        ratio = vals / l3
        # Separate p ≡ 1 (mod 24) from all primes
        m24 = (p_idx % 24 == 1)
        print(f"  m={m}: mean f_m/(log p)^3 = {ratio.mean():.5f} (median {np.median(ratio):.5f}, "
              f"min f_m={int(vals.min())}, max={int(vals.max())}) | "
              f"at p≡1 mod 24: mean = {ratio[m24].mean():.5f} (min f_m={int(vals[m24].min())})")

    print(f"\nTotal time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
