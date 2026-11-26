#!/usr/bin/env python3
"""
Exact computation of m(p) = 4·x_min(p) − p for all hard primes up to 10^10, using the
exact criteria for c = 3, 7 and 15 as shortcuts and the type criterion otherwise (note, §4).
"""

import csv
import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify import types_at, search, is_solution  # noqa: E402


# ---------------------------------------------------------------------------
# 64-bit arithmetic and exact factorization in Numba
# ---------------------------------------------------------------------------

@njit(inline="always")
def nb_gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(inline="always")
def mulmod64(a, b, mod):
    # Safe for mod <= 10^10 (< 2^34): use a decomposition if int64 could overflow
    a %= mod
    b %= mod
    if a < 3037000499 and b < 3037000499:  # sqrt(2^63 - 1)
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
def powmod64(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            res = mulmod64(res, base, mod)
        base = mulmod64(base, base, mod)
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
    # For n < 3.4 * 10^14 the bases {2, 3, 5, 7, 11, 13, 17} suffice (deterministic)
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a >= n:
            continue
        x = powmod64(a, d, n)
        if x == 1 or x == n - 1:
            continue
        comp = True
        for _ in range(s - 1):
            x = mulmod64(x, x, n)
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
    for c in range(1, 30):
        x = 2
        y = 2
        d = 1
        while d == 1:
            x = (mulmod64(x, x, n) + c) % n
            y = (mulmod64(y, y, n) + c) % n
            y = (mulmod64(y, y, n) + c) % n
            diff = x - y if x >= y else y - x
            d = nb_gcd(diff, n)
        if d != n:
            return d
    d = 3
    while d * d <= n:
        if n % d == 0:
            return d
        d += 2
    return n


@njit
def has_sol_c3(K, small_p):
    """Exact criterion for c = 3: does K have a prime factor ≡ 2 (mod 3)?"""
    # Remove factors 2 (2 ≡ 2 mod 3 -> immediate!)
    if (K & 1) == 0:
        return True
    # Try small odd primes (up to ~100)
    n = K
    for i in range(1, len(small_p)):
        q = small_p[i]
        if q * q > n:
            break
        if n % q == 0:
            if q % 3 == 2:
                return True
            while n % q == 0:
                n //= q
    if n == 1:
        return False
    if n % 3 == 2:
        return True  # n must have at least one prime factor ≡ 2 (mod 3)
    if is_prime_64(n):
        return False  # n is a prime ≡ 1 (mod 3)
    # Factor n by Pollard-rho (stack)
    stack = np.empty(32, dtype=np.int64)
    top = 0
    stack[top] = n
    top += 1
    while top > 0:
        top -= 1
        cur = stack[top]
        if cur == 1:
            continue
        if cur % 3 == 2:
            return True
        if is_prime_64(cur):
            continue
        d = pollard_rho(cur)
        stack[top] = d
        top += 1
        stack[top] = cur // d
        top += 1
    return False


@njit
def has_sol_c7(K, small_p):
    """Exact criterion for c = 7 (p ≡ 1 mod 8): does K have a prime with residue mod 7 in {3, 5, 6}?"""
    n = K
    # Remove powers of 2 (2 is a residue mod 7)
    while (n & 1) == 0:
        n >>= 1
    for i in range(1, len(small_p)):
        q = small_p[i]
        if q * q > n:
            break
        if n % q == 0:
            r = q % 7
            if r == 3 or r == 5 or r == 6:
                return True
            while n % q == 0:
                n //= q
    if n == 1:
        return False
    r = n % 7
    if r == 3 or r == 5 or r == 6:
        return True
    if is_prime_64(n):
        return False
    stack = np.empty(32, dtype=np.int64)
    top = 0
    stack[top] = n
    top += 1
    while top > 0:
        top -= 1
        cur = stack[top]
        if cur == 1:
            continue
        r = cur % 7
        if r == 3 or r == 5 or r == 6:
            return True
        if is_prime_64(cur):
            continue
        d = pollard_rho(cur)
        stack[top] = d
        top += 1
        stack[top] = cur // d
        top += 1
    return False


@njit
def has_sol_c15(K, small_p):
    """Exact criterion for c = 15 (p hard mod 840): does K have a prime with residue mod 15 in {7, 11, 13, 14}?"""
    n = K
    while (n & 1) == 0:
        n >>= 1
    for i in range(1, len(small_p)):
        q = small_p[i]
        if q * q > n:
            break
        if n % q == 0:
            r = q % 15
            if r == 7 or r == 11 or r == 13 or r == 14:
                return True
            while n % q == 0:
                n //= q
    if n == 1:
        return False
    r = n % 15
    if r == 7 or r == 11 or r == 13 or r == 14:
        return True
    if is_prime_64(n):
        return False
    stack = np.empty(32, dtype=np.int64)
    top = 0
    stack[top] = n
    top += 1
    while top > 0:
        top -= 1
        cur = stack[top]
        if cur == 1:
            continue
        r = cur % 15
        if r == 7 or r == 11 or r == 13 or r == 14:
            return True
        if is_prime_64(cur):
            continue
        d = pollard_rho(cur)
        stack[top] = d
        top += 1
        stack[top] = cur // d
        top += 1
    return False


@njit
def has_sol_general_c(p, c, small_p, dp, dp2, stack, primes_f):
    """
    Exact type criterion (note, §1) for any c ≡ 3 (mod 4) with p > c:
      Type II at K_c <=> there is r in S with (c - r) % c in S
      Type I  at K_c <=> there is r in S with ((-p) * r) % c in S
    where S = {d mod c : d | K_c}.
    """
    K = (p + c) // 4
    # Fully factor K into primes_f[:nf] (with multiplicity)
    n = K
    nf = 0
    while (n & 1) == 0:
        primes_f[nf] = 2
        nf += 1
        n >>= 1
    for i in range(1, len(small_p)):
        q = small_p[i]
        if q * q > n:
            break
        while n % q == 0:
            primes_f[nf] = q
            nf += 1
            n //= q
    if n > 1:
        top = 0
        stack[top] = n
        top += 1
        while top > 0:
            top -= 1
            cur = stack[top]
            if cur == 1:
                continue
            if is_prime_64(cur):
                primes_f[nf] = cur
                nf += 1
            else:
                d = pollard_rho(cur)
                stack[top] = d
                top += 1
                stack[top] = cur // d
                top += 1

    # Build the exact set of residues S mod c
    for r in range(c):
        dp[r] = False
    dp[1] = True
    for idx in range(nf):
        qmod = primes_f[idx] % c
        if qmod == 1:
            continue
        for r in range(c):
            dp2[r] = dp[r]
        for r in range(c):
            if dp[r]:
                dp2[(r * qmod) % c] = True
        for r in range(c):
            dp[r] = dp2[r]

    # Check Type II and Type I
    neg_p = (c - (p % c)) % c
    for r in range(1, c):
        if dp[r]:
            # Type II: -r mod c
            if dp[c - r]:
                return True
            # Type I: -p * r mod c
            if dp[(neg_p * r) % c]:
                return True
    return False


@njit
def compute_m_exact(p, small_p, dp, dp2, stack, primes_f):
    if has_sol_c3((p + 3) // 4, small_p):
        return 3
    if has_sol_c7((p + 7) // 4, small_p):
        return 7
    if has_sol_general_c(p, 11, small_p, dp, dp2, stack, primes_f):
        return 11
    if has_sol_c15((p + 15) // 4, small_p):
        return 15
    c = 19
    while c < 2000:
        if has_sol_general_c(p, c, small_p, dp, dp2, stack, primes_f):
            return c
        c += 4
    return -1  # should not be reached


@njit
def sieve_base_primes(limit):
    is_p = np.ones(limit + 1, dtype=np.bool_)
    is_p[0] = False
    is_p[1] = False
    lim_s = int(limit ** 0.5) + 1
    for i in range(2, lim_s):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
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


@njit(parallel=True)
def scan_hard_primes(N_max, base_primes, small_p, seg_size=4_000_000):
    """
    Sweeps ALL hard primes p <= N_max in parallel segments.
    Returns:
      - hist[c] = number of hard primes <= N_max with m(p) = c (for c < 2000)
      - seg_max_m[s], seg_max_p[s] = maximum m and smallest prime attaining it
        in segment s (we also store ALL primes with m >= 59, to rebuild
        the exact sequence of records in increasing order of p!).
    """
    n_segs = (N_max + seg_size - 1) // seg_size
    n_threads = 8
    hist_local = np.zeros((n_threads, 2000), dtype=np.int64)
    # Store record candidates: every prime with m(p) >= 59 (or, for p <= 200,000, every prime with m >= 3)
    # and also the maximum of each decade block, for the growth analysis.
    # Expected number of candidates with m >= 59 up to 10^10: a few thousand -> buffer of 50,000 per thread.
    CAP = 50_000
    cand_p = np.empty((n_threads, CAP), dtype=np.int64)
    cand_m = np.empty((n_threads, CAP), dtype=np.int32)
    cand_cnt = np.zeros(n_threads, dtype=np.int64)

    # Also count per decade [10^k, 10^(k+1))
    dec_counts = np.zeros((n_threads, 11), dtype=np.int64)
    dec_max_m = np.zeros((n_threads, 11), dtype=np.int32)
    dec_max_p = np.zeros((n_threads, 11), dtype=np.int64)
    dec_m3 = np.zeros((n_threads, 11), dtype=np.int64)

    # Membership table mod 840
    is_hard = np.zeros(840, dtype=np.bool_)
    for r in (1, 121, 169, 289, 361, 529):
        is_hard[r] = True

    n_base = len(base_primes)

    for tid in prange(n_threads):
        buf = np.empty(seg_size // 2 + 1, dtype=np.uint8)
        dp = np.empty(2048, dtype=np.bool_)
        dp2 = np.empty(2048, dtype=np.bool_)
        stack = np.empty(64, dtype=np.int64)
        primes_f = np.empty(64, dtype=np.int64)

        for seg_idx in range(tid, n_segs, n_threads):
            lo = seg_idx * seg_size
            if lo < 3:
                lo = 3
            if (lo & 1) == 0:
                lo += 1
            hi = (seg_idx + 1) * seg_size - 1
            if hi > N_max:
                hi = N_max
            if lo > hi:
                continue
            length = (hi - lo) // 2 + 1
            for i in range(length):
                buf[i] = 1

            # Sieve with the odd base_primes (base_primes[0] == 2, skip it)
            for bi in range(1, n_base):
                q = base_primes[bi]
                q2 = q * q
                if q2 > hi:
                    break
                start = q2
                if start < lo:
                    rem = lo % q
                    start = lo if rem == 0 else lo + (q - rem)
                    if (start & 1) == 0:
                        start += q
                idx = (start - lo) // 2
                while idx < length:
                    buf[idx] = 0
                    idx += q

            # Run through the hard primes in [lo, hi]
            for i in range(length):
                if buf[i]:
                    p = lo + 2 * i
                    if is_hard[p % 840]:
                        m = compute_m_exact(p, small_p, dp, dp2, stack, primes_f)
                        if m < 0:
                            m = 1999
                        hist_local[tid, m] += 1

                        # Decade
                        dec = 0
                        tmp = p
                        while tmp >= 10:
                            tmp //= 10
                            dec += 1
                        if dec > 10:
                            dec = 10
                        dec_counts[tid, dec] += 1
                        if m == 3:
                            dec_m3[tid, dec] += 1
                        if m > dec_max_m[tid, dec] or (m == dec_max_m[tid, dec] and (dec_max_p[tid, dec] == 0 or p < dec_max_p[tid, dec])):
                            dec_max_m[tid, dec] = m
                            dec_max_p[tid, dec] = p

                        # Store record candidates (all of them for p <= 200,000; only m >= 59 for p > 200,000)
                        if p <= 200_000 or m >= 59:
                            cidx = cand_cnt[tid]
                            if cidx < CAP:
                                cand_p[tid, cidx] = p
                                cand_m[tid, cidx] = m
                                cand_cnt[tid] = cidx + 1

    # Reduce the results
    hist = np.zeros(2000, dtype=np.int64)
    for tid in range(n_threads):
        for c in range(2000):
            hist[c] += hist_local[tid, c]

    tot_cands = 0
    for tid in range(n_threads):
        tot_cands += cand_cnt[tid]
    all_p = np.empty(tot_cands, dtype=np.int64)
    all_m = np.empty(tot_cands, dtype=np.int32)
    pos = 0
    for tid in range(n_threads):
        cnt = cand_cnt[tid]
        for j in range(cnt):
            all_p[pos] = cand_p[tid, j]
            all_m[pos] = cand_m[tid, j]
            pos += 1

    d_cnt = np.zeros(11, dtype=np.int64)
    d_m3 = np.zeros(11, dtype=np.int64)
    d_max_m = np.zeros(11, dtype=np.int32)
    d_max_p = np.zeros(11, dtype=np.int64)
    for dec in range(11):
        for tid in range(n_threads):
            d_cnt[dec] += dec_counts[tid, dec]
            d_m3[dec] += dec_m3[tid, dec]
            if dec_max_m[tid, dec] > d_max_m[dec] or (
                dec_max_m[tid, dec] == d_max_m[dec] and dec_max_p[tid, dec] > 0 and (d_max_p[dec] == 0 or dec_max_p[tid, dec] < d_max_p[dec])
            ):
                d_max_m[dec] = dec_max_m[tid, dec]
                d_max_p[dec] = dec_max_p[tid, dec]

    return hist, all_p, all_m, d_cnt, d_m3, d_max_m, d_max_p


@njit
def validate_array_m(ps, small_p):
    out = np.empty(len(ps), dtype=np.int32)
    dp = np.empty(2048, dtype=np.bool_)
    dp2 = np.empty(2048, dtype=np.bool_)
    stack = np.empty(64, dtype=np.int64)
    primes_f = np.empty(64, dtype=np.int64)
    for i in range(len(ps)):
        out[i] = compute_m_exact(ps[i], small_p, dp, dp2, stack, primes_f)
    return out


def verify_exact_m(p, m_claimed):
    """Checks with tools/verify.py (types_at) that every c < m_claimed fails and c = m_claimed works."""
    for c in range(3, m_claimed, 4):
        t1, t2 = types_at(p, c)
        if t1 or t2:
            return False, f"Earlier solution found at c={c}: {(t1, t2)}"
    t1, t2 = types_at(p, m_claimed)
    if not (t1 or t2):
        return False, f"No solution at c={m_claimed}"
    return True, (t1, t2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    t0 = time.time()

    N_MAX = 10_000_000_000  # 10^10
    base_primes = sieve_base_primes(int(N_MAX ** 0.5) + 100)
    small_p = base_primes[base_primes <= 150].copy()
    print(f"Base primes <= sqrt(10^10): {len(base_primes):,} | small_p <= 150: {len(small_p)} ({time.time() - t0:.2f} s)")

    # 1. Exact row-by-row validation against data/hard_primes_1e8.csv (179,468 primes <= 10^8)
    csv_1e8 = os.path.join(ROOT, "data", "hard_primes_1e8.csv")
    ps_1e8 = []
    ms_1e8 = []
    with open(csv_1e8, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ps_1e8.append(int(row["p"]))
            ms_1e8.append(int(row["m"]))
    ps_arr = np.array(ps_1e8, dtype=np.int64)
    ms_ref = np.array(ms_1e8, dtype=np.int32)
    t_v = time.time()
    ms_got = validate_array_m(ps_arr, small_p)
    disc = int((ms_got != ms_ref).sum())
    print(f"Validation against hard_primes_1e8.csv ({len(ps_arr):,} hard primes <= 10^8): "
          f"discrepancies = {disc} ({time.time() - t_v:.2f} s)")
    assert disc == 0

    # 2. Full sweep up to 10^10
    t_scan = time.time()
    hist, cand_p, cand_m, d_cnt, d_m3, d_max_m, d_max_p = scan_hard_primes(N_MAX, base_primes, small_p)
    t_scan_el = time.time() - t_scan
    tot_hard = int(hist.sum())
    print(f"\nEXACT sweep up to N = {N_MAX:,} finished in {t_scan_el:.2f} s")
    print(f"Total hard primes p <= 10^10: {tot_hard:,}")

    # Sort the candidates by p and build the full sequence of strict records
    order = np.argsort(cand_p)
    cand_p = cand_p[order]
    cand_m = cand_m[order]

    records = []
    cur_max = 0
    for p_val, m_val in zip(cand_p, cand_m):
        if m_val > cur_max:
            cur_max = int(m_val)
            records.append((int(p_val), cur_max))

    print("\n=== FULL SEQUENCE OF RECORDS OF m(p) UP TO 10^10 ===")
    for p_r, m_r in records:
        ok, info = verify_exact_m(p_r, m_r)
        lp = math.log(p_r)
        print(f"  p = {p_r:>13,} (mod 840 = {p_r % 840:>3}) | m(p) = {m_r:>3} | "
              f"log p = {lp:7.4f} | m/log p = {m_r/lp:6.3f} | m/(log p)^2 = {m_r/(lp*lp):.4f} | "
              f"verified={ok} {info}")
        assert ok

    # Top 15 largest m(p) over all of [1, 10^10]
    top_idx = np.argsort(-cand_m)[:15]
    print("\n=== TOP 15 VALUES OF m(p) UP TO 10^10 ===")
    for idx in top_idx:
        p_t, m_t = int(cand_p[idx]), int(cand_m[idx])
        print(f"  m = {m_t:>3}  en p = {p_t:>13,} (mod 840 = {p_t % 840:>3})")

    print("\n=== STATISTICS BY DECADE [10^k, 10^(k+1)) ===")
    for k in range(3, 10):
        cnt = int(d_cnt[k])
        if cnt == 0:
            continue
        f3 = 1.0 - int(d_m3[k]) / cnt
        x_mid = math.sqrt(10 ** k * 10 ** (k + 1))
        print(f"  [10^{k}, 10^{k+1}): primes = {cnt:>10,} | frac(m=3) = {100*int(d_m3[k])/cnt:5.2f}% | "
              f"F3*sqrt(log x_mid) = {f3*math.sqrt(math.log(x_mid)):.4f} | "
              f"max m = {int(d_max_m[k]):>3} at p = {int(d_max_p[k]):,}")

    print("\n=== FREQUENCY DISTRIBUTION OF m(p) UP TO 10^10 ===")
    nonzero_c = [c for c in range(3, 2000, 4) if hist[c] > 0]
    print(f"  Distinct values of m observed: {len(nonzero_c)} (maximum = {max(nonzero_c)})")
    for c in nonzero_c:
        print(f"  m = {c:>3}: {int(hist[c]):>10,} ({100.0 * int(hist[c]) / tot_hard:7.4f}%)")

    # Growth fits of the records against log p and a cautious extrapolation to 10^18
    print("\n=== GROWTH FIT OF THE RECORDS AGAINST log p ===")
    lps = np.array([math.log(p) for p, _ in records], dtype=float)
    ms_r = np.array([m for _, m in records], dtype=float)
    # 1. Linear in log p: m ≈ a * log p + b
    a1, b1 = np.polyfit(lps, ms_r, 1)
    r2_1 = 1.0 - ((ms_r - (a1 * lps + b1)) ** 2).sum() / ((ms_r - ms_r.mean()) ** 2).sum()
    # 2. Quadratic (log p)^2: m ≈ a * (log p)^2 + b
    a2, b2 = np.polyfit(lps ** 2, ms_r, 1)
    r2_2 = 1.0 - ((ms_r - (a2 * lps ** 2 + b2)) ** 2).sum() / ((ms_r - ms_r.mean()) ** 2).sum()
    # 3. Power law m ≈ c * (log p)^gamma (in log-log)
    g3, lc3 = np.polyfit(np.log(lps), np.log(ms_r), 1)
    c3 = math.exp(lc3)
    r2_3 = 1.0 - ((np.log(ms_r) - (g3 * np.log(lps) + lc3)) ** 2).sum() / ((np.log(ms_r) - np.log(ms_r).mean()) ** 2).sum()
    # 4. Linear in (log p)(log log p)
    u4 = lps * np.log(lps)
    a4, b4 = np.polyfit(u4, ms_r, 1)
    r2_4 = 1.0 - ((ms_r - (a4 * u4 + b4)) ** 2).sum() / ((ms_r - ms_r.mean()) ** 2).sum()

    lp18 = math.log(1e18)
    print(f"  Linear     m ≈ {a1:.3f} log p + ({b1:.2f})        [R^2={r2_1:.4f}] -> ext(10^18) ≈ {a1*lp18+b1:.0f}")
    print(f"  Quadratic  m ≈ {a2:.4f} (log p)^2 + ({b2:.2f})   [R^2={r2_2:.4f}] -> ext(10^18) ≈ {a2*lp18**2+b2:.0f}")
    print(f"  log*loglog m ≈ {a4:.3f} log p log log p + ({b4:.2f}) [R^2={r2_4:.4f}] -> ext(10^18) ≈ {a4*lp18*math.log(lp18)+b4:.0f}")
    print(f"  Power law  m ≈ {c3:.4f} (log p)^{g3:.3f}        [R^2_log={r2_3:.4f}] -> ext(10^18) ≈ {c3*(lp18**g3):.0f}")

    # Save the aggregated CSV (records + histogram) to data/records_m_1e10.csv
    out_csv = os.path.join(ROOT, "data", "records_m_1e10.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "m", "p_or_count", "extra"])
        for p_r, m_r in records:
            w.writerow(["record", m_r, p_r, f"mod840={p_r%840}"])
        for c in nonzero_c:
            w.writerow(["hist", c, int(hist[c]), f"{int(hist[c])/tot_hard:.8f}"])
    print(f"\nSaved: {out_csv} ({os.path.getsize(out_csv):,} bytes)")
    print(f"Total time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
