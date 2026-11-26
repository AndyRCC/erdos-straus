#!/usr/bin/env python3
"""
Extension of the exact sweep of m(p) to all hard primes up to 10^11, checked
against data/records_m_1e10.csv on [1, 10^10] (note, §5.1).
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
from verify import types_at  # noqa: E402


# ---------------------------------------------------------------------------
# Exact and fast 64-bit arithmetic for mod < 2^42
# ---------------------------------------------------------------------------

@njit(inline="always")
def nb_gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(inline="always")
def mulmod64_fast(a, b, mod):
    """Exact, without int64 overflow, for every mod < 2^42 (here mod <= 10^11 < 2^37)."""
    a %= mod
    b %= mod
    if a < 3037000499 and b < 3037000499:
        return (a * b) % mod
    # a = a_hi * 2^20 + a_lo; a_hi < 2^17, b < 2^37 => a_hi*b < 2^54 < 2^63
    # ((a_hi*b) % mod) < 2^37 => << 20 < 2^57 < 2^63; a_lo*b < 2^57 < 2^63
    a_hi = a >> 20
    a_lo = a & 1048575
    t1 = ((a_hi * b) % mod) << 20
    t2 = a_lo * b
    return (t1 + t2) % mod


@njit(inline="always")
def powmod64(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            res = mulmod64_fast(res, base, mod)
        base = mulmod64_fast(base, base, mod)
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
    # Deterministic for every n < 3.41 * 10^14 with bases {2, 3, 5, 7, 11, 13, 17}
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a >= n:
            continue
        x = powmod64(a, d, n)
        if x == 1 or x == n - 1:
            continue
        comp = True
        for _ in range(s - 1):
            x = mulmod64_fast(x, x, n)
            if x == n - 1:
                comp = False
                break
        if comp:
            return False
    return True


@njit
def pollard_rho(n):
    if (n & 1) == 0:
        return 2
    for c in range(1, 30):
        x = 2
        y = 2
        d = 1
        while d == 1:
            x = (mulmod64_fast(x, x, n) + c) % n
            y = (mulmod64_fast(y, y, n) + c) % n
            y = (mulmod64_fast(y, y, n) + c) % n
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
    if (K & 1) == 0:
        return True
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
    n = K
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
    K = (p + c) // 4
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

    neg_p = (c - (p % c)) % c
    for r in range(1, c):
        if dp[r]:
            if dp[c - r]:
                return True
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
    return -1


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


@njit
def modinv_prime(a, p):
    """Inverse of a mod a prime p via Fermat (a^(p-2) mod p)."""
    return powmod64(a, p - 2, p)


@njit(parallel=True)
def scan_mod840_range(N_lo, N_hi, sieve_primes, inv840, small_p, seg_k=500_000):
    """
    Sweeps EXACTLY all hard primes p in (N_lo, N_hi] using a direct sieve
    over the 6 classes p = 840*k + r (r in {1, 121, 169, 289, 361, 529}).
    """
    hard_res = np.array([1, 121, 169, 289, 361, 529], dtype=np.int64)
    k_min = N_lo // 840
    k_max = N_hi // 840
    n_k = k_max - k_min + 1
    n_segs = (n_k + seg_k - 1) // seg_k
    n_threads = 8

    hist_local = np.zeros((n_threads, 2000), dtype=np.int64)
    CAP = 50_000
    cand_p = np.empty((n_threads, CAP), dtype=np.int64)
    cand_m = np.empty((n_threads, CAP), dtype=np.int32)
    cand_cnt = np.zeros(n_threads, dtype=np.int64)

    n_sp = len(sieve_primes)

    for tid in prange(n_threads):
        buf = np.empty(seg_k, dtype=np.uint8)
        dp = np.empty(2048, dtype=np.bool_)
        dp2 = np.empty(2048, dtype=np.bool_)
        stack = np.empty(64, dtype=np.int64)
        primes_f = np.empty(64, dtype=np.int64)

        for seg_idx in range(tid, n_segs, n_threads):
            k0 = k_min + seg_idx * seg_k
            k1 = k0 + seg_k - 1
            if k1 > k_max:
                k1 = k_max
            length = k1 - k0 + 1
            p_max_seg = 840 * k1 + 529

            for ri in range(6):
                r = hard_res[ri]
                for i in range(length):
                    buf[i] = 1
                # Sieve the k in [k0, k1] for which 840*k + r is composite
                for pi in range(n_sp):
                    q = sieve_primes[pi]
                    q2 = q * q
                    if q2 > p_max_seg:
                        break
                    # 840*k + r ≡ 0 (mod q) <=> k ≡ (-r) * inv840[pi] (mod q)
                    rem = (q - (r % q)) % q
                    k_root = (rem * inv840[pi]) % q
                    # First k >= k0 with k ≡ k_root (mod q)
                    base = k0 % q
                    off = k_root - base
                    if off < 0:
                        off += q
                    k_first = k0 + off
                    # Do not cross out q itself (if 840*k_first + r == q)
                    if 840 * k_first + r == q:
                        off += q
                    pos = off
                    while pos < length:
                        buf[pos] = 0
                        pos += q

                for i in range(length):
                    if buf[i]:
                        p = 840 * (k0 + i) + r
                        if p <= N_lo or p > N_hi or p < 11:
                            continue
                        m = compute_m_exact(p, small_p, dp, dp2, stack, primes_f)
                        if m < 0:
                            m = 1999
                        hist_local[tid, m] += 1
                        if m >= 75:
                            cidx = cand_cnt[tid]
                            if cidx < CAP:
                                cand_p[tid, cidx] = p
                                cand_m[tid, cidx] = m
                                cand_cnt[tid] = cidx + 1

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
    return hist, all_p, all_m


def verify_exact_m(p, m_claimed):
    for c in range(3, m_claimed, 4):
        t1, t2 = types_at(p, c)
        if t1 or t2:
            return False, f"Earlier solution at c={c}: {(t1, t2)}"
    t1, t2 = types_at(p, m_claimed)
    if not (t1 or t2):
        return False, f"No solution at c={m_claimed}"
    return True, (t1, t2)


def main():
    t0 = time.time()
    N_10 = 10_000_000_000
    N_11 = 100_000_000_000
    base_primes = sieve_base_primes(int(N_11 ** 0.5) + 100)
    # Sieving primes q >= 11 (2, 3, 5, 7 are already excluded by gcd(r, 840) = 1)
    sieve_primes = base_primes[base_primes >= 11].copy()
    inv840 = np.array([pow(840 % int(q), int(q) - 2, int(q)) for q in sieve_primes], dtype=np.int64)
    small_p = base_primes[base_primes <= 150].copy()
    print(f"Base primes <= sqrt(10^11): {len(base_primes):,}, sieve_primes >= 11: {len(sieve_primes):,} ({time.time()-t0:.2f} s)", flush=True)

    # 1. Exact validation against records_m_1e10.csv on [1, 10^10]
    t_val = time.time()
    hist10, cp10, cm10 = scan_mod840_range(0, N_10, sieve_primes, inv840, small_p)
    tot10 = int(hist10.sum())
    ref_hist = {}
    with open(os.path.join(ROOT, "data", "records_m_1e10.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kind"] == "hist":
                ref_hist[int(row["m"])] = int(row["p_or_count"])
    disc = 0
    for c in range(3, 2000, 4):
        if int(hist10[c]) != ref_hist.get(c, 0):
            disc += 1
            print(f"DISCREPANCY at 10^10 for m={c}: got={hist10[c]}, ref={ref_hist.get(c, 0)}", flush=True)
    print(f"Validation on [1, 10^10]: {tot10:,} hard primes, histogram discrepancies = {disc} ({time.time()-t_val:.2f} s)", flush=True)
    assert disc == 0 and tot10 == 14_215_707

    # 2. EXACT sweep of the decade (10^10, 10^11] in blocks of 10^10, to report progress
    hist_dec = np.zeros(2000, dtype=np.int64)
    all_cp = [cp10]
    all_cm = [cm10]
    t_scan = time.time()
    for blk in range(1, 10):
        lo = blk * N_10
        hi = (blk + 1) * N_10
        tb = time.time()
        hb, cpb, cmb = scan_mod840_range(lo, hi, sieve_primes, inv840, small_p)
        hist_dec += hb
        all_cp.append(cpb)
        all_cm.append(cmb)
        b_max_m = int(cmb.max()) if len(cmb) > 0 else 0
        b_max_p = int(cpb[np.argmax(cmb)]) if len(cmb) > 0 else 0
        print(f"  Block ({blk}*10^10, {blk+1}*10^10]: {int(hb.sum()):>10,} primes, max m = {b_max_m:>3} at p = {b_max_p:,} ({time.time()-tb:.2f} s)", flush=True)

    hist_tot = hist10 + hist_dec
    tot_11 = int(hist_tot.sum())
    tot_dec = int(hist_dec.sum())
    cand_p = np.concatenate(all_cp)
    cand_m = np.concatenate(all_cm)
    order = np.argsort(cand_p)
    cand_p = cand_p[order]
    cand_m = cand_m[order]

    print(f"\nEXACT sweep up to 10^11 finished in {time.time() - t_scan:.2f} s", flush=True)
    print(f"Total hard primes <= 10^11: {tot_11:,} (in (10^10, 10^11]: {tot_dec:,})", flush=True)

    # Statistics of the decade [10^10, 10^11)
    f3_dec = 1.0 - int(hist_dec[3]) / tot_dec
    x_mid = math.sqrt(1e10 * 1e11)
    print(f"Decade [10^10, 10^11): frac(m=3) = {100*int(hist_dec[3])/tot_dec:.2f}%, F3*sqrt(log x_mid) = {f3_dec*math.sqrt(math.log(x_mid)):.4f}", flush=True)

    # Top 20 values of m(p) up to 10^11 + check with types_at of every m >= 83!
    top_idx = np.argsort(-cand_m)[:20]
    print("\n=== TOP 20 VALUES OF m(p) UP TO 10^11 (checked with types_at) ===", flush=True)
    for idx in top_idx:
        pt, mt = int(cand_p[idx]), int(cand_m[idx])
        ok, info = verify_exact_m(pt, mt)
        assert ok, (pt, mt, info)
        print(f"  m = {mt:>3} at p = {pt:>14,} (mod 840 = {pt % 840:>3}) | verified={ok} {info}", flush=True)

    # Check whether there is a new record > 107!
    max_m_11 = int(cand_m.max())
    print(f"\n>>> MAXIMUM m(p) UP TO 10^11: m = {max_m_11} <<<", flush=True)
    new_recs = [(int(p), int(m)) for p, m in zip(cand_p, cand_m) if p > N_10 and m >= 107]
    print(f"Primes in (10^10, 10^11] with m >= 107: {new_recs}", flush=True)

    # Full frequency distribution up to 10^11
    print("\n=== FREQUENCY DISTRIBUTION OF m(p) UP TO 10^11 ===", flush=True)
    nonzero_c = [c for c in range(3, 2000, 4) if hist_tot[c] > 0]
    for c in nonzero_c:
        print(f"  m = {c:>3}: {int(hist_tot[c]):>11,} ({100.0 * int(hist_tot[c]) / tot_11:8.5f}%)", flush=True)

    out_csv = os.path.join(ROOT, "data", "records_m_1e11.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "m", "p_or_count", "extra"])
        # Strict records (include the known ones <= 10^10 + any > 107)
        base_recs = [(1009, 3), (1129, 11), (1201, 23), (21169, 31), (118801, 59), (8803369, 107)]
        cur_max = 107
        for p, m in zip(cand_p, cand_m):
            if p > N_10 and m > cur_max:
                cur_max = int(m)
                base_recs.append((int(p), cur_max))
        for pr, mr in base_recs:
            w.writerow(["record", mr, pr, f"mod840={pr%840}"])
        for c in nonzero_c:
            w.writerow(["hist", c, int(hist_tot[c]), f"{int(hist_tot[c])/tot_11:.9f}"])
    print(f"\nSaved {out_csv}. Total time: {time.time() - t0:.2f} s", flush=True)


if __name__ == "__main__":
    main()
