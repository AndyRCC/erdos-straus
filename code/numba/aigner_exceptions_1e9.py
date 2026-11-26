#!/usr/bin/env python3
"""
Exceptions of the Aigner slice (type II with a = 1) for all primes p ≡ 1 (mod 24)
up to 10^9: filtering by progressions (cd <= 600) and an exhaustive O(√p log p) decision
(note, §3).
"""
import argparse
import csv
import math
import os
import platform
import sys
import time

import numba as nb
import numpy as np
from sympy import divisors, factorint, primerange

sys.path.insert(0, "tools")
from verify import is_solution, profile


@nb.njit(cache=True)
def _gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


@nb.njit(cache=True)
def _modinv(a, m):
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
    r = int(math.sqrt(float(limit)))
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
def has_aigner_II_fast(p):
    """
    EXHAUSTIVE decision in O(sqrt(p) log p), without factoring, for p ≡ 1 (mod 4) prime (p >= 5).
    Returns (ok, c, d, e) with e*(4*c*d - 1) = p + 4*d, or (False, 0, 0, 0) if there is NONE.
    """
    S = int(math.sqrt(float(p)))
    while (S + 1) * (S + 1) <= p:
        S += 1
    while S * S > p:
        S -= 1

    # Branch 1: e <= S, e ≡ 3 (mod 4)
    if S >= 3:
        S3 = S - ((S - 3) % 4)
        K_min = (p + 3) // 4
        K_max = (p + S3) // 4
        D_max = int(math.sqrt(float(K_max)))
        while (D_max + 1) * (D_max + 1) <= K_max:
            D_max += 1
        for d in range(1, D_max + 1):
            # First multiple of d in [K_min, K_max]
            first = ((K_min + d - 1) // d) * d
            K = first
            while K <= K_max:
                e = 4 * K - p
                q = K // d
                if (d + 1) % e == 0:
                    # b = d, d_param = q, c = (b+1)/e
                    return True, (d + 1) // e, q, e
                if (q + 1) % e == 0:
                    # b = q, d_param = d, c = (b+1)/e
                    return True, (q + 1) // e, d, e
                K += d

    # Branch 2: e > S => c*d < (3/8)*(sqrt(p) + 1) < (3*(S + 2)) / 8
    M = (3 * (S + 2)) // 8
    for c in range(1, M + 1):
        d_lim = M // c
        for d in range(1, d_lim + 1):
            denom = 4 * c * d - 1
            num = p + 4 * d
            if num % denom == 0:
                e = num // denom
                return True, c, d, e

    return False, 0, 0, 0


def has_aigner_II_bruteforce(p):
    """Brute-force search over every e ≡ 3 (mod 4), 3 <= e <= (p + 4)//3."""
    emax = (p + 4) // 3
    for e in range(3, emax + 1, 4):
        Ke = (p + e) // 4
        for b in divisors(Ke):
            if b % e == e - 1:
                return True, (b + 1) // e, Ke // b, e
    return False, 0, 0, 0


@nb.njit(cache=True)
def precompute_mark_pairs(cd_max):
    """Precomputes (m_step, r_k, n_min) for each (c, d) with c*d <= cd_max."""
    # Count valid pairs
    cnt = 0
    for c in range(1, cd_max + 1):
        for d in range(1, cd_max // c + 1):
            denom = 4 * c * d - 1
            # n = e*denom - 4*d ≡ -4*d (mod denom): compatible with n ≡ 1 (mod 24)?
            # denom is odd; if 3 | denom, it requires (-4*d) ≡ 1 (mod 3) <=> d ≡ 2 (mod 3)
            if denom % 3 == 0 and (d % 3 != 2):
                continue
            cnt += 1
    steps = np.empty(cnt, dtype=np.int64)
    rems = np.empty(cnt, dtype=np.int64)
    nmins = np.empty(cnt, dtype=np.int64)
    idx = 0
    for c in range(1, cd_max + 1):
        for d in range(1, cd_max // c + 1):
            denom = 4 * c * d - 1
            if denom % 3 == 0:
                if d % 3 != 2:
                    continue
                m_red = denom // 3
                # 24*k + 1 + 4*d ≡ 0 (mod denom)  =>  8*k + (1+4d)//3 ≡ 0 (mod m_red)
                if m_red == 1:
                    r_k = 0
                else:
                    inv8 = _modinv(8, m_red)
                    rhs = (-((1 + 4 * d) // 3)) % m_red
                    r_k = (rhs * inv8) % m_red
                step_k = m_red
            else:
                inv24 = _modinv(24, denom)
                rhs = (-(1 + 4 * d)) % denom
                r_k = (rhs * inv24) % denom
                step_k = denom
            # We require e >= 3 (e ≡ 3 mod 4 is guaranteed by n ≡ 1 mod 4 and denom ≡ 3 mod 4)
            n_min = 3 * denom - 4 * d
            steps[idx] = step_k
            rems[idx] = r_k
            nmins[idx] = n_min
            idx += 1
    return steps, rems, nmins


@nb.njit(cache=True)
def process_segment_T2(seg_idx, seg_k_len, limit_n, small_primes, steps, rems, nmins):
    """
    Phase 1 + Phase 2 on a segment of k (n = 24*k + 1 <= limit_n).
    Returns: (n_primes_seg, n_survivors_phase1, array_of_exceptions)
    """
    k0 = seg_idx * seg_k_len
    k1 = k0 + seg_k_len
    n_lo = 24 * k0 + 1
    if n_lo > limit_n:
        return 0, 0, np.empty(0, dtype=np.int64)
    max_k_valid = (limit_n - 1) // 24
    eff_len = seg_k_len
    if k0 + eff_len - 1 > max_k_valid:
        eff_len = max_k_valid - k0 + 1
    n_hi = 24 * (k0 + eff_len - 1) + 1

    # 1. Sieve of the primes n = 24*k + 1
    is_prime = np.ones(eff_len, dtype=np.uint8)
    if k0 == 0:
        is_prime[0] = 0
    lim_p = int(math.sqrt(float(n_hi))) + 1
    for i in range(len(small_primes)):
        p = small_primes[i]
        if p > lim_p:
            break
        if p <= 3:
            continue
        inv24 = _modinv(24, p)
        r = (p - inv24) % p
        rem = k0 % p
        off = r - rem
        if off < 0:
            off += p
        if 24 * (k0 + off) + 1 == p:
            off += p
        for idx in range(off, eff_len, p):
            is_prime[idx] = 0

    n_primes = 0
    for idx in range(eff_len):
        if is_prime[idx]:
            n_primes += 1

    # 2. Phase 1: mark every n_k = e*(4*c*d - 1) - 4*d with c*d <= cd_max (e >= 3)
    solved = np.zeros(eff_len, dtype=np.uint8)
    for j in range(len(steps)):
        st = steps[j]
        rk = rems[j]
        nm = nmins[j]
        rem = k0 % st
        off = rk - rem
        if off < 0:
            off += st
        while off < eff_len and (24 * (k0 + off) + 1) < nm:
            off += st
        for idx in range(off, eff_len, st):
            solved[idx] = 1

    # 3. Phase 2: EXHAUSTIVE O(sqrt(p) log p) decision for each surviving prime
    n_surv = 0
    exc_buf = np.empty(64, dtype=np.int64)
    n_exc = 0
    for idx in range(eff_len):
        if is_prime[idx] and solved[idx] == 0:
            n_surv += 1
            p = 24 * (k0 + idx) + 1
            ok, c_v, d_v, e_v = has_aigner_II_fast(p)
            if not ok:
                if n_exc == len(exc_buf):
                    new_buf = np.empty(len(exc_buf) * 2, dtype=np.int64)
                    new_buf[:n_exc] = exc_buf
                    exc_buf = new_buf
                exc_buf[n_exc] = p
                n_exc += 1

    return n_primes, n_surv, exc_buf[:n_exc]


@nb.njit(parallel=True, cache=True)
def run_T2_parallel(n_segs, seg_k_len, limit_n, small_primes, steps, rems, nmins):
    prim_cnts = np.zeros(n_segs, dtype=np.int64)
    surv_cnts = np.zeros(n_segs, dtype=np.int64)
    exc_cnts = np.zeros(n_segs, dtype=np.int64)
    # Store the exceptions per segment (reasonable maximum per segment; re-checked if it ever fills up)
    exc_mat = np.zeros((n_segs, 64), dtype=np.int64)
    overflow = np.zeros(n_segs, dtype=np.int64)
    for s in nb.prange(n_segs):
        np_s, ns_s, ex = process_segment_T2(s, seg_k_len, limit_n, small_primes, steps, rems, nmins)
        prim_cnts[s] = np_s
        surv_cnts[s] = ns_s
        exc_cnts[s] = len(ex)
        if len(ex) > 64:
            overflow[s] = 1
        else:
            for i in range(len(ex)):
                exc_mat[s, i] = ex[i]
    return prim_cnts.sum(), surv_cnts.sum(), exc_cnts, exc_mat, overflow.sum()


def solve_dahan_alg2(p):
    umax = (1 + math.isqrt(4 * p + 1)) // 4
    for u in range(1, umax + 1):
        mod4u = 4 * u
        target = mod4u - 1
        for s in divisors(p + u):
            if s % mod4u == target:
                alpha = (s + 1) // mod4u
                v = (p + u) // s
                x, y, z = alpha * u * v, p * alpha * v, p * alpha * u
                assert is_solution(p, x, y, z)
                return u, s, alpha, v, (x, y, z)
    return None


def solve_type1_a1(p):
    for e in range(3, (p + 4) // 3 + 1, 4):
        Me = (e * p + 1) // 4
        for B in divisors(Me):
            if B % e == e - 1:
                c = (B + 1) // e
                d = Me // B
                x, y, z = B * d * p, c * d, B * c * d
                assert is_solution(p, x, y, z)
                return e, B, c, d, (x, y, z)
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1_000_000_000)
    ap.add_argument("--cd-mark", type=int, default=600)
    ap.add_argument("--csv", default="data/aigner_II_exceptions_1e9.csv")
    args = ap.parse_args()

    t_all = time.time()
    print(f"Hardware: {platform.processor()} | {os.cpu_count()} logical cores | Numba {nb.__version__}")

    # --- Exhaustive comparison of the O(sqrt(p) log p) method against brute force < 200,000 ---
    print("\n=== Validation of O(sqrt(p) log p) against brute force e <= (p+4)/3 (p < 200,000, p ≡ 1 mod 24) ===")
    t_v0 = time.time()
    primes_2e5 = [p for p in primerange(2, 200_000) if p % 24 == 1]
    discrepancies = 0
    exc_2e5 = []
    for p in primes_2e5:
        ok_f, cf, df, ef = has_aigner_II_fast(p)
        ok_b, cb, db, eb = has_aigner_II_bruteforce(p)
        if ok_f != ok_b:
            print(f"  DISCREPANCY at p={p}: fast={ok_f}, brute={ok_b}")
            discrepancies += 1
        if ok_f:
            # Check the exact identity and is_solution
            assert ef * (4 * cf * df - 1) == p + 4 * df
            bf = cf * ef - 1
            assert is_solution(p, bf * df, cf * df * p, bf * cf * df * p)
        else:
            exc_2e5.append(p)
    print(f"  Primes checked: {len(primes_2e5):,} | Time: {time.time() - t_v0:.2f} s | Discrepancies: {discrepancies}")
    print(f"  Exceptions with p < 200,000 ({len(exc_2e5)}): {exc_2e5}")
    assert discrepancies == 0

    # --- Exhaustive search over ALL primes p <= 10^9, p ≡ 1 (mod 24) ---
    print(f"\n=== Exhaustive search for exceptions up to N = {args.limit:,} (cd_mark = {args.cd_mark}) ===")
    t0 = time.time()
    small_p = sieve_small_primes(int(math.isqrt(args.limit + 100)) + 10)
    steps, rems, nmins = precompute_mark_pairs(args.cd_mark)
    print(f"  Phase 1 progressions precomputed: {len(steps):,}")

    seg_k_len = 100_000  # 2,400,000 integers per segment
    total_k = ((args.limit - 1) // 24) + 1
    n_segs = (total_k + seg_k_len - 1) // seg_k_len

    tot_p, tot_surv, exc_cnts, exc_mat, ovf = run_T2_parallel(n_segs, seg_k_len, args.limit, small_p, steps, rems, nmins)
    assert ovf == 0, "Overflow in the exceptions buffer!"
    dt = time.time() - t0

    exceptions = []
    for s in range(n_segs):
        for i in range(exc_cnts[s]):
            exceptions.append(int(exc_mat[s, i]))
    exceptions.sort()

    print(f"  Total search time (Phase 1 + exhaustive Phase 2): {dt:.2f} s")
    print(f"  Total primes p <= {args.limit:,} with p ≡ 1 (mod 24): {tot_p:,}")
    print(f"  Survivors of Phase 1 (checked exhaustively in Phase 2): {tot_surv:,} ({100*tot_surv/tot_p:.4f}%)")
    print(f"  TOTAL Aigner Type II (a_ET = 1) EXCEPTIONS up to {args.limit:,}: {len(exceptions)}")
    print(f"  Full list: {exceptions}")

    # Independent double check (brute force for p < 10^6, fast method for all) + solutions in Dahan and in Type I a=1
    HARD_SET = {1, 121, 169, 289, 361, 529}
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    print("\n=== Details and alternative solutions (Dahan Alg. 2 and Type I a=1) of each exception ===")
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p", "p_mod_840", "is_hard_mod840", "m_profile", "dahan_u", "dahan_sol", "type1_a1_e", "type1_a1_B", "type1_a1_sol"])
        for p in exceptions:
            ok_f, *_ = has_aigner_II_fast(p)
            assert not ok_f
            if p < 500_000:
                ok_b, *_ = has_aigner_II_bruteforce(p)
                assert not ok_b
            m_p, _, c1_p, c2_p = profile(p)
            dah = solve_dahan_alg2(p)
            t1 = solve_type1_a1(p)
            hard = (p % 840) in HARD_SET
            print(
                f"  p={p:>10d} (mod 840={p%840:3d}, hard={str(hard):5s}, m={m_p:2d}): "
                f"Dahan Alg.2 u={dah[0]} {dah[4]} | Type I a=1 e={t1[0]}, B={t1[1]} {t1[4]}"
            )
            w.writerow([p, p % 840, hard, m_p, dah[0], str(dah[4]), t1[0], t1[1], str(t1[4])])

    print(f"\nExceptions CSV saved to {args.csv}")
    print(f"Total script time: {time.time() - t_all:.2f} s")


if __name__ == "__main__":
    main()
