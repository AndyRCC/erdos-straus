#!/usr/bin/env python3
"""
Blind sets at fixed shifts c ≡ 3 (mod 4) up to 10^9: B_c^QR, B_c^exact and the
primes given by the form B·Q(x, y) − c of the theorem (note, §5).
"""

import csv
import math
import os
import sys
import time
import numpy as np
from numba import njit, prange
import sympy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from verify import types_at


N_MAX = 10**9
K_MAX = (N_MAX + 99) // 4 + 2  # 250_000_026
CHECKPOINTS = np.array([10**4, 10**5, 10**6, 10**7, 10**8, 10**9], dtype=np.int64)


# ---------------------------------------------------------------------------
# 1. Prime sieve (bitset of odd numbers up to 10^9, ~62.5 MB)
#    and SPF sieve of odd numbers up to K_MAX (~250 MB)
# ---------------------------------------------------------------------------

@njit
def build_odd_prime_bitset(limit):
    """Bitset of odd composites: bit i corresponds to the odd number 2*i + 1."""
    n_odds = (limit // 2) + 1
    comp = np.zeros((n_odds + 7) // 8, dtype=np.uint8)
    comp[0] |= np.uint8(1)  # 1 is not prime
    lim_p = int(math.sqrt(limit)) + 1
    for p in range(3, lim_p + 1, 2):
        idx = p >> 1
        if (comp[idx >> 3] & (1 << (idx & 7))) == 0:
            start = (p * p) >> 1
            for j in range(start, n_odds, p):
                comp[j >> 3] |= np.uint8(1 << (j & 7))
    return comp


@njit
def is_odd_prime(n, comp):
    if n < 3 or (n & 1) == 0:
        return False
    idx = n >> 1
    return (comp[idx >> 3] & (1 << (idx & 7))) == 0


@njit
def build_odd_spf(k_max):
    """Smallest odd prime factor of each odd 2*i + 1 <= k_max (0 => prime)."""
    n_odds = (k_max // 2) + 1
    spf = np.zeros(n_odds, dtype=np.uint16)
    lim_p = int(math.sqrt(k_max)) + 1
    for p in range(3, lim_p + 1, 2):
        if spf[p >> 1] == 0:
            start = (p * p) >> 1
            for j in range(start, n_odds, p):
                if spf[j] == 0:
                    spf[j] = np.uint16(p)
    return spf


@njit
def isqrt_i64(n):
    if n <= 0:
        return 0
    r = int(math.sqrt(float(n)))
    while (r + 1) * (r + 1) <= n:
        r += 1
    while r * r > n:
        r -= 1
    return r


@njit
def gcd_i64(a, b):
    if a < 0:
        a = -a
    if b < 0:
        b = -b
    while b != 0:
        a, b = b, a % b
    return a


@njit
def jacobi_sym(a, n):
    """Jacobi symbol (a/n) for odd n > 0."""
    a %= n
    res = 1
    while a != 0:
        while (a & 1) == 0:
            a >>= 1
            r = n & 7
            if r == 3 or r == 5:
                res = -res
        a, n = n, a
        if (a & 3) == 3 and (n & 3) == 3:
            res = -res
        a %= n
    return res if n == 1 else 0


@njit
def extract_primes_1mod24(comp, lim):
    """Extracts the contiguous array of the 6.35M primes n ≡ 1 (mod 24) <= lim."""
    # Count first
    cnt = 0
    n = 25
    while n <= lim:
        if is_odd_prime(n, comp):
            cnt += 1
        n += 24
    arr = np.empty(cnt, dtype=np.int64)
    k = 0
    n = 25
    while n <= lim:
        if is_odd_prime(n, comp):
            arr[k] = n
            k += 1
        n += 24
    return arr


@njit
def factor_and_check_blind_fast(n, c, c0, spf, qr_table):
    """
    Zero heap allocations! Uses two uint64 bitmasks (lo, hi) for c <= 127.
    Returns (is_qr_blind, is_exact_blind) for a prime n > c, n ≡ 1 (mod 24).
    """
    x = (n + c) // 4
    qr_ok = True
    # Bitmasks of the residues mod c of the divisors of K_c (c <= 100 < 128)
    lo = np.uint64(2)  # bit 1 set (divisor 1)
    hi = np.uint64(0)
    neg_n = (c - (n % c)) % c

    # Factor 2
    if (x & 1) == 0:
        if qr_table[2 % c0] != 1:
            qr_ok = False
        e = 0
        while (x & 1) == 0:
            x >>= 1
            e += 1
        # Update the bitmask with q_mod = 2 % c (only if we do not need to postpone it;
        # in fact doing it on the fly is always very fast, whether qr_ok is False or not)
        # Better: first check QR quickly, and build D only if QR fails!
        # Should we keep the original x to rebuild it? No: storing up to 10 factors in
        # registers is no faster than re-factoring only when !qr_ok.
        # Re-factoring takes only 5 lookups!
        pass

    # Ultra-fast step 1: only check QR (no bitmasks!)
    x0 = (n + c) // 4
    if (x0 & 1) == 0:
        if qr_table[2 % c0] != 1:
            qr_ok = False
        else:
            while (x0 & 1) == 0:
                x0 >>= 1
    if qr_ok:
        while x0 > 1:
            p = int(spf[x0 >> 1])
            if p == 0:
                p = x0
            if qr_table[p % c0] != 1:
                qr_ok = False
                break
            while x0 % p == 0:
                x0 //= p

    if qr_ok:
        # If every prime satisfies (q/c0) == +1, both branches fail automatically.
        return True, True

    # Step 2 (only if !qr_ok): for c in {3, 7, 15} the exact collapse is a theorem (note, §4),
    # but we still compute D by bitmask (c <= 100) for an exact check without shortcuts.
    x = (n + c) // 4
    # Factor 2
    if (x & 1) == 0:
        e = 0
        while (x & 1) == 0:
            x >>= 1
            e += 1
        q_mod = 2 % c
        if q_mod == c - 1 or q_mod == neg_n:
            return False, False
        if q_mod != 1:
            for _ in range(e):
                nlo = lo
                nhi = hi
                # Run through the set bits in lo and hi
                m = lo
                while m != 0:
                    # lowest bit
                    lsb = m & (~m + np.uint64(1))
                    # position of the bit
                    r = 0
                    tmp = lsb
                    while tmp > np.uint64(1):
                        tmp >>= np.uint64(1)
                        r += 1
                    nr = (r * q_mod) % c
                    if nr < 64:
                        nlo |= np.uint64(1) << np.uint64(nr)
                    else:
                        nhi |= np.uint64(1) << np.uint64(nr - 64)
                    m ^= lsb
                m = hi
                while m != 0:
                    lsb = m & (~m + np.uint64(1))
                    r = 64
                    tmp = lsb
                    while tmp > np.uint64(1):
                        tmp >>= np.uint64(1)
                        r += 1
                    nr = (r * q_mod) % c
                    if nr < 64:
                        nlo |= np.uint64(1) << np.uint64(nr)
                    else:
                        nhi |= np.uint64(1) << np.uint64(nr - 64)
                    m ^= lsb
                if nlo == lo and nhi == hi:
                    break
                lo = nlo
                hi = nhi

    # Odd factors
    while x > 1:
        p = int(spf[x >> 1])
        if p == 0:
            p = x
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        q_mod = p % c
        if q_mod == c - 1 or q_mod == neg_n:
            return False, False
        if q_mod == 1:
            continue
        for _ in range(e):
            nlo = lo
            nhi = hi
            m = lo
            while m != 0:
                lsb = m & (~m + np.uint64(1))
                r = 0
                tmp = lsb
                while tmp > np.uint64(1):
                    tmp >>= np.uint64(1)
                    r += 1
                nr = (r * q_mod) % c
                if nr < 64:
                    nlo |= np.uint64(1) << np.uint64(nr)
                else:
                    nhi |= np.uint64(1) << np.uint64(nr - 64)
                m ^= lsb
            m = hi
            while m != 0:
                lsb = m & (~m + np.uint64(1))
                r = 64
                tmp = lsb
                while tmp > np.uint64(1):
                    tmp >>= np.uint64(1)
                    r += 1
                nr = (r * q_mod) % c
                if nr < 64:
                    nlo |= np.uint64(1) << np.uint64(nr)
                else:
                    nhi |= np.uint64(1) << np.uint64(nr - 64)
                m ^= lsb
            if nlo == lo and nhi == hi:
                break
            lo = nlo
            hi = nhi
        # Early Type II check after each prime factor:
        # check whether some set r has (c - r) set
        m = lo
        while m != 0:
            lsb = m & (~m + np.uint64(1))
            r = 0
            tmp = lsb
            while tmp > np.uint64(1):
                tmp >>= np.uint64(1)
                r += 1
            nr = c - r
            if nr < 64:
                if (lo & (np.uint64(1) << np.uint64(nr))) != 0:
                    return False, False
            else:
                if (hi & (np.uint64(1) << np.uint64(nr - 64))) != 0:
                    return False, False
            m ^= lsb

    # Final check of Type II (including bits in hi) and Type I
    m = lo
    while m != 0:
        lsb = m & (~m + np.uint64(1))
        r = 0
        tmp = lsb
        while tmp > np.uint64(1):
            tmp >>= np.uint64(1)
            r += 1
        nr = c - r
        if nr < 64:
            if (lo & (np.uint64(1) << np.uint64(nr))) != 0:
                return False, False
        else:
            if (hi & (np.uint64(1) << np.uint64(nr - 64))) != 0:
                return False, False
        t1 = (neg_n * r) % c
        if t1 < 64:
            if (lo & (np.uint64(1) << np.uint64(t1))) != 0:
                return False, False
        else:
            if (hi & (np.uint64(1) << np.uint64(t1 - 64))) != 0:
                return False, False
        m ^= lsb

    m = hi
    while m != 0:
        lsb = m & (~m + np.uint64(1))
        r = 64
        tmp = lsb
        while tmp > np.uint64(1):
            tmp >>= np.uint64(1)
            r += 1
        nr = c - r
        if nr < 64:
            if (lo & (np.uint64(1) << np.uint64(nr))) != 0:
                return False, False
        else:
            if (hi & (np.uint64(1) << np.uint64(nr - 64))) != 0:
                return False, False
        t1 = (neg_n * r) % c
        if t1 < 64:
            if (lo & (np.uint64(1) << np.uint64(t1))) != 0:
                return False, False
        else:
            if (hi & (np.uint64(1) << np.uint64(t1 - 64))) != 0:
                return False, False
        m ^= lsb

    return False, True


@njit(parallel=True)
def scan_blind_parallel(primes_arr, c, c0, spf, checkpoints):
    """
    Sweeps in parallel the 6.35M primes n ≡ 1 (mod 24) <= 10^9.
    """
    qr_table = np.zeros(c0, dtype=np.int8)
    for r in range(c0):
        qr_table[r] = np.int8(jacobi_sym(r, c0))

    n_cp = len(checkpoints)
    # Find the cut-off indices in primes_arr for each checkpoint
    cp_bounds = np.zeros(n_cp + 1, dtype=np.int64)
    n_tot = len(primes_arr)
    cur = 0
    for b in range(n_cp):
        lim_b = checkpoints[b]
        while cur < n_tot and primes_arr[cur] <= lim_b:
            cur += 1
        cp_bounds[b + 1] = cur

    cnt_qr = np.zeros(n_cp, dtype=np.int64)
    cnt_ex = np.zeros(n_cp, dtype=np.int64)
    acc_qr = 0
    acc_ex = 0
    for b in range(n_cp):
        lo_i = cp_bounds[b]
        hi_i = cp_bounds[b + 1]
        sub_qr = 0
        sub_ex = 0
        for i in prange(lo_i, hi_i):
            n = primes_arr[i]
            if n > c:
                ok_qr, ok_ex = factor_and_check_blind_fast(n, c, c0, spf, qr_table)
                if ok_qr:
                    sub_qr += 1
                if ok_ex:
                    sub_ex += 1
        acc_qr += sub_qr
        acc_ex += sub_ex
        cnt_qr[b] = acc_qr
        cnt_ex[b] = acc_ex
    return cnt_qr, cnt_ex


@njit
def enum_form_primes(c, c0, a_f, b_f, c_f, B, filter_mod3, comp, spf, checkpoints):
    """
    Enumerates ALL coprime pairs (u, v) with v >= 0 (and u = 1 if v = 0)
    such that n = B*(a_f*u^2 + b_f*u*v + c_f*v^2) - c <= checkpoints[-1], n > c, n prime.
    For c = 3 (a_f=b_f=c_f=1) it uses the symmetry u >= v >= 0 (exactly 6x faster).
    """
    lim = checkpoints[len(checkpoints) - 1]
    M = (lim + c) // B
    absD = 4 * a_f * c_f - b_f * b_f

    qr_table = np.zeros(c0, dtype=np.int8)
    for r in range(c0):
        qr_table[r] = np.int8(jacobi_sym(r, c0))

    n_slots = (lim // 24) + 2
    seen = np.zeros((n_slots + 7) // 8, dtype=np.uint8)

    bad_mod24 = 0
    bad_qr = 0
    bad_exact = 0

    # v = 0 -> u = 1
    n0 = B * a_f - c
    if n0 > c and n0 <= lim and is_odd_prime(n0, comp):
        if (not filter_mod3) or (n0 % 3 == 1):
            if n0 % 24 != 1:
                bad_mod24 += 1
            else:
                idx = (n0 - 1) // 24
                if (seen[idx >> 3] & (1 << (idx & 7))) == 0:
                    seen[idx >> 3] |= np.uint8(1 << (idx & 7))
                    ok_qr, ok_ex = factor_and_check_blind_fast(n0, c, c0, spf, qr_table)
                    if not ok_qr:
                        bad_qr += 1
                    if not ok_ex:
                        bad_exact += 1

    four_a_M = 4 * a_f * M
    two_a = 2 * a_f

    if a_f == 1 and b_f == 1 and c_f == 1:
        # Exact symmetry of x^2 + xy + y^2: fundamental domain u >= v >= 1, gcd(u, v) = 1
        # (u=1, v=0 already covered above; u=1, v=1 covered here).
        # Largest v with 3*v^2 <= M:
        v_max = isqrt_i64(M // 3)
        for v in range(1, v_max + 1):
            # u^2 + v*u + (v^2 - M) <= 0 => (2u + v)^2 <= 4M - 3v^2
            rem = 4 * M - 3 * v * v
            s = isqrt_i64(rem)
            u_hi = (s - v) // 2
            for u in range(v, u_hi + 1):
                if gcd_i64(u, v) != 1:
                    continue
                qval = u * u + u * v + v * v
                n = B * qval - c
                if n <= c or n > lim:
                    continue
                if not is_odd_prime(n, comp):
                    continue
                if filter_mod3 and (n % 3 != 1):
                    continue
                if n % 24 != 1:
                    bad_mod24 += 1
                    continue
                idx = (n - 1) // 24
                if (seen[idx >> 3] & (1 << (idx & 7))) == 0:
                    seen[idx >> 3] |= np.uint8(1 << (idx & 7))
                    ok_qr, ok_ex = factor_and_check_blind_fast(n, c, c0, spf, qr_table)
                    if not ok_qr:
                        bad_qr += 1
                    if not ok_ex:
                        bad_exact += 1
    else:
        v_max = isqrt_i64(four_a_M // absD)
        for v in range(1, v_max + 1):
            rem = four_a_M - absD * v * v
            if rem < 0:
                break
            s = isqrt_i64(rem)
            num_lo = -b_f * v - s
            num_hi = -b_f * v + s
            u_lo = -((-num_lo) // two_a) if num_lo < 0 else (num_lo + two_a - 1) // two_a
            u_hi = num_hi // two_a
            for u in range(u_lo, u_hi + 1):
                if gcd_i64(u, v) != 1:
                    continue
                qval = a_f * u * u + b_f * u * v + c_f * v * v
                n = B * qval - c
                if n <= c or n > lim:
                    continue
                if not is_odd_prime(n, comp):
                    continue
                if filter_mod3 and (n % 3 != 1):
                    continue
                if n % 24 != 1:
                    bad_mod24 += 1
                    continue
                idx = (n - 1) // 24
                if (seen[idx >> 3] & (1 << (idx & 7))) == 0:
                    seen[idx >> 3] |= np.uint8(1 << (idx & 7))
                    ok_qr, ok_ex = factor_and_check_blind_fast(n, c, c0, spf, qr_table)
                    if not ok_qr:
                        bad_qr += 1
                    if not ok_ex:
                        bad_exact += 1

    n_cp = len(checkpoints)
    cnt = np.zeros(n_cp, dtype=np.int64)
    cp_idx = 0
    total = 0
    max_idx = (lim - 1) // 24
    for idx in range(max_idx + 1):
        n = 24 * idx + 1
        while cp_idx < n_cp and n > checkpoints[cp_idx]:
            cnt[cp_idx] = total
            cp_idx += 1
        if (seen[idx >> 3] & (1 << (idx & 7))) != 0:
            total += 1
    while cp_idx < n_cp:
        cnt[cp_idx] = total
        cp_idx += 1

    return cnt, bad_mod24, bad_qr, bad_exact, seen


@njit
def verify_c3_exact_equality(primes_arr, spf, seen_c3):
    qr_table = np.zeros(3, dtype=np.int8)
    qr_table[1] = 1
    qr_table[2] = -1

    in_form_not_qr = 0
    in_qr_not_form = 0
    qr_vs_ex_diff = 0
    total_both = 0

    for i in range(len(primes_arr)):
        n = primes_arr[i]
        ok_qr, ok_ex = factor_and_check_blind_fast(n, 3, 3, spf, qr_table)
        if ok_qr != ok_ex:
            qr_vs_ex_diff += 1
        idx = (n - 1) // 24
        in_form = (seen_c3[idx >> 3] & (1 << (idx & 7))) != 0
        if in_form and not ok_qr:
            in_form_not_qr += 1
        elif ok_qr and not in_form:
            in_qr_not_form += 1
        elif in_form and ok_qr:
            total_both += 1

    return total_both, in_form_not_qr, in_qr_not_form, qr_vs_ex_diff


def squarefree_kernel(c):
    c0 = 1
    for p, e in sympy.factorint(c).items():
        if e % 2 == 1:
            c0 *= p
    return c0


def form_params(c):
    c0 = squarefree_kernel(c)
    assert c % 4 == 3 and c0 % 4 == 3
    mod8 = c % 8
    B_base = 4 if mod8 == 3 else 16
    if c % 3 == 0:
        if c0 % 3 == 0:
            return c0, 1, 1, (c0 + 1) // 4, B_base, 0, f"{B_base}*(x^2+xy+{(c0+1)//4}y^2)-{c} [Thm 1.1(1)]"
        else:
            return c0, 1, 1, (c0 + 1) // 4, B_base, 1, f"{B_base}*(x^2+xy+{(c0+1)//4}y^2)-{c} [Thm 1.1(2), m=3]"
    elif c % 3 == 1:
        return c0, 1, 1, (c0 + 1) // 4, B_base, 0, f"{B_base}*(x^2+xy+{(c0+1)//4}y^2)-{c} [Thm 1.1(1)]"
    else:
        assert (c0 + 1) % 12 == 0
        B = 3 * B_base
        return c0, 3, 1, (c0 + 1) // 12, B, 0, f"{B}*(3u^2+uv+{(c0+1)//12}v^2)-{c} [Thm 1.1(1)]"


def main():
    t0 = time.time()
    print("Building the prime sieve up to 10^9 and the SPF table up to 2.5e8...", flush=True)
    comp = build_odd_prime_bitset(N_MAX)
    spf = build_odd_spf(K_MAX)
    primes_arr = extract_primes_1mod24(comp, N_MAX)
    print(f"Sieves ready in {time.time() - t0:.2f} s ({len(primes_arr)} primes n ≡ 1 mod 24 <= 10^9).", flush=True)

    c_list = [3, 7, 11, 19, 23, 31, 43, 15, 27, 35, 63, 99]

    # Exact validation against tools/verify.types_at up to 10^5
    t_val = time.time()
    val_checks = 0
    val_disc = 0
    for c in c_list:
        c0 = squarefree_kernel(c)
        qr_table = np.array([jacobi_sym(r, c0) for r in range(c0)], dtype=np.int8)
        for n in primes_arr:
            if n > 100_000:
                break
            if n > c:
                ok_qr, ok_ex = factor_and_check_blind_fast(int(n), c, c0, spf, qr_table)
                t1, t2 = types_at(int(n), c)
                ref_blind = (not t1) and (not t2)
                val_checks += 1
                if ok_ex != ref_blind or (ok_qr and not ref_blind):
                    val_disc += 1
                    print(f"DISCREPANCY: c={c}, n={n}, ok_qr={ok_qr}, ok_ex={ok_ex}, ref={ref_blind}", flush=True)
    print(f"types_at validation (n <= 10^5, 12 values of c): {val_checks} pairs, {val_disc} discrepancies ({time.time() - t_val:.2f} s).", flush=True)
    assert val_disc == 0

    rows = []
    print("\n=== RESULTS UP TO N = 10^9 ===", flush=True)
    for c in c_list:
        tc = time.time()
        c0, a_f, b_f, c_f, B, filt3, desc = form_params(c)
        cnt_qr, cnt_ex = scan_blind_parallel(primes_arr, c, c0, spf, CHECKPOINTS)
        cnt_form, b24, bqr, bex, seen = enum_form_primes(
            c, c0, a_f, b_f, c_f, B, filt3, comp, spf, CHECKPOINTS
        )
        assert b24 == 0 and bqr == 0 and bex == 0, (c, b24, bqr, bex)

        kind = "prime" if sympy.isprime(c) else "composite"
        print(f"\nc = {c} ({kind}, c0 = {c0}, form: {desc}) [{time.time() - tc:.2f} s]", flush=True)
        print(f"  Check of the form up to 10^9: bad_mod24={b24}, bad_qr={bqr}, bad_exact={bex}", flush=True)
        print(f"  {'N':>10s} | {'B_QR':>9s} | {'B_exact':>9s} | {'B_form':>9s} | {'QR/scale':>8s} | {'Ex/scale':>8s} | {'Form/sc':>8s} | {'(Ex-QR)/QR':>10s}", flush=True)
        for i, N in enumerate(CHECKPOINTS):
            scale = N / (math.log(N) ** 1.5)
            r_qr = cnt_qr[i] / scale
            r_ex = cnt_ex[i] / scale
            r_fm = cnt_form[i] / scale
            rel_diff = (cnt_ex[i] - cnt_qr[i]) / cnt_qr[i] if cnt_qr[i] > 0 else 0.0
            print(f"  {N:10d} | {cnt_qr[i]:9d} | {cnt_ex[i]:9d} | {cnt_form[i]:9d} | {r_qr:8.4f} | {r_ex:8.4f} | {r_fm:8.4f} | {rel_diff:10.4f}", flush=True)
            rows.append({
                "c": c,
                "kind": kind,
                "c0": c0,
                "form": desc,
                "N": int(N),
                "B_QR": int(cnt_qr[i]),
                "B_exact": int(cnt_ex[i]),
                "B_form": int(cnt_form[i]),
                "ratio_QR": f"{r_qr:.6f}",
                "ratio_exact": f"{r_ex:.6f}",
                "ratio_form": f"{r_fm:.6f}",
                "frac_nonQR_over_QR": f"{rel_diff:.6f}",
            })

        if c == 3:
            tot_eq, fnq, qnf, qed = verify_c3_exact_equality(primes_arr, spf, seen)
            print(f"  >>> EXACT EQUALITY c = 3 up to 10^9: total={tot_eq}, in_form_not_QR={fnq}, in_QR_not_form={qnf}, QR_vs_exact={qed}", flush=True)
            assert fnq == 0 and qnf == 0 and qed == 0

    out_csv = os.path.join(os.path.dirname(__file__), "..", "..", "data", "blind_primes_1e9.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_csv} ({len(rows)} rows). Total time: {time.time() - t0:.2f} s.", flush=True)


if __name__ == "__main__":
    main()
