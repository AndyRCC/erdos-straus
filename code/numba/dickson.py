#!/usr/bin/env python3
"""
Explicit primes for the construction conditional on Dickson (note, §5.1): if p = L·t + 1 and
q_j = 1 + (L/4j)·t are all prime, then m(p) > 4J − 1. Segmented sieve, exact m(p) with
tools/verify.types_at, and comparison with Bateman–Horn. Table in data/dickson_primes.csv.
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
from verify import types_at, is_solution


# ---------------------------------------------------------------------------
# 1. Construction of L_basic(J) and L_opt(J) + the exact Bateman-Horn constant
# ---------------------------------------------------------------------------

def build_L(J, mode="basic"):
    C = 4 * J - 1
    lcm_J = 1
    for j in range(1, J + 1):
        lcm_J = int(sympy.ilcm(lcm_J, j))
    if mode == "basic":
        prim = 1
        for p in sympy.primerange(2, C + 1):
            prim *= int(p)
        return 4 * lcm_J * prim
    else:
        # Smallest valid L for Lemma A:
        # - 4*j | L for j = 1..J and 2 | L/(4j) (i.e. v2(L) >= 3 + floor(log2 J))
        # - c0(4j-1) | L for j = 1..J (since gcd(j, c0)=1, c0 | L/(4j) automatically)
        L = 8 * lcm_J
        for j in range(1, J + 1):
            c = 4 * j - 1
            c0 = 1
            for q, e in sympy.factorint(c).items():
                if e % 2 == 1:
                    c0 *= int(q)
            L = int(sympy.ilcm(L, c0))
        k2 = J.bit_length() - 1
        pow2 = 1 << (3 + k2)
        if L % pow2 != 0:
            L = int(sympy.ilcm(L, pow2))
        # Include small prime factors r <= 17 if they increase the Bateman-Horn density
        for r in [3, 5, 7, 11, 13, 17]:
            def om(Ltest):
                roots = set()
                for a in [Ltest] + [Ltest // (4 * j) for j in range(1, J + 1)]:
                    if a % r != 0:
                        roots.add((-pow(a % r, -1, r)) % r)
                return len(roots)
            w0 = om(L)
            if w0 > 0:
                w1 = om(L * r)
                gain = (1.0 - w1 / r) / (1.0 - w0 / r)
                logp = math.log(L) + 18.0
                cost = (logp / (logp + math.log(r))) ** (J + 1)
                if gain * cost > 1.0:
                    L *= r
        return L


def bateman_horn_prediction(J, L, T, p_limit=100_000):
    """Computes the constant S(J, L) and the exact Bateman-Horn integral over [1, T]."""
    a_list = [L] + [L // (4 * j) for j in range(1, J + 1)]
    k = J + 1
    S = 1.0
    for p in sympy.primerange(2, p_limit):
        p = int(p)
        roots = set()
        for a in a_list:
            am = a % p
            if am != 0:
                roots.add((-pow(am, -1, p)) % p)
        w = len(roots)
        assert w < p, (J, p, w)
        S *= (1.0 - w / p) / ((1.0 - 1.0 / p) ** k)

    # Numerical Simpson integration over [1, T] on a logarithmic scale (u = log t)
    if T <= 1:
        return S, 0.0
    n_steps = 2000
    u_max = math.log(T)
    du = u_max / n_steps
    log_a = [math.log(a) for a in a_list]
    acc = 0.0
    for i in range(n_steps + 1):
        u = i * du
        t_val = math.exp(u)
        denom = 1.0
        for la in log_a:
            denom *= (la + u)
        val = t_val / denom  # dt = exp(u) du
        w_s = 1.0 if (i == 0 or i == n_steps) else (4.0 if (i & 1) else 2.0)
        acc += w_s * val
    integral = acc * du / 3.0
    return S, S * integral


# ---------------------------------------------------------------------------
# 2. Parallel segmented sieve in t (Numba)
# ---------------------------------------------------------------------------

def build_sieve_table(a_list, p_max):
    """
    For each prime p <= p_max builds the list of forbidden roots t ≡ r (mod p)
    such that a_i * t + 1 ≡ 0 (mod p) for some i.
    Returns (primes_arr, offsets_arr, roots_arr).
    """
    primes = []
    offsets = [0]
    roots = []
    for p in sympy.primerange(2, p_max + 1):
        p = int(p)
        rset = set()
        for a in a_list:
            am = a % p
            if am != 0:
                rset.add((-pow(am, -1, p)) % p)
        if rset:
            primes.append(p)
            for r in sorted(rset):
                roots.append(r)
            offsets.append(len(roots))
    return (
        np.array(primes, dtype=np.int64),
        np.array(offsets, dtype=np.int64),
        np.array(roots, dtype=np.int64),
    )


@njit
def sieve_segment(t_start, seg_len, primes, offsets, roots):
    """
    Sieves the block [t_start, t_start + seg_len) and returns the surviving t.
    """
    mark = np.zeros(seg_len, dtype=np.uint8)
    n_p = len(primes)
    for pi in range(n_p):
        p = primes[pi]
        lo = offsets[pi]
        hi = offsets[pi + 1]
        base = t_start % p
        for ri in range(lo, hi):
            r = roots[ri]
            off = r - base
            if off < 0:
                off += p
            pos = off
            while pos < seg_len:
                mark[pos] = 1
                pos += p
    cnt = 0
    for i in range(seg_len):
        if mark[i] == 0:
            cnt += 1
    out = np.empty(cnt, dtype=np.int64)
    k = 0
    for i in range(seg_len):
        if mark[i] == 0:
            out[k] = t_start + i
            k += 1
    return out


@njit(parallel=True)
def sieve_range_parallel(t_lo, t_hi, seg_len, primes, offsets, roots, max_out_per_seg):
    """
    Sieves [t_lo, t_hi) in parallel in segments of size seg_len.
    """
    n_segs = (t_hi - t_lo + seg_len - 1) // seg_len
    buf = np.empty((n_segs, max_out_per_seg), dtype=np.int64)
    counts = np.zeros(n_segs, dtype=np.int64)
    overflow = np.zeros(n_segs, dtype=np.int8)

    for si in prange(n_segs):
        ts = t_lo + si * seg_len
        sl = seg_len
        if ts + sl > t_hi:
            sl = t_hi - ts
        surv = sieve_segment(ts, sl, primes, offsets, roots)
        c = len(surv)
        if c > max_out_per_seg:
            overflow[si] = 1
            c = max_out_per_seg
        counts[si] = c
        for j in range(c):
            buf[si, j] = surv[j]

    return buf, counts, overflow


def exact_m_of_prime(p, c_min_guaranteed):
    """
    Computes the EXACT m(p) with tools/verify.types_at, starting at c = 3,
    to check independently that ALL c <= c_min_guaranteed fail on both branches!
    """
    # 1. Confirm blindness at every c <= c_min_guaranteed
    for c in range(3, c_min_guaranteed + 1, 4):
        t1, t2 = types_at(p, c)
        assert (not t1) and (not t2), f"BLINDNESS FAILURE: p={p}, c={c}, types={(t1, t2)}"
    # 2. Find the first c > c_min_guaranteed with a solution
    c = c_min_guaranteed + 4
    while True:
        t1, t2 = types_at(p, c)
        if t1 or t2:
            return c, (t1, t2)
        c += 4


def search_dickson(J, mode, T_max, p_sieve_max=6000, seg_len=1_000_000, batch_segs=64, max_hits_show=5):
    t0 = time.time()
    C = 4 * J - 1
    L = build_L(J, mode=mode)
    a_list = [L] + [L // (4 * j) for j in range(1, J + 1)]
    S, bh_exp = bateman_horn_prediction(J, L, T_max)
    primes_s, offsets_s, roots_s = build_sieve_table(a_list, p_sieve_max)

    # Estimate the survival rate after sieving up to p_sieve_max
    surv_rate = 1.0
    for pi in range(len(primes_s)):
        w = offsets_s[pi + 1] - offsets_s[pi]
        surv_rate *= (1.0 - w / primes_s[pi])
    max_out = max(1000, int(seg_len * surv_rate * 2.5) + 500)

    # For ultra-fast rejection in Python, sort the forms from smallest to largest coefficient (q_J first)
    a_check = sorted(a_list)

    hits = []
    t_cur = 1
    total_surv = 0
    while t_cur <= T_max:
        t_end = min(T_max + 1, t_cur + batch_segs * seg_len)
        buf, counts, overflow = sieve_range_parallel(
            t_cur, t_end, seg_len, primes_s, offsets_s, roots_s, max_out
        )
        assert np.all(overflow == 0), f"Overflow in the sieve J={J}!"
        for si in range(len(counts)):
            c_n = int(counts[si])
            total_surv += c_n
            row = buf[si]
            for idx in range(c_n):
                tv = int(row[idx])
                ok = True
                for a in a_check:
                    cand = a * tv + 1
                    # Fast base-2 Fermat test before sympy.isprime
                    if pow(2, cand - 1, cand) != 1:
                        ok = False
                        break
                if ok:
                    if all(sympy.isprime(a * tv + 1) for a in a_check):
                        hits.append(tv)
        t_cur = t_end

    elapsed = time.time() - t0
    print(f"\nJ = {J:2d} (C = {C:2d}, mode = {mode:6s}, L = {L}, digits(L) = {len(str(L))})", flush=True)
    print(f"  T_max = {T_max:,}, p_sieve <= {p_sieve_max}, survivors = {total_surv:,} ({elapsed:.2f} s)", flush=True)
    print(f"  Bateman-Horn: S = {S:.4e}, expected in [1, T_max] = {bh_exp:.2f}, found = {len(hits)}", flush=True)

    detailed = []
    for tv in hits[:max_hits_show]:
        p = L * tv + 1
        digs = len(str(p))
        # If p <= 10^26 we can compute the EXACT m(p) quickly with types_at!
        t_m = time.time()
        if digs <= 26:
            m_val, types_m = exact_m_of_prime(p, C)
            exact_str = f"EXACT m(p) = {m_val} (types={types_m}, {time.time()-t_m:.2f}s)"
        else:
            m_val, types_m = None, None
            exact_str = f"guaranteed m(p) > {C} (Lemma A)"
        print(f"    t={tv}: p = {p} ({digs} digits, mod 840={p%840}) -> {exact_str}", flush=True)
        detailed.append((tv, p, digs, m_val))

    return {
        "J": J,
        "C": C,
        "mode": mode,
        "digits_L": len(str(L)),
        "T_max": T_max,
        "S_BH": f"{S:.6e}",
        "BH_expected": f"{bh_exp:.3f}",
        "found": len(hits),
        "first_t": detailed[0][0] if detailed else "",
        "first_p": str(detailed[0][1]) if detailed else "",
        "first_p_digits": detailed[0][2] if detailed else "",
        "first_m_exact": detailed[0][3] if detailed else "",
        "elapsed_s": f"{elapsed:.2f}",
    }


def main():
    t_all = time.time()
    results = []

    # 1. Validation: J = 4 and J = 6 with L_basic
    results.append(search_dickson(4, "basic", T_max=10_000, p_sieve_max=500, seg_len=10_000, batch_segs=1, max_hits_show=5))
    results.append(search_dickson(6, "basic", T_max=2_000_000, p_sieve_max=1500, seg_len=200_000, batch_segs=10, max_hits_show=4))

    # 2. J = 8: with both L_basic and L_opt!
    results.append(search_dickson(8, "basic", T_max=600_000_000, p_sieve_max=4000, seg_len=1_000_000, batch_segs=64, max_hits_show=5))
    results.append(search_dickson(8, "opt",    T_max=400_000_000, p_sieve_max=4000, seg_len=1_000_000, batch_segs=64, max_hits_show=5))

    # 3. J = 9 with L_opt (expect ~1 every 1.01e9 -> we sweep 3e9)
    results.append(search_dickson(9, "opt",    T_max=3_000_000_000, p_sieve_max=6000, seg_len=2_000_000, batch_segs=64, max_hits_show=5))

    # 4. J = 10 with L_opt (expect ~1 every 1.75e10 -> we sweep 2.5e10)
    results.append(search_dickson(10, "opt",   T_max=25_000_000_000, p_sieve_max=8000, seg_len=2_000_000, batch_segs=128, max_hits_show=5))

    out_csv = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dickson_primes.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {out_csv} ({len(results)} rows). Total time: {time.time() - t_all:.2f} s.", flush=True)


if __name__ == "__main__":
    main()
