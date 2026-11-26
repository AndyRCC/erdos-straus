#!/usr/bin/env python3
"""
Independent check (sympy) of the data up to 10^8: m(p), the a = 1 slices of type I and
type II (Aigner and Dahan), and the local criteria at c = 3 and c = 7. Compares row by row
with data/hard_primes_1e8.csv (Numba implementation).
"""
import csv
import os
import sys
import time
from collections import Counter
from math import isqrt, log, sqrt
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify import profile, types_at, is_solution  # noqa: E402
from sympy import divisors, legendre_symbol, primerange, factorint  # noqa: E402

DIF = {1, 121, 169, 289, 361, 529}
CAP_E = 6000          # search cap for II1A before marking it "pending"
Q34 = [q for q in primerange(3, 20000) if q % 4 == 3]


def e_I1(p):
    for e in range(3, (p + 4) // 3 + 1, 4):
        M = (e * p + 1) // 4
        if any(B % e == e - 1 for B in divisors(M)):
            return e
    return None


def e_II1A(p, cap=CAP_E):
    top = min((p + 4) // 3, cap)
    for e in range(3, top + 1, 4):
        K = (p + e) // 4
        if any(b % e == e - 1 for b in divisors(K)):
            return e
    return None if top == (p + 4) // 3 else "pending"


def u_II1D(p):
    for u in range(1, (1 + isqrt(4 * p + 1)) // 4 + 2):
        if any(s % (4 * u) == 4 * u - 1 for s in divisors(p + u)):
            return u
    return None


def k_q34(p):
    k = 0
    for q in Q34:
        if legendre_symbol(p % q, q) == -1:
            k += 1
            if any(types_at(p, q)):
                return k, q
    return None, None


def compute_row(p):
    m, _, c1, c2 = profile(p)
    k, q = k_q34(p)
    return (p, m, c1, c2, e_I1(p), u_II1D(p), e_II1A(p), k, q)


def hard_primes(N):
    s = np.ones(N // 2 + 1, dtype=bool)     # s[i] <-> 2i+1
    s[0] = False
    for i in range(1, (isqrt(N) - 1) // 2 + 1):
        if s[i]:
            p = 2 * i + 1
            s[p * p // 2::p] = False
    ps = 2 * np.nonzero(s)[0] + 1
    ps = ps[ps <= N]
    return [int(p) for p in ps if int(p) % 840 in DIF]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    t0 = time.time()

    print("=== type II slices with a=1 on the 14 wild primes and on 2521, 66529 ===")
    wild = [409, 577, 5569, 9601, 23929, 83449, 102001, 329617, 712321,
            1134241, 1724209, 1726201, 5212561, 8813281]
    for p in wild + [2521, 66529]:
        print(f"  p={p:>8}: II1D (Dahan Alg.2) u={u_II1D(p)},  II1A (Aigner a_ET=1) e={e_II1A(p, cap=10**9)}")

    print("\n=== local criteria at c=3 and c=7 (hard primes <= 10^7) ===")
    P7 = hard_primes(10 ** 7)
    a3 = b3 = a7 = b7 = d3 = d7 = 0
    for p in P7:
        t3 = any(types_at(p, 3))
        k3 = any(q % 3 == 2 for q in factorint((p + 3) // 4))
        t1, t2 = types_at(p, 7)
        k7 = any(q % 7 in (3, 5, 6) for q in factorint((p + 7) // 4))
        a3 += t3; b3 += k3; d3 += t3 != k3
        a7 += t1 and t2; b7 += k7; d7 += (t1 != k7) + (t2 != k7)
    print(f"  {len(P7)} primes. c=3: types={a3}, criterion={b3}, discrepancies={d3}")
    print(f"  c=7: both types={a7}, criterion={b7}, discrepancies={d7}")

    print("\n=== full recomputation up to 10^8 (8 processes) ===")
    P = hard_primes(10 ** 8)
    print(f"  {len(P)} hard primes <= 10^8 ({time.time() - t0:.0f} s)")
    with Pool(8) as pool:
        table = pool.map(compute_row, P, chunksize=400)
    print(f"  computation finished ({time.time() - t0:.0f} s)")
    mx = max(table, key=lambda f: f[1])
    high_rows = [f for f in table if f[0] > 10 ** 7]
    mx2 = max(high_rows, key=lambda f: f[1])
    print(f"  max m = {mx[1]} at p = {mx[0]};  max m in (10^7,10^8] = {mx2[1]} at p = {mx2[0]}")
    print(f"  primes with m >= 59 in (10^7,10^8]: {sorted((f[1], f[0]) for f in high_rows if f[1] >= 59)}")
    for name, idx in [("I1", 4), ("II1D", 5), ("II1A", 6)]:
        missing = [f[0] for f in table if f[idx] is None]
        pending = [f[0] for f in table if f[idx] == "pending"]
        finished = [f for f in table if isinstance(f[idx], int)]
        top = max(finished, key=lambda f: f[idx])
        print(f"  {name}: no solution = {missing}; pending = {pending}; max = {top[idx]} at p = {top[0]}")
    print(f"  neither I1 nor II1D: {[f[0] for f in table if f[4] is None and f[5] is None]}")
    kq = Counter(f[7] for f in table)
    print(f"  k_q34: {sorted(kq.items(), key=lambda kv: (kv[0] is None, kv[0]))}")
    print(f"  k_q34 maximum at: {[(f[0], f[8], f[1]) for f in table if f[7] == max(k for k in kq if k)]}")

    print("\n  F_3 * sqrt(log x_mid) by decade:")
    for a in range(3, 8):
        sub = [f for f in table if 10 ** a < f[0] <= 10 ** (a + 1)]
        F3 = sum(1 for f in sub if f[1] > 3) / len(sub)
        print(f"    (10^{a}, 10^{a + 1}]: n={len(sub):>7}  F_3={F3:.4f}  F_3*sqrt(log x_mid)={F3 * sqrt(log(10 ** (a + 0.5))):.4f}")

    print("\n=== Row-by-row comparison with data/hard_primes_1e8.csv ===")
    with open("data/hard_primes_1e8.csv", encoding="utf-8") as fh:
        G = {int(r["p"]): r for r in csv.DictReader(fh)}
    print(f"  same p: {set(G) == set(P)} ({len(G)} vs {len(P)})")
    cols = ["m", "min_c_I", "min_c_II", "e_min_I1", "u_min_II1", "e_min_II_aET1", "k_q34", "q_hit"]
    dif = Counter()
    examples = {}
    for f in table:
        g = G.get(f[0])
        if g is None:
            continue
        for c, v in zip(cols, f[1:]):
            gv = g[c]
            ok = (str(v) == gv) or (v is None and gv in ("", "None", "-1", "inf", "nan"))
            if not ok:
                dif[c] += 1
                examples.setdefault(c, (f[0], v, gv))
    print(f"  discrepancies by column: {dict(dif) or 'none'}")
    for c, e in examples.items():
        print(f"    example {c}: p={e[0]} here={e[1]} CSV={e[2]!r}")
    print(f"\nTotal time: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
