#!/usr/bin/env python3
"""
Independent check (sympy + numpy) of results 1 and 2 of the note up to 10^9:
Dahan's E_1(N;J) and the exceptions of the Aigner slice; plus an exhaustive local check of the
criteria for c = 7 and c = 15.
"""
import os
import sys
import time
from itertools import combinations_with_replacement
from math import gcd, isqrt, log
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify import types_at  # noqa: E402
from sympy import divisors, isprime, factorint, jacobi_symbol  # noqa: E402

N = 10 ** 9
JMAX = 20
CD = 1000


def primes_24(N):
    s = np.ones(N // 2, dtype=bool)          # s[i] <-> 2i+1 < N
    s[0] = False
    for i in range(1, (isqrt(N) - 1) // 2 + 1):
        if s[i]:
            p = 2 * i + 1
            s[p * p // 2::p] = False
    ps = 2 * np.nonzero(s)[0].astype(np.int64) + 1
    return ps[ps % 24 == 1]


def umin(n):
    for u in range(1, JMAX + 1):
        m = 4 * u
        if any(s % m == m - 1 for s in divisors(n + u)):
            return u
    return JMAX + 1


def umin_block(block):
    return [umin(int(n)) for n in block]


def ii1a_exhaustive(p):
    """Are there c,d,e >= 1 with e(4cd-1) = p+4d?  Exhaustive (branches e <= E0 and e > E0)."""
    E0 = isqrt(p) + 1
    def branch_A(e_lo, e_hi):
        for e in range(e_lo, e_hi + 1, 4):
            if any(b % e == e - 1 for b in divisors((p + e) // 4)):
                return True
        return False
    if branch_A(3, min(199, E0)):
        return True
    d = 1
    while E0 * (4 * d - 1) < p + 4 * d:
        c = 1
        while E0 * (4 * c * d - 1) < p + 4 * d:
            s = 4 * c * d - 1
            if (p + 4 * d) % s == 0 and (p + 4 * d) // s > E0:
                return True
            c += 1
        d += 1
    return E0 > 199 and branch_A(203, E0)


def fit_line(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (a * x + b)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2


def local_exhaustive(c, p_res, size_max):
    """For p ≡ p_res (mod c), runs through all multisets of residues of primes of K_c
    (including at least one 2, because 2 | K_c) with product ≡ K_c (mod c). Returns
    the discrepancies between: type II, type I and 'some prime with Jacobi (q/c) = -1'."""
    G = [r for r in range(1, c) if gcd(r, c) == 1]
    inv4 = pow(4, -1, c)
    K = (p_res * inv4) % c
    bad = []
    for k in range(0, size_max):
        for remainder in combinations_with_replacement(G, k):
            ms = (2,) + remainder
            prod = 1
            for r in ms:
                prod = prod * r % c
            if prod != K:
                continue
            S = {1}
            for r in ms:
                S |= {s * r % c for s in S}
            R = {a * pow(b, -1, c) % c for a in S for b in S}
            t2 = (c - 1) in R
            t1 = (-p_res) % c in R
            nr = any(jacobi_symbol(r, c) == -1 for r in ms)
            if not (t1 == t2 == nr):
                bad.append((ms, t1, t2, nr))
    return bad


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    t0 = time.time()

    # ---------------------------------------------------------------- C (fast)
    print("=== C) Exhaustive local check (multisets of up to 8 primes, including 2) ===")
    for c, residues in [(7, [1, 2, 4]), (15, [1, 4])]:
        for pr in residues:
            m = local_exhaustive(c, pr, 8)
            print(f"  c={c}, p ≡ {pr} (mod {c}): discrepancies = {len(m)}{'  e.g.: ' + str(m[0]) if m else ''}")
    print("  (control) c=15, p ≡ 13 (mod 15), which is not hard: ", end="")
    m = local_exhaustive(15, 13, 6)
    print(f"{len(m)} discrepancies (expected > 0: there only type II can fail)")
    examples = [(97, 23, False), (8929, 23, True), (73, 31, False), (1801, 31, True), (73, 39, False),
                (1009, 39, True), (73, 47, False), (2521, 47, True), (97, 55, False), (1009, 55, True),
                (73, 63, False), (2521, 63, True), (97, 71, False), (1801, 71, True), (193, 79, False),
                (1009, 79, True)]
    ok = 0
    for p, c, hard_flag in examples:
        K = (p + c) // 4
        nr = [q for q in factorint(K) if jacobi_symbol(q, c) == -1]
        good = (isprime(p) and p % 24 == 1 and (p % 840 in {1, 121, 169, 289, 361, 529}) == hard_flag
                and nr and types_at(p, c) == (False, False))
        ok += bool(good)
        if not good:
            print(f"  ✗ counterexample NOT confirmed: p={p}, c={c}")
    print(f"  counterexamples (c ≡ 7 mod 8, c >= 23) confirmed: {ok}/{len(examples)}")
    DIF = {1, 121, 169, 289, 361, 529}
    from sympy import primerange
    d15 = n15 = 0
    for p in primerange(2, 10 ** 7 + 1):
        if p % 840 in DIF:
            K = (p + 15) // 4
            nr = any(jacobi_symbol(q, 15) == -1 for q in factorint(K))
            t1, t2 = types_at(p, 15)
            n15 += nr
            d15 += not (t1 == t2 == nr)
    print(f"  c=15 on hard primes <= 10^7: with a prime (q/15)=-1: {n15}, discrepancies: {d15}")

    # ---------------------------------------------------------------- primes
    P = primes_24(N)
    print(f"\nPrimes p ≡ 1 (mod 24) < 10^9: {len(P):,}  ({time.time() - t0:.0f} s)")
    print(f"  of which < 2·10^7: {(P < 2 * 10 ** 7).sum():,}")

    # ---------------------------------------------------------------- B
    print("\n=== B) Aigner slice, type II (a_ET=1): exceptions up to 10^9 ===")
    alive = P.copy()
    pairs = sorted(((4 * c * d - 1, c, d) for c in range(1, CD + 1) for d in range(1, CD // c + 1)))
    for s, c, d in pairs:
        sol = ((alive + 4 * d) % s == 0) & (alive + 4 * d >= s)
        if sol.any():
            alive = alive[~sol]
    print(f"  survivors of phase 1 (cd <= {CD}): {len(alive):,}  ({time.time() - t0:.0f} s)")
    with Pool(8) as pool:
        res = pool.map(ii1a_exhaustive, [int(p) for p in alive], chunksize=20)
    exc = [int(p) for p, r in zip(alive, res) if not r]
    print(f"  EXCEPTIONS: {exc}  ({time.time() - t0:.0f} s)")

    # ---------------------------------------------------------------- A
    print("\n=== A) Dahan E_1(N;J) up to 10^9 ===")
    blocks = np.array_split(P, 4000)
    with Pool(8) as pool:
        U = np.concatenate([np.array(r, dtype=np.int16) for r in pool.map(umin_block, blocks)])
    print(f"  u_min computed for {len(U):,} primes ({time.time() - t0:.0f} s)")
    windows = [(1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 3e6), (3e6, 1e7), (1e7, 2e7),
                (2e7, 5e7), (5e7, 1e8), (1e8, 3e8), (3e8, 1e9)]
    table = []
    Js = np.arange(1, JMAX + 1)
    for A, B in windows:
        sel = (P >= A) & (P < B)
        u = U[sel]
        f = np.array([(u > J).mean() for J in Js])
        ok = f > 0
        beta, _, r2 = fit_line(np.log(Js[ok]), np.log(f[ok]))
        lm = 0.5 * (log(A) + log(B))
        table.append((A, B, sel.sum(), lm, -beta, r2, f))
        print(f"  [{A:.0e},{B:.0e}): n={sel.sum():>9,}  β={-beta:.3f}  β/log n_mid={-beta / lm:.4f}  R²={r2:.4f}"
              f"{'' if ok.all() else '  (J with f=0 omitted)'}")
    for name, idx in [("Dahan's 6 windows", range(0, 6)), ("10 windows", range(0, 10)),
                        ("9 windows without [1e3,1e4)", range(1, 10)), ("5 windows [1e7,1e9)", range(5, 10)),
                        ("4 new windows [2e7,1e9)", range(6, 10))]:
        xs = [log(table[i][3]) for i in idx]
        slopes = [fit_line(xs, [log(table[i][6][J - 1]) for i in idx])[0] for J in range(1, 7)]
        bl = fit_line([table[i][3] for i in idx], [table[i][4] for i in idx])
        print(f"  {name:<28} slopes J=1..6: {', '.join(f'{v:+.3f}' for v in slopes)}"
              f"   | β = {bl[0]:.4f} log n + {bl[1]:+.3f} (R²={bl[2]:.4f})")
    print(f"\nTotal time: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
