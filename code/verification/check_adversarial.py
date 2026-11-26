#!/usr/bin/env python3
"""
Independent check of the note, §7: the family p ≡ 1 (mod L(47)) with exact m against a
control group, exact m of adversarial primes, and a sample of f_m(p) against
data/fm_primes_2e5.csv.
"""
import csv
import os
import random
import sys
from collections import defaultdict
from math import gcd, log, prod

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from verify import types_at, is_solution  # noqa: E402
from sympy import divisors, isprime, legendre_symbol, primerange  # noqa: E402

DIF = (1, 121, 169, 289, 361, 529)


def m_of(p, cap_limit=2000):
    for m in range(3, cap_limit, 4):
        if any(types_at(p, m)):
            return m
    return None


def qmin(p):
    return next(q for q in primerange(2, 10 ** 6) if legendre_symbol(q % p, p) == -1)


def summary(ms):
    ms = sorted(ms)
    n = len(ms)
    return (f"n={n} mean={sum(ms) / n:.2f} median={ms[n // 2]} p90={ms[int(0.9 * n)]} max={ms[-1]} "
            f"P(m>3)={sum(m > 3 for m in ms) / n:.0%} P(m>11)={sum(m > 11 for m in ms) / n:.0%} "
            f"P(m>23)={sum(m > 23 for m in ms) / n:.0%}")


def f_ordered(m, p):
    """Number of ORDERED triples (x, y, z) with m/p = 1/x + 1/y + 1/z (own enumeration)."""
    tot = 0
    for x in range(p // m + 1, (3 * p) // m + 1):
        A, B = m * x - p, p * x
        g = gcd(A, B)
        A, B = A // g, B // g
        D = divisors(B)
        by_res = defaultdict(list)
        for d in D:
            by_res[d % A].append(d)
        for yp in D:
            for zp in by_res[(-yp) % A]:
                if zp < yp or gcd(yp, zp) != 1 or B % (yp * zp):
                    continue
                k = (yp + zp) // A
                gg = B * k // (yp * zp)
                y, z = gg * yp, gg * zp
                if y < x:
                    continue
                assert m * x * y * z == p * (x * y + y * z + z * x)
                tot += 6 if (x < y < z) else (1 if x == y == z else 3)
    return tot


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(7)

    print("=== A) Family A_modL1 with Y = 47 (exact m) against our own control ===")
    L = 8 * prod(primerange(3, 48))
    A = []
    k = 1
    while len(A) < 100:
        if isprime(k * L + 1):
            A.append(k * L + 1)
        k += 1
    print("  A_modL1 :", summary([m_of(p) for p in A]), " [Numba: mean 8.28, p90 23, max 39, P(m>11) 17%, P(m>23) 8%]")
    C = []
    while len(C) < 100:
        c = rng.randint(L // 840, 100 * L // 840) * 840 + rng.choice(DIF)
        if isprime(c):
            C.append(c)
    print("  control :", summary([m_of(p) for p in C]), " [other sample: mean 5.08, max 23, P(m>11) 2%]")

    print("\n=== B) exact m of adversarial primes with a high upper bound  ===")
    for p, mub, Y in [(499290503461855024921, 55, 47), (307444891294245705001, 39, 47),
                      (65041293118675673809, 39, 47), (367231949708083241494729, 47, 59)]:
        m = m_of(p)
        print(f"  p={p}: prime={isprime(p)}, hard={p % 840 in DIF}, q_min={qmin(p)} (>{Y}: {qmin(p) > Y}), "
              f"exact m={m} (bound from the search {mub}; consistent: {m is not None and m <= mub})")

    print("\n=== C) ordered f_m(p): own enumeration against data/fm_primes_2e5.csv ===")
    G = {}
    with open("data/fm_primes_2e5.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            G[int(r["p"])] = r
    sample_ps = [2, 3, 5, 7, 11, 17, 131, 241, 1009, 2521] + rng.sample([p for p in G if 1000 < p < 6000], 12)
    bad = 0
    for p in sample_ps:
        for m in (4, 8):
            f = f_ordered(m, p)
            g = int(G[p][f"f{m}"])
            bad += f != g
            if f != g:
                print(f"  ✗ p={p}, m={m}: here {f} vs CSV {g}")
    print(f"  {len(sample_ps)} primes × m ∈ {{4, 8}}: discrepancies = {bad}")
    print(f"  E–T control p. 2: f_4(2,3,5,7) = {[f_ordered(4, p) for p in (2, 3, 5, 7)]} (expected [3, 12, 12, 36])")
    class_rows = [(p, int(r["f4"]), int(r["f8"])) for p, r in G.items() if p >= 100 and p % 24 == 1]
    r4 = sum(f / log(p) ** 3 for p, f, _ in class_rows) / len(class_rows)
    r8 = sum(f / log(p) ** 3 for p, _, f in class_rows) / len(class_rows)
    print(f"  p ≡ 1 (mod 24), p >= 100: mean f4/log³ = {r4:.4f}, f8/log³ = {r8:.4f}, ratio = {r4 / r8:.2f}")
    exc8 = [p for p, r in G.items() if int(r["f8"]) == 0]
    print(f"  CSV: primes with f_8 = 0: {exc8}")

    print("\n=== D) mean of f4/(log p)³ by q_min(p) for p ≡ 1 (mod 24), p >= 1000 (CSV) ===")
    by_q = defaultdict(list)
    for p, r in G.items():
        if p >= 1000 and p % 24 == 1:
            by_q[qmin(p)].append(int(r["f4"]) / log(p) ** 3)
    for q in sorted(by_q):
        if len(by_q[q]) >= 5:
            print(f"  q_min={q:>3}: n={len(by_q[q]):>5}  mean={sum(by_q[q]) / len(by_q[q]):.4f}")


if __name__ == "__main__":
    main()
