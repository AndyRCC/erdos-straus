#!/usr/bin/env python3
"""
Independent check of the note, §5.1: certificate of the quadratic polynomial (E_231),
primes from the Dickson construction with J = 9, and sub-records of m(p) up to 10^11.
"""
import json
import os
import sys
from math import gcd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify import types_at, profile  # noqa: E402
from sympy import isprime, legendre_symbol, primerange  # noqa: E402


def m_exact(p, cap_limit=4000):
    return next(c for c in range(3, cap_limit, 4) if any(types_at(p, c)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)

    print("=== A) E_231 certificate → quadratic polynomial  ===")
    cert = json.load(open("data/e231_certificate.json", encoding="utf-8"))
    D, a0, a1, a2, X0, t0 = (int(cert[k]) for k in ("D", "a0", "a1", "a2", "X0", "t0"))
    print("  7a1² − 3a0² = 4D²:", 7 * a1 * a1 - 3 * a0 * a0 == 4 * D * D)
    print("  11a2² − 3a0² = 8D²:", 11 * a2 * a2 - 3 * a0 * a0 == 8 * D * D)
    q1 = 12541
    print(f"  D odd: {D % 2 == 1};  gcd(a0, D) = {gcd(a0, D)};  12541 | D: {D % q1 == 0};  "
          f"cofactor prime: {isprime(D // q1)};  12541 prime: {isprime(q1)};  "
          f"both ≡ 1 (mod 3): {q1 % 3 == 1 and (D // q1) % 3 == 1};  D = 12541·cofactor: {q1 * (D // q1) == D}")
    print("  X0² + 3a0² ≡ 0 (mod 4D²):", (X0 * X0 + 3 * a0 * a0) % (4 * D * D) == 0)
    X = 2 * D * D * t0 + X0
    print("  gcd(X(t0), a0·a1·a2) = 1:", gcd(X, a0 * a1 * a2) == 1)
    p0 = (X * X + 3 * a0 * a0) // (D * D) - 3
    print(f"  p0 = P(0): {len(str(p0))} digits, prime: {isprime(p0)}, p0 mod 24 = {p0 % 24}, p0 mod 840 = {p0 % 840}")
    K3, K7, K11 = (p0 + 3) // 4, (p0 + 7) // 4, (p0 + 11) // 4
    print("  4D²K3 = X²+3a0²:", 4 * D * D * K3 == X * X + 3 * a0 * a0,
          "| 4D²K7 = X²+7a1²:", 4 * D * D * K7 == X * X + 7 * a1 * a1,
          "| 4D²K11 = X²+11a2²:", 4 * D * D * K11 == X * X + 11 * a2 * a2)
    bad, seen_vals = 0, 0
    for K, c in ((K3, 3), (K7, 7), (K11, 11)):
        for q in primerange(2, 3 * 10 ** 5):
            if K % q == 0:
                seen_vals += 1
                good = (q % 3 == 1) if c == 3 else (legendre_symbol(q % c, c) == 1)
                bad += not good
    print(f"  prime factors < 3·10^5 of K3, K7, K11: {seen_vals} found, with the wrong residue: {bad}")

    print("\n=== B) Dickson primes with J = 9  ===")
    L = 1991773143360
    for t, m_g in ((1358978576, 59), (360830309, 43)):
        p = L * t + 1
        forms = [L // (4 * j) * t + 1 for j in range(1, 10)]
        integers = all(L % (4 * j) == 0 for j in range(1, 10))
        all_primes_list = isprime(p) and all(isprime(f) for f in forms)
        m = m_exact(p)
        print(f"  t={t}: p={p} ({len(str(p))} digits), L/(4j) integers: {integers}, 10 prime forms: {all_primes_list}, "
              f"exact m(p) = {m} (reference: {m_g}), > 35: {m > 35}")

    print("\n=== C) Sub-records up to 10^11  ===")
    for p, m_g in ((45195081409, 103), (92159707201, 99), (18483104929, 95), (8803369, 107)):
        m, types, c1, c2 = profile(p)
        print(f"  p={p}: prime={isprime(p)}, mod 840={p % 840}, m(p)={m} (reference: {m_g}), match={m == m_g}")

    print("\n=== D) E_231: y² = x³ − 231²x ===")
    x, y = -63, 1764
    print(f"  P0 = (−63, 1764) on the curve: {y * y == x ** 3 - 231 ** 2 * x};  "
          f"y ≠ 0, so it is not 2-torsion, and the torsion of E_n is (ℤ/2)², hence P0 has infinite order and the rank is ≥ 1")


if __name__ == "__main__":
    main()
