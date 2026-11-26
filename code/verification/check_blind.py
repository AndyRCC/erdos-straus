#!/usr/bin/env python3
"""
Independent check of the note, §5 up to 10^7: blind sets B_QR, B_exact and B_form
for c ∈ {3, 7, 11, 19, 23}, the one-parameter family and the quartic family at c = 3 and c = 7.
"""
import os
import sys
from math import gcd, isqrt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify import types_at  # noqa: E402
from sympy import factorint, isprime, legendre_symbol, primerange, symbols, expand  # noqa: E402

N = 10 ** 7
# c: (a, b, cc, B) with Q = a x² + b xy + cc y² of discriminant −c (or −c0) and n = B·Q − c
CONSTR = {3: (1, 1, 1, 4), 7: (1, 1, 2, 16), 11: (3, 1, 1, 12), 19: (1, 1, 5, 4), 23: (3, 1, 2, 48)}
REFERENCE = {  # counts from code/numba/blind_1e9.py
    3: (35750, 35750, 35750), 7: (18732, 18732, 9590), 11: (21262, 25242, 21262),
          19: (25959, 46440, 25959), 23: (16371, 17255, 7715)}


def qr(q, c):
    return legendre_symbol(q % c, c) == 1


def form_values(a, b, cc, B, c, N):
    M = (N + c) // B
    D = b * b - 4 * a * cc
    out = set()
    ymax = isqrt(4 * a * M // (-D)) + 2
    for y in range(-ymax, ymax + 1):
        # a x² + b y x + (cc y² − M) <= 0
        disc = (b * y) ** 2 - 4 * a * (cc * y * y - M)
        if disc < 0:
            continue
        r = isqrt(disc)
        x0, x1 = (-b * y - r) // (2 * a) - 1, (-b * y + r) // (2 * a) + 1
        for x in range(x0, x1 + 1):
            if gcd(x, y) != 1:
                continue
            Q = a * x * x + b * x * y + cc * y * y
            n = B * Q - c
            if c < n <= N and isprime(n):
                out.add(n)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    P = [p for p in primerange(2, N + 1) if p % 24 == 1]
    print(f"=== A) Blind sets up to 10^7 ({len(P):,} primes p ≡ 1 mod 24) ===")
    for c, (a, b, cc, B) in CONSTR.items():
        Bqr = {p for p in P if p > c and all(qr(q, c) for q in factorint((p + c) // 4))}
        Bex = {p for p in P if p > c and not any(types_at(p, c))}
        Bfo = form_values(a, b, cc, B, c, N)
        ok = Bfo <= Bqr <= Bex and all(n % 24 == 1 for n in Bfo)
        g = REFERENCE[c]
        print(f"  c={c:>2}: QR={len(Bqr):>6}  exact={len(Bex):>6}  form={len(Bfo):>6}  "
              f"inclusions={'OK' if ok else 'FAILED'}  form=QR: {Bfo == Bqr}  "
              f"Numba={g}  match={(len(Bqr), len(Bex), len(Bfo)) == g}")

    print("\n=== B) Frontier c = 3 and c = 7 ===")
    s, t = symbols("s t")
    X, Y, U, V = s**2 - 84 * t**2, 14 * s * t + 1, s**2 + 84 * t**2, 6 * s * t + 1
    print("  identity U²+7V²−X²−3Y² = 4:", expand(U**2 + 7 * V**2 - X**2 - 3 * Y**2) == 4)
    print("  p = X²+3Y²−3 =", expand(X**2 + 3 * Y**2 - 3))
    # one-parameter family: (u,v) = (t+5, 19) in f7 = u²+uv+2v², (x,y) = (t, 29) in f3 = x²+xy+y²
    bad, blind = 0, []
    for tt in range(0, 200):
        f3 = tt * tt + 29 * tt + 841
        f7 = (tt + 5) ** 2 + 19 * (tt + 5) + 722
        assert f7 == f3 + 1
        p = 4 * f3 - 3
        assert p == (2 * tt + 29) ** 2 + 2520
        if gcd(tt, 29) == 1 and gcd(tt + 5, 19) == 1 and isprime(p):
            b3, b7 = not any(types_at(p, 3)), not any(types_at(p, 7))
            if b3 and b7:
                blind.append(p)
            else:
                bad += 1
    print(f"  1 parameter, 0 <= t < 200: {len(blind)} primes blind at c=3 and c=7 (Numba: 35), "
          f"primes of the family that are NOT blind: {bad}; first ones: {blind[:4]}")
    bad, blind = 0, set()
    for ss in range(1, 60, 2):
        for tt in list(range(-20, 0)) + list(range(1, 21)):
            Xv, Yv, Uv, Vv = ss * ss - 84 * tt * tt, 14 * ss * tt + 1, ss * ss + 84 * tt * tt, 6 * ss * tt + 1
            x, y, u, v = (Xv - Yv) // 2, Yv, (Uv - Vv) // 2, Vv
            p = Xv * Xv + 3 * Yv * Yv - 3
            if gcd(x, y) == 1 and gcd(u, v) == 1 and p > 7 and isprime(p):
                if not any(types_at(p, 3)) and not any(types_at(p, 7)):
                    blind.add(p)
                else:
                    bad += 1
    print(f"  quartic (odd s < 60, 0 < |t| <= 20): {len(blind)} distinct blind primes (Numba: 111), NOT blind: {bad}")


if __name__ == "__main__":
    main()
