#!/usr/bin/env python3
"""
Independent check of the note, §6: R² of the table in §2, the Bright–Loughran theorems
with our own Hilbert symbol, exceptions of 8/n, composites without types I/II, and U_{8,241}.
"""
import csv
import math
import os
import sys
from fractions import Fraction as F
from math import gcd, log

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
from sympy import divisors, factorint, isprime, legendre_symbol, primerange  # noqa: E402


# ----------------------------------------------------------------- utilities
def vp(x, p):
    x, k = abs(x), 0
    while x % p == 0:
        x //= p
        k += 1
    return k


def hilbert(a, b, p):
    """(a,b)_p for p an odd prime and a, b nonzero rationals (Serre, Cours d'arithmétique, III)."""
    a, b = F(a), F(b)
    al = vp(a.numerator, p) - vp(a.denominator, p)
    be = vp(b.numerator, p) - vp(b.denominator, p)
    u = a / F(p) ** al
    v = b / F(p) ** be
    def leg(w):
        return int(legendre_symbol(w.numerator % p, p)) * int(legendre_symbol(w.denominator % p, p))
    # everything in integers: with negative exponents (-1)**k gives floats, and in sympy >= 1.13 Float(-1.0) != -1
    s = -1 if (al * be * ((p - 1) // 2)) % 2 else 1
    return s * (leg(u) if be % 2 else 1) * (leg(v) if al % 2 else 1)


def hilbert_inf(a, b):
    return -1 if (a < 0 and b < 0) else 1


def solutions(m, n):
    """All positive solutions x <= y <= z of m/n = 1/x + 1/y + 1/z."""
    out = []
    for x in range(n // m + 1, (3 * n) // m + 1):
        A, B = m * x - n, n * x
        g = gcd(A, B)
        A, B = A // g, B // g
        if A <= 0:
            continue
        D = divisors(B)
        for yp in D:
            for zp in D:
                if zp < yp or gcd(yp, zp) != 1 or B % (yp * zp) or (yp + zp) % A:
                    continue
                k = (yp + zp) // A
                gg = B * k // (yp * zp)
                y, z = gg * yp, gg * zp
                if y >= x:
                    out.append((x, y, z))
    return out


def is_sol(m, n, u):
    x, y, z = u
    return m * x * y * z == n * (x * y + y * z + z * x)


def main():
    # ------------------------------------------------------------- A
    print("=== A) R² of the window [2e7, 5e7) (from data/dahan_E1_1e9.csv) ===")
    with open("data/dahan_E1_1e9.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["lo"].startswith("2000000") and r["hi"].startswith("5000000"):
                n = int(r["n_primes"])
                xs = [log(J) for J in range(1, 21)]
                ys = [log(int(r[f"E1_J{J}"]) / n) for J in range(1, 21)]
                mx, my = sum(xs) / 20, sum(ys) / 20
                b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
                a0 = my - b * mx
                r2 = 1 - sum((c - (a0 + b * a)) ** 2 for a, c in zip(xs, ys)) / sum((c - my) ** 2 for c in ys)
                print(f"  n={n}, β={-b:.4f}, R²={r2:.8f} -> 3 decimals: {r2:.3f}")

    # ------------------------------------------------------------- B
    print("\n=== B) Bright–Loughran ===")
    failures12 = total12 = 0
    for n in range(3, 152, 2):
        for u in solutions(4, n):
            assert is_sol(4, n, u)
            prod = 1
            for p in factorint(n):
                prod *= hilbert(F(-u[0], u[2]), F(-u[1], u[2]), p)
            total12 += 1
            failures12 += prod != -1
    print(f"  Thm 1.2 (all positive solutions, odd n 3..151): {total12} solutions, failures = {failures12}")
    f13 = t13 = 0
    for p in primerange(3, 400):
        for u in solutions(4, p):
            t13 += 1
            ok = any(vp(u[i], p) == vp(u[j], p) and legendre_symbol((-F(u[i], u[j])).numerator * pow((-F(u[i], u[j])).denominator, -1, p) % p, p) == -1
                     for i in range(3) for j in range(3) if i != j)
            f13 += not ok
    print(f"  Cor 1.3 (primes 3..397): {t13} solutions, failures = {f13}")
    for n in [9, 25, 49, 121, 169]:
        S = solutions(4, n)
        t12 = sum(1 for u in S if sum(1 for d in u if d % n == 0) in (1, 2)
                  and all(gcd(n, d) == 1 for d in u if d % n))
        print(f"  Cor 1.4: n={n}: {len(S)} positive solutions, of type I/II: {t12}")
    f15 = t15 = 0
    for n in range(3, 52, 2):
        for x in range(1, 300):
            for y in range(x, 300):
                r = F(4, n) - F(1, x) - F(1, y)
                if r < 0 and r.numerator == -1:
                    u = (x, y, -r.denominator)
                    assert is_sol(4, n, u)
                    prod = 1
                    for p in factorint(n):
                        prod *= hilbert(F(-u[0], u[2]), F(-u[1], u[2]), p)
                    t15 += 1
                    f15 += prod != 1
    print(f"  Thm 1.5 (solutions with z < 0, odd n 3..51): {t15} solutions, failures = {f15}")

    # ------------------------------------------------------------- C
    print("\n=== C) Numerator barrier: exceptions of 8/n, 2 <= n <= 300 ===")
    exc8 = [n for n in range(2, 301) if not solutions(8, n)]
    print(f"  8/n with no positive solution: {exc8}  (Pomerance, slide 19: 1, 2, 3, 11, 17, 131, 241)")

    # ------------------------------------------------------------- D
    print("\n=== D) Composites with no solution of type I or II ===")
    for n in [105, 345, 561, 15, 21, 33]:
        S = solutions(4, n)
        types = 0
        for u in S:
            div = [d for d in u if d % n == 0]
            cop = [d for d in u if gcd(d, n) == 1]
            if len(div) in (1, 2) and len(div) + len(cop) == 3:
                types += 1
        print(f"  n={n} = {dict(factorint(n))}: {len(S)} solutions (x<=y<=z), of type I/II: {types}")

    # ------------------------------------------------------------- E
    print("\n=== E) U_{8,241}: 8 u1 u2 u3 = 241 (u1u2 + u1u3 + u2u3) ===")
    m, n = 8, 241
    uZ = (30, -14460, -14460)
    print(f"  uZ on the surface: {is_sol(m, n, uZ)}")
    a, b = F(-uZ[0], uZ[2]), F(-uZ[1], uZ[2])
    inv = {"∞": hilbert_inf(a, b)}
    odd_list = set(factorint(a.numerator)) | set(factorint(a.denominator)) | set(factorint(b.numerator)) | set(factorint(b.denominator)) | {241}
    for p in sorted(q for q in odd_list if q > 2):
        inv[p] = hilbert(a, b, p)
    prod_imp = math.prod(v for k, v in inv.items() if k != "∞")
    inv[2] = inv["∞"] * prod_imp   # Hilbert reciprocity: total product = 1
    print(f"  local invariants of uZ: {inv}")
    # 241-adically integral points with u1 = 241 a1, 8 a1 ≡ 1 (mod 241)
    seen_vals = set()
    for s in range(0, 5):
        a1 = 211 + 241 * s
        t = (8 * a1 - 1) // 241
        if t % 241 == 0:
            continue
        for u2 in range(1, 60):
            den = t * u2 - a1
            if den % 241 == 0:
                continue
            u = (F(241 * a1), F(u2), F(a1 * u2, den))
            assert 8 * u[0] * u[1] * u[2] == 241 * (u[0] * u[1] + u[0] * u[2] + u[1] * u[2])
            assert u[2].denominator % 241 and u[2].numerator % 241
            seen_vals.add(hilbert(-u[0] / u[2], -u[1] / u[2], 241))
    print(f"  values of inv_241 at the 241-adically integral points constructed: {sorted(seen_vals)}")
    print(f"  inv_∞ on U(R)+ (all coordinates positive): {hilbert_inf(-1, -1)}")


if __name__ == "__main__":
    main()
