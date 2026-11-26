#!/usr/bin/env python3
"""
Comprobación independiente de la nota, §6: R² de la tabla de §2, teoremas de Bright–Loughran
con un símbolo de Hilbert propio, excepciones de 8/n, compuestos sin tipos I/II y U_{8,241}.
"""
import csv
import math
import os
import sys
from fractions import Fraction as F
from math import gcd, log

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(RAIZ)
from sympy import divisors, factorint, isprime, legendre_symbol, primerange  # noqa: E402


# ----------------------------------------------------------------- utilidades
def vp(x, p):
    x, k = abs(x), 0
    while x % p == 0:
        x //= p
        k += 1
    return k


def hilbert(a, b, p):
    """(a,b)_p para p primo impar y a, b racionales no nulos (Serre, Cours d'arithmétique, III)."""
    a, b = F(a), F(b)
    al = vp(a.numerator, p) - vp(a.denominator, p)
    be = vp(b.numerator, p) - vp(b.denominator, p)
    u = a / F(p) ** al
    v = b / F(p) ** be
    def leg(w):
        return int(legendre_symbol(w.numerator % p, p)) * int(legendre_symbol(w.denominator % p, p))
    # todo en enteros: con exponentes negativos, (-1)**k da floats y en sympy >= 1.13 Float(-1.0) != -1
    s = -1 if (al * be * ((p - 1) // 2)) % 2 else 1
    return s * (leg(u) if be % 2 else 1) * (leg(v) if al % 2 else 1)


def hilbert_inf(a, b):
    return -1 if (a < 0 and b < 0) else 1


def soluciones(m, n):
    """Todas las soluciones positivas x <= y <= z de m/n = 1/x + 1/y + 1/z."""
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


def es_sol(m, n, u):
    x, y, z = u
    return m * x * y * z == n * (x * y + y * z + z * x)


def main():
    # ------------------------------------------------------------- A
    print("=== A) R² de la ventana [2e7, 5e7) (desde datos/dahan_E1_1e9.csv) ===")
    with open("datos/dahan_E1_1e9.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["lo"].startswith("2000000") and r["hi"].startswith("5000000"):
                n = int(r["n_primes"])
                xs = [log(J) for J in range(1, 21)]
                ys = [log(int(r[f"E1_J{J}"]) / n) for J in range(1, 21)]
                mx, my = sum(xs) / 20, sum(ys) / 20
                b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
                a0 = my - b * mx
                r2 = 1 - sum((c - (a0 + b * a)) ** 2 for a, c in zip(xs, ys)) / sum((c - my) ** 2 for c in ys)
                print(f"  n={n}, β={-b:.4f}, R²={r2:.8f} -> 3 decimales: {r2:.3f}")

    # ------------------------------------------------------------- B
    print("\n=== B) Bright–Loughran ===")
    fallos12 = total12 = 0
    for n in range(3, 152, 2):
        for u in soluciones(4, n):
            assert es_sol(4, n, u)
            prod = 1
            for p in factorint(n):
                prod *= hilbert(F(-u[0], u[2]), F(-u[1], u[2]), p)
            total12 += 1
            fallos12 += prod != -1
    print(f"  Thm 1.2 (todas las soluciones positivas, n impar 3..151): {total12} soluciones, fallos = {fallos12}")
    f13 = t13 = 0
    for p in primerange(3, 400):
        for u in soluciones(4, p):
            t13 += 1
            ok = any(vp(u[i], p) == vp(u[j], p) and legendre_symbol((-F(u[i], u[j])).numerator * pow((-F(u[i], u[j])).denominator, -1, p) % p, p) == -1
                     for i in range(3) for j in range(3) if i != j)
            f13 += not ok
    print(f"  Cor 1.3 (primos 3..397): {t13} soluciones, fallos = {f13}")
    for n in [9, 25, 49, 121, 169]:
        S = soluciones(4, n)
        t12 = sum(1 for u in S if sum(1 for d in u if d % n == 0) in (1, 2)
                  and all(gcd(n, d) == 1 for d in u if d % n))
        print(f"  Cor 1.4: n={n}: {len(S)} soluciones positivas, de tipo I/II: {t12}")
    f15 = t15 = 0
    for n in range(3, 52, 2):
        for x in range(1, 300):
            for y in range(x, 300):
                r = F(4, n) - F(1, x) - F(1, y)
                if r < 0 and r.numerator == -1:
                    u = (x, y, -r.denominator)
                    assert es_sol(4, n, u)
                    prod = 1
                    for p in factorint(n):
                        prod *= hilbert(F(-u[0], u[2]), F(-u[1], u[2]), p)
                    t15 += 1
                    f15 += prod != 1
    print(f"  Thm 1.5 (soluciones con z < 0, n impar 3..51): {t15} soluciones, fallos = {f15}")

    # ------------------------------------------------------------- C
    print("\n=== C) Barrera del numerador: excepciones de 8/n, 2 <= n <= 300 ===")
    exc8 = [n for n in range(2, 301) if not soluciones(8, n)]
    print(f"  8/n sin solución positiva: {exc8}  (Pomerance, diap. 19: 1, 2, 3, 11, 17, 131, 241)")

    # ------------------------------------------------------------- D
    print("\n=== D) Compuestos sin soluciones de tipo I ni II ===")
    for n in [105, 345, 561, 15, 21, 33]:
        S = soluciones(4, n)
        tipos = 0
        for u in S:
            div = [d for d in u if d % n == 0]
            cop = [d for d in u if gcd(d, n) == 1]
            if len(div) in (1, 2) and len(div) + len(cop) == 3:
                tipos += 1
        print(f"  n={n} = {dict(factorint(n))}: {len(S)} soluciones (x<=y<=z), de tipo I/II: {tipos}")

    # ------------------------------------------------------------- E
    print("\n=== E) U_{8,241}: 8 u1 u2 u3 = 241 (u1u2 + u1u3 + u2u3) ===")
    m, n = 8, 241
    uZ = (30, -14460, -14460)
    print(f"  uZ en la superficie: {es_sol(m, n, uZ)}")
    a, b = F(-uZ[0], uZ[2]), F(-uZ[1], uZ[2])
    inv = {"∞": hilbert_inf(a, b)}
    impares = set(factorint(a.numerator)) | set(factorint(a.denominator)) | set(factorint(b.numerator)) | set(factorint(b.denominator)) | {241}
    for p in sorted(q for q in impares if q > 2):
        inv[p] = hilbert(a, b, p)
    prod_imp = math.prod(v for k, v in inv.items() if k != "∞")
    inv[2] = inv["∞"] * prod_imp   # reciprocidad de Hilbert: producto total = 1
    print(f"  invariantes locales de uZ: {inv}")
    # puntos 241-ádicamente enteros con u1 = 241 a1, 8 a1 ≡ 1 (mod 241)
    vistos = set()
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
            vistos.add(hilbert(-u[0] / u[2], -u[1] / u[2], 241))
    print(f"  valores de inv_241 en puntos 241-ádicamente enteros construidos: {sorted(vistos)}")
    print(f"  inv_∞ en U(R)+ (todas las coordenadas positivas): {hilbert_inf(-1, -1)}")


if __name__ == "__main__":
    main()
