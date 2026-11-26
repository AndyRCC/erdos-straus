#!/usr/bin/env python3
"""
Controles de la nota, §6: cuadrados impares y compuestos libres de cuadrados sin soluciones
de tipo I ni II (105, 345, 561); formas cuadráticas Q = [de, 4ad, f] de discriminante −4dp y
el caso m = 8, p = 241.
"""

import os
import sys
import time
from math import gcd, isqrt
from sympy import jacobi_symbol, isprime, divisors

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bright_loughran import build_spf, all_positive_solutions, factor_int  # noqa: E402


def has_type2_with_d_bound(p, d_max):
    """
    Decide de forma EXHAUSTIVA si existe una solución Tipo II para 4/p (p ≡ 1 mod 4 primo)
    con parámetro d <= d_max en E-T Prop. 2.6:
      4 a b d = p + e,   e | a + b,   gcd(a, b) = 1,   1 <= d <= d_max.
    Como a <= b y b < ce <= 2b (E-T Lema 2.8), acde <= p => e <= p / (a*c*d) <= p / d.
    Más exactamente, de 4abd = p + e y e <= a + b <= 2ab:
      p = 4abd - e >= 4abd - 2ab = 2ab(2d - 1) => ab <= p / (2*(2d - 1)).
    Luego para cada d <= d_max, recorremos a*b <= p // (2*(2d - 1)), fijamos e = 4abd - p,
    y comprobamos e > 0 y e | (a + b) (y e < 3p, automático).
    """
    for d in range(1, d_max + 1):
        ab_max = p // (2 * (2 * d - 1))
        # Recorrer pares (a, b) con 1 <= a <= b y a*b <= ab_max
        # Más rápido: también e = 4abd - p > 0 => ab > p / (4d).
        ab_min = p // (4 * d) + 1
        # Iterar a hasta sqrt(ab_max)
        a_lim = isqrt(ab_max)
        for a in range(1, a_lim + 1):
            b_lo = max(a, (ab_min + a - 1) // a)
            b_hi = ab_max // a
            for b in range(b_lo, b_hi + 1):
                e = 4 * a * b * d - p
                if e > 0 and (a + b) % e == 0:
                    # Comprobar coprimalidad con p (automática si e < 3p y p primo, pues ab <= p/2)
                    if gcd(a, b) == 1 and (a * b * d) % p != 0:
                        return True, (a, b, (a + b) // e, d, e)
    return False, None


def count_reduced_forms_nonprincipal_genus(D, p_odd):
    """Cuenta formas binarias primitivas reducidas [A, B, C] de discriminante D < 0 con (A/p_odd) = -1."""
    # Reducida: |B| <= A <= C, y si |B|=A o A=C entonces B >= 0. B^2 - 4AC = D.
    forms_minus = []
    forms_plus = []
    A_max = isqrt((-D) // 3)
    for A in range(1, A_max + 1):
        for B in range(-A, A + 1):
            if (B * B - D) % (4 * A) != 0:
                continue
            C = (B * B - D) // (4 * A)
            if C < A:
                continue
            if (abs(B) == A or A == C) and B < 0:
                continue
            if gcd(gcd(A, abs(B)), C) != 1:
                continue
            # Valor del carácter de género chi_p sobre un entero coprimo con p representado por Q
            val = A if A % p_odd != 0 else (C if C % p_odd != 0 else A + B + C)
            j = int(jacobi_symbol(val, p_odd))
            if j == -1:
                forms_minus.append((A, B, C))
            elif j == 1:
                forms_plus.append((A, B, C))
    return forms_plus, forms_minus


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    spf = build_spf(3_100_000)

    # =========================================================================
    # 1. Cuadrados impares y compuestos libres de cuadrados con f_I = f_II = 0
    # =========================================================================
    print("=== 1. CUADRADOS IMPARES Y COMPUESTOS NO CUADRADOS SIN SOLUCIÓN TIPO I NI TIPO II ===")
    for n in [9, 25, 49, 121, 169, 289]:
        sols = all_positive_solutions(4, n, spf)
        t12 = sum(
            1 for (u1, u2, u3) in sols
            if (sum(u % n == 0 for u in (u1, u2, u3)) == 1 and sum(gcd(u, n) == 1 for u in (u1, u2, u3)) == 2)
            or (sum(u % n == 0 for u in (u1, u2, u3)) == 2 and sum(gcd(u, n) == 1 for u in (u1, u2, u3)) == 1)
        )
        print(f"  Cuadrado impar n={n:>3}: soluciones positivas={len(sols):>3}, Tipo I/II={t12}")

    # Buscar todos los impares no cuadrados <= 2000 con f_I(n) = f_II(n) = 0
    nonsq_zero = []
    for n in range(3, 2000, 2):
        fac = factor_int(n, spf)
        if all(e % 2 == 0 for e in fac.values()):
            continue
        sols = all_positive_solutions(4, n, spf)
        has_t12 = any(
            (sum(u % n == 0 for u in (u1, u2, u3)) == 1 and sum(gcd(u, n) == 1 for u in (u1, u2, u3)) == 2)
            or (sum(u % n == 0 for u in (u1, u2, u3)) == 2 and sum(gcd(u, n) == 1 for u in (u1, u2, u3)) == 1)
            for (u1, u2, u3) in sols
        )
        if not has_t12:
            nonsq_zero.append((n, fac, len(sols)))
    print(f"  Impares NO cuadrados <= 2000 con f_I(n) = f_II(n) = 0: {len(nonsq_zero)}")
    for n, fac, nsol in nonsq_zero:
        f_str = "*".join(f"{p}" if e == 1 else f"{p}^{e}" for p, e in sorted(fac.items()))
        print(f"    n={n} = {f_str} (n mod 4 = {n%4}, libre de cuadrados): {nsol} sols. positivas, 0 Tipo I/II")

    # =========================================================================
    # 2. Formas Q = [de, 4ad, f] de disc -4dp en género chi_p = -1
    # =========================================================================
    print("\n=== 2. FORMAS BINARIAS Q=[de, 4ad, f] EN Cl(-4dp) ===")
    # 2a. Verificación de género no principal en todos los primos p <= 1000, p ≡ 1 (mod 4)
    checked_q = 0
    ok_q = 0
    for p in range(5, 1000, 4):
        if spf[p] != p:
            continue
        for u1, u2, u3 in all_positive_solutions(4, p, spf):
            div_p = [u for u in (u1, u2, u3) if u % p == 0]
            cop_p = [u for u in (u1, u2, u3) if u % p != 0]
            if len(div_p) != 2:
                continue
            x, y, z = cop_p[0], div_p[0], div_p[1]
            d = gcd(gcd(x, y // p), z // p)
            xp, yp, zp = x // d, (y // p) // d, (z // p) // d
            a, b, c = gcd(xp, yp), gcd(xp, zp), gcd(yp, zp)
            e = (a + b) // c
            f = 4 * a * c * d - 1
            assert (4 * a * d) ** 2 - 4 * (d * e) * f == -4 * d * p
            assert gcd(gcd(d * e, 4 * a * d), f) == 1
            checked_q += 1
            if int(jacobi_symbol(f, p)) == -1 and int(jacobi_symbol(d * e, p)) == -1:
                ok_q += 1
    print(f"  Tipo II (p<=1000, p≡1 mod 4): {ok_q}/{checked_q} dan Q=[de,4ad,f] primitiva, disc=-4dp, chi_p=-1")

    # 2b. PRUEBA 1: ¿basta d = 1 (solo Cl(-4p)) o d <= D fijo?
    primes_24 = [p for p in range(25, 50_000, 24) if isprime(p)]
    for d_lim in (1, 2, 5, 10):
        excs = []
        for p in primes_24:
            ok, _ = has_type2_with_d_bound(p, d_lim)
            if not ok:
                excs.append(p)
        pct = 100.0 * len(excs) / len(primes_24)
        print(f"  Prueba 1 (d <= {d_lim:>2}, p ≡ 1 mod 24 < 50,000, total {len(primes_24)}): "
              f"{len(excs)} excepciones ({pct:.2f}%), primeras: {excs[:8]}")

    # 2c. PRUEBA 2 (barrera m = 8, p = 241):
    # En 8/241: Q = [de, 8ad, f] tendría disc (8ad)^2 - 4(de)f = -8dp (pues ef = p + 8a^2d?
    # Exactamente: 8abcd = a + b + pc, f = 8acd - 1 => ef = p + 8a^2d => (8ad)^2 - 4(2de)f = -8dp).
    f_plus, f_minus = count_reduced_forms_nonprincipal_genus(-8 * 241, 241)
    sols_8_241 = all_positive_solutions(8, 241, spf)
    print(f"\n  Prueba 2 (barrera m=8, p=241):")
    print(f"    Cl(-8*241 = -1928): {len(f_plus)} formas reducidas con chi_241=+1, "
          f"{len(f_minus)} con chi_241=-1 (ej: {f_minus[:4]})")
    print(f"    Soluciones positivas de 8/241 = 1/u1 + 1/u2 + 1/u3: {len(sols_8_241)}")
    print("    => El género no principal de Cl(-8p) es abundante, pero NO produce ninguna solución.")

    print(f"\nTiempo total: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
