#!/usr/bin/env python3
"""
Comprobación numérica de Bright–Loughran (arXiv:1908.02526): Thm. 1.2, Thm. 1.5, Cor. 1.3 y
Cor. 1.4, con un símbolo de Hilbert exacto, y el ejemplo U_{8,241} (nota, §6). También
contiene utilidades (criba SPF, enumeración de soluciones) que usan otros scripts.
"""

import os
import sys
import time
from fractions import Fraction
from math import gcd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
from verificar import es_solucion  # noqa: E402


# ---------------------------------------------------------------------------
# Factorización rápida con cribado SPF
# ---------------------------------------------------------------------------

def build_spf(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def factor_int(n, spf):
    """Factorización prima de n > 0 usando tabla SPF (con división de respaldo si n > len(spf)-1)."""
    res = {}
    lim = len(spf) - 1
    while n > lim:
        # extraer factores pequeños
        d = 2
        while d * d <= n and n > lim:
            if n % d == 0:
                c = 0
                while n % d == 0:
                    n //= d
                    c += 1
                res[d] = res.get(d, 0) + c
            d = 3 if d == 2 else d + 2
        if n > lim:
            res[n] = res.get(n, 0) + 1
            return res
    while n > 1:
        p = spf[n]
        c = 0
        while n % p == 0:
            n //= p
            c += 1
        res[p] = res.get(p, 0) + c
    return res


def divisors_of_square_upto(B, fac_B):
    """Genera todos los divisores positivos d de B^2 con d <= B."""
    divs = [1]
    for p, e in fac_B.items():
        cur = []
        pe = 1
        for _ in range(2 * e + 1):
            for x in divs:
                v = x * pe
                if v <= B:
                    cur.append(v)
            pe *= p
            if pe > B:
                break
        divs = cur
    return divs


def all_divisors_of_square(fac_B):
    """Genera todos los divisores positivos d de B^2."""
    divs = [1]
    for p, e in fac_B.items():
        cur = []
        pe = 1
        for _ in range(2 * e + 1):
            for x in divs:
                cur.append(x * pe)
            pe *= p
        divs = cur
    return divs


# ---------------------------------------------------------------------------
# Símbolo de Legendre y símbolos de Hilbert (Serre, Thm. III.1)
# ---------------------------------------------------------------------------

def legendre(a, p):
    """Símbolo de Legendre (a/p) para p primo impar y p ∤ a."""
    r = a % p
    if r == 0:
        raise ValueError(f"p={p} divide a a={a}")
    v = pow(r, (p - 1) // 2, p)
    return 1 if v == 1 else -1


def vp_int(x, p):
    """Valuación p-ádica de un entero no nulo x y su parte prima a p."""
    c = 0
    while x % p == 0:
        x //= p
        c += 1
    return c, x


def hilbert_odd(a: Fraction, b: Fraction, p: int) -> int:
    """
    Símbolo de Hilbert (a, b)_p para p primo impar (Serre, A Course in Arithmetic, Thm. III.1):
    Escribiendo a = p^alpha * u, b = p^beta * v con u, v en Z_p^*,
      (a, b)_p = (-1)^(alpha * beta * (p-1)/2) * (u/p)^beta * (v/p)^alpha.
    Nota: para r = r_num / r_den en Z_p^*, (r/p) = (r_num * r_den / p).
    """
    assert p > 2 and a != 0 and b != 0
    va_n, an = vp_int(a.numerator, p)
    va_d, ad = vp_int(a.denominator, p)
    alpha = va_n - va_d

    vb_n, bn = vp_int(b.numerator, p)
    vb_d, bd = vp_int(b.denominator, p)
    beta = vb_n - vb_d

    sign = 1
    if (alpha & 1) and (beta & 1) and ((p - 1) // 2) & 1:
        sign = -sign
    if beta & 1:
        sign *= legendre(an * ad, p)
    if alpha & 1:
        sign *= legendre(bn * bd, p)
    return sign


def hilbert_2(a: Fraction, b: Fraction) -> int:
    """
    Símbolo de Hilbert (a, b)_2 (Serre, Thm. III.1):
    a = 2^alpha * u, b = 2^beta * v con u, v en Z_2^*,
      (a, b)_2 = (-1)^(eps(u)*eps(v) + alpha*omega(v) + beta*omega(u)),
    donde eps(x) = (x - 1)/2 mod 2 y omega(x) = (x^2 - 1)/8 mod 2.
    """
    assert a != 0 and b != 0
    va_n, an = vp_int(a.numerator, 2)
    va_d, ad = vp_int(a.denominator, 2)
    alpha = va_n - va_d
    # u mod 8: como ad es impar, ad^-1 ≡ ad (mod 8) pues ad^2 ≡ 1 (mod 8)
    u = (an * ad) % 8

    vb_n, bn = vp_int(b.numerator, 2)
    vb_d, bd = vp_int(b.denominator, 2)
    beta = vb_n - vb_d
    v = (bn * bd) % 8

    eps = lambda x: ((x - 1) // 2) & 1
    omg = lambda x: ((x * x - 1) // 8) & 1
    exp = (eps(u) * eps(v) + (alpha & 1) * omg(v) + (beta & 1) * omg(u)) & 1
    return -1 if exp else 1


def hilbert_inf(a: Fraction, b: Fraction) -> int:
    """Símbolo de Hilbert real (a, b)_∞ = -1 sii a < 0 y b < 0."""
    return -1 if (a < 0 and b < 0) else 1


# ---------------------------------------------------------------------------
# Enumeración exhaustiva de soluciones positivas y búsqueda de no positivas
# ---------------------------------------------------------------------------

def all_positive_solutions(m, n, spf):
    """
    Devuelve TODAS las soluciones enteras positivas 1 <= u1 <= u2 <= u3 de
      m/n = 1/u1 + 1/u2 + 1/u3.
    """
    sols = []
    u1_min = n // m + 1
    u1_max = (3 * n) // m
    for u1 in range(u1_min, u1_max + 1):
        num = m * u1 - n
        den = n * u1
        g = gcd(num, den)
        A = num // g
        B = den // g
        fac_B = factor_int(B, spf)
        # (A*u2 - B)(A*u3 - B) = B^2, con 1 <= d <= B y d ≡ -B (mod A)
        target = (-B) % A
        for d in divisors_of_square_upto(B, fac_B):
            if d % A == target:
                u2 = (B + d) // A
                if u2 >= u1:
                    u3 = (B + (B * B) // d) // A
                    sols.append((u1, u2, u3))
    return sols


def find_nonpositive_solutions(n, spf, u1_bound=12, max_per_n=25):
    r"""
    Busca soluciones enteras NO positivas u = (u1, u2, u3) en Z^3 \ N^3 con u1*u2*u3 != 0
    para 4/n = 1/u1 + 1/u2 + 1/u3 (n impar).
    Incluye una familia explícita garantizada + búsqueda por u1 pequeño (positivo y negativo).
    """
    found = set()
    # Familia explícita garantizada para n impar >= 3:
    # - Si n ≡ 1 (mod 4): u1 = (n-1)/4 da 4/n - 4/(n-1) = -4/(n(n-1)) = 1/(-n(n-1)/2) + 1/(-n(n-1)/2)
    #   y también u1 = (n-1)/2 da 4/n - 2/(n-1) -> no; u = ((n-1)/2, (n-1)/2, -n(n-1)/4) tiene 2 pos, 1 neg!
    if n % 4 == 1:
        k = (n - 1) // 4
        found.add(tuple(sorted((k, -n * (n - 1) // 2, -n * (n - 1) // 2))))
        found.add(tuple(sorted((2 * k, 2 * k, -n * k))))
    else:
        # n ≡ 3 (mod 4): u1 = (n+1)/4 da 4/n - 4/(n+1) = 4/(n(n+1)) (positiva).
        # Para no positiva: 4/n = 1/((n-3)/4) ... probamos u1 = -n o pequeños u1 y también
        # identidad válida para todo n ≡ 3 (mod 4) >= 3:
        #   4/n = 1/((n+1)/4) + 1/(-n(n+1)/4) + ... no;
        #   4/n = 1/((n-1)/2) + ... mejor búsqueda sistemática sobre pequeños u1 + u1 cercanos a n/4!
        pass

    cand_u1 = list(range(-u1_bound, 0)) + list(range(1, u1_bound + 1))
    # Añadir valores cercanos a n/4 y -n para cubrir ambos signos (1 neg y 2 neg)
    for delta in (-3, -2, -1, 0, 1, 2, 3):
        v = n // 4 + delta
        if v != 0:
            cand_u1.append(v)
    cand_u1.extend([-n, n, -(n - 1) // 2, (n - 1) // 2, -(n + 1) // 2, (n + 1) // 2])

    for u1 in cand_u1:
        if len(found) >= max_per_n:
            break
        num = 4 * u1 - n
        if num == 0:
            continue
        den = n * u1
        g = gcd(abs(num), abs(den))
        A = num // g
        B = den // g
        if A < 0:
            A, B = -A, -B
        # (A*u2 - B)(A*u3 - B) = B^2, permitiendo divisores d de B^2 positivos y negativos
        fac_B = factor_int(abs(B), spf)
        B2 = B * B
        for d_pos in all_divisors_of_square(fac_B):
            for d in (d_pos, -d_pos):
                if (B + d) % A == 0:
                    u2 = (B + d) // A
                    u3 = (B + B2 // d) // A
                    if u2 == 0 or u3 == 0:
                        continue
                    if u1 > 0 and u2 > 0 and u3 > 0:
                        continue
                    # Verificar exactitud
                    if 4 * u1 * u2 * u3 == n * (u1 * u2 + u1 * u3 + u2 * u3):
                        found.add(tuple(sorted((u1, u2, u3))))
                        if len(found) >= max_per_n:
                            break
            if len(found) >= max_per_n:
                break
    return sorted(found)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    N_MAX = 2000
    # Máximo B posible en all_positive_solutions para n <= 2000 es n * (3n/4) <= 3,000,000
    spf = build_spf(3_100_000)
    print(f"Tabla SPF construida hasta 3,100,000 en {time.time() - t0:.2f} s")

    # =========================================================================
    # 1. Teorema 1.2 + Corolario 1.3 + Corolario 1.4 para TODO n impar <= 2000
    # =========================================================================
    total_odd = 0
    total_primes = 0
    total_composites = 0
    total_odd_squares = 0

    total_pos_sols = 0
    thm12_fails = 0
    s3_inv_fails = 0
    lem38_fails = 0
    lem35_fails = 0
    hilbert_prod_fails = 0

    cor13_sols = 0
    cor13_fails = 0
    t1c_type1_checked = 0
    t1c_type1_fails = 0
    t1c_type2_checked = 0
    t1c_type2_fails = 0

    cor14_squares_checked = 0
    cor14_sols = 0
    cor14_type1_or_2_found = 0
    square_examples = []

    for n in range(3, N_MAX + 1, 2):
        total_odd += 1
        fac_n = factor_int(n, spf)
        primes_n = sorted(fac_n.keys())
        is_prime_n = (len(primes_n) == 1 and fac_n[primes_n[0]] == 1)
        is_square_n = all(e % 2 == 0 for e in fac_n.values())

        if is_prime_n:
            total_primes += 1
        else:
            total_composites += 1
        if is_square_n:
            total_odd_squares += 1
            cor14_squares_checked += 1

        sols = all_positive_solutions(4, n, spf)
        for u1, u2, u3 in sols:
            assert es_solucion(n, u1, u2, u3), f"Solución inválida: n={n}, {(u1, u2, u3)}"
            total_pos_sols += 1

            a = Fraction(-u1, u3)
            b = Fraction(-u2, u3)

            # Producto sobre p | n (Thm. 1.2)
            prod_n = 1
            for p in primes_n:
                h_p = hilbert_odd(a, b, p)
                prod_n *= h_p
                # Comprobar invarianza S_3 (las 3 elecciones de denominador)
                h_p2 = hilbert_odd(Fraction(-u1, u2), Fraction(-u3, u2), p)
                h_p3 = hilbert_odd(Fraction(-u2, u1), Fraction(-u3, u1), p)
                if not (h_p == h_p2 == h_p3):
                    s3_inv_fails += 1
            if prod_n != -1:
                thm12_fails += 1

            # Lema 3.8 (p = 2)
            h2 = hilbert_2(a, b)
            if h2 != 1:
                lem38_fails += 1

            # Lema 3.5 (primos buenos q | u1*u2*u3 con q ∤ 2n) y reciprocidad global de Hilbert
            prod_good = 1
            bad_set = set(primes_n) | {2}
            extra_primes = set()
            for u in (u1, u2, u3):
                for q in factor_int(u, spf):
                    if q not in bad_set:
                        extra_primes.add(q)
            for q in extra_primes:
                hq = hilbert_odd(a, b, q)
                if hq != 1:
                    lem35_fails += 1
                prod_good *= hq
            if hilbert_inf(a, b) * h2 * prod_n * prod_good != 1:
                hilbert_prod_fails += 1

            # Corolario 1.3 para primos impares
            if is_prime_n:
                p = n
                cor13_sols += 1
                vs = [vp_int(u, p)[0] for u in (u1, u2, u3)]
                us = (u1, u2, u3)
                # Buscar par (i, j) con v_p(u_i) == v_p(u_j)
                found_pair = False
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    if vs[i] == vs[j]:
                        found_pair = True
                        r = Fraction(-us[i], us[j])
                        # r está en Z_p^*
                        val = legendre(r.numerator * r.denominator, p)
                        if val != -1:
                            cor13_fails += 1
                if not found_pair:
                    cor13_fails += 1

                # Verificación directa de las fórmulas de para p ≡ 1 (mod 4) y x = u1 < p
                if p % 4 == 1 and u1 < p:
                    c = 4 * u1 - p
                    assert c % 4 == 3 and (p + c) // 4 == u1
                    if u2 % p == 0 and u3 % p == 0:
                        # Tipo II en x = K_c: u2 = p*x*k/d1, u3 = p*x*k/d2 => u2/u3 = d2/d1
                        g23 = gcd(u2, u3)
                        d2_red = u2 // g23
                        d1_red = u3 // g23
                        t1c_type2_checked += 1
                        if legendre(d1_red * d2_red, p) != -1:
                            t1c_type2_fails += 1
                    else:
                        # Tipo I en x = K_c: p ∤ u1 y p divide a exactamente uno de u2, u3
                        t1c_type1_checked += 1
                        y_cop = u2 if u2 % p != 0 else u3
                        z_mul = u3 if u2 % p != 0 else u2
                        assert z_mul % p == 0 and y_cop % p != 0
                        # u1/y_cop ≡ c * d1 / d2 (mod p), y (-u1/y_cop | p) = -1
                        # Como y_cop = x*k/d1 y z_mul = p*x*k/d2 => (z_mul/p)/y_cop = d1/d2
                        zp = z_mul // p
                        g_yz = gcd(zp, y_cop)
                        d1_r = zp // g_yz
                        d2_r = y_cop // g_yz
                        lhs = legendre(-u1 * y_cop, p)
                        rhs = legendre(c * d1_r * d2_r, p)
                        if not (lhs == -1 and rhs == -1):
                            t1c_type1_fails += 1

            # Corolario 1.4 para cuadrados impares
            if is_square_n:
                cor14_sols += 1
                # Tipo I (n | u_k, gcd(n, prod otros) = 1) o Tipo II (gcd(n, u_k)=1, n | otros dos)
                div_flags = [u % n == 0 for u in (u1, u2, u3)]
                cop_flags = [gcd(u, n) == 1 for u in (u1, u2, u3)]
                is_t1 = (sum(div_flags) == 1 and sum(cop_flags) == 2)
                is_t2 = (sum(div_flags) == 2 and sum(cop_flags) == 1)
                if is_t1 or is_t2:
                    cor14_type1_or_2_found += 1
                if n in (9, 25, 49) and len(square_examples) < 6:
                    p0 = primes_n[0]
                    v_tuple = tuple(vp_int(u, p0)[0] for u in (u1, u2, u3))
                    square_examples.append((n, (u1, u2, u3), v_tuple))

    print("\n=== 1. TEOREMA 1.2 (todas las soluciones positivas, n impar, 3 <= n <= 2000) ===")
    print(f"  n impares analizados : {total_odd} ({total_primes} primos, {total_composites} compuestos, {total_odd_squares} cuadrados impares)")
    print(f"  Soluciones positivas totales (u1 <= u2 <= u3): {total_pos_sols:,}")
    print(f"  Fallos Thm. 1.2 (prod_{{p|n}} != -1)         : {thm12_fails}")
    print(f"  Fallos invarianza S_3                        : {s3_inv_fails}")
    print(f"  Fallos Lema 3.8 (inv_2 != +1)                : {lem38_fails}")
    print(f"  Fallos Lema 3.5 (inv_q != +1, q ∤ 2n)        : {lem35_fails}")
    print(f"  Fallos fórmula del producto de Hilbert       : {hilbert_prod_fails}")

    print("\n=== 2. COROLARIO 1.3 (todos los primos impares p <= 2000) y CONEXIÓN ===")
    print(f"  Primos impares <= 2000                       : {total_primes}")
    print(f"  Soluciones positivas en primos               : {cor13_sols:,}")
    print(f"  Fallos Cor. 1.3 ((-u_i/u_j | p) != -1)       : {cor13_fails}")
    print(f"  Tipo II en x=K_c (p≡1 mod 4): {t1c_type2_checked:,} comprobadas, fallos = {t1c_type2_fails}")
    print(f"  Tipo I  en x=K_c (p≡1 mod 4): {t1c_type1_checked:,} comprobadas, fallos = {t1c_type1_fails}")

    print("\n=== 3. COROLARIO 1.4 (todos los cuadrados impares n <= 2000) ===")
    print(f"  Cuadrados impares <= 2000                    : {cor14_squares_checked}")
    print(f"  Soluciones positivas totales en cuadrados    : {cor14_sols:,}")
    print(f"  Soluciones de Tipo I o Tipo II encontradas   : {cor14_type1_or_2_found}")
    print("  Ejemplos de soluciones con valuación intermedia en cuadrados:")
    for n_sq, sol, vt in square_examples:
        print(f"    n={n_sq}: u={sol}, valuaciones v_p={vt}")

    # =========================================================================
    # 2. Teorema 1.5 (soluciones enteras NO positivas)
    # =========================================================================
    print("\n=== 4. TEOREMA 1.5 (soluciones enteras NO positivas, n impar <= 2000) ===")
    # Primero los ejemplos explícitos de Bright-Loughran (p. 2)
    bl_examples = [
        (9, (-18, 4, 4)),
        (9, (-9, 2, 18)),
        (5, (-5, 2, 2)),
    ]
    for n_ex, (u1, u2, u3) in bl_examples:
        assert 4 * u1 * u2 * u3 == n_ex * (u1 * u2 + u1 * u3 + u2 * u3)
        a, b = Fraction(-u1, u3), Fraction(-u2, u3)
        primes_n = sorted(factor_int(n_ex, spf).keys())
        prod_n = 1
        for p in primes_n:
            prod_n *= hilbert_odd(a, b, p)
        h_inf = hilbert_inf(a, b)
        h_2 = hilbert_2(a, b)
        print(f"  Ejemplo B-L: n={n_ex}, u={(u1, u2, u3)} -> inv_∞={h_inf:+d}, inv_2={h_2:+d}, prod_{{p|n}}={prod_n:+d}")
        assert prod_n == 1 and h_inf == 1 and h_2 == 1

    nonpos_total = 0
    nonpos_1neg = 0
    nonpos_2neg = 0
    nonpos_covered_n = 0
    thm15_fails = 0
    thm15_inf_fails = 0
    thm15_h2_fails = 0
    sample_nonpos = []

    for n in range(3, N_MAX + 1, 2):
        sols_np = find_nonpositive_solutions(n, spf, u1_bound=12, max_per_n=15)
        if sols_np:
            nonpos_covered_n += 1
        primes_n = sorted(factor_int(n, spf).keys())
        for u1, u2, u3 in sols_np:
            nonpos_total += 1
            n_neg = sum(1 for u in (u1, u2, u3) if u < 0)
            if n_neg == 1:
                nonpos_1neg += 1
            elif n_neg == 2:
                nonpos_2neg += 1
            a, b = Fraction(-u1, u3), Fraction(-u2, u3)
            prod_n = 1
            for p in primes_n:
                prod_n *= hilbert_odd(a, b, p)
            if prod_n != 1:
                thm15_fails += 1
            if hilbert_inf(a, b) != 1:
                thm15_inf_fails += 1
            if hilbert_2(a, b) != 1:
                thm15_h2_fails += 1
            if n in (7, 15, 25, 193, 241) and len([x for x in sample_nonpos if x[0] == n]) < 2:
                sample_nonpos.append((n, (u1, u2, u3), prod_n))

    print(f"  n impares <= 2000 con soluciones no positivas halladas: {nonpos_covered_n}/{total_odd}")
    print(f"  Total soluciones enteras no positivas verificadas     : {nonpos_total:,} (1 neg: {nonpos_1neg:,}, 2 neg: {nonpos_2neg:,})")
    print(f"  Fallos Thm. 1.5 (prod_{{p|n}} != +1)                  : {thm15_fails}")
    print(f"  Fallos inv_∞ != +1                                    : {thm15_inf_fails}")
    print(f"  Fallos inv_2 != +1                                    : {thm15_h2_fails}")
    print("  Muestra de soluciones no positivas (primos y compuestos):")
    for n_s, u_s, pr in sample_nonpos:
        print(f"    n={n_s:>3}, u={str(u_s):<26} -> prod_{{p|n}} = {pr:+d}")

    # =========================================================================
    # 3. Control U_{8, 241} (m=8, n=241) — sin obstrucción BM pero sin U(N)
    # =========================================================================
    print("\n=== 5. CONTROL SUPERFICIE U_{8, 241} (8/241 = 1/u1 + 1/u2 + 1/u3) ===")
    sols_8_241 = all_positive_solutions(8, 241, spf)
    print(f"  Soluciones positivas de 8/241 (búsqueda exhaustiva): {len(sols_8_241)}")
    assert len(sols_8_241) == 0

    # Punto entero no positivo en U_{8, 241}(Z): (30, -14460, -14460)
    u_z = (30, -14460, -14460)
    assert 8 * u_z[0] * u_z[1] * u_z[2] == 241 * (u_z[0]*u_z[1] + u_z[0]*u_z[2] + u_z[1]*u_z[2])
    az, bz = Fraction(-u_z[0], u_z[2]), Fraction(-u_z[1], u_z[2])
    h_inf_z = hilbert_inf(az, bz)
    h_2_z = hilbert_2(az, bz)
    h_241_z = hilbert_odd(az, bz, 241)
    # Comprobar todos los primos divisores de 30 y 14460 (2, 3, 5, 241)
    other_ps = [3, 5]
    h_other = {q: hilbert_odd(az, bz, q) for q in other_ps}
    print(f"  Punto en U_{{8,241}}(Z): u={u_z}")
    print(f"    inv_∞={h_inf_z:+d}, inv_2={h_2_z:+d}, inv_241={h_241_z:+d}, inv_3={h_other[3]:+d}, inv_5={h_other[5]:+d}")
    assert h_inf_z * h_2_z * h_241_z * h_other[3] * h_other[5] == 1

    # Comprobar que en Z_{241} la elevación de Hensel con u1 = 241*a1, 8*a1 ≡ 1 (mod 241)
    # y u2, u3 ≢ 0 (mod 241) realiza ambos signos de inv_{241} = (-u2/u3 | 241)^1:
    a1_mod = pow(8, -1, 241)  # 8*a1 ≡ 1 mod 241
    assert (8 * a1_mod - 1) % 241 == 0
    # Para (u2, u3) = (1, 1): (-1/241) = +1 (pues 241 ≡ 1 mod 4)
    # Para (u2, u3) = (7, 1): (-7/241) = (7/241) = (241/7) = (3/7) = -1
    assert legendre(-1, 241) == 1 and legendre(-7, 241) == -1
    print(f"  Hensel en p=241 (a1 ≡ {a1_mod} mod 241): (-1/241)={legendre(-1, 241):+d}, (-7/241)={legendre(-7, 241):+d}")
    print("  => (U_{8,241}(R)_+ x prod_p U_{8,241}(Z_p))^Br != vacío, pero U_{8,241}(N) = vacío.")

    print(f"\nTiempo total: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
