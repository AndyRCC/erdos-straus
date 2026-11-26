#!/usr/bin/env python3
"""
Número exacto de soluciones f_m(p) (ternas ordenadas) para todos los primos p <= 2·10^5 y
m = 4, 5, 6, 7, 8 (nota, §7). Guarda datos/fm_primos_2e5.csv.
"""

import csv
import math
import os
import sys
import time
from math import gcd

import numpy as np
from numba import njit, prange

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bright_loughran import build_spf, all_positive_solutions  # noqa: E402


# ---------------------------------------------------------------------------
# Aritmética entera rápida en Numba (64 bits; para M <= 10^11 usamos __int128-safe mulmod)
# ---------------------------------------------------------------------------

@njit(inline="always")
def nb_gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(inline="always")
def mulmod(a, b, mod):
    # Seguro para mod < 2^62 sin desbordar uint64 usando descomposición en 30 bits
    a %= mod
    b %= mod
    if a < (1 << 31) and b < (1 << 31):
        return (a * b) % mod
    res = 0
    while b > 0:
        if b & 1:
            res += a
            if res >= mod:
                res -= mod
        a <<= 1
        if a >= mod:
            a -= mod
        b >>= 1
    return res


@njit(inline="always")
def powmod(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            res = mulmod(res, base, mod)
        base = mulmod(base, base, mod)
        exp >>= 1
    return res


@njit
def is_prime_64(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
    # Determinista para todo n < 3.317 * 10^24 con bases <= 37 (aquí n < 10^12 basta {2, 3, 5, 7, 11, 13})
    for a in (2, 3, 5, 7, 11, 13):
        if a >= n:
            continue
        x = powmod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        comp = True
        for _ in range(s - 1):
            x = mulmod(x, x, n)
            if x == n - 1:
                comp = False
                break
        if comp:
            return False
    return True


@njit
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    for c in range(1, 20):
        x = 2
        y = 2
        d = 1
        while d == 1:
            x = (mulmod(x, x, n) + c) % n
            y = (mulmod(y, y, n) + c) % n
            y = (mulmod(y, y, n) + c) % n
            diff = x - y if x >= y else y - x
            d = nb_gcd(diff, n)
        if d != n:
            return d
    # Fallback trial division (no debería ocurrir)
    d = 3
    while d * d <= n:
        if n % d == 0:
            return d
        d += 2
    return n


@njit
def factor_64(n, spf, spf_lim, primes_out, exps_out):
    """Factoriza n >= 1 en primes_out[:k], exps_out[:k]. Devuelve k."""
    k = 0
    if n <= 1:
        return 0
    # Si cabe en SPF, instantáneo
    if n <= spf_lim:
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            primes_out[k] = p
            exps_out[k] = e
            k += 1
        return k

    # Extraer pequeños factores
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            primes_out[k] = p
            exps_out[k] = e
            k += 1
            if n <= spf_lim:
                break

    if n <= 1:
        return k
    if n <= spf_lim:
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            primes_out[k] = p
            exps_out[k] = e
            k += 1
        return k

    # Pila para factores restantes via Pollard-rho
    stack = np.empty(64, dtype=np.int64)
    top = 0
    stack[top] = n
    top += 1

    raw = np.empty(64, dtype=np.int64)
    nraw = 0
    while top > 0:
        top -= 1
        cur = stack[top]
        if cur == 1:
            continue
        if cur <= spf_lim:
            while cur > 1:
                p = spf[cur]
                raw[nraw] = p
                nraw += 1
                cur //= p
        elif is_prime_64(cur):
            raw[nraw] = cur
            nraw += 1
        else:
            d = pollard_rho(cur)
            stack[top] = d
            top += 1
            stack[top] = cur // d
            top += 1

    # Agrupar factores primos de raw[:nraw] (insertion sort)
    for i in range(1, nraw):
        v = raw[i]
        j = i - 1
        while j >= 0 and raw[j] > v:
            raw[j + 1] = raw[j]
            j -= 1
        raw[j + 1] = v

    i = 0
    while i < nraw:
        p = raw[i]
        e = 1
        i += 1
        while i < nraw and raw[i] == p:
            e += 1
            i += 1
        primes_out[k] = p
        exps_out[k] = e
        k += 1
    return k


@njit
def build_spf_nb(limit):
    spf = np.arange(limit + 1, dtype=np.int32)
    lim_s = int(limit ** 0.5) + 1
    for i in range(2, lim_s):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


@njit
def sieve_primes_bool(limit):
    is_p = np.ones(limit + 1, dtype=np.bool_)
    is_p[0] = False
    is_p[1] = False
    lim_s = int(limit ** 0.5) + 1
    for i in range(2, lim_s):
        if is_p[i]:
            for j in range(i * i, limit + 1, i):
                is_p[j] = False
    return is_p


@njit(parallel=True)
def count_type2_all_primes(m, N, is_p, spf):
    """
    Cuenta 3 * f_{m, II}(p) (ternas ordenadas) para todos los primos m < p <= N.
    Ecuación: m*a*b*d = p + e, c*e = a + b, 1 <= a <= b, gcd(a, b) = 1, a*b*d <= N // (m - 2).
    """
    X2 = N // (m - 2)
    a_max = int(X2 ** 0.5)
    # Particionar por a en paralelo con acumuladores por bloque o atómico-libre (acumular por hilo)
    n_threads = 8
    local_counts = np.zeros((n_threads, N + 1), dtype=np.int32)
    spf_lim = len(spf) - 1

    for tid in prange(n_threads):
        pr = np.empty(32, dtype=np.int64)
        ex = np.empty(32, dtype=np.int64)
        divs = np.empty(2048, dtype=np.int64)
        for a in range(1 + tid, a_max + 1, n_threads):
            b_max = X2 // a
            for b in range(a, b_max + 1):
                if nb_gcd(a, b) != 1:
                    continue
                s = a + b
                # Factorizar s y generar todos los divisores e | s
                nf = factor_64(s, spf, spf_lim, pr, ex)
                nd = 1
                divs[0] = 1
                for fi in range(nf):
                    pf = pr[fi]
                    ef = ex[fi]
                    old_nd = nd
                    pe = 1
                    for _ in range(ef):
                        pe *= pf
                        for idx in range(old_nd):
                            divs[nd] = divs[idx] * pe
                            nd += 1
                ab = a * b
                d_max = X2 // ab
                weight = 3 if a == b else 6
                for d in range(1, d_max + 1):
                    mabd = m * ab * d
                    for di in range(nd):
                        e = divs[di]
                        p = mabd - e
                        if m < p <= N and is_p[p]:
                            c = s // e
                            if a % p != 0 and b % p != 0 and c % p != 0 and d % p != 0:
                                local_counts[tid, p] += weight

    counts = np.zeros(N + 1, dtype=np.int32)
    for p in range(m + 1, N + 1):
        if is_p[p]:
            tot = 0
            for tid in range(n_threads):
                tot += local_counts[tid, p]
            counts[p] = tot
    return counts


@njit(parallel=True)
def count_type1_all_primes(m, N, is_p, spf):
    """
    Cuenta 3 * f_{m, I}(p) (ternas ordenadas) para todos los primos m < p <= N.
    Ecuación: a*c*d <= 2N // (m - 1), f | (m*a^2*d + 1), e = (m*a^2*d + 1)/f,
    b = c*e - a >= a, gcd(a, c) = gcd(a, b) = 1, p = m*a*c*d - f.
    """
    X1 = (2 * N) // (m - 1)
    n_threads = 8
    local_counts = np.zeros((n_threads, N + 1), dtype=np.int32)
    spf_lim = len(spf) - 1

    for tid in prange(n_threads):
        pr = np.empty(32, dtype=np.int64)
        ex = np.empty(32, dtype=np.int64)
        divs = np.empty(4096, dtype=np.int64)
        # Reparto cíclico sobre a para balancear carga (a grande tiene pocos d pero M más grande)
        for a in range(1 + tid, X1 + 1, n_threads):
            d_max = X1 // a
            for d in range(1, d_max + 1):
                ad = a * d
                c_max = X1 // ad
                f_upper = m * ad * c_max - (m + 1)
                if f_upper < 1:
                    continue
                M = m * a * ad + 1
                nf = factor_64(M, spf, spf_lim, pr, ex)
                # Generar divisores f | M con f <= f_upper
                nd = 1
                divs[0] = 1
                for fi in range(nf):
                    pf = pr[fi]
                    ef = ex[fi]
                    old_nd = nd
                    pe = 1
                    for _ in range(ef):
                        pe *= pf
                        if pe > f_upper:
                            break
                        for idx in range(old_nd):
                            v = divs[idx] * pe
                            if v <= f_upper:
                                divs[nd] = v
                                nd += 1

                mad = m * ad
                two_a = 2 * a
                for di in range(nd):
                    f = divs[di]
                    e = M // f
                    # Necesitamos c*e >= 2*a y m*a*d*c - f > m y m*a*d*c - f <= N
                    c_lo1 = (two_a + e - 1) // e
                    c_lo2 = (f + m + 1 + mad - 1) // mad
                    c_lo = c_lo1 if c_lo1 > c_lo2 else c_lo2
                    if c_lo < 1:
                        c_lo = 1
                    c_hi1 = (N + f) // mad
                    c_hi = c_hi1 if c_hi1 < c_max else c_max
                    if c_lo > c_hi:
                        continue
                    for c in range(c_lo, c_hi + 1):
                        if nb_gcd(a, c) != 1:
                            continue
                        p = mad * c - f
                        if m < p <= N and is_p[p]:
                            b = c * e - a
                            if b >= a and nb_gcd(a, b) == 1:
                                if a % p != 0 and b % p != 0 and c % p != 0 and d % p != 0:
                                    local_counts[tid, p] += 3 if a == b else 6

    counts = np.zeros(N + 1, dtype=np.int32)
    for p in range(m + 1, N + 1):
        if is_p[p]:
            tot = 0
            for tid in range(n_threads):
                tot += local_counts[tid, p]
            counts[p] = tot
    return counts


def brute_ordered_fm(m, p, spf_py):
    """Cuenta ternas ORDENADAS (x, y, z) en N^3 con m/p = 1/x + 1/y + 1/z por fuerza bruta."""
    sols = all_positive_solutions(m, p, spf_py)
    tot = 0
    for u1, u2, u3 in sols:
        if u1 == u2 == u3:
            tot += 1
        elif u1 == u2 or u2 == u3:
            tot += 3
        else:
            tot += 6
    return tot


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(RAIZ)
    t0 = time.time()

    N = 200_000
    VAL_N = 5_000
    MS = (4, 5, 6, 7, 8)

    spf_nb = build_spf_nb(1_000_000)
    is_p = sieve_primes_bool(N)
    primes = [int(p) for p in np.nonzero(is_p)[0]]
    print(f"Primos <= {N:,}: {len(primes):,} (SPF y criba en {time.time() - t0:.2f} s)")

    # 1. Calcular exacto con Numba para N = 200,000 y m = 4..8
    spf_py_small = build_spf(200_000)
    results = {}
    for m in MS:
        tm = time.time()
        c1 = count_type1_all_primes(m, N, is_p, spf_nb)
        c2 = count_type2_all_primes(m, N, is_p, spf_nb)
        tot = c1 + c2
        # Para p <= m (p in {2, 3, 5, 7}), usar fuerza bruta exacta
        for p in (2, 3, 5, 7):
            if p <= m:
                tot[p] = brute_ordered_fm(m, p, spf_py_small)
                c1[p] = -1
                c2[p] = -1
        results[m] = (tot, c1, c2)
        excs = [p for p in primes if tot[p] == 0]
        print(f"  m={m}: calculado en {time.time() - tm:.2f} s | "
              f"sum f_m(p)={int(tot[primes].sum()):,} | excepciones primas <= {N:,}: {excs}")

    # 2. Validación exhaustiva contra fuerza bruta en TODOS los primos p <= 5,000
    t_val = time.time()
    # Máximo B en all_positive_solutions para p <= 5000, m >= 4: 5000 * 3750 = 18.75M -> usar spf de 5000*4000?
    # Mejor: con factor_int de t1_bright_loughran (spf=200k tiene fallback) o spf=5M (en 1 s)
    spf_val = build_spf(5_000_000)
    val_primes = [p for p in primes if p <= VAL_N]
    disc = 0
    for m in MS:
        tot = results[m][0]
        for p in val_primes:
            bf = brute_ordered_fm(m, p, spf_val)
            if int(tot[p]) != bf:
                print(f"  DISCREPANCIA m={m}, p={p}: numba={int(tot[p])}, bf={bf}")
                disc += 1
                if disc >= 10:
                    break
    # Comprobar también los valores explícitos de Elsholtz-Tao p. 2: f_4(2)=3, f_4(3)=12, f_4(5)=12, f_4(7)=36
    et_check = {p: int(results[4][0][p]) for p in (2, 3, 5, 7)}
    print(f"\nValidación contra fuerza bruta (todos los {len(val_primes)} primos <= {VAL_N:,}, m=4..8, "
          f"{len(val_primes)*len(MS)} pares): discrepancias = {disc} ({time.time() - t_val:.2f} s)")
    print(f"Control Elsholtz-Tao p. 2: f_4(2,3,5,7) = {et_check} (esperado {{2: 3, 3: 12, 5: 12, 7: 36}})")
    assert disc == 0 and et_check == {2: 3, 3: 12, 5: 12, 7: 36}

    # 3. Guardar CSV en datos/fm_primos_2e5.csv
    out_csv = os.path.join(RAIZ, "datos", "fm_primos_2e5.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p", "f4", "f4_I", "f4_II", "f5", "f6", "f7", "f8"])
        for p in primes:
            w.writerow([
                p,
                int(results[4][0][p]), int(results[4][1][p]), int(results[4][2][p]),
                int(results[5][0][p]), int(results[6][0][p]),
                int(results[7][0][p]), int(results[8][0][p]),
            ])
    print(f"Guardado: {out_csv} ({os.path.getsize(out_csv):,} bytes)")

    # 4. Resumen estadístico rápido por m (primos p <= 200,000, p > m)
    print("\n=== RESUMEN ESTADÍSTICO f_m(p) / (log p)^3 (primos 100 <= p <= 200,000) ===")
    p_arr = np.array([p for p in primes if p >= 100], dtype=np.float64)
    l3 = np.log(p_arr) ** 3
    p_idx = p_arr.astype(np.int64)
    for m in MS:
        vals = results[m][0][p_idx].astype(np.float64)
        ratio = vals / l3
        # Separar p ≡ 1 (mod 24) vs todos
        m24 = (p_idx % 24 == 1)
        print(f"  m={m}: media f_m/(log p)^3 = {ratio.mean():.5f} (mediana {np.median(ratio):.5f}, "
              f"min f_m={int(vals.min())}, max={int(vals.max())}) | "
              f"en p≡1 mod 24: media = {ratio[m24].mean():.5f} (min f_m={int(vals[m24].min())})")

    print(f"\nTiempo total: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
