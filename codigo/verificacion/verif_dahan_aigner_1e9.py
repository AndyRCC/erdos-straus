#!/usr/bin/env python3
"""
Comprobación independiente (sympy + numpy) de los resultados 1 y 2 de la nota hasta 10^9:
E_1(N;J) de Dahan y excepciones del corte de Aigner; y comprobación local exhaustiva de los
criterios de c = 7 y c = 15.
"""
import os
import sys
import time
from itertools import combinations_with_replacement
from math import gcd, isqrt, log
from multiprocessing import Pool

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
from verificar import tipos_en  # noqa: E402
from sympy import divisors, isprime, factorint, jacobi_symbol  # noqa: E402

N = 10 ** 9
JMAX = 20
CD = 1000


def primos_24(N):
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


def umin_bloque(bloque):
    return [umin(int(n)) for n in bloque]


def ii1a_exhaustivo(p):
    """¿Existe c,d,e >= 1 con e(4cd-1) = p+4d?  Exhaustivo (ramas e <= E0 y e > E0)."""
    E0 = isqrt(p) + 1
    def rama_A(e_lo, e_hi):
        for e in range(e_lo, e_hi + 1, 4):
            if any(b % e == e - 1 for b in divisors((p + e) // 4)):
                return True
        return False
    if rama_A(3, min(199, E0)):
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
    return E0 > 199 and rama_A(203, E0)


def ajuste(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (a * x + b)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2


def local_exhaustivo(c, p_res, tam_max):
    """Para p ≡ p_res (mod c), recorre todos los multiconjuntos de restos de primos de K_c
    (incluyendo al menos un 2, porque 2 | K_c) con producto ≡ K_c (mod c). Devuelve
    las discrepancias entre: tipo II, tipo I y 'algún primo con Jacobi (q/c) = -1'."""
    G = [r for r in range(1, c) if gcd(r, c) == 1]
    inv4 = pow(4, -1, c)
    K = (p_res * inv4) % c
    malos = []
    for k in range(0, tam_max):
        for resto in combinations_with_replacement(G, k):
            ms = (2,) + resto
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
                malos.append((ms, t1, t2, nr))
    return malos


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(RAIZ)
    t0 = time.time()

    # ---------------------------------------------------------------- C (rápido)
    print("=== C) Comprobación local exhaustiva (multiconjuntos de hasta 8 primos, incl. el 2) ===")
    for c, residuos in [(7, [1, 2, 4]), (15, [1, 4])]:
        for pr in residuos:
            m = local_exhaustivo(c, pr, 8)
            print(f"  c={c}, p ≡ {pr} (mod {c}): discrepancias = {len(m)}{'  ej: ' + str(m[0]) if m else ''}")
    print("  (control) c=15, p ≡ 13 (mod 15), que no es difícil: ", end="")
    m = local_exhaustivo(15, 13, 6)
    print(f"{len(m)} discrepancias (se esperan > 0: allí tipo II solo puede fallar)")
    ejemplos = [(97, 23, False), (8929, 23, True), (73, 31, False), (1801, 31, True), (73, 39, False),
                (1009, 39, True), (73, 47, False), (2521, 47, True), (97, 55, False), (1009, 55, True),
                (73, 63, False), (2521, 63, True), (97, 71, False), (1801, 71, True), (193, 79, False),
                (1009, 79, True)]
    ok = 0
    for p, c, dificil in ejemplos:
        K = (p + c) // 4
        nr = [q for q in factorint(K) if jacobi_symbol(q, c) == -1]
        bien = (isprime(p) and p % 24 == 1 and (p % 840 in {1, 121, 169, 289, 361, 529}) == dificil
                and nr and tipos_en(p, c) == (False, False))
        ok += bool(bien)
        if not bien:
            print(f"  ✗ contraejemplo NO confirmado: p={p}, c={c}")
    print(f"  contraejemplos (c ≡ 7 mod 8, c >= 23) confirmados: {ok}/{len(ejemplos)}")
    DIF = {1, 121, 169, 289, 361, 529}
    from sympy import primerange
    d15 = n15 = 0
    for p in primerange(2, 10 ** 7 + 1):
        if p % 840 in DIF:
            K = (p + 15) // 4
            nr = any(jacobi_symbol(q, 15) == -1 for q in factorint(K))
            t1, t2 = tipos_en(p, 15)
            n15 += nr
            d15 += not (t1 == t2 == nr)
    print(f"  c=15 en primos difíciles <= 10^7: con primo (q/15)=-1: {n15}, discrepancias: {d15}")

    # ---------------------------------------------------------------- primos
    P = primos_24(N)
    print(f"\nPrimos p ≡ 1 (mod 24) < 10^9: {len(P):,}  ({time.time() - t0:.0f} s)")
    print(f"  de ellos < 2·10^7: {(P < 2 * 10 ** 7).sum():,}")

    # ---------------------------------------------------------------- B
    print("\n=== B) Corte de Aigner tipo II (a_ET=1): excepciones hasta 10^9 ===")
    vivos = P.copy()
    pares = sorted(((4 * c * d - 1, c, d) for c in range(1, CD + 1) for d in range(1, CD // c + 1)))
    for s, c, d in pares:
        sol = ((vivos + 4 * d) % s == 0) & (vivos + 4 * d >= s)
        if sol.any():
            vivos = vivos[~sol]
    print(f"  supervivientes a la fase 1 (cd <= {CD}): {len(vivos):,}  ({time.time() - t0:.0f} s)")
    with Pool(8) as pool:
        res = pool.map(ii1a_exhaustivo, [int(p) for p in vivos], chunksize=20)
    exc = [int(p) for p, r in zip(vivos, res) if not r]
    print(f"  EXCEPCIONES: {exc}  ({time.time() - t0:.0f} s)")

    # ---------------------------------------------------------------- A
    print("\n=== A) Dahan E_1(N;J) hasta 10^9 ===")
    bloques = np.array_split(P, 4000)
    with Pool(8) as pool:
        U = np.concatenate([np.array(r, dtype=np.int16) for r in pool.map(umin_bloque, bloques)])
    print(f"  u_min calculado para {len(U):,} primos ({time.time() - t0:.0f} s)")
    ventanas = [(1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 3e6), (3e6, 1e7), (1e7, 2e7),
                (2e7, 5e7), (5e7, 1e8), (1e8, 3e8), (3e8, 1e9)]
    filas = []
    Js = np.arange(1, JMAX + 1)
    for A, B in ventanas:
        sel = (P >= A) & (P < B)
        u = U[sel]
        f = np.array([(u > J).mean() for J in Js])
        ok = f > 0
        beta, _, r2 = ajuste(np.log(Js[ok]), np.log(f[ok]))
        lm = 0.5 * (log(A) + log(B))
        filas.append((A, B, sel.sum(), lm, -beta, r2, f))
        print(f"  [{A:.0e},{B:.0e}): n={sel.sum():>9,}  β={-beta:.3f}  β/log n_mid={-beta / lm:.4f}  R²={r2:.4f}"
              f"{'' if ok.all() else '  (J con f=0 omitidos)'}")
    for nombre, idx in [("6 ventanas de Dahan", range(0, 6)), ("10 ventanas", range(0, 10)),
                        ("9 ventanas sin [1e3,1e4)", range(1, 10)), ("5 ventanas [1e7,1e9)", range(5, 10)),
                        ("4 ventanas nuevas [2e7,1e9)", range(6, 10))]:
        xs = [log(filas[i][3]) for i in idx]
        pend = [ajuste(xs, [log(filas[i][6][J - 1]) for i in idx])[0] for J in range(1, 7)]
        bl = ajuste([filas[i][3] for i in idx], [filas[i][4] for i in idx])
        print(f"  {nombre:<28} pendientes J=1..6: {', '.join(f'{v:+.3f}' for v in pend)}"
              f"   | β = {bl[0]:.4f} log n + {bl[1]:+.3f} (R²={bl[2]:.4f})")
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
