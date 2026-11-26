#!/usr/bin/env python3
"""
Comprobación independiente (sympy) de los datos hasta 10^8: m(p), cortes con a = 1 de tipo I
y de tipo II (Aigner y Dahan) y criterios locales en c = 3 y c = 7. Compara fila a fila con
datos/clases_dificiles_1e8.csv (implementación Numba).
"""
import csv
import os
import sys
import time
from collections import Counter
from math import isqrt, log, sqrt
from multiprocessing import Pool

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
from verificar import perfil, tipos_en, es_solucion  # noqa: E402
from sympy import divisors, legendre_symbol, primerange, factorint  # noqa: E402

DIF = {1, 121, 169, 289, 361, 529}
CAP_E = 6000          # tope de búsqueda para II1A antes de marcar "pendiente"
Q34 = [q for q in primerange(3, 20000) if q % 4 == 3]


def e_I1(p):
    for e in range(3, (p + 4) // 3 + 1, 4):
        M = (e * p + 1) // 4
        if any(B % e == e - 1 for B in divisors(M)):
            return e
    return None


def e_II1A(p, cap=CAP_E):
    top = min((p + 4) // 3, cap)
    for e in range(3, top + 1, 4):
        K = (p + e) // 4
        if any(b % e == e - 1 for b in divisors(K)):
            return e
    return None if top == (p + 4) // 3 else "pendiente"


def u_II1D(p):
    for u in range(1, (1 + isqrt(4 * p + 1)) // 4 + 2):
        if any(s % (4 * u) == 4 * u - 1 for s in divisors(p + u)):
            return u
    return None


def k_q34(p):
    k = 0
    for q in Q34:
        if legendre_symbol(p % q, q) == -1:
            k += 1
            if any(tipos_en(p, q)):
                return k, q
    return None, None


def fila(p):
    m, _, c1, c2 = perfil(p)
    k, q = k_q34(p)
    return (p, m, c1, c2, e_I1(p), u_II1D(p), e_II1A(p), k, q)


def primos_dificiles(N):
    s = np.ones(N // 2 + 1, dtype=bool)     # s[i] <-> 2i+1
    s[0] = False
    for i in range(1, (isqrt(N) - 1) // 2 + 1):
        if s[i]:
            p = 2 * i + 1
            s[p * p // 2::p] = False
    ps = 2 * np.nonzero(s)[0] + 1
    ps = ps[ps <= N]
    return [int(p) for p in ps if int(p) % 840 in DIF]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(RAIZ)
    t0 = time.time()

    print("=== cortes de tipo II con a=1 sobre los 14 wild y sobre 2521, 66529 ===")
    wild = [409, 577, 5569, 9601, 23929, 83449, 102001, 329617, 712321,
            1134241, 1724209, 1726201, 5212561, 8813281]
    for p in wild + [2521, 66529]:
        print(f"  p={p:>8}: II1D (Dahan Alg.2) u={u_II1D(p)},  II1A (Aigner a_ET=1) e={e_II1A(p, cap=10**9)}")

    print("\n=== criterios locales en c=3 y c=7 (primos difíciles <= 10^7) ===")
    P7 = primos_dificiles(10 ** 7)
    a3 = b3 = a7 = b7 = d3 = d7 = 0
    for p in P7:
        t3 = any(tipos_en(p, 3))
        k3 = any(q % 3 == 2 for q in factorint((p + 3) // 4))
        t1, t2 = tipos_en(p, 7)
        k7 = any(q % 7 in (3, 5, 6) for q in factorint((p + 7) // 4))
        a3 += t3; b3 += k3; d3 += t3 != k3
        a7 += t1 and t2; b7 += k7; d7 += (t1 != k7) + (t2 != k7)
    print(f"  {len(P7)} primos. c=3: tipos={a3}, criterio={b3}, discrepancias={d3}")
    print(f"  c=7: ambos tipos={a7}, criterio={b7}, discrepancias={d7}")

    print("\n=== recálculo completo hasta 10^8 (8 procesos) ===")
    P = primos_dificiles(10 ** 8)
    print(f"  {len(P)} primos difíciles <= 10^8 ({time.time() - t0:.0f} s)")
    with Pool(8) as pool:
        filas = pool.map(fila, P, chunksize=400)
    print(f"  cálculo terminado ({time.time() - t0:.0f} s)")
    mx = max(filas, key=lambda f: f[1])
    alto = [f for f in filas if f[0] > 10 ** 7]
    mx2 = max(alto, key=lambda f: f[1])
    print(f"  max m = {mx[1]} en p = {mx[0]};  max m en (10^7,10^8] = {mx2[1]} en p = {mx2[0]}")
    print(f"  primos con m >= 59 en (10^7,10^8]: {sorted((f[1], f[0]) for f in alto if f[1] >= 59)}")
    for nombre, idx in [("I1", 4), ("II1D", 5), ("II1A", 6)]:
        sin = [f[0] for f in filas if f[idx] is None]
        pend = [f[0] for f in filas if f[idx] == "pendiente"]
        fin = [f for f in filas if isinstance(f[idx], int)]
        top = max(fin, key=lambda f: f[idx])
        print(f"  {nombre}: sin solución = {sin}; pendientes = {pend}; max = {top[idx]} en p = {top[0]}")
    print(f"  sin I1 ni II1D: {[f[0] for f in filas if f[4] is None and f[5] is None]}")
    kq = Counter(f[7] for f in filas)
    print(f"  k_q34: {sorted(kq.items(), key=lambda kv: (kv[0] is None, kv[0]))}")
    print(f"  k_q34 máximo en: {[(f[0], f[8], f[1]) for f in filas if f[7] == max(k for k in kq if k)]}")

    print("\n  F_3 * sqrt(log x_mid) por décadas:")
    for a in range(3, 8):
        sub = [f for f in filas if 10 ** a < f[0] <= 10 ** (a + 1)]
        F3 = sum(1 for f in sub if f[1] > 3) / len(sub)
        print(f"    (10^{a}, 10^{a + 1}]: n={len(sub):>7}  F_3={F3:.4f}  F_3*sqrt(log x_mid)={F3 * sqrt(log(10 ** (a + 0.5))):.4f}")

    print("\n=== Comparación fila a fila con datos/clases_dificiles_1e8.csv ===")
    with open("datos/clases_dificiles_1e8.csv", encoding="utf-8") as fh:
        G = {int(r["p"]): r for r in csv.DictReader(fh)}
    print(f"  mismas p: {set(G) == set(P)} ({len(G)} vs {len(P)})")
    cols = ["m", "min_c_I", "min_c_II", "e_min_I1", "u_min_II1", "e_min_II_aET1", "k_q34", "q_hit"]
    dif = Counter()
    ejemplos = {}
    for f in filas:
        g = G.get(f[0])
        if g is None:
            continue
        for c, v in zip(cols, f[1:]):
            gv = g[c]
            ok = (str(v) == gv) or (v is None and gv in ("", "None", "-1", "inf", "nan"))
            if not ok:
                dif[c] += 1
                ejemplos.setdefault(c, (f[0], v, gv))
    print(f"  discrepancias por columna: {dict(dif) or 'ninguna'}")
    for c, e in ejemplos.items():
        print(f"    ejemplo {c}: p={e[0]} aquí={e[1]} CSV={e[2]!r}")
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
