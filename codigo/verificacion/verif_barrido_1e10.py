#!/usr/bin/env python3
"""
Comprobación independiente (criba numpy + sympy) de m(p) para todos los primos difíciles
hasta 10^10 (nota, §4).
"""
import os
import sys
import time
from collections import Counter
from math import isqrt, log, sqrt
from multiprocessing import Pool

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
from verificar import tipos_en  # noqa: E402
from sympy import primerange  # noqa: E402

N = 10 ** 10
SEG = 2 * 10 ** 8
DIF = np.array([1, 121, 169, 289, 361, 529])
BASE = np.array(list(primerange(3, isqrt(N) + 1)), dtype=np.int64)


def m_de(p):
    m = 3
    while True:
        t1, t2 = tipos_en(p, m)
        if t1 or t2:
            return m
        m += 4


def bloque(ps):
    return [(p, m_de(p)) for p in ps]


def primos_segmento(lo, hi):
    """Primos impares en [lo, hi), lo par."""
    s = np.ones((hi - lo) // 2, dtype=bool)          # s[i] <-> lo + 2i + 1
    for q in BASE:
        q = int(q)
        if q * q >= hi:
            break
        st = max(q * q, ((lo + q - 1) // q) * q)
        if st % 2 == 0:
            st += q
        s[(st - lo - 1) // 2::q] = False
    ps = lo + 2 * np.nonzero(s)[0].astype(np.int64) + 1
    if lo == 0:
        ps = ps[ps > 1]
    return ps


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    cuenta = Counter()
    record, records, altos = 0, [], []
    por_decada = Counter()
    m3_decada = Counter()
    total = 0
    with Pool(8) as pool:
        for lo in range(0, N, SEG):
            ps = primos_segmento(lo, min(lo + SEG, N + 1))
            ps = ps[np.isin(ps % 840, DIF)]
            trozos = [list(map(int, c)) for c in np.array_split(ps, 64) if len(c)]
            for res in pool.map(bloque, trozos):
                for p, m in res:
                    total += 1
                    cuenta[m] += 1
                    k = int(log(p, 10))
                    por_decada[k] += 1
                    m3_decada[k] += m == 3
                    if m > record:
                        record = m
                        records.append((p, m))
                    if m >= 75:
                        altos.append((m, p))
            print(f"  hasta {min(lo + SEG, N):,}: {total:,} primos difíciles, récord m = {record}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"\nTOTAL primos difíciles <= 10^10: {total:,}")
    print("Récords:", records)
    print("m >= 75:", sorted(altos, reverse=True))
    print("Distribución:", sorted(cuenta.items()))
    for k in sorted(por_decada):
        if k >= 3:
            F3 = 1 - m3_decada[k] / por_decada[k]
            print(f"  [10^{k}, 10^{k + 1}): n={por_decada[k]:,}  frac(m=3)={m3_decada[k] / por_decada[k]:.4f}  F3*sqrt(log x_mid)={F3 * sqrt(log(10 ** (k + 0.5))):.4f}")
    print(f"Tiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
