#!/usr/bin/env python3
"""
Comprobación independiente de la nota, §7: familia p ≡ 1 (mod L(47)) con m exacto frente a un
grupo de control, m exacto de primos adversarios y muestra de f_m(p) frente a
datos/fm_primos_2e5.csv.
"""
import csv
import os
import random
import sys
from collections import defaultdict
from math import gcd, log, prod

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
os.chdir(RAIZ)
from verificar import tipos_en, es_solucion  # noqa: E402
from sympy import divisors, isprime, legendre_symbol, primerange  # noqa: E402

DIF = (1, 121, 169, 289, 361, 529)


def m_de(p, tope=2000):
    for m in range(3, tope, 4):
        if any(tipos_en(p, m)):
            return m
    return None


def qmin(p):
    return next(q for q in primerange(2, 10 ** 6) if legendre_symbol(q % p, p) == -1)


def resumen(ms):
    ms = sorted(ms)
    n = len(ms)
    return (f"n={n} media={sum(ms) / n:.2f} mediana={ms[n // 2]} p90={ms[int(0.9 * n)]} max={ms[-1]} "
            f"P(m>3)={sum(m > 3 for m in ms) / n:.0%} P(m>11)={sum(m > 11 for m in ms) / n:.0%} "
            f"P(m>23)={sum(m > 23 for m in ms) / n:.0%}")


def f_ordenadas(m, p):
    """Número de ternas ORDENADAS (x, y, z) con m/p = 1/x + 1/y + 1/z (enumeración propia)."""
    tot = 0
    for x in range(p // m + 1, (3 * p) // m + 1):
        A, B = m * x - p, p * x
        g = gcd(A, B)
        A, B = A // g, B // g
        D = divisors(B)
        por_res = defaultdict(list)
        for d in D:
            por_res[d % A].append(d)
        for yp in D:
            for zp in por_res[(-yp) % A]:
                if zp < yp or gcd(yp, zp) != 1 or B % (yp * zp):
                    continue
                k = (yp + zp) // A
                gg = B * k // (yp * zp)
                y, z = gg * yp, gg * zp
                if y < x:
                    continue
                assert m * x * y * z == p * (x * y + y * z + z * x)
                tot += 6 if (x < y < z) else (1 if x == y == z else 3)
    return tot


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(7)

    print("=== A) Familia A_modL1 con Y = 47 (m exacto) frente a un control propio ===")
    L = 8 * prod(primerange(3, 48))
    A = []
    k = 1
    while len(A) < 100:
        if isprime(k * L + 1):
            A.append(k * L + 1)
        k += 1
    print("  A_modL1 :", resumen([m_de(p) for p in A]), " [Numba: media 8.28, p90 23, max 39, P(m>11) 17%, P(m>23) 8%]")
    C = []
    while len(C) < 100:
        c = rng.randint(L // 840, 100 * L // 840) * 840 + rng.choice(DIF)
        if isprime(c):
            C.append(c)
    print("  control :", resumen([m_de(p) for p in C]), " [otra muestra: media 5.08, max 23, P(m>11) 2%]")

    print("\n=== B) m exacto de primos adversarios con cota superior alta  ===")
    for p, mub, Y in [(499290503461855024921, 55, 47), (307444891294245705001, 39, 47),
                      (65041293118675673809, 39, 47), (367231949708083241494729, 47, 59)]:
        m = m_de(p)
        print(f"  p={p}: primo={isprime(p)}, difícil={p % 840 in DIF}, q_min={qmin(p)} (>{Y}: {qmin(p) > Y}), "
              f"m exacto={m} (cota de la búsqueda {mub}; coherente: {m is not None and m <= mub})")

    print("\n=== C) f_m(p) ordenadas: enumeración propia frente a datos/fm_primos_2e5.csv ===")
    G = {}
    with open("datos/fm_primos_2e5.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            G[int(r["p"])] = r
    muestra = [2, 3, 5, 7, 11, 17, 131, 241, 1009, 2521] + rng.sample([p for p in G if 1000 < p < 6000], 12)
    malos = 0
    for p in muestra:
        for m in (4, 8):
            f = f_ordenadas(m, p)
            g = int(G[p][f"f{m}"])
            malos += f != g
            if f != g:
                print(f"  ✗ p={p}, m={m}: aquí {f} vs CSV {g}")
    print(f"  {len(muestra)} primos × m ∈ {{4, 8}}: discrepancias = {malos}")
    print(f"  control E–T p. 2: f_4(2,3,5,7) = {[f_ordenadas(4, p) for p in (2, 3, 5, 7)]} (esperado [3, 12, 12, 36])")
    clase = [(p, int(r["f4"]), int(r["f8"])) for p, r in G.items() if p >= 100 and p % 24 == 1]
    r4 = sum(f / log(p) ** 3 for p, f, _ in clase) / len(clase)
    r8 = sum(f / log(p) ** 3 for p, _, f in clase) / len(clase)
    print(f"  p ≡ 1 (mod 24), p >= 100: media f4/log³ = {r4:.4f}, f8/log³ = {r8:.4f}, cociente = {r4 / r8:.2f}")
    exc8 = [p for p, r in G.items() if int(r["f8"]) == 0]
    print(f"  CSV: primos con f_8 = 0: {exc8}")

    print("\n=== D) media de f4/(log p)³ por q_min(p) en p ≡ 1 (mod 24), p >= 1000 (CSV) ===")
    por_q = defaultdict(list)
    for p, r in G.items():
        if p >= 1000 and p % 24 == 1:
            por_q[qmin(p)].append(int(r["f4"]) / log(p) ** 3)
    for q in sorted(por_q):
        if len(por_q[q]) >= 5:
            print(f"  q_min={q:>3}: n={len(por_q[q]):>5}  media={sum(por_q[q]) / len(por_q[q]):.4f}")


if __name__ == "__main__":
    main()
