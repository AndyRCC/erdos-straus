#!/usr/bin/env python3
"""
Comprobación independiente de la nota, §5.1: certificado del polinomio cuadrático (E_231),
primos de la construcción de Dickson con J = 9 y subrécords de m(p) hasta 10^11.
"""
import json
import os
import sys
from math import gcd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
from verificar import tipos_en, perfil  # noqa: E402
from sympy import isprime, legendre_symbol, primerange  # noqa: E402


def m_exacto(p, tope=4000):
    return next(c for c in range(3, tope, 4) if any(tipos_en(p, c)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(RAIZ)

    print("=== A) Certificado E_231 → polinomio cuadrático  ===")
    cert = json.load(open("datos/e231_certificado.json", encoding="utf-8"))
    D, a0, a1, a2, X0, t0 = (int(cert[k]) for k in ("D", "a0", "a1", "a2", "X0", "t0"))
    print("  7a1² − 3a0² = 4D²:", 7 * a1 * a1 - 3 * a0 * a0 == 4 * D * D)
    print("  11a2² − 3a0² = 8D²:", 11 * a2 * a2 - 3 * a0 * a0 == 8 * D * D)
    q1 = 12541
    print(f"  D impar: {D % 2 == 1};  gcd(a0, D) = {gcd(a0, D)};  12541 | D: {D % q1 == 0};  "
          f"cofactor primo: {isprime(D // q1)};  12541 primo: {isprime(q1)};  "
          f"ambos ≡ 1 (mod 3): {q1 % 3 == 1 and (D // q1) % 3 == 1};  D = 12541·cofactor: {q1 * (D // q1) == D}")
    print("  X0² + 3a0² ≡ 0 (mod 4D²):", (X0 * X0 + 3 * a0 * a0) % (4 * D * D) == 0)
    X = 2 * D * D * t0 + X0
    print("  gcd(X(t0), a0·a1·a2) = 1:", gcd(X, a0 * a1 * a2) == 1)
    p0 = (X * X + 3 * a0 * a0) // (D * D) - 3
    print(f"  p0 = P(0): {len(str(p0))} cifras, primo: {isprime(p0)}, p0 mod 24 = {p0 % 24}, p0 mod 840 = {p0 % 840}")
    K3, K7, K11 = (p0 + 3) // 4, (p0 + 7) // 4, (p0 + 11) // 4
    print("  4D²K3 = X²+3a0²:", 4 * D * D * K3 == X * X + 3 * a0 * a0,
          "| 4D²K7 = X²+7a1²:", 4 * D * D * K7 == X * X + 7 * a1 * a1,
          "| 4D²K11 = X²+11a2²:", 4 * D * D * K11 == X * X + 11 * a2 * a2)
    malos, vistos = 0, 0
    for K, c in ((K3, 3), (K7, 7), (K11, 11)):
        for q in primerange(2, 3 * 10 ** 5):
            if K % q == 0:
                vistos += 1
                bien = (q % 3 == 1) if c == 3 else (legendre_symbol(q % c, c) == 1)
                malos += not bien
    print(f"  factores primos < 3·10^5 de K3, K7, K11: {vistos} encontrados, con residuo incorrecto: {malos}")

    print("\n=== B) Primos de Dickson con J = 9  ===")
    L = 1991773143360
    for t, m_g in ((1358978576, 59), (360830309, 43)):
        p = L * t + 1
        formas = [L // (4 * j) * t + 1 for j in range(1, 10)]
        enteros = all(L % (4 * j) == 0 for j in range(1, 10))
        todos_primos = isprime(p) and all(isprime(f) for f in formas)
        m = m_exacto(p)
        print(f"  t={t}: p={p} ({len(str(p))} cifras), L/(4j) enteros: {enteros}, 10 formas primas: {todos_primos}, "
              f"m(p) exacto = {m} (referencia: {m_g}), > 35: {m > 35}")

    print("\n=== C) Subrécords hasta 10^11  ===")
    for p, m_g in ((45195081409, 103), (92159707201, 99), (18483104929, 95), (8803369, 107)):
        m, tipos, c1, c2 = perfil(p)
        print(f"  p={p}: primo={isprime(p)}, mod 840={p % 840}, m(p)={m} (referencia: {m_g}), coincide={m == m_g}")

    print("\n=== D) E_231: y² = x³ − 231²x ===")
    x, y = -63, 1764
    print(f"  P0 = (−63, 1764) en la curva: {y * y == x ** 3 - 231 ** 2 * x};  "
          f"y ≠ 0, así que no es de 2-torsión, y la torsión de E_n es (ℤ/2)², luego P0 tiene orden infinito y el rango es ≥ 1")


if __name__ == "__main__":
    main()
