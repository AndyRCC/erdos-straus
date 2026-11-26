#!/usr/bin/env python3
"""
Búsqueda dirigida por encima de 10^18 en primos "casi cuadrados" (q_min(p) > Y), con un
grupo de control (nota, §7). Cada solución encontrada se verifica en aritmética exacta.
Por encima de 10^18 la búsqueda solo puede encontrar soluciones, no certificar contraejemplos.
"""

import csv
import math
import os
import random
import sys
import time
from math import gcd

import numpy as np
from sympy import isprime, primerange, legendre_symbol

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))
from verificar import es_solucion, tipos_en  # noqa: E402


# Primos pequeños precalculados para división por tentativa rápida
SMALL_PRIMES_20K = list(primerange(2, 20_000))
SMALL_PRIMES_100K = list(primerange(2, 100_000))
QS_300 = list(primerange(3, 300))
HARD_840 = (1, 121, 169, 289, 361, 529)


def bounded_brent_rho(n, max_iters=400):
    """Pollard-rho de Brent acotado (atrapa factores hasta ~10^7 en < 0.5 ms)."""
    if n % 2 == 0:
        return 2
    for c in (1, 2, 3):
        y, m_step = 2, 32
        g, r, q = 1, 1, 1
        iters = 0
        while g == 1 and iters < max_iters:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1 and iters < max_iters:
                ys = y
                lim = min(m_step, r - k)
                for _ in range(lim):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = gcd(q, n)
                k += lim
                iters += lim
            r <<= 1
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
        if 1 < g < n:
            return g
    return 1


def partial_factor_kc(K, trial_primes, rho_iters=400):
    """
    Extrae todos los factores primos q en trial_primes que dividen a K, y aplica
    Pollard-rho acotado al cofactor compuesto para atrapar factores medianos (~10^7).
    Devuelve (lista de factores primos hallados con multiplicidad, lista de bloques cofactores, es_completa).
    Si todos los bloques restantes son 1 o primos, la factorización es 100% completa.
    """
    factors = []
    n = K
    while (n & 1) == 0:
        factors.append(2)
        n >>= 1
    for q in trial_primes[1:]:
        if q * q > n:
            break
        while n % q == 0:
            factors.append(q)
            n //= q
    if n == 1:
        return factors, [], True
    if isprime(n):
        factors.append(n)
        return factors, [], True

    # Intentar dividir n mediante Pollard-rho acotado (pila)
    stack = [n]
    unfactored = []
    while stack:
        cur = stack.pop()
        if cur == 1:
            continue
        if isprime(cur):
            factors.append(cur)
            continue
        d = bounded_brent_rho(cur, max_iters=rho_iters) if rho_iters > 0 else 1
        if 1 < d < cur:
            stack.append(d)
            stack.append(cur // d)
        else:
            unfactored.append(cur)

    return factors, unfactored, (len(unfactored) == 0)


def find_solution_at_c(p, c, trial_primes, rho_iters=400):
    """
    Intenta hallar una solución explícita (x, y, z) en x = K_c = (p + c) // 4 usando
    los factores pequeños/medianos de K_c más cada bloque cofactor R como divisor real.
    Devuelve (ok, tipo_str, (x, y, z), factorizacion_completa).
    """
    K = (p + c) // 4
    factors, unfactored, is_complete = partial_factor_kc(K, trial_primes, rho_iters=rho_iters)
    blocks = factors + unfactored

    # DP sobre residuos mod c: dp[r] guarda un divisor real d | K con d ≡ r (mod c)
    dp = {1: 1}
    for blk in blocks:
        bmod = blk % c
        if bmod == 1:
            continue
        new_items = {}
        for r, d in dp.items():
            nr = (r * bmod) % c
            if nr not in dp and nr not in new_items:
                new_items[nr] = d * blk
        dp.update(new_items)

    # Comprobar Tipo II: d1 + d2 ≡ 0 (mod c)
    for r, d1 in dp.items():
        comp = (c - r) % c
        if comp in dp:
            d2 = dp[comp]
            k = (d1 + d2) // c
            # 4/p - 1/K = c/(p*K) = (d1 + d2)/(p*K*k) = 1/(p*K*k/d1) + 1/(p*K*k/d2)
            y = (p * K * k) // d1
            z = (p * K * k) // d2
            sol = (K, y, z)
            assert es_solucion(p, *sol), f"Solución Tipo II inválida en p={p}, c={c}"
            return True, "II", sol, is_complete

    # Comprobar Tipo I: p*d1 + d2 ≡ 0 (mod c)
    neg_p = (-p) % c
    for r, d1 in dp.items():
        comp = (neg_p * r) % c
        if comp in dp:
            d2 = dp[comp]
            k = (p * d1 + d2) // c
            # 4/p - 1/K = c/(p*K) = (p*d1 + d2)/(p*K*k) = 1/(K*k/d1) + 1/(p*K*k/d2)
            y = (K * k) // d1
            z = (p * K * k) // d2
            sol = (K, y, z)
            assert es_solucion(p, *sol), f"Solución Tipo I inválida en p={p}, c={c}"
            return True, "I", sol, is_complete

    return False, None, None, is_complete


def q_min_exact(p):
    """Menor primo q tal que (q/p) = -1 (para primo impar p)."""
    if p % 8 in (3, 5):
        return 2
    for q in primerange(3, 2000):
        if legendre_symbol(p % q, q) == -1:
            return q
    return -1


def local_S4_score(p):
    """Factor local S_4(p) de /(d) sobre q <= 300 (para p ≡ 1 mod 24)."""
    L_inv = 1.0
    pm1 = 1.0
    for q in QS_300:
        r = p % q
        if r == 0:
            continue
        leg = 1 if pow(r, (q - 1) // 2, q) == 1 else -1
        L_inv *= (1.0 - leg / q)
        if r == 1:
            pm1 *= (1.0 - 1.0 / q)
    return (L_inv ** 0.571) * (pm1 ** 0.604)


def solve_prime_targeted(p, c_max_base=400, c_max_hard=2000):
    """
    Busca una solución para 4/p recorriendo c = 3, 7, 11, ...
    Devuelve (m_bound, is_exact, tipo, sol, effort_level).
    """
    # Esfuerzo 1: primos hasta 20,000 + Pollard-rho acotado (400 iters) + bloque R, c <= c_max_base
    all_complete = True
    for c in range(3, c_max_base + 1, 4):
        ok, tipo, sol, comp = find_solution_at_c(p, c, SMALL_PRIMES_20K, rho_iters=400)
        if ok:
            return c, all_complete, tipo, sol, "E1(20k+rho400)"
        if not comp:
            all_complete = False

    # Esfuerzo 2 (si resistió E1): primos hasta 100,000 + Pollard-rho (2000 iters), c <= c_max_hard
    all_complete = True
    for c in range(3, c_max_hard + 1, 4):
        ok, tipo, sol, comp = find_solution_at_c(p, c, SMALL_PRIMES_100K, rho_iters=2000)
        if ok:
            return c, all_complete, tipo, sol, "E2(100k+rho2000)"
        if not comp:
            all_complete = False

    return -1, False, None, None, f"NO_RESUELTO(E2:100k+rho2000,c<={c_max_hard})"


def crt_pair(a1, m1, a2, m2):
    """Combina x ≡ a1 (mod m1) y x ≡ a2 (mod m2) con gcd(m1, m2) = 1."""
    inv = pow(m1, -1, m2)
    k = ((a2 - a1) * inv) % m2
    return a1 + m1 * k, m1 * m2


def build_L(Y):
    L = 8
    for q in primerange(3, Y + 1):
        L *= q
    return L


def generate_subfamily_A(Y, count):
    """Primeros `count` primos p ≡ 1 (mod L(Y))."""
    L = build_L(Y)
    res = []
    k = 1
    while len(res) < count:
        cand = k * L + 1
        if isprime(cand):
            res.append(cand)
        k += 1
    return L, res


def generate_subfamily_B(Y, count, rng):
    """
    `count` primos con p ≡ 1 (mod 8), p ≡ 1 (mod 3), y para cada primo 5 <= q <= Y
    un resto cuadrático aleatorio r_q != 1 mod q (evitando q | p - 1).
    """
    qs = list(primerange(5, Y + 1))
    # Precomputar restos cuadráticos != 1 mod q
    qrs = {}
    for q in qs:
        s = sorted({(x * x) % q for x in range(1, q)} - {1})
        qrs[q] = s

    res = []
    seen = set()
    L = build_L(Y)
    while len(res) < count:
        # Construir residuo aleatorio r0 mod L(Y) por CRT
        r0, mod0 = 1, 24
        for q in qs:
            rq = rng.choice(qrs[q])
            r0, mod0 = crt_pair(r0, mod0, rq, q)
        assert mod0 == L
        # Buscar el menor k >= 1 (o aleatorio pequeño) con k*L + r0 primo
        k = rng.randint(1, 50)
        while True:
            cand = k * L + r0
            if cand not in seen and isprime(cand):
                seen.add(cand)
                res.append(cand)
                break
            k += 1
    return res


def generate_subfamily_C_worst_E4(Y, pool_size, pick_count):
    """De `pool_size` primos p ≡ 1 (mod L(Y)), elige los `pick_count` con menor S_4(p)."""
    _, pool = generate_subfamily_A(Y, pool_size)
    scored = sorted(((local_S4_score(p), p) for p in pool), key=lambda x: x[0])
    return [p for _, p in scored[:pick_count]], [s for s, _ in scored[:pick_count]]


def generate_control(Y, L, count, rng):
    """`count` primos difíciles aleatorios (mod 840 en las 6 clases) en [L, 100 * L]."""
    res = []
    seen = set()
    lo_q = L // 840 + 1
    hi_q = (100 * L) // 840
    while len(res) < count:
        r = rng.choice(HARD_840)
        q_idx = rng.randint(lo_q, hi_q)
        cand = q_idx * 840 + r
        if cand not in seen and isprime(cand):
            seen.add(cand)
            res.append(cand)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(RAIZ)
    t0 = time.time()
    rng = random.Random(20261126)  # semilla fija para reproducibilidad 100% exacta

    # =========================================================================
    # 0. Validación sobre 3 primos de control
    # =========================================================================
    print("=== 0. VALIDACIÓN SOBRE 3 PRIMOS DE CONTROL ===")
    controls = [
        (43, 313986271960080721, 59, 7),
        (47, 68867655649911037921, 67, 3),
        (53, 912496437361321252441, 59, 3),
    ]
    for Y, p_ref, qm_ref, m_ref in controls:
        _, ps = generate_subfamily_A(Y, 1)
        p = ps[0]
        qm = q_min_exact(p)
        # Validación exacta con herramientas/verificar.tipos_en
        m_exact_ver = next(c for c in range(3, 100, 4) if any(tipos_en(p, c)))
        # También comprobar nuestra búsqueda con factorización parcial
        m_got, is_ex, tipo, sol, eff = solve_prime_targeted(p)
        print(f"  Y={Y}: p==ref:{p==p_ref} ({p}) | q_min={qm} (ref {qm_ref}) | "
              f"m_tipos_en={m_exact_ver} (ref {m_ref}) | m_parcial={m_got} (exact={is_ex}, tipo={tipo}) | es_solucion={es_solucion(p, *sol)}")
        assert p == p_ref and qm == qm_ref and m_exact_ver == m_ref and m_got >= m_ref and es_solucion(p, *sol)

    # =========================================================================
    # 1. Experimento de control EXACTO en Y = 47 (~10^20, 100 vs 100 vs 100, factorint completo)
    # =========================================================================
    print("\n=== 1. CONTROL EXACTO (SIN SESGO DE FACTORIZACIÓN PARCIAL) EN Y = 47 (~10^20) ===")
    L47, A47 = generate_subfamily_A(47, 100)
    B47 = generate_subfamily_B(47, 100, rng)
    K47 = generate_control(47, L47, 100, rng)
    print(f"  {'Familia (Y=47)':<14} | {'n':>3} | {'media m':>7} | {'mediana':>7} | {'p90':>5} | {'max':>3} | {'P(m>3)':>7} | {'P(m>11)':>7} | {'P(m>23)':>7}")
    for fname, plist in [("A_modL1", A47), ("B_QR_rand", B47), ("Control", K47)]:
        exact_ms = []
        for p in plist:
            m_ex = next(c for c in range(3, 400, 4) if any(tipos_en(p, c)))
            exact_ms.append(m_ex)
        arr = np.array(exact_ms, dtype=float)
        print(f"  {fname:<14} | {len(arr):>3} | {arr.mean():>7.2f} | {np.median(arr):>7.1f} | "
              f"{np.percentile(arr, 90):>5.1f} | {int(arr.max()):>3} | {100*(arr>3).mean():>6.1f}% | "
              f"{100*(arr>11).mean():>6.1f}% | {100*(arr>23).mean():>6.1f}%")

    # =========================================================================
    # 2. Búsqueda dirigida para Y = 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101
    # =========================================================================
    Y_LIST = [47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
    N_PER_Y = 40   # 40 primos por (Y, familia) => 480 primos por familia A, B, Control
    N_C_PICK = 15  # 15 peores E_4 de un pool de 300 en cada Y => 180 primos en familia C

    print(f"\n=== 2. BÚSQUEDA DIRIGIDA SOBRE 10^18..10^43 ({len(Y_LIST)} niveles de Y, factorización parcial) ===")
    print(f"  Esfuerzo E1: primos <= 20,000 + Pollard-rho(400) + bloque R, c <= 400; escalado E2: 100k+rho(2000), c <= 2000.")
    print(f"  {'Y':>3} {'dig':>3} | {'A: p≡1 mod L(Y) (med/max/ex)':>28} | {'B: QR aleat (med/max/ex)':>25} | "
          f"{'C: peor E4 (med/max/ex)':>24} | {'Control (med/max/ex)':>22}")

    agg_rows = []
    all_by_fam = {"A_modL1": [], "B_QR_rand": [], "C_worst_E4": [], "Control": []}
    unresolved = []
    hardest_primes = []

    for Y in Y_LIST:
        L, fam_A = generate_subfamily_A(Y, N_PER_Y)
        fam_B = generate_subfamily_B(Y, N_PER_Y, rng)
        fam_C, _ = generate_subfamily_C_worst_E4(Y, 300, N_C_PICK)
        fam_Ctrl = generate_control(Y, L, N_PER_Y, rng)
        digits = len(str(fam_A[0]))

        summaries = {}
        for fname, plist in [("A_modL1", fam_A), ("B_QR_rand", fam_B), ("C_worst_E4", fam_C), ("Control", fam_Ctrl)]:
            ms = []
            n_exact = 0
            n_unres = 0
            for p in plist:
                m_val, is_ex, tipo, sol, eff = solve_prime_targeted(p)
                if m_val < 0:
                    n_unres += 1
                    unresolved.append((Y, fname, p, eff))
                else:
                    ms.append(m_val)
                    all_by_fam[fname].append((Y, p, m_val, is_ex))
                    if is_ex:
                        n_exact += 1
                    if m_val >= 35:
                        hardest_primes.append((m_val, is_ex, Y, fname, p, tipo, sol))
            arr = np.array(ms, dtype=float)
            med = float(np.median(arr)) if len(arr) else -1.0
            mean_v = float(arr.mean()) if len(arr) else -1.0
            max_v = int(arr.max()) if len(arr) else -1
            f_gt3 = float((arr > 3).mean()) if len(arr) else -1.0
            f_gt11 = float((arr > 11).mean()) if len(arr) else -1.0
            summaries[fname] = (mean_v, med, max_v, n_exact, len(plist), n_unres, f_gt3, f_gt11)
            agg_rows.append([Y, digits, fname, len(plist), n_unres, n_exact, f"{mean_v:.2f}", f"{med:.1f}", max_v, f"{f_gt3:.3f}", f"{f_gt11:.3f}"])

        sA, sB, sC, sK = summaries["A_modL1"], summaries["B_QR_rand"], summaries["C_worst_E4"], summaries["Control"]
        print(f"  {Y:>3} {digits:>3} | {sA[0]:5.1f} / {sA[1]:4.1f} / {sA[2]:>3} (ex {sA[3]:>2}/{sA[4]}) | "
              f"{sB[0]:5.1f} / {sB[1]:4.1f} / {sB[2]:>3} (ex {sB[3]:>2}) | "
              f"{sC[0]:5.1f} / {sC[1]:4.1f} / {sC[2]:>3} (ex {sC[3]:>2}) | "
              f"{sK[0]:5.1f} / {sK[1]:4.1f} / {sK[2]:>3} (ex {sK[3]:>2})")

    # Resumen global agregado comparando las 4 familias
    print("\n=== 2. COMPARACIÓN GLOBAL ENTRE FAMILIAS (TODOS LOS Y = 47..101, ~10^19 A ~10^40) ===")
    print(f"  {'Familia':<12} | {'n':>4} | {'No res.':>7} | {'Exactos':>7} | {'media m_ub':>10} | {'mediana':>7} | "
          f"{'p90':>5} | {'max':>4} | {'P(m>3)':>7} | {'P(m>11)':>7} | {'P(m>23)':>7}")
    for fname in ("A_modL1", "B_QR_rand", "C_worst_E4", "Control"):
        vals = all_by_fam[fname]
        arr = np.array([x[2] for x in vals], dtype=float)
        nex = sum(1 for x in vals if x[3])
        nun = sum(1 for x in unresolved if x[1] == fname)
        print(f"  {fname:<12} | {len(arr):>4} | {nun:>7} | {nex:>7} | {arr.mean():>10.2f} | {np.median(arr):>7.1f} | "
              f"{np.percentile(arr, 90):>5.1f} | {int(arr.max()):>4} | {100*(arr>3).mean():>6.1f}% | "
              f"{100*(arr>11).mean():>6.1f}% | {100*(arr>23).mean():>6.1f}%")

    print(f"\nPrimos NO resueltos con esfuerzo máximo E: {len(unresolved)}")
    for item in unresolved:
        print("  NO RESUELTO:", item)

    hardest_primes.sort(key=lambda x: (-x[0], x[2]))
    print("\n=== 3. TOP 10 MAYORES COTAS m ENCONTRADAS POR ENCIMA DE 10^18 ===")
    for m_val, is_ex, Y, fname, p, tipo, sol in hardest_primes[:10]:
        ex_lbl = "EXACTO" if is_ex else "cota sup."
        print(f"  m = {m_val:>3} ({ex_lbl:<9}) | Y={Y:>3} ({len(str(p)):>2} dig) | {fname:<10} | tipo={tipo} | p = {p}")

    # Guardar tabla agregada en CSV
    out_csv = os.path.join(RAIZ, "datos", "busqueda_adversaria_1e18.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Y", "digits", "familia", "n", "no_resueltos", "exactos", "media_m", "mediana_m", "max_m", "frac_gt3", "frac_gt11"])
        w.writerows(agg_rows)
    print(f"\nGuardado: {out_csv} ({os.path.getsize(out_csv):,} bytes)")
    print(f"Tiempo total: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
