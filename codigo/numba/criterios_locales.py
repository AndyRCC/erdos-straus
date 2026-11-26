#!/usr/bin/env python3
"""
Criterios locales exactos en c = 3 y c = 7, y comparación de los dos cortes "a = 1" de tipo II
(Aigner y Dahan) (nota, §4).
"""
import csv
import math
import sys
from collections import Counter

from sympy import divisors, factorint, isprime, primerange

sys.path.insert(0, "herramientas")
from verificar import buscar, es_solucion, perfil, tipos_en

HARD_SET = {1, 121, 169, 289, 361, 529}


def has_prime_2mod3(n):
    return any(q % 3 == 2 for q in factorint(n))


def has_nqr_mod7(n):
    return any(q % 7 in (3, 5, 6) for q in factorint(n))


def min_e_type2_aET1(p, emax=500):
    """Aigner Tipo II con a_ET = 1: e | 1+b, 4b | p+e  <=>  b | (p+e)/4 con b ≡ -1 (mod e)."""
    for e in range(3, emax + 1, 4):
        Ke = (p + e) // 4
        for b in divisors(Ke):
            if b % e == e - 1:
                return e, b
    return None, None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("============================================================")
    print("COMPARACIÓN DE LOS CORTES a = 1 DE TIPO II")
    print("============================================================")

    # 1. Aigner Tipo II con a_ET = 1 vs Dahan Alg. 2 (a_D = c_ET = 1) en los 14 primos wild
    print("\n--- Aigner Tipo II con a_ET=1 (e|1+b, 4b|p+e) vs Dahan Alg. 2 (c_ET=1) en los 14 primos wild ---")
    wild14 = [
        409, 577, 5569, 9601, 23929, 83449, 102001, 329617, 712321,
        1134241, 1724209, 1726201, 5212561, 8813281,
    ]
    resueltos_aET1 = 0
    for p in wild14:
        e, b = min_e_type2_aET1(p, emax=500)
        if e is not None:
            d = (p + e) // (4 * b)
            c = (1 + b) // e
            ok = es_solucion(p, b * d, c * d * p, b * c * d * p)
            resueltos_aET1 += 1
            print(f"  p={p:8d} (mod 840={p%840:3d}, hard={p%840 in HARD_SET}): Aigner Tipo II (a_ET=1) RESUELTO con e={e:3d}, b={b}, c_ET=a_D={c}, es_sol={ok}")
        else:
            print(f"  p={p:8d} (mod 840={p%840:3d}, hard={p%840 in HARD_SET}): NO resuelto para e<=500")
    print(f"  Total de los 14 primos wild de Dahan resueltos por Aigner Tipo II (a_ET=1, e<=500): {resueltos_aET1}/14")

    print("\n============================================================")
    print("T1. CRITERIOS LOCALES EXACTOS EN c = 3 Y c = 7")
    print("============================================================")

    rows = list(csv.DictReader(open("datos/clases_dificiles_1e7.csv", encoding="utf-8")))
    N = len(rows)

    # m(p) = 3 <=> K_3 tiene factor primo ≡ 2 (mod 3)
    cnt_m3 = 0
    cnt_p2mod3 = 0
    cnt_t1_3 = 0
    cnt_t2_3 = 0
    disc_a = 0
    disc_t12_3 = 0

    # solución en m = 7 <=> K_7 tiene factor primo ≡ 3, 5, 6 (mod 7)
    cnt_t1_7 = 0
    cnt_t2_7 = 0
    cnt_nqr7 = 0
    disc_b_t1 = 0
    disc_b_t2 = 0
    cnt_m7 = 0
    cnt_m7_crit = 0
    disc_m7 = 0
    by_p7 = Counter()
    by_p7_hit = Counter()

    for r in rows:
        p = int(r["p"])
        m = int(r["m"])
        K3 = (p + 3) // 4
        K7 = (p + 7) // 4

        t1_3, t2_3 = tipos_en(p, 3)
        c_a = has_prime_2mod3(K3)
        if m == 3:
            cnt_m3 += 1
        if c_a:
            cnt_p2mod3 += 1
        if t1_3:
            cnt_t1_3 += 1
        if t2_3:
            cnt_t2_3 += 1
        if t1_3 != t2_3:
            disc_t12_3 += 1
        if (m == 3) != c_a or (t1_3 != c_a):
            disc_a += 1

        t1_7, t2_7 = tipos_en(p, 7)
        c_b = has_nqr_mod7(K7)
        if t1_7:
            cnt_t1_7 += 1
        if t2_7:
            cnt_t2_7 += 1
        if c_b:
            cnt_nqr7 += 1
        if t1_7 != c_b:
            disc_b_t1 += 1
        if t2_7 != c_b:
            disc_b_t2 += 1

        by_p7[p % 7] += 1
        if c_b:
            by_p7_hit[p % 7] += 1

        m7_crit = (not c_a) and c_b
        if m == 7:
            cnt_m7 += 1
        if m7_crit:
            cnt_m7_crit += 1
        if (m == 7) != m7_crit:
            disc_m7 += 1

    print(f"\n--- sobre los {N} primos difíciles <= 10^7 ---")
    print(f"  m(p) = 3:                         {cnt_m3}")
    print(f"  K_3 tiene primo ≡ 2 (mod 3):      {cnt_p2mod3}")
    print(f"  tipos_en(p, 3) Tipo I / Tipo II:  {cnt_t1_3} / {cnt_t2_3} (discrepancias I vs II: {disc_t12_3})")
    print(f"  Discrepancias totales {disc_a}")

    print(f"\n--- sobre los {N} primos difíciles <= 10^7 ---")
    print(f"  tipos_en(p, 7) Tipo I:            {cnt_t1_7}")
    print(f"  tipos_en(p, 7) Tipo II:           {cnt_t2_7}")
    print(f"  K_7 tiene primo ≡ 3, 5, 6 (mod 7):{cnt_nqr7}")
    print(f"  Discrepancias Tipo I vs criterio: {disc_b_t1}")
    print(f"  Discrepancias Tipo II vs criterio:{disc_b_t2}")
    print(f"  Corolario m(p) = 7: {cnt_m7} vs criterio {cnt_m7_crit} (discrepancias: {disc_m7})")
    for res7 in [1, 2, 4]:
        print(f"    p ≡ {res7} (mod 7): total={by_p7[res7]}, resueltos en m=7: {by_p7_hit[res7]} ({100*by_p7_hit[res7]/by_p7[res7]:.2f}%)")

    # Comprobar p ≡ 1 (mod 24) no difíciles con p mod 7 ∈ {3, 5, 6} (<= 10^6)
    nonhard_nqr7 = [p for p in primerange(25, 1_000_001) if p % 24 == 1 and p % 7 in (3, 5, 6)]
    ok_nonhard = sum(1 for p in nonhard_nqr7 if tipos_en(p, 7) == (True, True))
    print(f"  Bonus: primos p <= 10^6 con p ≡ 1 (mod 24) y p mod 7 ∈ {{3,5,6}}: {ok_nonhard}/{len(nonhard_nqr7)} tienen (I y II) en m=7")

    # Comprobar el contraste con p ≡ 5 (mod 8) (donde 2 ∤ K_7) para ilustrar por qué p ≡ 1 (mod 8) es esencial
    ej_5mod8 = []
    for p in primerange(13, 5000):
        if p % 8 == 5 and p % 7 in (1, 2, 4):
            t1, t2 = tipos_en(p, 7)
            if t1 != t2 or (has_nqr_mod7((p + 7) // 4) and not (t1 and t2)):
                ej_5mod8.append((p, p % 7, factorint((p + 7) // 4), t1, t2))
                if len(ej_5mod8) >= 5:
                    break
    print("  Contraste p ≡ 5 (mod 8) (donde 2 ∤ K_7, el colapso FALLA):")
    for p, r7, fk, t1, t2 in ej_5mod8:
        print(f"    p={p} (mod 7={r7}): K_7={(p+7)//4}={fk}, tipos_en(p,7)=(I:{t1}, II:{t2})")

    # Tasa de fallo en c = 3 vs (log X)^(-1/2)
    print(f"\n--- Tasa de fallo F_3 = P(m(p) > 3) en los primos difíciles <= 10^7 ---")
    bounds = [10**3, 10**4, 10**5, 10**6, 10**7]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = [r for r in rows if a < int(r["p"]) <= b]
        fail = sum(1 for r in sub if int(r["m"]) > 3)
        rate = fail / len(sub)
        xmid = math.sqrt(a * b)
        print(f"  ({a:>8,}, {b:>9,}]: n={len(sub):5d}, fallos={fail:5d}, F_3={rate:.4f}, F_3*sqrt(log x_mid)={rate*math.sqrt(math.log(xmid)):.4f}")

    print("  Acumulado p <= X (6 clases difíciles):")
    for X in [10**4, 10**5, 10**6, 10**7]:
        sub = [r for r in rows if int(r["p"]) <= X]
        fail = sum(1 for r in sub if int(r["m"]) > 3)
        rate = fail / len(sub)
        print(f"  X={X:>9,}: n={len(sub):5d}, fallos={fail:5d}, F_3={rate:.4f}, F_3*sqrt(log X)={rate*math.sqrt(math.log(X)):.4f}")


if __name__ == "__main__":
    main()
