#!/usr/bin/env python3
"""
Exact local criteria at c = 3 and c = 7, and comparison of the two type II "a = 1" slices
(Aigner and Dahan) (note, §4).
"""
import csv
import math
import sys
from collections import Counter

from sympy import divisors, factorint, isprime, primerange

sys.path.insert(0, "tools")
from verify import search, is_solution, profile, types_at

HARD_SET = {1, 121, 169, 289, 361, 529}


def has_prime_2mod3(n):
    return any(q % 3 == 2 for q in factorint(n))


def has_nqr_mod7(n):
    return any(q % 7 in (3, 5, 6) for q in factorint(n))


def min_e_type2_aET1(p, emax=500):
    """Aigner Type II with a_ET = 1: e | 1+b, 4b | p+e  <=>  b | (p+e)/4 with b ≡ -1 (mod e)."""
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
    print("COMPARISON OF THE TYPE II a = 1 SLICES")
    print("============================================================")

    # 1. Aigner Type II with a_ET = 1 vs Dahan Alg. 2 (a_D = c_ET = 1) on the 14 wild primes
    print("\n--- Aigner Type II with a_ET=1 (e|1+b, 4b|p+e) vs Dahan Alg. 2 (c_ET=1) on the 14 wild primes ---")
    wild14 = [
        409, 577, 5569, 9601, 23929, 83449, 102001, 329617, 712321,
        1134241, 1724209, 1726201, 5212561, 8813281,
    ]
    solved_aET1 = 0
    for p in wild14:
        e, b = min_e_type2_aET1(p, emax=500)
        if e is not None:
            d = (p + e) // (4 * b)
            c = (1 + b) // e
            ok = is_solution(p, b * d, c * d * p, b * c * d * p)
            solved_aET1 += 1
            print(f"  p={p:8d} (mod 840={p%840:3d}, hard={p%840 in HARD_SET}): Aigner Type II (a_ET=1) SOLVED with e={e:3d}, b={b}, c_ET=a_D={c}, is_sol={ok}")
        else:
            print(f"  p={p:8d} (mod 840={p%840:3d}, hard={p%840 in HARD_SET}): NOT solved for e<=500")
    print(f"  Total of Dahan's 14 wild primes solved by Aigner Type II (a_ET=1, e<=500): {solved_aET1}/14")

    print("\n============================================================")
    print("T1. EXACT LOCAL CRITERIA AT c = 3 AND c = 7")
    print("============================================================")

    rows = list(csv.DictReader(open("data/hard_primes_1e7.csv", encoding="utf-8")))
    N = len(rows)

    # m(p) = 3 <=> K_3 has a prime factor ≡ 2 (mod 3)
    cnt_m3 = 0
    cnt_p2mod3 = 0
    cnt_t1_3 = 0
    cnt_t2_3 = 0
    disc_a = 0
    disc_t12_3 = 0

    # solution at m = 7 <=> K_7 has a prime factor ≡ 3, 5, 6 (mod 7)
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

        t1_3, t2_3 = types_at(p, 3)
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

        t1_7, t2_7 = types_at(p, 7)
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

    print(f"\n--- over the {N} hard primes <= 10^7 ---")
    print(f"  m(p) = 3:                         {cnt_m3}")
    print(f"  K_3 has a prime ≡ 2 (mod 3):      {cnt_p2mod3}")
    print(f"  types_at(p, 3) Type I / Type II:  {cnt_t1_3} / {cnt_t2_3} (discrepancies I vs II: {disc_t12_3})")
    print(f"  Total discrepancies {disc_a}")

    print(f"\n--- over the {N} hard primes <= 10^7 ---")
    print(f"  types_at(p, 7) Type I:            {cnt_t1_7}")
    print(f"  types_at(p, 7) Type II:           {cnt_t2_7}")
    print(f"  K_7 has a prime ≡ 3, 5, 6 (mod 7):{cnt_nqr7}")
    print(f"  Discrepancies Type I vs criterion: {disc_b_t1}")
    print(f"  Discrepancies Type II vs criterion:{disc_b_t2}")
    print(f"  Corollary m(p) = 7: {cnt_m7} vs criterion {cnt_m7_crit} (discrepancies: {disc_m7})")
    for res7 in [1, 2, 4]:
        print(f"    p ≡ {res7} (mod 7): total={by_p7[res7]}, solved at m=7: {by_p7_hit[res7]} ({100*by_p7_hit[res7]/by_p7[res7]:.2f}%)")

    # Check the non-hard p ≡ 1 (mod 24) with p mod 7 ∈ {3, 5, 6} (<= 10^6)
    nonhard_nqr7 = [p for p in primerange(25, 1_000_001) if p % 24 == 1 and p % 7 in (3, 5, 6)]
    ok_nonhard = sum(1 for p in nonhard_nqr7 if types_at(p, 7) == (True, True))
    print(f"  Bonus: primes p <= 10^6 with p ≡ 1 (mod 24) and p mod 7 ∈ {{3,5,6}}: {ok_nonhard}/{len(nonhard_nqr7)} have (I and II) at m=7")

    # Check the contrast with p ≡ 5 (mod 8) (where 2 ∤ K_7) to show why p ≡ 1 (mod 8) is essential
    ex_5mod8 = []
    for p in primerange(13, 5000):
        if p % 8 == 5 and p % 7 in (1, 2, 4):
            t1, t2 = types_at(p, 7)
            if t1 != t2 or (has_nqr_mod7((p + 7) // 4) and not (t1 and t2)):
                ex_5mod8.append((p, p % 7, factorint((p + 7) // 4), t1, t2))
                if len(ex_5mod8) >= 5:
                    break
    print("  Contrast p ≡ 5 (mod 8) (where 2 ∤ K_7, the collapse FAILS):")
    for p, r7, fk, t1, t2 in ex_5mod8:
        print(f"    p={p} (mod 7={r7}): K_7={(p+7)//4}={fk}, types_at(p,7)=(I:{t1}, II:{t2})")

    # Failure rate at c = 3 vs (log X)^(-1/2)
    print(f"\n--- Failure rate F_3 = P(m(p) > 3) among the hard primes <= 10^7 ---")
    bounds = [10**3, 10**4, 10**5, 10**6, 10**7]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = [r for r in rows if a < int(r["p"]) <= b]
        fail = sum(1 for r in sub if int(r["m"]) > 3)
        rate = fail / len(sub)
        xmid = math.sqrt(a * b)
        print(f"  ({a:>8,}, {b:>9,}]: n={len(sub):5d}, failures={fail:5d}, F_3={rate:.4f}, F_3*sqrt(log x_mid)={rate*math.sqrt(math.log(xmid)):.4f}")

    print("  Cumulative p <= X (6 hard classes):")
    for X in [10**4, 10**5, 10**6, 10**7]:
        sub = [r for r in rows if int(r["p"]) <= X]
        fail = sum(1 for r in sub if int(r["m"]) > 3)
        rate = fail / len(sub)
        print(f"  X={X:>9,}: n={len(sub):5d}, failures={fail:5d}, F_3={rate:.4f}, F_3*sqrt(log X)={rate*math.sqrt(math.log(X)):.4f}")


if __name__ == "__main__":
    main()
