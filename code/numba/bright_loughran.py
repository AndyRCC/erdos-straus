#!/usr/bin/env python3
"""
Numerical check of Bright–Loughran (arXiv:1908.02526): Thm. 1.2, Thm. 1.5, Cor. 1.3 and
Cor. 1.4, with an exact Hilbert symbol, and the example U_{8,241} (note, §6). It also
contains utilities (SPF sieve, enumeration of solutions) used by other scripts.
"""

import os
import sys
import time
from fractions import Fraction
from math import gcd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify import is_solution  # noqa: E402


# ---------------------------------------------------------------------------
# Fast factorization with an SPF sieve
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
    """Prime factorization of n > 0 using the SPF table (with fallback trial division if n > len(spf)-1)."""
    res = {}
    lim = len(spf) - 1
    while n > lim:
        # extract small factors
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
    """Generates all positive divisors d of B^2 with d <= B."""
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
    """Generates all positive divisors d of B^2."""
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
# Legendre symbol and Hilbert symbols (Serre, Thm. III.1)
# ---------------------------------------------------------------------------

def legendre(a, p):
    """Legendre symbol (a/p) for p an odd prime and p ∤ a."""
    r = a % p
    if r == 0:
        raise ValueError(f"p={p} divide a a={a}")
    v = pow(r, (p - 1) // 2, p)
    return 1 if v == 1 else -1


def vp_int(x, p):
    """p-adic valuation of a nonzero integer x and its part prime to p."""
    c = 0
    while x % p == 0:
        x //= p
        c += 1
    return c, x


def hilbert_odd(a: Fraction, b: Fraction, p: int) -> int:
    """
    Hilbert symbol (a, b)_p for p an odd prime (Serre, A Course in Arithmetic, Thm. III.1):
    Writing a = p^alpha * u, b = p^beta * v with u, v in Z_p^*,
      (a, b)_p = (-1)^(alpha * beta * (p-1)/2) * (u/p)^beta * (v/p)^alpha.
    Note: for r = r_num / r_den in Z_p^*, (r/p) = (r_num * r_den / p).
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
    Hilbert symbol (a, b)_2 (Serre, Thm. III.1):
    a = 2^alpha * u, b = 2^beta * v with u, v in Z_2^*,
      (a, b)_2 = (-1)^(eps(u)*eps(v) + alpha*omega(v) + beta*omega(u)),
    where eps(x) = (x - 1)/2 mod 2 and omega(x) = (x^2 - 1)/8 mod 2.
    """
    assert a != 0 and b != 0
    va_n, an = vp_int(a.numerator, 2)
    va_d, ad = vp_int(a.denominator, 2)
    alpha = va_n - va_d
    # u mod 8: since ad is odd, ad^-1 ≡ ad (mod 8) because ad^2 ≡ 1 (mod 8)
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
    """Real Hilbert symbol (a, b)_∞ = -1 iff a < 0 and b < 0."""
    return -1 if (a < 0 and b < 0) else 1


# ---------------------------------------------------------------------------
# Exhaustive enumeration of positive solutions and search for non-positive ones
# ---------------------------------------------------------------------------

def all_positive_solutions(m, n, spf):
    """
    Returns ALL positive integer solutions 1 <= u1 <= u2 <= u3 of
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
        # (A*u2 - B)(A*u3 - B) = B^2, with 1 <= d <= B and d ≡ -B (mod A)
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
    Searches for NON-positive integer solutions u = (u1, u2, u3) in Z^3 \ N^3 with u1*u2*u3 != 0
    of 4/n = 1/u1 + 1/u2 + 1/u3 (n odd).
    Includes a guaranteed explicit family + a search over small u1 (positive and negative).
    """
    found = set()
    # Guaranteed explicit family for odd n >= 3:
    # - If n ≡ 1 (mod 4): u1 = (n-1)/4 gives 4/n - 4/(n-1) = -4/(n(n-1)) = 1/(-n(n-1)/2) + 1/(-n(n-1)/2)
    #   and u1 = (n-1)/2 gives 4/n - 2/(n-1) -> no; u = ((n-1)/2, (n-1)/2, -n(n-1)/4) has 2 positive, 1 negative!
    if n % 4 == 1:
        k = (n - 1) // 4
        found.add(tuple(sorted((k, -n * (n - 1) // 2, -n * (n - 1) // 2))))
        found.add(tuple(sorted((2 * k, 2 * k, -n * k))))
    else:
        # n ≡ 3 (mod 4): u1 = (n+1)/4 gives 4/n - 4/(n+1) = 4/(n(n+1)) (positive).
        # For a non-positive one: 4/n = 1/((n-3)/4) ... we try u1 = -n or small u1, and also an
        # identity valid for every n ≡ 3 (mod 4) >= 3:
        #   4/n = 1/((n+1)/4) + 1/(-n(n+1)/4) + ... no;
        #   4/n = 1/((n-1)/2) + ... better: a systematic search over small u1 + u1 close to n/4!
        pass

    cand_u1 = list(range(-u1_bound, 0)) + list(range(1, u1_bound + 1))
    # Add values close to n/4 and -n to cover both signs (1 negative and 2 negative)
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
        # (A*u2 - B)(A*u3 - B) = B^2, allowing positive and negative divisors d of B^2
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
                    # Check exactness
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
    # The largest possible B in all_positive_solutions for n <= 2000 is n * (3n/4) <= 3,000,000
    spf = build_spf(3_100_000)
    print(f"SPF table built up to 3,100,000 in {time.time() - t0:.2f} s")

    # =========================================================================
    # 1. Theorem 1.2 + Corollary 1.3 + Corollary 1.4 for EVERY odd n <= 2000
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
            assert is_solution(n, u1, u2, u3), f"Invalid solution: n={n}, {(u1, u2, u3)}"
            total_pos_sols += 1

            a = Fraction(-u1, u3)
            b = Fraction(-u2, u3)

            # Product over p | n (Thm. 1.2)
            prod_n = 1
            for p in primes_n:
                h_p = hilbert_odd(a, b, p)
                prod_n *= h_p
                # Check S_3 invariance (the 3 choices of denominator)
                h_p2 = hilbert_odd(Fraction(-u1, u2), Fraction(-u3, u2), p)
                h_p3 = hilbert_odd(Fraction(-u2, u1), Fraction(-u3, u1), p)
                if not (h_p == h_p2 == h_p3):
                    s3_inv_fails += 1
            if prod_n != -1:
                thm12_fails += 1

            # Lemma 3.8 (p = 2)
            h2 = hilbert_2(a, b)
            if h2 != 1:
                lem38_fails += 1

            # Lemma 3.5 (good primes q | u1*u2*u3 with q ∤ 2n) and global Hilbert reciprocity
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

            # Corollary 1.3 for odd primes
            if is_prime_n:
                p = n
                cor13_sols += 1
                vs = [vp_int(u, p)[0] for u in (u1, u2, u3)]
                us = (u1, u2, u3)
                # Find a pair (i, j) with v_p(u_i) == v_p(u_j)
                found_pair = False
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    if vs[i] == vs[j]:
                        found_pair = True
                        r = Fraction(-us[i], us[j])
                        # r is in Z_p^*
                        val = legendre(r.numerator * r.denominator, p)
                        if val != -1:
                            cor13_fails += 1
                if not found_pair:
                    cor13_fails += 1

                # Direct check of the formulas for p ≡ 1 (mod 4) and x = u1 < p
                if p % 4 == 1 and u1 < p:
                    c = 4 * u1 - p
                    assert c % 4 == 3 and (p + c) // 4 == u1
                    if u2 % p == 0 and u3 % p == 0:
                        # Type II at x = K_c: u2 = p*x*k/d1, u3 = p*x*k/d2 => u2/u3 = d2/d1
                        g23 = gcd(u2, u3)
                        d2_red = u2 // g23
                        d1_red = u3 // g23
                        t1c_type2_checked += 1
                        if legendre(d1_red * d2_red, p) != -1:
                            t1c_type2_fails += 1
                    else:
                        # Type I at x = K_c: p ∤ u1 and p divides exactly one of u2, u3
                        t1c_type1_checked += 1
                        y_cop = u2 if u2 % p != 0 else u3
                        z_mul = u3 if u2 % p != 0 else u2
                        assert z_mul % p == 0 and y_cop % p != 0
                        # u1/y_cop ≡ c * d1 / d2 (mod p), and (-u1/y_cop | p) = -1
                        # Since y_cop = x*k/d1 and z_mul = p*x*k/d2 => (z_mul/p)/y_cop = d1/d2
                        zp = z_mul // p
                        g_yz = gcd(zp, y_cop)
                        d1_r = zp // g_yz
                        d2_r = y_cop // g_yz
                        lhs = legendre(-u1 * y_cop, p)
                        rhs = legendre(c * d1_r * d2_r, p)
                        if not (lhs == -1 and rhs == -1):
                            t1c_type1_fails += 1

            # Corollary 1.4 for odd squares
            if is_square_n:
                cor14_sols += 1
                # Type I (n | u_k, gcd(n, product of the others) = 1) or Type II (gcd(n, u_k)=1, n | the other two)
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

    print("\n=== 1. THEOREM 1.2 (all positive solutions, odd n, 3 <= n <= 2000) ===")
    print(f"  odd n analysed       : {total_odd} ({total_primes} primes, {total_composites} composites, {total_odd_squares} odd squares)")
    print(f"  Total positive solutions (u1 <= u2 <= u3): {total_pos_sols:,}")
    print(f"  Thm. 1.2 failures (prod_{{p|n}} != -1)       : {thm12_fails}")
    print(f"  S_3 invariance failures                      : {s3_inv_fails}")
    print(f"  Lemma 3.8 failures (inv_2 != +1)             : {lem38_fails}")
    print(f"  Lemma 3.5 failures (inv_q != +1, q ∤ 2n)     : {lem35_fails}")
    print(f"  Hilbert product formula failures             : {hilbert_prod_fails}")

    print("\n=== 2. COROLLARY 1.3 (all odd primes p <= 2000) and CONNECTION ===")
    print(f"  Odd primes <= 2000                           : {total_primes}")
    print(f"  Positive solutions for primes                : {cor13_sols:,}")
    print(f"  Cor. 1.3 failures ((-u_i/u_j | p) != -1)     : {cor13_fails}")
    print(f"  Type II at x=K_c (p≡1 mod 4): {t1c_type2_checked:,} checked, failures = {t1c_type2_fails}")
    print(f"  Type I  at x=K_c (p≡1 mod 4): {t1c_type1_checked:,} checked, failures = {t1c_type1_fails}")

    print("\n=== 3. COROLLARY 1.4 (all odd squares n <= 2000) ===")
    print(f"  Odd squares <= 2000                          : {cor14_squares_checked}")
    print(f"  Total positive solutions for squares         : {cor14_sols:,}")
    print(f"  Type I or Type II solutions found            : {cor14_type1_or_2_found}")
    print("  Examples of solutions with intermediate valuation at squares:")
    for n_sq, sol, vt in square_examples:
        print(f"    n={n_sq}: u={sol}, valuations v_p={vt}")

    # =========================================================================
    # 2. Theorem 1.5 (NON-positive integer solutions)
    # =========================================================================
    print("\n=== 4. THEOREM 1.5 (NON-positive integer solutions, odd n <= 2000) ===")
    # First the explicit examples of Bright-Loughran (p. 2)
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
        print(f"  B-L example: n={n_ex}, u={(u1, u2, u3)} -> inv_∞={h_inf:+d}, inv_2={h_2:+d}, prod_{{p|n}}={prod_n:+d}")
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

    print(f"  odd n <= 2000 with non-positive solutions found: {nonpos_covered_n}/{total_odd}")
    print(f"  Total non-positive integer solutions checked        : {nonpos_total:,} (1 negative: {nonpos_1neg:,}, 2 negative: {nonpos_2neg:,})")
    print(f"  Thm. 1.5 failures (prod_{{p|n}} != +1)                : {thm15_fails}")
    print(f"  inv_∞ != +1 failures                                  : {thm15_inf_fails}")
    print(f"  inv_2 != +1 failures                                  : {thm15_h2_fails}")
    print("  Sample of non-positive solutions (primes and composites):")
    for n_s, u_s, pr in sample_nonpos:
        print(f"    n={n_s:>3}, u={str(u_s):<26} -> prod_{{p|n}} = {pr:+d}")

    # =========================================================================
    # 3. Control U_{8, 241} (m=8, n=241) — no BM obstruction, yet U(N) is empty
    # =========================================================================
    print("\n=== 5. CONTROL SURFACE U_{8, 241} (8/241 = 1/u1 + 1/u2 + 1/u3) ===")
    sols_8_241 = all_positive_solutions(8, 241, spf)
    print(f"  Positive solutions of 8/241 (exhaustive search): {len(sols_8_241)}")
    assert len(sols_8_241) == 0

    # Non-positive integral point on U_{8, 241}(Z): (30, -14460, -14460)
    u_z = (30, -14460, -14460)
    assert 8 * u_z[0] * u_z[1] * u_z[2] == 241 * (u_z[0]*u_z[1] + u_z[0]*u_z[2] + u_z[1]*u_z[2])
    az, bz = Fraction(-u_z[0], u_z[2]), Fraction(-u_z[1], u_z[2])
    h_inf_z = hilbert_inf(az, bz)
    h_2_z = hilbert_2(az, bz)
    h_241_z = hilbert_odd(az, bz, 241)
    # Check all prime divisors of 30 and 14460 (2, 3, 5, 241)
    other_ps = [3, 5]
    h_other = {q: hilbert_odd(az, bz, q) for q in other_ps}
    print(f"  Point on U_{{8,241}}(Z): u={u_z}")
    print(f"    inv_∞={h_inf_z:+d}, inv_2={h_2_z:+d}, inv_241={h_241_z:+d}, inv_3={h_other[3]:+d}, inv_5={h_other[5]:+d}")
    assert h_inf_z * h_2_z * h_241_z * h_other[3] * h_other[5] == 1

    # Check that in Z_{241} the Hensel lift with u1 = 241*a1, 8*a1 ≡ 1 (mod 241)
    # and u2, u3 ≢ 0 (mod 241) realizes both signs of inv_{241} = (-u2/u3 | 241)^1:
    a1_mod = pow(8, -1, 241)  # 8*a1 ≡ 1 mod 241
    assert (8 * a1_mod - 1) % 241 == 0
    # For (u2, u3) = (1, 1): (-1/241) = +1 (since 241 ≡ 1 mod 4)
    # For (u2, u3) = (7, 1): (-7/241) = (7/241) = (241/7) = (3/7) = -1
    assert legendre(-1, 241) == 1 and legendre(-7, 241) == -1
    print(f"  Hensel at p=241 (a1 ≡ {a1_mod} mod 241): (-1/241)={legendre(-1, 241):+d}, (-7/241)={legendre(-7, 241):+d}")
    print("  => (U_{8,241}(R)_+ x prod_p U_{8,241}(Z_p))^Br != empty, but U_{8,241}(N) = empty.")

    print(f"\nTotal time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
