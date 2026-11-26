#!/usr/bin/env python3
"""
Is c = 7 special among the shifts c ≡ 7 (mod 8)? Counterexamples for c >= 23 and an
exact analysis of c = 15 (note, §4).
"""
import math
import sys
from sympy import factorint, isprime, jacobi_symbol, primerange

sys.path.insert(0, "tools")
from verify import types_at

HARD_SET = {1, 121, 169, 289, 361, 529}


def squares_mod(c):
    return {(x * x) % c for x in range(c) if math.gcd(x, c) == 1}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    primes_1mod24 = [p for p in primerange(25, 200_000) if p % 24 == 1]

    print("=== T3: Check for c ≡ 7 (mod 8), 7 <= c <= 79 ===")
    print("  Looking for concrete counterexamples where K_c has a prime factor q with (q/c) = -1")
    print("  (and, for composite c, also without requiring (q/c)=-1: a non-square mod c is enough),")
    print("  yet there is NO solution (neither Type I nor Type II) at c:\n")

    for c in range(7, 80, 8):
        sq_c = squares_mod(c)
        ce_any = None       # smallest prime p ≡ 1 (mod 24) with total failure (neither I nor II) and (q/c) = -1
        ce_hard = None      # smallest hard prime mod 840 with total failure and (q/c) = -1
        ce_type2_only = None  # smallest p ≡ 1 (mod 24) where Type II fails even though (q/c) = -1
        ce_nonsq = None     # for composite c: total failure with q a non-square mod c

        for p in primes_1mod24:
            if p <= c or p % c == 0:
                continue
            Kc = (p + c) // 4
            pf = factorint(Kc)
            has_jac_neg = any(jacobi_symbol(q, c) == -1 for q in pf if q > 2)
            has_nonsq = any((q % c) not in sq_c for q in pf)
            if not (has_jac_neg or has_nonsq):
                continue
            t1, t2 = types_at(p, c)
            if has_jac_neg and not t2 and ce_type2_only is None:
                ce_type2_only = (p, dict(pf), (t1, t2))
            if has_jac_neg and not t1 and not t2:
                if ce_any is None:
                    ce_any = (p, dict(pf), (t1, t2))
                if (p % 840) in HARD_SET and ce_hard is None:
                    ce_hard = (p, dict(pf), (t1, t2))
            if has_nonsq and not t1 and not t2 and ce_nonsq is None:
                ce_nonsq = (p, dict(pf), (t1, t2))
            if ce_any and ce_hard and ce_type2_only and ce_nonsq:
                break

        kind = "prime" if isprime(c) else f"comp={dict(factorint(c))}"
        print(f"c = {c:2d} ({kind:16s}):")
        if c == 7:
            print(f"  Counterexamples (p < 200,000): none (impossible by the exact criterion for c = 7)")
        else:
            if ce_any:
                p, pf, tp = ce_any
                q_neg = [q for q in pf if q > 2 and jacobi_symbol(q, c) == -1]
                print(f"  I and II fail with (q/c)=-1 [p≡1 mod 24]:  p={p} (mod {c}={p%c}), K_c={(p+c)//4}={pf}, q with (q/c)=-1: {q_neg}, types={tp}")
            else:
                print(f"  I and II fail with (q/c)=-1 [p≡1 mod 24]:  NONE for p < 200,000")
            if ce_hard:
                p, pf, tp = ce_hard
                q_neg = [q for q in pf if q > 2 and jacobi_symbol(q, c) == -1]
                print(f"  I and II fail with (q/c)=-1 [hard 840]: p={p} (mod 840={p%840}, mod {c}={p%c}), K_c={(p+c)//4}={pf}, q with (q/c)=-1: {q_neg}, types={tp}")
            elif not isprime(c) and ce_nonsq:
                p, pf, tp = ce_nonsq
                q_ns = [q for q in pf if (q % c) not in sq_c]
                print(f"  I and II fail with q a non-square mod {c}:  p={p} (mod {c}={p%c}), K_c={(p+c)//4}={pf}, non-square q: {q_ns}, types={tp}")

    # Exact algebraic and numerical check of c = 15
    print("\n=== Exact algebraic and numerical analysis of c = 15 in (Z/15Z)* ===")
    for q in [7, 11, 13, 14]:
        S = {1, 2, q % 15, (2 * q) % 15}
        invs = {a: pow(a, -1, 15) for a in S}
        SSinv = {(a * invs[b]) % 15 for a in S for b in S}
        miss = sorted({1, 2, 4, 7, 8, 11, 13, 14} - SSinv)
        print(f"  q ≡ {q:2d} (mod 15) [(q/15)={int(jacobi_symbol(q, 15)):+d}]: S={sorted(S)}, S*S^-1={sorted(SSinv)}, missing={miss}")

    # Check every p < 200,000, p ≡ 1 (mod 24) at c = 15
    t2_fails_15 = []
    any_fails_15 = []
    hard_t12_disc_15 = 0
    hard_jac_hits_15 = 0
    for p in primes_1mod24:
        if p % 15 == 0:
            continue
        K15 = (p + 15) // 4
        pf = factorint(K15)
        has_jac = any(jacobi_symbol(q, 15) == -1 for q in pf if q > 2)
        t1, t2 = types_at(p, 15)
        if has_jac and not t2:
            t2_fails_15.append((p, p % 15, dict(pf), (t1, t2)))
        if has_jac and not (t1 or t2):
            any_fails_15.append(p)
        if (p % 840) in HARD_SET:
            if has_jac:
                hard_jac_hits_15 += 1
            if has_jac != t1 or has_jac != t2:
                hard_t12_disc_15 += 1
    print(f"  p < 200,000, p ≡ 1 (mod 24) with (q/15)=-1 and Type II failing: {len(t2_fails_15)} cases; first 5: {t2_fails_15[:5]}")
    print(f"  p < 200,000, p ≡ 1 (mod 24) with (q/15)=-1 and (I and II) failing: {len(any_fails_15)} cases")
    print(f"  Hard primes mod 840 < 200,000: with (q/15)=-1: {hard_jac_hits_15}, discrepancies with (I and II): {hard_t12_disc_15}")


if __name__ == "__main__":
    main()
