#!/usr/bin/env python3
"""
¿Es especial c = 7 entre los desplazamientos c ≡ 7 (mod 8)? Contraejemplos para c >= 23 y
análisis exacto de c = 15 (nota, §4).
"""
import math
import sys
from sympy import factorint, isprime, jacobi_symbol, primerange

sys.path.insert(0, "herramientas")
from verificar import tipos_en

HARD_SET = {1, 121, 169, 289, 361, 529}


def squares_mod(c):
    return {(x * x) % c for x in range(c) if math.gcd(x, c) == 1}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    primes_1mod24 = [p for p in primerange(25, 200_000) if p % 24 == 1]

    print("=== T3: Comprobación para c ≡ 7 (mod 8), 7 <= c <= 79 ===")
    print("  Buscando contraejemplos concretos donde K_c tiene un factor primo q con (q/c) = -1")
    print("  (y para c compuesto, también sin exigir (q/c)=-1, basta no-cuadrado mod c),")
    print("  pero NO existe solución (ni Tipo I ni Tipo II) en c:\n")

    for c in range(7, 80, 8):
        sq_c = squares_mod(c)
        ce_any = None       # menor primo p ≡ 1 (mod 24) con fallo total (ni I ni II) y (q/c) = -1
        ce_hard = None      # menor primo difícil mod 840 con fallo total y (q/c) = -1
        ce_type2_only = None  # menor p ≡ 1 (mod 24) donde falla Tipo II aunque (q/c) = -1
        ce_nonsq = None     # para c compuesto: fallo total con q no-cuadrado mod c

        for p in primes_1mod24:
            if p <= c or p % c == 0:
                continue
            Kc = (p + c) // 4
            pf = factorint(Kc)
            has_jac_neg = any(jacobi_symbol(q, c) == -1 for q in pf if q > 2)
            has_nonsq = any((q % c) not in sq_c for q in pf)
            if not (has_jac_neg or has_nonsq):
                continue
            t1, t2 = tipos_en(p, c)
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

        kind = "primo" if isprime(c) else f"comp={dict(factorint(c))}"
        print(f"c = {c:2d} ({kind:16s}):")
        if c == 7:
            print(f"  Contraejemplos (p < 200,000): ninguno (imposible por el criterio exacto de c = 7)")
        else:
            if ce_any:
                p, pf, tp = ce_any
                q_neg = [q for q in pf if q > 2 and jacobi_symbol(q, c) == -1]
                print(f"  Fallo I y II con (q/c)=-1 [p≡1 mod 24]:  p={p} (mod {c}={p%c}), K_c={(p+c)//4}={pf}, q con (q/c)=-1: {q_neg}, tipos={tp}")
            else:
                print(f"  Fallo I y II con (q/c)=-1 [p≡1 mod 24]:  NINGUNO en p < 200,000")
            if ce_hard:
                p, pf, tp = ce_hard
                q_neg = [q for q in pf if q > 2 and jacobi_symbol(q, c) == -1]
                print(f"  Fallo I y II con (q/c)=-1 [difícil 840]: p={p} (mod 840={p%840}, mod {c}={p%c}), K_c={(p+c)//4}={pf}, q con (q/c)=-1: {q_neg}, tipos={tp}")
            elif not isprime(c) and ce_nonsq:
                p, pf, tp = ce_nonsq
                q_ns = [q for q in pf if (q % c) not in sq_c]
                print(f"  Fallo I y II con q no-cuadrado mod {c}:  p={p} (mod {c}={p%c}), K_c={(p+c)//4}={pf}, q no-cuad: {q_ns}, tipos={tp}")

    # Comprobación algebraica y numérica exacta de c = 15
    print("\n=== Análisis algebraico y numérico exacto de c = 15 en (Z/15Z)* ===")
    for q in [7, 11, 13, 14]:
        S = {1, 2, q % 15, (2 * q) % 15}
        invs = {a: pow(a, -1, 15) for a in S}
        SSinv = {(a * invs[b]) % 15 for a in S for b in S}
        miss = sorted({1, 2, 4, 7, 8, 11, 13, 14} - SSinv)
        print(f"  q ≡ {q:2d} (mod 15) [(q/15)={int(jacobi_symbol(q, 15)):+d}]: S={sorted(S)}, S*S^-1={sorted(SSinv)}, faltan={miss}")

    # Comprobar todos los p < 200,000, p ≡ 1 (mod 24) en c = 15
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
        t1, t2 = tipos_en(p, 15)
        if has_jac and not t2:
            t2_fails_15.append((p, p % 15, dict(pf), (t1, t2)))
        if has_jac and not (t1 or t2):
            any_fails_15.append(p)
        if (p % 840) in HARD_SET:
            if has_jac:
                hard_jac_hits_15 += 1
            if has_jac != t1 or has_jac != t2:
                hard_t12_disc_15 += 1
    print(f"  p < 200,000, p ≡ 1 (mod 24) con (q/15)=-1 y fallo Tipo II: {len(t2_fails_15)} casos; primeros 5: {t2_fails_15[:5]}")
    print(f"  p < 200,000, p ≡ 1 (mod 24) con (q/15)=-1 y fallo (I y II): {len(any_fails_15)} casos")
    print(f"  Primos difíciles mod 840 < 200,000: con (q/15)=-1: {hard_jac_hits_15}, discrepancias con (I y II): {hard_t12_disc_15}")


if __name__ == "__main__":
    main()
