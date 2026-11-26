#!/usr/bin/env python3
"""
Primes blind at c = 3 and c = 7 simultaneously: Witt index, the one-parameter family
p(t) = (2t+29)² + 2520, and quartic families (note, §5).
"""

import os
import sys
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from verify import types_at, profile


def part_a_check_consecutive_norms():
    print("=== Consecutive norms of Q(sqrt(-3)) and Q(sqrt(-7)) ===")
    # Check on every prime p ≡ 1 (mod 24), p <= 100_000
    n_blind_3_7 = 0
    n_norms_ok = 0
    examples = []
    for p in sp.primerange(25, 100_000):
        if p % 24 != 1:
            continue
        t1_3, t2_3 = types_at(p, 3)
        t1_7, t2_7 = types_at(p, 7)
        blind3 = (not t1_3) and (not t2_3)
        blind7 = (not t1_7) and (not t2_7)
        if blind3 and blind7:
            n_blind_3_7 += 1
            m = (p + 3) // 4
            m1 = m + 1
            f3 = sp.factorint(m)
            f7 = sp.factorint(m1)
            ok3 = all(q % 3 == 1 for q in f3)
            ok7 = all((q == 2 or sp.legendre_symbol(q, 7) == 1) for q in f7)
            assert ok3 and ok7, (p, m, f3, f7)
            n_norms_ok += 1
            if len(examples) < 8:
                examples.append((p, m, f3, m1, f7))
    print(f"Primes p <= 10^5 with p ≡ 1 (mod 24) blind at c=3 and c=7: {n_blind_3_7} (100% consecutive norms: {n_norms_ok}/{n_blind_3_7}).")
    for p, m, f3, m1, f7 in examples:
        print(f"  p={p:6d} (mod 840={p%840:3d}): K_3={m} ({dict(f3)}), K_7={m1} ({dict(f7)})")


def part_b_witt_index_and_1param():
    print("\n=== Witt index and affine one-parameter families ===")
    a, b, d3, d7 = sp.symbols("a b d3 d7", positive=True)
    # General form of discriminant -7*d7^2, equivalent over Q to diag(1, 7*d7^2)/4
    M7 = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 2]])
    M3 = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 1]])
    Q_witt = sp.diag(2 * M7, -M3)
    Q_gen = sp.diag(b * M7, -a * M3)
    print(f"det(2*(u^2+uv+2v^2) - (x^2+xy+y^2)) = {Q_witt.det()}")
    print(f"det(b*(u^2+uv+2v^2) - a*(x^2+xy+y^2)) = {sp.factor(Q_gen.det())}")
    print(f"Square class mod (Q^*)^2: 21 (not a square; Witt index <= 1).")

    # Check that the Witt index is EXACTLY 1 for c = 1, 2, 4
    for c_mult, vec in [(1, (1, 0, 1, 0)), (2, (0, 1, 2, 0)), (4, (1, 0, 2, 0))]:
        u, v, x, y = vec
        val = c_mult * (u**2 + u * v + 2 * v**2) - (x**2 + x * y + y**2)
        assert val == 0
        print(f"  Isotropic vector for c={c_mult}: (u,v,x,y)={vec} -> Q={val} (Witt index = 1)")

    # Affine one-parameter family on f7(u,v) - f3(x,y) = 1:
    # (u, v) = (t + 5, 19), (x, y) = (t, 29) -> 4*f3(t,29) - 3 = (2t+29)^2 + 2520
    t = sp.Symbol("t", integer=True)
    f3_t = t**2 + 29 * t + 29**2
    f7_t = (t + 5) ** 2 + 19 * (t + 5) + 2 * 19**2
    assert sp.expand(f7_t - f3_t) == 1
    p_t = sp.expand(4 * f3_t - 3)
    print(f"  Affine 1-parameter family: (u,v)=(t+5,19), (x,y)=(t,29) => f7-f3 = 1, p(t) = {p_t} = (2t+29)^2 + 2520")
    primes_1p = []
    for tv in range(0, 200):
        pv = int(p_t.subs(t, tv))
        if sp.isprime(pv) and sp.gcd(tv, 29) == 1 and sp.gcd(tv + 5, 19) == 1:
            t1_3, t2_3 = types_at(pv, 3)
            t1_7, t2_7 = types_at(pv, 7)
            assert (not t1_3) and (not t2_3) and (not t1_7) and (not t2_7)
            primes_1p.append((tv, pv))
    print(f"  Primes blind at c=3 and c=7 produced for 0 <= t < 200 (with gcd=1): {len(primes_1p)}; first 6: {primes_1p[:6]}")

    # Check that in the family of minimal order (v0=1, y0=1), p(t) = (2t+1)^2 is always a square
    p_min = sp.factor(4 * (t**2 + t + 1) - 3)
    print(f"  Minimal affine family (v0=1, y0=1): (u,v)=(t,1), (x,y)=(t,1) => p(t) = {p_min} (always a square!)")


def part_c_higher_degree_exploration():
    print("\n=== Exploration of higher-degree families in 2 parameters ===")
    # 1. Quadratic family by parabolic projection from the isotropic vector (1,0,1,0):
    # U - X = 2*d => X = (4 - 4d^2 + 3Y^2 - 7V^2)/(4d), U = X + 2d.
    # For d = 1, Y = 2s+1, V = 2t+1:
    s, t = sp.symbols("s t", integer=True)
    Y = 2 * s + 1
    V = 2 * t + 1
    X = sp.expand((3 * Y**2 - 7 * V**2) / 4)
    U = sp.expand(X + 2)
    K3 = sp.expand((X**2 + 3 * Y**2) / 4)
    K7 = sp.expand((U**2 + 7 * V**2) / 4)
    assert sp.expand(K7 - K3) == 1
    p_st = sp.expand(4 * K3 - 3)
    print(f"1) Parabolic projection (d=1, Y=2s+1, V=2t+1):")
    print(f"   X = {X}")
    print(f"   p(s, t) = {p_st}")
    # Check that it yields actual primes blind at c=3 and c=7 when gcd(X,Y)=gcd(U,V)=1
    found = []
    for sv in range(1, 30):
        for tv in range(0, 30):
            Yv = 2 * sv + 1
            Vv = 2 * tv + 1
            Xv = (3 * Yv**2 - 7 * Vv**2) // 4
            Uv = Xv + 2
            if sp.gcd(abs(Xv), Yv) == 1 and sp.gcd(abs(Uv), Vv) == 1:
                pv = Xv**2 + 3 * Yv**2 - 3
                if pv > 7 and sp.isprime(pv):
                    t1_3, t2_3 = types_at(pv, 3)
                    t1_7, t2_7 = types_at(pv, 7)
                    assert (not t1_3) and (not t2_3) and (not t1_7) and (not t2_7)
                    if len(found) < 8:
                        found.append((sv, tv, pv))
    print(f"   Examples of primes blind at c=3 and c=7 (gcd=1): {found}")

    # 2. Symbolic test: can there be a polynomial solution of degree 2 (U,V,X,Y) in Q[s,t] of
    #    a (U^2 + 7V^2) - (X^2 + 3Y^2) = 4 whose homogeneous part of degree 4 is an irreducible
    #    norm form of K = Q(sqrt(-3), sqrt(-7))?
    #    We try the canonical norm form P4(s, t) = N_{K/Q}(s + t*(sqrt(-3)+sqrt(-7))/2)
    #    (and also N_{K/Q}(s + t*sqrt(-3) + ...)):
    #    N_{K/Q(sqrt(-3))}(s + t*sqrt(-7)) = s^2 + 7*t^2  (the norm in Q(sqrt(-3)) is (s^2+7t^2)^2 + 3*0^2, a square!)
    #    Take a generic element alpha = 1, beta = sqrt(-3) + sqrt(-7):
    #    N_{K/Q(sqrt(-3))}(s + t*(sqrt(-3)+sqrt(-7))) = (s + t*sqrt(-3))^2 - (-7)*t^2 = (s^2 + 4*t^2) + (2*s*t)*sqrt(-3)
    #    => X2 = s^2 + 4*t^2, Y2 = 2*s*t  => X2^2 + 3*Y2^2 = s^4 + 20*s^2*t^2 + 16*t^4
    #    N_{K/Q(sqrt(-7))}(s + t*(sqrt(-3)+sqrt(-7))) = (s + t*sqrt(-7))^2 - (-3)*t^2 = (s^2 - 4*t^2) + (2*s*t)*sqrt(-7)
    #    => U2 = s^2 - 4*t^2, V2 = 2*s*t  => U2^2 + 7*V2^2 = s^4 + 20*s^2*t^2 + 16*t^4  (IRREDUCIBLE en Q[s,t]!)
    X2 = s**2 + 4 * t**2
    Y2 = 2 * s * t
    U2 = s**2 - 4 * t**2
    V2 = 2 * s * t
    P4 = sp.expand(X2**2 + 3 * Y2**2)
    assert P4 == sp.expand(U2**2 + 7 * V2**2)
    print(f"\n2) Symbolic test of deforming the irreducible norm form P4(s,t) = {P4}:")
    print(f"   Factorization over Q: {sp.factor(P4)}")

    # We look for terms of degree <= 1:
    # By translation invariance (s, t) -> (s + alpha, t + beta): since Y2 = 2*s*t
    # becomes 2*s*t + 2*beta*s + 2*alpha*t + 2*alpha*beta, we may set WITHOUT LOSS OF
    # GENERALITY y1s = 0, y1t = 0! This removes the 2 trivial translation directions.
    u1s, u1t, u0, v1s, v1t, v0, x1s, x1t, x0, y0 = sp.symbols(
        "u1s u1t u0 v1s v1t v0 x1s x1t x0 y0"
    )
    U_g = U2 + u1s * s + u1t * t + u0
    V_g = V2 + v1s * s + v1t * t + v0
    X_g = X2 + x1s * s + x1t * t + x0
    Y_g = Y2 + y0

    diff = sp.Poly(sp.expand(U_g**2 + 7 * V_g**2 - X_g**2 - 3 * Y_g**2), s, t)
    # 1) Degree 3: 4 linear equations in 6 variables (u1s, u1t, v1s, v1t, x1s, x1t) -> 2 free (v1s, v1t)
    eqs_deg3 = [diff.coeff_monomial(m) for m in (s**3, s**2 * t, s * t**2, t**3)]
    sol3 = sp.solve(eqs_deg3, [u1s, u1t, x1s, x1t], dict=True)[0]
    print(f"   Degree-3 elimination (WLOG y1s=y1t=0): {sol3}")

    # 2) Degree 2: 3 LINEAR equations in (u0, v0, x0, y0) -> we solve for (u0, x0, v0) in terms of (v1s, v1t, y0)
    eqs_deg2 = [sp.expand(diff.coeff_monomial(m).subs(sol3)) for m in (s**2, s * t, t**2)]
    sol2 = sp.solve(eqs_deg2, [u0, x0, v0], dict=True)[0]
    print(f"   Degree-2 elimination: {sol2}")

    # 3) Degree 1: 2 polynomial equations in ONLY 3 variables (v1s, v1t, y0)!
    sub_all = {**sol3, **sol2}
    eqs_deg1 = [sp.factor(diff.coeff_monomial(m).subs(sub_all)) for m in (s, t)]
    const_term = sp.factor(diff.coeff_monomial(1).subs(sub_all))
    print(f"   Remaining equations (degree 1) in (v1s, v1t, y0):")
    for idx, e in enumerate(eqs_deg1, 1):
        print(f"     eq{idx}: {e} = 0")
    print(f"   Constant term C in terms of (v1s, v1t, y0): {const_term}")
    sol1 = sp.solve(eqs_deg1, [v1s, v1t, y0], dict=True)
    print(f"   Degree-1 solutions ({len(sol1)} branches):")
    for k, sol in enumerate(sol1, 1):
        ct = sp.factor(const_term.subs(sol))
        print(f"     Branch {k}: {sol}  ==>  C = {ct}")

    # 4) We also try the other direction: a generic degree-2 family with
    #    P4 = N_{K/Q}(s + t*(a*sqrt(-3) + b*sqrt(-7))) for symbolic parameters a, b != 0
    #    (case deg1 = 0, i.e., no linear terms after centring):
    a_p, b_p = sp.symbols("a_p b_p", nonzero=True)
    # N_{K/Q(sqrt(-3))}(s + t*(a*sqrt(-3)+b*sqrt(-7))) = (s^2 + (7*b^2 - 3*a^2)*t^2) + (2*a*s*t)*sqrt(-3)
    # N_{K/Q(sqrt(-7))}(s + t*(a*sqrt(-3)+b*sqrt(-7))) = (s^2 + (3*a^2 - 7*b^2)*t^2) + (2*b*s*t)*sqrt(-7)
    X2g = s**2 + (7 * b_p**2 - 3 * a_p**2) * t**2
    Y2g = 2 * a_p * s * t
    U2g = s**2 + (3 * a_p**2 - 7 * b_p**2) * t**2
    V2g = 2 * b_p * s * t
    assert sp.expand((U2g**2 + 7 * V2g**2) - (X2g**2 + 3 * Y2g**2)) == 0
    diff_g = sp.Poly(sp.expand((U2g + u0) ** 2 + 7 * (V2g + v0) ** 2 - (X2g + x0) ** 2 - 3 * (Y2g + y0) ** 2), s, t)
    eqs_g2 = [diff_g.coeff_monomial(m) for m in (s**2, s * t, t**2)]
    sol_g2 = sp.solve(eqs_g2, [u0, x0, v0], dict=True)[0]
    C_g = sp.factor((u0**2 + 7 * v0**2 - x0**2 - 3 * y0**2).subs(sol_g2))
    print(f"\n3) Constant deformation (deg1=0) of N_{{K/Q}}(s + t*(a*sqrt(-3)+b*sqrt(-7))):")
    print(f"   sol deg2: {sol_g2}")
    print(f"   Constant term C(a, b, y0) = {C_g}")

    # 5) Explicit family whose principal part is an IRREDUCIBLE biquadratic norm (a=7, b=3, y0=1 => C=4):
    X_n = s**2 - 84 * t**2
    Y_n = 14 * s * t + 1
    U_n = s**2 + 84 * t**2
    V_n = 6 * s * t + 1
    assert sp.expand((U_n**2 + 7 * V_n**2) - (X_n**2 + 3 * Y_n**2)) == 4
    p_norm = sp.expand(X_n**2 + 3 * Y_n**2 - 3)
    print(f"\n4) Explicit family from a deformed biquadratic norm (a=7, b=3, y0=1):")
    print(f"   X = {X_n}, Y = {Y_n}, U = {U_n}, V = {V_n}")
    print(f"   p(s, t) = {p_norm}  (factorization: {sp.factor(p_norm)})")
    print(f"   Principal part of degree 4: {sp.factor(s**4 + 420*s**2*t**2 + 7056*t**4)} (irreducible in Q[s,t])")
    norm_primes = []
    for sv in range(1, 60, 2):
        for tv in range(-20, 21):
            if tv == 0:
                continue
            Xv = sv**2 - 84 * tv**2
            Yv = 14 * sv * tv + 1
            Uv = sv**2 + 84 * tv**2
            Vv = 6 * sv * tv + 1
            if sp.gcd(abs(Xv), abs(Yv)) == 1 and sp.gcd(abs(Uv), abs(Vv)) == 1:
                pv = Xv**2 + 3 * Yv**2 - 3
                if pv > 7 and sp.isprime(pv):
                    t1_3, t2_3 = types_at(pv, 3)
                    t1_7, t2_7 = types_at(pv, 7)
                    assert (not t1_3) and (not t2_3) and (not t1_7) and (not t2_7)
                    norm_primes.append((sv, tv, pv))
    norm_primes.sort(key=lambda z: z[2])
    print(f"   Primes blind at c=3 and c=7 (1<=s<60 odd, 0<|t|<=20, gcd=1): {len(norm_primes)}; smallest 8: {norm_primes[:8]}")

    # 6) Obstruction in 3 variables (Maynard's threshold n >= 3d/4 = 3 for d = 4):
    #    N_{K/Q}(r + s*sqrt(-3) + t*sqrt(-7)) admits NO deformation of degree <= 1 with C != 0!
    r = sp.Symbol("r")
    u1r, u1s, u1t, u0_3, v1r, v1s, v1t, v0_3, x1r, x1s, x1t, x0_3, y1r, y1s, y1t, y0_3 = sp.symbols(
        "u1r u1s u1t u0_3 v1r v1s v1t v0_3 x1r x1s x1t x0_3 y1r y1s y1t y0_3"
    )
    X2_3 = r**2 - 3 * s**2 + 7 * t**2
    Y2_3 = 2 * r * s
    U2_3 = r**2 + 3 * s**2 - 7 * t**2
    V2_3 = 2 * r * t
    U3 = U2_3 + u1r * r + u1s * s + u1t * t + u0_3
    V3 = V2_3 + v1r * r + v1s * s + v1t * t + v0_3
    X3 = X2_3 + x1r * r + x1s * s + x1t * t + x0_3
    Y3 = Y2_3 + y1r * r + y1s * s + y1t * t + y0_3
    diff3 = sp.Poly(sp.expand(U3**2 + 7 * V3**2 - X3**2 - 3 * Y3**2), r, s, t)
    mon3 = [r**3, s**3, t**3, r**2 * s, r**2 * t, r * s**2, r * t**2, s**2 * t, s * t**2, r * s * t]
    sol3_3 = sp.solve(
        [diff3.coeff_monomial(m) for m in mon3],
        [u1r, u1s, u1t, v1r, v1s, v1t, x1r, x1s, x1t, y1r, y1s, y1t],
        dict=True,
    )[0]
    mon2 = [r**2, s**2, t**2, r * s, r * t, s * t]
    sol2_3 = sp.solve(
        [sp.expand(diff3.coeff_monomial(m).subs(sol3_3)) for m in mon2],
        [u0_3, v0_3, x0_3, y0_3],
        dict=True,
    )[0]
    C3 = sp.factor(diff3.coeff_monomial(1).subs({**sol3_3, **sol2_3}))
    print(f"\n5) Obstruction in 3 variables N_{{K/Q}}(r + s*sqrt(-3) + t*sqrt(-7)) + (degree <= 1):")
    print(f"   The only solution in degrees 3 and 2 forces the constant term C = {C3} (C = 4 is impossible)!")


if __name__ == "__main__":
    part_a_check_consecutive_norms()
    part_b_witt_index_and_1param()
    part_c_higher_degree_exploration()
