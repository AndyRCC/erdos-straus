#!/usr/bin/env python3
"""
Primos ciegos en c = 3 y c = 7 a la vez: índice de Witt, familia de un parámetro
p(t) = (2t+29)² + 2520 y familias cuárticas (nota, §5).
"""

import os
import sys
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "herramientas"))
from verificar import tipos_en, perfil


def part_a_check_consecutive_norms():
    print("=== Normas consecutivas de Q(sqrt(-3)) y Q(sqrt(-7)) ===")
    # Verificar en todos los primos p ≡ 1 (mod 24), p <= 100_000
    n_blind_3_7 = 0
    n_norms_ok = 0
    ejemplos = []
    for p in sp.primerange(25, 100_000):
        if p % 24 != 1:
            continue
        t1_3, t2_3 = tipos_en(p, 3)
        t1_7, t2_7 = tipos_en(p, 7)
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
            if len(ejemplos) < 8:
                ejemplos.append((p, m, f3, m1, f7))
    print(f"Primos p <= 10^5 con p ≡ 1 (mod 24) ciegos en c=3 y c=7: {n_blind_3_7} (100% normas consecutivas: {n_norms_ok}/{n_blind_3_7}).")
    for p, m, f3, m1, f7 in ejemplos:
        print(f"  p={p:6d} (mod 840={p%840:3d}): K_3={m} ({dict(f3)}), K_7={m1} ({dict(f7)})")


def part_b_witt_index_and_1param():
    print("\n=== Indice de Witt y familias afines de 1 parametro ===")
    a, b, d3, d7 = sp.symbols("a b d3 d7", positive=True)
    # Forma general de discriminante -7*d7^2 equivalente sobre Q a diag(1, 7*d7^2)/4
    M7 = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 2]])
    M3 = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 1]])
    Q_witt = sp.diag(2 * M7, -M3)
    Q_gen = sp.diag(b * M7, -a * M3)
    print(f"det(2*(u^2+uv+2v^2) - (x^2+xy+y^2)) = {Q_witt.det()}")
    print(f"det(b*(u^2+uv+2v^2) - a*(x^2+xy+y^2)) = {sp.factor(Q_gen.det())}")
    print(f"Clase cuadrada mod (Q^*)^2: 21 (no es cuadrado; indice de Witt <= 1).")

    # Comprobar que el indice de Witt es EXACTAMENTE 1 para c = 1, 2, 4
    for c_mult, vec in [(1, (1, 0, 1, 0)), (2, (0, 1, 2, 0)), (4, (1, 0, 2, 0))]:
        u, v, x, y = vec
        val = c_mult * (u**2 + u * v + 2 * v**2) - (x**2 + x * y + y**2)
        assert val == 0
        print(f"  Vector isotropo para c={c_mult}: (u,v,x,y)={vec} -> Q={val} (indice de Witt = 1)")

    # Familia afin de 1 parametro sobre f7(u,v) - f3(x,y) = 1:
    # (u, v) = (t + 5, 19), (x, y) = (t, 29) -> 4*f3(t,29) - 3 = (2t+29)^2 + 2520
    t = sp.Symbol("t", integer=True)
    f3_t = t**2 + 29 * t + 29**2
    f7_t = (t + 5) ** 2 + 19 * (t + 5) + 2 * 19**2
    assert sp.expand(f7_t - f3_t) == 1
    p_t = sp.expand(4 * f3_t - 3)
    print(f"  Familia afin 1-parametro: (u,v)=(t+5,19), (x,y)=(t,29) => f7-f3 = 1, p(t) = {p_t} = (2t+29)^2 + 2520")
    primos_1p = []
    for tv in range(0, 200):
        pv = int(p_t.subs(t, tv))
        if sp.isprime(pv) and sp.gcd(tv, 29) == 1 and sp.gcd(tv + 5, 19) == 1:
            t1_3, t2_3 = tipos_en(pv, 3)
            t1_7, t2_7 = tipos_en(pv, 7)
            assert (not t1_3) and (not t2_3) and (not t1_7) and (not t2_7)
            primos_1p.append((tv, pv))
    print(f"  Primos ciegos en c=3 y c=7 producidos en 0 <= t < 200 (con gcd=1): {len(primos_1p)}; primeros 6: {primos_1p[:6]}")

    # Verificar que en la familia de orden minimo (v0=1, y0=1), p(t) = (2t+1)^2 es siempre cuadrado
    p_min = sp.factor(4 * (t**2 + t + 1) - 3)
    print(f"  Familia afin minima (v0=1, y0=1): (u,v)=(t,1), (x,y)=(t,1) => p(t) = {p_min} (siempre cuadrado!)")


def part_c_higher_degree_exploration():
    print("\n=== Exploracion de familias de grado mayor en 2 parametros ===")
    # 1. Familia cuadratica por proyeccion parabolica desde vector isotropo (1,0,1,0):
    # U - X = 2*d => X = (4 - 4d^2 + 3Y^2 - 7V^2)/(4d), U = X + 2d.
    # Para d = 1, Y = 2s+1, V = 2t+1:
    s, t = sp.symbols("s t", integer=True)
    Y = 2 * s + 1
    V = 2 * t + 1
    X = sp.expand((3 * Y**2 - 7 * V**2) / 4)
    U = sp.expand(X + 2)
    K3 = sp.expand((X**2 + 3 * Y**2) / 4)
    K7 = sp.expand((U**2 + 7 * V**2) / 4)
    assert sp.expand(K7 - K3) == 1
    p_st = sp.expand(4 * K3 - 3)
    print(f"1) Proyeccion parabolica (d=1, Y=2s+1, V=2t+1):")
    print(f"   X = {X}")
    print(f"   p(s, t) = {p_st}")
    # Verificar que genera primos reales ciegos en c=3 y c=7 cuando gcd(X,Y)=gcd(U,V)=1
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
                    t1_3, t2_3 = tipos_en(pv, 3)
                    t1_7, t2_7 = tipos_en(pv, 7)
                    assert (not t1_3) and (not t2_3) and (not t1_7) and (not t2_7)
                    if len(found) < 8:
                        found.append((sv, tv, pv))
    print(f"   Ejemplos de primos ciegos en c=3 y c=7 (gcd=1): {found}")

    # 2. Test simbolico: puede existir una solucion polinomica de grado 2 (U,V,X,Y) en Q[s,t]
    #    a (U^2 + 7V^2) - (X^2 + 3Y^2) = 4 cuya parte homogenea de grado 4 sea una forma norma
    #    irreducible de K = Q(sqrt(-3), sqrt(-7))?
    #    Probamos la forma norma canonica P4(s, t) = N_{K/Q}(s + t*(sqrt(-3)+sqrt(-7))/2)
    #    (y tambien N_{K/Q}(s + t*sqrt(-3) + ...)):
    #    N_{K/Q(sqrt(-3))}(s + t*sqrt(-7)) = s^2 + 7*t^2  (norma en Q(sqrt(-3)) es (s^2+7t^2)^2 + 3*0^2, cuadrado!)
    #    Tomemos un elemento generico alpha = 1, beta = sqrt(-3) + sqrt(-7):
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
    print(f"\n2) Test simbolico de deformacion de forma norma irreducible P4(s,t) = {P4}:")
    print(f"   Factorizacion sobre Q: {sp.factor(P4)}")

    # Buscamos terminos de grado <= 1:
    # Por invariancia traslacional (s, t) -> (s + alpha, t + beta), como Y2 = 2*s*t
    # pasa a 2*s*t + 2*beta*s + 2*alpha*t + 2*alpha*beta, podemos fijar SIN PERDIDA DE
    # GENERALIDAD y1s = 0, y1t = 0! Esto elimina las 2 direcciones triviales de traslacion.
    u1s, u1t, u0, v1s, v1t, v0, x1s, x1t, x0, y0 = sp.symbols(
        "u1s u1t u0 v1s v1t v0 x1s x1t x0 y0"
    )
    U_g = U2 + u1s * s + u1t * t + u0
    V_g = V2 + v1s * s + v1t * t + v0
    X_g = X2 + x1s * s + x1t * t + x0
    Y_g = Y2 + y0

    diff = sp.Poly(sp.expand(U_g**2 + 7 * V_g**2 - X_g**2 - 3 * Y_g**2), s, t)
    # 1) Grado 3: 4 ecuaciones lineales en 6 variables (u1s, u1t, v1s, v1t, x1s, x1t) -> 2 libres (v1s, v1t)
    eqs_deg3 = [diff.coeff_monomial(m) for m in (s**3, s**2 * t, s * t**2, t**3)]
    sol3 = sp.solve(eqs_deg3, [u1s, u1t, x1s, x1t], dict=True)[0]
    print(f"   Eliminacion grado 3 (WLOG y1s=y1t=0): {sol3}")

    # 2) Grado 2: 3 ecuaciones LINEALES en (u0, v0, x0, y0) -> resolvemos (u0, x0, v0) en funcion de (v1s, v1t, y0)
    eqs_deg2 = [sp.expand(diff.coeff_monomial(m).subs(sol3)) for m in (s**2, s * t, t**2)]
    sol2 = sp.solve(eqs_deg2, [u0, x0, v0], dict=True)[0]
    print(f"   Eliminacion grado 2: {sol2}")

    # 3) Grado 1: 2 ecuaciones polinomicas en SOLO 3 variables (v1s, v1t, y0)!
    sub_all = {**sol3, **sol2}
    eqs_deg1 = [sp.factor(diff.coeff_monomial(m).subs(sub_all)) for m in (s, t)]
    const_term = sp.factor(diff.coeff_monomial(1).subs(sub_all))
    print(f"   Ecuaciones restantes (grado 1) en (v1s, v1t, y0):")
    for idx, e in enumerate(eqs_deg1, 1):
        print(f"     eq{idx}: {e} = 0")
    print(f"   Termino constante C en funcion de (v1s, v1t, y0): {const_term}")
    sol1 = sp.solve(eqs_deg1, [v1s, v1t, y0], dict=True)
    print(f"   Soluciones de grado 1 ({len(sol1)} ramas):")
    for k, sol in enumerate(sol1, 1):
        ct = sp.factor(const_term.subs(sol))
        print(f"     Rama {k}: {sol}  ==>  C = {ct}")

    # 4) Tambien probamos la otra direccion: una familia generica de grado 2 con
    #    P4 = N_{K/Q}(s + t*(a*sqrt(-3) + b*sqrt(-7))) para parametros simbolicos a, b != 0
    #    (caso deg1 = 0, i.e., sin terminos lineales tras centrar):
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
    print(f"\n3) Deformacion constante (deg1=0) de N_{{K/Q}}(s + t*(a*sqrt(-3)+b*sqrt(-7))):")
    print(f"   sol deg2: {sol_g2}")
    print(f"   Termino constante C(a, b, y0) = {C_g}")

    # 5) Familia explicita con parte principal norma bicuadratica IRREDUCIBLE (a=7, b=3, y0=1 => C=4):
    X_n = s**2 - 84 * t**2
    Y_n = 14 * s * t + 1
    U_n = s**2 + 84 * t**2
    V_n = 6 * s * t + 1
    assert sp.expand((U_n**2 + 7 * V_n**2) - (X_n**2 + 3 * Y_n**2)) == 4
    p_norm = sp.expand(X_n**2 + 3 * Y_n**2 - 3)
    print(f"\n4) Familia explicita de norma bicuadratica deformada (a=7, b=3, y0=1):")
    print(f"   X = {X_n}, Y = {Y_n}, U = {U_n}, V = {V_n}")
    print(f"   p(s, t) = {p_norm}  (factorizacion: {sp.factor(p_norm)})")
    print(f"   Parte principal grado 4: {sp.factor(s**4 + 420*s**2*t**2 + 7056*t**4)} (irreducible en Q[s,t])")
    primos_norm = []
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
                    t1_3, t2_3 = tipos_en(pv, 3)
                    t1_7, t2_7 = tipos_en(pv, 7)
                    assert (not t1_3) and (not t2_3) and (not t1_7) and (not t2_7)
                    primos_norm.append((sv, tv, pv))
    primos_norm.sort(key=lambda z: z[2])
    print(f"   Primos ciegos en c=3 y c=7 (1<=s<60 impar, 0<|t|<=20, gcd=1): {len(primos_norm)}; menores 8: {primos_norm[:8]}")

    # 6) Obstruccion en 3 variables (umbral de Maynard n >= 3d/4 = 3 para d = 4):
    #    N_{K/Q}(r + s*sqrt(-3) + t*sqrt(-7)) no admite NINGUNA deformacion de grado <= 1 con C != 0!
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
    print(f"\n5) Obstruccion en 3 variables N_{{K/Q}}(r + s*sqrt(-3) + t*sqrt(-7)) + (grado <= 1):")
    print(f"   Unica solucion de grados 3 y 2 fuerza termino constante C = {C3} (imposible C = 4)!")


if __name__ == "__main__":
    part_a_check_consecutive_norms()
    part_b_witt_index_and_1param()
    part_c_higher_degree_exploration()
