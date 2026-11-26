#!/usr/bin/env python3
"""
Herramienta de verificación del proyecto Erdős–Straus.

    4/n = 1/x + 1/y + 1/z      con x, y, z enteros positivos
    (equivalente:  4xyz = n(xy + yz + zx))

Toda comprobación usa aritmética entera exacta (nunca floats).

Subcomandos
-----------
  sol N X Y Z
      Verifica una solución concreta.

  buscar N
      Busca una solución para N. La búsqueda es EXHAUSTIVA: si dice que no
      hay solución, es que no la hay (criterio exacto: A/B = 1/y + 1/z con
      A/B irreducible ⟺ existen divisores u, v de B con A | u + v).

  rango A B [--todos]
      Verifica que cada primo de [A, B] tiene solución (los compuestos se
      deducen de sus factores primos; con --todos se comprueban todos los n).

  identidad "n(t)" "x(t)" "y(t)" "z(t)" [--tmin 0] [--tmax 2000]
      Verifica una identidad paramétrica en t: primero simbólicamente
      (sympy) y luego que x, y, z son enteros positivos para cada t del rango.
      Solo se aceptan expresiones con dígitos, t, + - * / ^ ( ) y espacios.

  conocidas
      Autotest: verifica las identidades clásicas y las de Mordell.

  mod840
      Calcula los cuadrados invertibles módulo 840 y comprueba que son
      exactamente las clases que no cubren las identidades de 'conocidas'.

  perfil P
      Para P primo impar: m(P) = 4·x_min − P, qué tipos (I/II) existen en
      x_min y el menor m con solución de cada tipo.
"""

import argparse
import re
import sys
import time
from math import gcd

try:
    import sympy
    from sympy import factorint, primerange
    from sympy.ntheory.residue_ntheory import is_quad_residue
except ImportError:  # pragma: no cover
    sys.exit("Falta sympy:  pip install sympy")


# ---------------------------------------------------------------------------
# Núcleo
# ---------------------------------------------------------------------------

def es_solucion(n, x, y, z):
    """Comprobación exacta de 4/n = 1/x + 1/y + 1/z."""
    return (n >= 2 and x > 0 and y > 0 and z > 0
            and 4 * x * y * z == n * (x * y + y * z + z * x))


def _divisores(fact):
    divs = [1]
    for p, e in fact.items():
        divs = [d * p ** k for d in divs for k in range(e + 1)]
    return divs


def dos_unitarias(A, B, fact_B):
    """
    Para A/B irreducible devuelve (y, z) con A/B = 1/y + 1/z, o None.

    Criterio exacto: existe solución si y solo si hay
    divisores u, v de B con A | u + v.  Entonces k = (u+v)/A,
    y = Bk/u, z = Bk/v.
    """
    divs = _divisores(fact_B)
    por_residuo = {}
    for d in divs:
        por_residuo.setdefault(d % A, d)
    for u in divs:
        v = por_residuo.get((-u) % A)
        if v is not None:
            k = (u + v) // A
            return B * k // u, B * k // v
    return None


def buscar(n):
    """
    Devuelve (x, y, z) con 4/n = 1/x+1/y+1/z, o None si NO existe ninguna.

    Si x es el menor denominador:  n/4 < x <= 3n/4.  Para cada x se aplica el
    criterio exacto de dos fracciones unitarias, así que recorrer ese rango
    completo es una decisión exhaustiva.
    """
    if n < 2:
        raise ValueError("n debe ser >= 2")
    fact_n = factorint(n)
    for x in range(n // 4 + 1, (3 * n) // 4 + 1):
        A, B = 4 * x - n, n * x
        g = gcd(A, B)
        A //= g
        B //= g
        fB = dict(fact_n)
        for p, e in factorint(x).items():
            fB[p] = fB.get(p, 0) + e
        for p in list(fB):           # quitar de fB los primos de g
            while g % p == 0:
                g //= p
                fB[p] -= 1
            if fB[p] == 0:
                del fB[p]
        r = dos_unitarias(A, B, fB)
        if r is not None:
            sol = tuple(sorted((x, r[0], r[1])))
            assert es_solucion(n, *sol), (n, sol)
            return sol
    return None


def tipo(p, sol):
    """
    Tipo I / II (Elsholtz–Tao) de UNA solución concreta, para p primo impar.
    Ojo: el tipo de la solución que devuelve buscar() depende del orden en que
    se prueban los divisores; para saber qué tipos EXISTEN usa tipos_en().
    """
    k = sum(1 for d in sol if d % p == 0)
    return {1: "I", 2: "II"}.get(k, f"?({k})")


def tipos_en(p, m):
    """
    Para p primo impar y x = (p+m)/4 < p: devuelve (hay_tipo_I, hay_tipo_II).
    Criterio exacto:
      tipo II  <=>  existen d1, d2 | x con  d1 + d2   ≡ 0 (mod m)
      tipo I   <=>  existen d1, d2 | x con  p*d1 + d2 ≡ 0 (mod m)
    """
    x = (p + m) // 4
    assert 4 * x - p == m and 0 < m and x < p
    D = {d % m for d in sympy.divisors(x)}
    return (any((-p * d) % m in D for d in D), any((-d) % m in D for d in D))


def perfil(p, cap=10 ** 4):
    """m(p) = 4·x_min − p, tipos en x_min, y menor m con tipo I / tipo II."""
    m = (-p) % 4 or 4
    m_min = tipos_min = c1 = c2 = None
    while (c1 is None or c2 is None) and m <= cap:
        t1, t2 = tipos_en(p, m)
        if (t1 or t2) and m_min is None:
            m_min, tipos_min = m, (t1, t2)
        if t1 and c1 is None:
            c1 = m
        if t2 and c2 is None:
            c2 = m
        m += 4
    return m_min, tipos_min, c1, c2


# ---------------------------------------------------------------------------
# Identidades paramétricas
# ---------------------------------------------------------------------------

_PERMITIDO = re.compile(r"^[0-9t+\-*/^() ]+$")
t = sympy.Symbol("t", integer=True)


def _parsear(s):
    if not _PERMITIDO.match(s):
        raise ValueError(f"Expresión no permitida: {s!r}")
    return sympy.sympify(s.replace("^", "**"), locals={"t": t})


def verificar_identidad(ns, xs, ys, zs, tmin=0, tmax=2000, silencioso=False):
    n_e, x_e, y_e, z_e = (_parsear(s) for s in (ns, xs, ys, zs))
    out = []
    diff = sympy.cancel(sympy.together(4 / n_e - (1 / x_e + 1 / y_e + 1 / z_e)))
    simb_ok = diff == 0
    out.append(f"  Identidad simbólica: {'OK' if simb_ok else 'FALLA  (diferencia = ' + str(diff) + ')'}")

    fallos = []
    for tv in range(tmin, tmax + 1):
        vals = [e.subs(t, tv) for e in (n_e, x_e, y_e, z_e)]
        if not all(v.is_Integer for v in vals):
            fallos.append((tv, "no entero", vals))
        elif not es_solucion(*(int(v) for v in vals)):
            fallos.append((tv, "no positivo o no cumple la ecuación", vals))
        if len(fallos) >= 5:
            break
    num_ok = not fallos
    out.append(f"  Comprobación numérica t = {tmin}..{tmax}: "
               + ("OK" if num_ok else f"FALLA, primeros casos: {fallos}"))

    poly = sympy.Poly(n_e, t)
    if poly.degree() == 1 and all(c.is_Integer for c in poly.all_coeffs()):
        a, b = (int(c) for c in poly.all_coeffs())
        a = abs(a)
        b %= a
        out.append(f"  Cubre la progresión n ≡ {b} (mod {a}).")
        if gcd(a, b) == 1 and is_quad_residue(b, a):
            out.append("  ⚠ ATENCIÓN: esa clase es un residuo cuadrático. Según la obstrucción\n"
                       "    de Mordell–Schinzel (Elsholtz–Tao, Prop. 1.6) no debería existir una\n"
                       "    identidad polinomial así. Revisar con muchísimo cuidado.")
    if not silencioso:
        print("\n".join(out))
    return simb_ok and num_ok


IDENTIDADES_CONOCIDAS = [
    ("n ≡ 3 (mod 4)", "4*t+3", "t+2", "(t+1)*(t+2)", "(4*t+3)*(t+1)"),
    ("n ≡ 2 (mod 3)", "3*t+2", "3*t+2", "t+1", "(3*t+2)*(t+1)"),
    ("n ≡ 5 (mod 8)", "8*t+5", "2*t+2", "(8*t+5)*(2*t+2)", "(8*t+5)*(t+1)"),
    # Identidades de Mordell (clases módulo 5 y módulo 7)
    ("n ≡ 17 (mod 20)", "20*t+17", "5*(t+1)", "2*(t+1)*(20*t+17)", "10*(t+1)*(20*t+17)"),
    ("n ≡ 13 (mod 20)", "20*t+13", "5*(t+1)", "5*(3*t+2)*(t+1)", "5*(3*t+2)*(20*t+13)"),
    ("n ≡ 3 (mod 7)", "7*t+3", "2*t+1", "2*(7*t+3)", "2*(2*t+1)*(7*t+3)"),
    ("n ≡ 6 (mod 7)", "7*t+6", "2*(t+1)", "2*(7*t+6)", "2*(t+1)*(7*t+6)"),
    ("n ≡ 33 (mod 56)", "56*t+33", "2*(7*t+5)", "2*(8*t+5)*(7*t+5)", "(8*t+5)*(7*t+5)*(56*t+33)"),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    try:  # la consola de Windows a veces no es UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Verificador Erdős–Straus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sol"); s.add_argument("nums", type=int, nargs=4)
    s = sub.add_parser("buscar"); s.add_argument("n", type=int)
    s = sub.add_parser("rango"); s.add_argument("a", type=int); s.add_argument("b", type=int)
    s.add_argument("--todos", action="store_true")
    s = sub.add_parser("identidad"); s.add_argument("expr", nargs=4)
    s.add_argument("--tmin", type=int, default=0); s.add_argument("--tmax", type=int, default=2000)
    sub.add_parser("conocidas")
    sub.add_parser("mod840")
    s = sub.add_parser("perfil"); s.add_argument("p", type=int)
    a = ap.parse_args()

    if a.cmd == "sol":
        n, x, y, z = a.nums
        ok = es_solucion(n, x, y, z)
        print(f"4/{n} = 1/{x} + 1/{y} + 1/{z}:  {'CORRECTO' if ok else 'INCORRECTO'}")
        sys.exit(0 if ok else 1)

    if a.cmd == "buscar":
        sol = buscar(a.n)
        if sol is None:
            print(f"*** n = {a.n}: NO EXISTE SOLUCIÓN (búsqueda exhaustiva). ¡Contraejemplo! Revisar a mano. ***")
            sys.exit(2)
        x, y, z = sol
        extra = ""
        if sympy.isprime(a.n) and a.n > 2:
            extra = f"   [tipo {tipo(a.n, sol)}, m = 4x - n = {4 * x - a.n}]"
        print(f"4/{a.n} = 1/{x} + 1/{y} + 1/{z}{extra}")
        return

    if a.cmd == "rango":
        t0 = time.time()
        candidatos = range(max(2, a.a), a.b + 1) if a.todos else primerange(max(2, a.a), a.b + 1)
        cuenta = 0
        for n in candidatos:
            if buscar(n) is None:
                print(f"*** n = {n}: NO EXISTE SOLUCIÓN. ¡Contraejemplo! ***")
                sys.exit(2)
            cuenta += 1
        que = "enteros" if a.todos else "primos"
        print(f"OK: los {cuenta} {que} de [{a.a}, {a.b}] tienen solución ({time.time() - t0:.1f} s).")
        return

    if a.cmd == "identidad":
        try:
            ok = verificar_identidad(*a.expr, tmin=a.tmin, tmax=a.tmax)
        except (ValueError, sympy.SympifyError) as e:
            sys.exit(f"Error: {e}")
        print("RESULTADO:", "VÁLIDA" if ok else "NO VÁLIDA")
        sys.exit(0 if ok else 1)

    if a.cmd == "conocidas":
        todo = True
        for nombre, *e in IDENTIDADES_CONOCIDAS:
            print(f"{nombre}:  n={e[0]},  x={e[1]},  y={e[2]},  z={e[3]}")
            todo &= verificar_identidad(*e, tmax=500)
        print("RESULTADO:", "todas válidas" if todo else "HAY FALLOS")
        sys.exit(0 if todo else 1)

    if a.cmd == "perfil":
        p = a.p
        if not (sympy.isprime(p) and p > 2):
            sys.exit("perfil necesita un primo impar")
        m, (t1, t2), c1, c2 = perfil(p)
        tipos = " y ".join(n for n, b in (("I", t1), ("II", t2)) if b)
        print(f"p = {p} (mod 840 = {p % 840}): m(p) = {m}, x_min = {(p + m) // 4}, "
              f"tipos en x_min: {tipos};  menor m con tipo I: {c1}, con tipo II: {c2}")
        return

    if a.cmd == "mod840":
        m = 840
        cuad = sorted({(s * s) % m for s in range(m) if gcd(s, m) == 1})
        print(f"Cuadrados invertibles módulo {m} ({len(cuad)}): {cuad}")
        progresiones = []
        for nombre, ns, *_ in IDENTIDADES_CONOCIDAS:
            q, b = (int(c) for c in sympy.Poly(_parsear(ns), t).all_coeffs())
            progresiones.append((q, b % q))
        sin = [r for r in range(m) if gcd(r, m) == 1 and not any(r % q == b for q, b in progresiones)]
        print(f"Clases invertibles NO cubiertas por las {len(progresiones)} identidades de 'conocidas': {sin}")
        print("Coinciden con los cuadrados." if sin == cuad else "*** NO coinciden con los cuadrados ***")


if __name__ == "__main__":
    main()
