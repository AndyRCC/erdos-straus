#!/usr/bin/env python3
"""
Verification tool for the Erdős–Straus project.

    4/n = 1/x + 1/y + 1/z      with x, y, z positive integers
    (equivalently:  4xyz = n(xy + yz + zx))

Every check uses exact integer arithmetic (never floats).

Subcommands
-----------
  check N X Y Z
      Checks one given solution.

  search N
      Searches for a solution for N. The search is EXHAUSTIVE: if it says there
      is no solution, there is none (exact criterion: A/B = 1/y + 1/z with
      A/B in lowest terms ⟺ there are divisors u, v of B with A | u + v).

  range A B [--all-integers]
      Checks that every prime in [A, B] has a solution (composites follow
      from their prime factors; with --all-integers every n is checked).

  identity "n(t)" "x(t)" "y(t)" "z(t)" [--tmin 0] [--tmax 2000]
      Checks a parametric identity in t: first symbolically (sympy), then
      that x, y, z are positive integers for every t in the range.
      Only expressions made of digits, t, + - * / ^ ( ) and spaces are accepted.

  known
      Self-test: checks the classical identities and Mordell's identities.

  mod840
      Computes the invertible squares modulo 840 and checks that they are
      exactly the classes not covered by the identities of 'known'.

  profile P
      For an odd prime P: m(P) = 4·x_min − P, which types (I/II) exist at
      x_min, and the smallest m with a solution of each type.
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
    sys.exit("sympy is missing:  pip install sympy")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def is_solution(n, x, y, z):
    """Exact check of 4/n = 1/x + 1/y + 1/z."""
    return (n >= 2 and x > 0 and y > 0 and z > 0
            and 4 * x * y * z == n * (x * y + y * z + z * x))


def _divisors(fact):
    divs = [1]
    for p, e in fact.items():
        divs = [d * p ** k for d in divs for k in range(e + 1)]
    return divs


def two_unit_fractions(A, B, fact_B):
    """
    For A/B in lowest terms, returns (y, z) with A/B = 1/y + 1/z, or None.

    Exact criterion: there is a solution if and only if there are
    divisors u, v of B with A | u + v.  Then k = (u+v)/A,
    y = Bk/u, z = Bk/v.
    """
    divs = _divisors(fact_B)
    by_residue = {}
    for d in divs:
        by_residue.setdefault(d % A, d)
    for u in divs:
        v = by_residue.get((-u) % A)
        if v is not None:
            k = (u + v) // A
            return B * k // u, B * k // v
    return None


def search(n):
    """
    Returns (x, y, z) with 4/n = 1/x+1/y+1/z, or None if there is NONE.

    If x is the smallest denominator:  n/4 < x <= 3n/4.  For each x the exact
    two-unit-fraction criterion is applied, so running through that whole
    range is an exhaustive decision.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    fact_n = factorint(n)
    for x in range(n // 4 + 1, (3 * n) // 4 + 1):
        A, B = 4 * x - n, n * x
        g = gcd(A, B)
        A //= g
        B //= g
        fB = dict(fact_n)
        for p, e in factorint(x).items():
            fB[p] = fB.get(p, 0) + e
        for p in list(fB):           # remove from fB the primes of g
            while g % p == 0:
                g //= p
                fB[p] -= 1
            if fB[p] == 0:
                del fB[p]
        r = two_unit_fractions(A, B, fB)
        if r is not None:
            sol = tuple(sorted((x, r[0], r[1])))
            assert is_solution(n, *sol), (n, sol)
            return sol
    return None


def sol_type(p, sol):
    """
    Type I / II (Elsholtz–Tao) of ONE given solution, for p an odd prime.
    Note: the type of the solution returned by search() depends on the order in
    which divisors are tried; to know which types EXIST, use types_at().
    """
    k = sum(1 for d in sol if d % p == 0)
    return {1: "I", 2: "II"}.get(k, f"?({k})")


def types_at(p, m):
    """
    For p an odd prime and x = (p+m)/4 < p: returns (has_type_I, has_type_II).
    Exact criterion:
      type II  <=>  there are d1, d2 | x with  d1 + d2   ≡ 0 (mod m)
      type I   <=>  there are d1, d2 | x with  p*d1 + d2 ≡ 0 (mod m)
    """
    x = (p + m) // 4
    assert 4 * x - p == m and 0 < m and x < p
    D = {d % m for d in sympy.divisors(x)}
    return (any((-p * d) % m in D for d in D), any((-d) % m in D for d in D))


def profile(p, cap=10 ** 4):
    """m(p) = 4·x_min − p, types at x_min, and smallest m with type I / type II."""
    m = (-p) % 4 or 4
    m_min = types_min = c1 = c2 = None
    while (c1 is None or c2 is None) and m <= cap:
        t1, t2 = types_at(p, m)
        if (t1 or t2) and m_min is None:
            m_min, types_min = m, (t1, t2)
        if t1 and c1 is None:
            c1 = m
        if t2 and c2 is None:
            c2 = m
        m += 4
    return m_min, types_min, c1, c2


# ---------------------------------------------------------------------------
# Parametric identities
# ---------------------------------------------------------------------------

_ALLOWED = re.compile(r"^[0-9t+\-*/^() ]+$")
t = sympy.Symbol("t", integer=True)


def _parse(s):
    if not _ALLOWED.match(s):
        raise ValueError(f"Expression not allowed: {s!r}")
    return sympy.sympify(s.replace("^", "**"), locals={"t": t})


def verify_identity(ns, xs, ys, zs, tmin=0, tmax=2000, quiet=False):
    n_e, x_e, y_e, z_e = (_parse(s) for s in (ns, xs, ys, zs))
    out = []
    diff = sympy.cancel(sympy.together(4 / n_e - (1 / x_e + 1 / y_e + 1 / z_e)))
    symb_ok = diff == 0
    out.append(f"  Symbolic identity: {'OK' if symb_ok else 'FAILS  (difference = ' + str(diff) + ')'}")

    failures = []
    for tv in range(tmin, tmax + 1):
        vals = [e.subs(t, tv) for e in (n_e, x_e, y_e, z_e)]
        if not all(v.is_Integer for v in vals):
            failures.append((tv, "not an integer", vals))
        elif not is_solution(*(int(v) for v in vals)):
            failures.append((tv, "not positive or does not satisfy the equation", vals))
        if len(failures) >= 5:
            break
    num_ok = not failures
    out.append(f"  Numerical check t = {tmin}..{tmax}: "
               + ("OK" if num_ok else f"FAILS, first cases: {failures}"))

    poly = sympy.Poly(n_e, t)
    if poly.degree() == 1 and all(c.is_Integer for c in poly.all_coeffs()):
        a, b = (int(c) for c in poly.all_coeffs())
        a = abs(a)
        b %= a
        out.append(f"  Covers the progression n ≡ {b} (mod {a}).")
        if gcd(a, b) == 1 and is_quad_residue(b, a):
            out.append("  ⚠ WARNING: that class is a quadratic residue. By the Mordell–Schinzel\n"
                       "    obstruction (Elsholtz–Tao, Prop. 1.6) no such polynomial identity\n"
                       "    should exist. Check very carefully.")
    if not quiet:
        print("\n".join(out))
    return symb_ok and num_ok


KNOWN_IDENTITIES = [
    ("n ≡ 3 (mod 4)", "4*t+3", "t+2", "(t+1)*(t+2)", "(4*t+3)*(t+1)"),
    ("n ≡ 2 (mod 3)", "3*t+2", "3*t+2", "t+1", "(3*t+2)*(t+1)"),
    ("n ≡ 5 (mod 8)", "8*t+5", "2*t+2", "(8*t+5)*(2*t+2)", "(8*t+5)*(t+1)"),
    # Mordell's identities (classes modulo 5 and modulo 7)
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
    try:  # the Windows console is sometimes not UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Erdős–Straus verifier")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check"); s.add_argument("nums", type=int, nargs=4)
    s = sub.add_parser("search"); s.add_argument("n", type=int)
    s = sub.add_parser("range"); s.add_argument("a", type=int); s.add_argument("b", type=int)
    s.add_argument("--all-integers", action="store_true")
    s = sub.add_parser("identity"); s.add_argument("expr", nargs=4)
    s.add_argument("--tmin", type=int, default=0); s.add_argument("--tmax", type=int, default=2000)
    sub.add_parser("known")
    sub.add_parser("mod840")
    s = sub.add_parser("profile"); s.add_argument("p", type=int)
    a = ap.parse_args()

    if a.cmd == "check":
        n, x, y, z = a.nums
        ok = is_solution(n, x, y, z)
        print(f"4/{n} = 1/{x} + 1/{y} + 1/{z}:  {'CORRECT' if ok else 'INCORRECT'}")
        sys.exit(0 if ok else 1)

    if a.cmd == "search":
        sol = search(a.n)
        if sol is None:
            print(f"*** n = {a.n}: NO SOLUTION EXISTS (exhaustive search). Counterexample! Check by hand. ***")
            sys.exit(2)
        x, y, z = sol
        extra = ""
        if sympy.isprime(a.n) and a.n > 2:
            extra = f"   [type {sol_type(a.n, sol)}, m = 4x - n = {4 * x - a.n}]"
        print(f"4/{a.n} = 1/{x} + 1/{y} + 1/{z}{extra}")
        return

    if a.cmd == "range":
        t0 = time.time()
        candidates = range(max(2, a.a), a.b + 1) if a.all_integers else primerange(max(2, a.a), a.b + 1)
        tally = 0
        for n in candidates:
            if search(n) is None:
                print(f"*** n = {n}: NO SOLUTION EXISTS. Counterexample! ***")
                sys.exit(2)
            tally += 1
        what = "integers" if a.all_integers else "primes"
        print(f"OK: all {tally} {what} in [{a.a}, {a.b}] have a solution ({time.time() - t0:.1f} s).")
        return

    if a.cmd == "identity":
        try:
            ok = verify_identity(*a.expr, tmin=a.tmin, tmax=a.tmax)
        except (ValueError, sympy.SympifyError) as e:
            sys.exit(f"Error: {e}")
        print("RESULT:", "VALID" if ok else "NOT VALID")
        sys.exit(0 if ok else 1)

    if a.cmd == "known":
        everything = True
        for name, *e in KNOWN_IDENTITIES:
            print(f"{name}:  n={e[0]},  x={e[1]},  y={e[2]},  z={e[3]}")
            everything &= verify_identity(*e, tmax=500)
        print("RESULT:", "all valid" if everything else "THERE ARE FAILURES")
        sys.exit(0 if everything else 1)

    if a.cmd == "profile":
        p = a.p
        if not (sympy.isprime(p) and p > 2):
            sys.exit("profile needs an odd prime")
        m, (t1, t2), c1, c2 = profile(p)
        types = " and ".join(n for n, b in (("I", t1), ("II", t2)) if b)
        print(f"p = {p} (mod 840 = {p % 840}): m(p) = {m}, x_min = {(p + m) // 4}, "
              f"types at x_min: {types};  smallest m with type I: {c1}, with type II: {c2}")
        return

    if a.cmd == "mod840":
        m = 840
        squares = sorted({(s * s) % m for s in range(m) if gcd(s, m) == 1})
        print(f"Invertible squares modulo {m} ({len(squares)}): {squares}")
        progressions = []
        for name, ns, *_ in KNOWN_IDENTITIES:
            q, b = (int(c) for c in sympy.Poly(_parse(ns), t).all_coeffs())
            progressions.append((q, b % q))
        missing = [r for r in range(m) if gcd(r, m) == 1 and not any(r % q == b for q, b in progressions)]
        print(f"Invertible classes NOT covered by the {len(progressions)} identities of 'known': {missing}")
        print("They coincide with the squares." if missing == squares else "*** They do NOT coincide with the squares ***")


if __name__ == "__main__":
    main()
