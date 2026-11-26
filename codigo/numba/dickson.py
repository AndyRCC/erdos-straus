#!/usr/bin/env python3
"""
Primos explícitos de la construcción condicional a Dickson (nota, §5.1): p = L·t + 1 y
q_j = 1 + (L/4j)·t todos primos implica m(p) > 4J − 1. Criba segmentada, m(p) exacto con
herramientas/verificar.tipos_en y comparación con Bateman–Horn. Tabla en datos/dickson_primos.csv.
"""

import csv
import math
import os
import sys
import time
import numpy as np
from numba import njit, prange
import sympy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "herramientas"))
from verificar import tipos_en, es_solucion


# ---------------------------------------------------------------------------
# 1. Construccion de L_basica(J) y L_opt(J) + constante exacta Bateman-Horn
# ---------------------------------------------------------------------------

def build_L(J, mode="basica"):
    C = 4 * J - 1
    lcm_J = 1
    for j in range(1, J + 1):
        lcm_J = int(sympy.ilcm(lcm_J, j))
    if mode == "basica":
        prim = 1
        for p in sympy.primerange(2, C + 1):
            prim *= int(p)
        return 4 * lcm_J * prim
    else:
        # L minimo valido para el Lema A:
        # - 4*j | L para j = 1..J y 2 | L/(4j) (i.e. v2(L) >= 3 + floor(log2 J))
        # - c0(4j-1) | L para j = 1..J (como gcd(j, c0)=1, c0 | L/(4j) automatico)
        L = 8 * lcm_J
        for j in range(1, J + 1):
            c = 4 * j - 1
            c0 = 1
            for q, e in sympy.factorint(c).items():
                if e % 2 == 1:
                    c0 *= int(q)
            L = int(sympy.ilcm(L, c0))
        k2 = J.bit_length() - 1
        pow2 = 1 << (3 + k2)
        if L % pow2 != 0:
            L = int(sympy.ilcm(L, pow2))
        # Incluir factores primos pequenos r <= 17 si aumentan la densidad Bateman-Horn
        for r in [3, 5, 7, 11, 13, 17]:
            def om(Ltest):
                roots = set()
                for a in [Ltest] + [Ltest // (4 * j) for j in range(1, J + 1)]:
                    if a % r != 0:
                        roots.add((-pow(a % r, -1, r)) % r)
                return len(roots)
            w0 = om(L)
            if w0 > 0:
                w1 = om(L * r)
                gain = (1.0 - w1 / r) / (1.0 - w0 / r)
                logp = math.log(L) + 18.0
                cost = (logp / (logp + math.log(r))) ** (J + 1)
                if gain * cost > 1.0:
                    L *= r
        return L


def bateman_horn_prediction(J, L, T, p_limit=100_000):
    """Calcula la constante S(J, L) y la integral exacta de Bateman-Horn en [1, T]."""
    a_list = [L] + [L // (4 * j) for j in range(1, J + 1)]
    k = J + 1
    S = 1.0
    for p in sympy.primerange(2, p_limit):
        p = int(p)
        roots = set()
        for a in a_list:
            am = a % p
            if am != 0:
                roots.add((-pow(am, -1, p)) % p)
        w = len(roots)
        assert w < p, (J, p, w)
        S *= (1.0 - w / p) / ((1.0 - 1.0 / p) ** k)

    # Integracion numerica de Simpson sobre [1, T] en escala logaritmica (u = log t)
    if T <= 1:
        return S, 0.0
    n_steps = 2000
    u_max = math.log(T)
    du = u_max / n_steps
    log_a = [math.log(a) for a in a_list]
    acc = 0.0
    for i in range(n_steps + 1):
        u = i * du
        t_val = math.exp(u)
        denom = 1.0
        for la in log_a:
            denom *= (la + u)
        val = t_val / denom  # dt = exp(u) du
        w_s = 1.0 if (i == 0 or i == n_steps) else (4.0 if (i & 1) else 2.0)
        acc += w_s * val
    integral = acc * du / 3.0
    return S, S * integral


# ---------------------------------------------------------------------------
# 2. Criba segmentada paralela en t (Numba)
# ---------------------------------------------------------------------------

def build_sieve_table(a_list, p_max):
    """
    Para cada primo p <= p_max construye la lista de raices prohibidas t ≡ r (mod p)
    tales que a_i * t + 1 ≡ 0 (mod p) para algun i.
    Devuelve (primes_arr, offsets_arr, roots_arr).
    """
    primes = []
    offsets = [0]
    roots = []
    for p in sympy.primerange(2, p_max + 1):
        p = int(p)
        rset = set()
        for a in a_list:
            am = a % p
            if am != 0:
                rset.add((-pow(am, -1, p)) % p)
        if rset:
            primes.append(p)
            for r in sorted(rset):
                roots.append(r)
            offsets.append(len(roots))
    return (
        np.array(primes, dtype=np.int64),
        np.array(offsets, dtype=np.int64),
        np.array(roots, dtype=np.int64),
    )


@njit
def sieve_segment(t_start, seg_len, primes, offsets, roots):
    """
    Criba el bloque [t_start, t_start + seg_len) devolviendo los t supervivientes.
    """
    mark = np.zeros(seg_len, dtype=np.uint8)
    n_p = len(primes)
    for pi in range(n_p):
        p = primes[pi]
        lo = offsets[pi]
        hi = offsets[pi + 1]
        base = t_start % p
        for ri in range(lo, hi):
            r = roots[ri]
            off = r - base
            if off < 0:
                off += p
            pos = off
            while pos < seg_len:
                mark[pos] = 1
                pos += p
    cnt = 0
    for i in range(seg_len):
        if mark[i] == 0:
            cnt += 1
    out = np.empty(cnt, dtype=np.int64)
    k = 0
    for i in range(seg_len):
        if mark[i] == 0:
            out[k] = t_start + i
            k += 1
    return out


@njit(parallel=True)
def sieve_range_parallel(t_lo, t_hi, seg_len, primes, offsets, roots, max_out_per_seg):
    """
    Criba [t_lo, t_hi) en paralelo por segmentos de tamano seg_len.
    """
    n_segs = (t_hi - t_lo + seg_len - 1) // seg_len
    buf = np.empty((n_segs, max_out_per_seg), dtype=np.int64)
    counts = np.zeros(n_segs, dtype=np.int64)
    overflow = np.zeros(n_segs, dtype=np.int8)

    for si in prange(n_segs):
        ts = t_lo + si * seg_len
        sl = seg_len
        if ts + sl > t_hi:
            sl = t_hi - ts
        surv = sieve_segment(ts, sl, primes, offsets, roots)
        c = len(surv)
        if c > max_out_per_seg:
            overflow[si] = 1
            c = max_out_per_seg
        counts[si] = c
        for j in range(c):
            buf[si, j] = surv[j]

    return buf, counts, overflow


def exact_m_of_prime(p, c_min_guaranteed):
    """
    Calcula el m(p) EXACTO con herramientas/verificar.tipos_en, empezando en c = 3
    para verificar independientemente que TODOS los c <= c_min_guaranteed fallan en ambas ramas!
    """
    # 1. Confirmar ceguera en todos los c <= c_min_guaranteed
    for c in range(3, c_min_guaranteed + 1, 4):
        t1, t2 = tipos_en(p, c)
        assert (not t1) and (not t2), f"FALLO CEGUERA: p={p}, c={c}, tipos={(t1, t2)}"
    # 2. Hallar el primer c > c_min_guaranteed con solucion
    c = c_min_guaranteed + 4
    while True:
        t1, t2 = tipos_en(p, c)
        if t1 or t2:
            return c, (t1, t2)
        c += 4


def search_dickson(J, mode, T_max, p_sieve_max=6000, seg_len=1_000_000, batch_segs=64, max_hits_show=5):
    t0 = time.time()
    C = 4 * J - 1
    L = build_L(J, mode=mode)
    a_list = [L] + [L // (4 * j) for j in range(1, J + 1)]
    S, bh_exp = bateman_horn_prediction(J, L, T_max)
    primes_s, offsets_s, roots_s = build_sieve_table(a_list, p_sieve_max)

    # Estimar tasa de supervivencia tras criba hasta p_sieve_max
    surv_rate = 1.0
    for pi in range(len(primes_s)):
        w = offsets_s[pi + 1] - offsets_s[pi]
        surv_rate *= (1.0 - w / primes_s[pi])
    max_out = max(1000, int(seg_len * surv_rate * 2.5) + 500)

    # Para rechazo ultrarrapido en Python, ordenar formas de menor a mayor coeficiente (q_J primero)
    a_check = sorted(a_list)

    hits = []
    t_cur = 1
    total_surv = 0
    while t_cur <= T_max:
        t_end = min(T_max + 1, t_cur + batch_segs * seg_len)
        buf, counts, overflow = sieve_range_parallel(
            t_cur, t_end, seg_len, primes_s, offsets_s, roots_s, max_out
        )
        assert np.all(overflow == 0), f"Overflow en criba J={J}!"
        for si in range(len(counts)):
            c_n = int(counts[si])
            total_surv += c_n
            row = buf[si]
            for idx in range(c_n):
                tv = int(row[idx])
                ok = True
                for a in a_check:
                    cand = a * tv + 1
                    # Fermat base 2 rapido antes de sympy.isprime
                    if pow(2, cand - 1, cand) != 1:
                        ok = False
                        break
                if ok:
                    if all(sympy.isprime(a * tv + 1) for a in a_check):
                        hits.append(tv)
        t_cur = t_end

    elapsed = time.time() - t0
    print(f"\nJ = {J:2d} (C = {C:2d}, modo = {mode:6s}, L = {L}, digitos(L) = {len(str(L))})", flush=True)
    print(f"  T_max = {T_max:,}, p_sieve <= {p_sieve_max}, supervivientes = {total_surv:,} ({elapsed:.2f} s)", flush=True)
    print(f"  Bateman-Horn: S = {S:.4e}, esperados en [1, T_max] = {bh_exp:.2f}, encontrados = {len(hits)}", flush=True)

    detailed = []
    for tv in hits[:max_hits_show]:
        p = L * tv + 1
        digs = len(str(p))
        # Si p <= 10^26 podemos calcular m(p) EXACTO con tipos_en rapidamente!
        t_m = time.time()
        if digs <= 26:
            m_val, tipos_m = exact_m_of_prime(p, C)
            exact_str = f"EXACTO m(p) = {m_val} (tipos={tipos_m}, {time.time()-t_m:.2f}s)"
        else:
            m_val, tipos_m = None, None
            exact_str = f"garantizado m(p) > {C} (Lema A)"
        print(f"    t={tv}: p = {p} ({digs} digitos, mod 840={p%840}) -> {exact_str}", flush=True)
        detailed.append((tv, p, digs, m_val))

    return {
        "J": J,
        "C": C,
        "mode": mode,
        "digits_L": len(str(L)),
        "T_max": T_max,
        "S_BH": f"{S:.6e}",
        "BH_expected": f"{bh_exp:.3f}",
        "found": len(hits),
        "first_t": detailed[0][0] if detailed else "",
        "first_p": str(detailed[0][1]) if detailed else "",
        "first_p_digits": detailed[0][2] if detailed else "",
        "first_m_exact": detailed[0][3] if detailed else "",
        "elapsed_s": f"{elapsed:.2f}",
    }


def main():
    t_all = time.time()
    results = []

    # 1. Validacion: J = 4 y J = 6 con L_basica
    results.append(search_dickson(4, "basica", T_max=10_000, p_sieve_max=500, seg_len=10_000, batch_segs=1, max_hits_show=5))
    results.append(search_dickson(6, "basica", T_max=2_000_000, p_sieve_max=1500, seg_len=200_000, batch_segs=10, max_hits_show=4))

    # 2. J = 8: tanto con L_basica como con L_opt!
    results.append(search_dickson(8, "basica", T_max=600_000_000, p_sieve_max=4000, seg_len=1_000_000, batch_segs=64, max_hits_show=5))
    results.append(search_dickson(8, "opt",    T_max=400_000_000, p_sieve_max=4000, seg_len=1_000_000, batch_segs=64, max_hits_show=5))

    # 3. J = 9 con L_opt (espera ~1 cada 1.01e9 -> barremos 3e9)
    results.append(search_dickson(9, "opt",    T_max=3_000_000_000, p_sieve_max=6000, seg_len=2_000_000, batch_segs=64, max_hits_show=5))

    # 4. J = 10 con L_opt (espera ~1 cada 1.75e10 -> barremos 2.5e10)
    results.append(search_dickson(10, "opt",   T_max=25_000_000_000, p_sieve_max=8000, seg_len=2_000_000, batch_segs=128, max_hits_show=5))

    out_csv = os.path.join(os.path.dirname(__file__), "..", "..", "datos", "dickson_primos.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nGuardado {out_csv} ({len(results)} filas). Tiempo total: {time.time() - t_all:.2f} s.", flush=True)


if __name__ == "__main__":
    main()
