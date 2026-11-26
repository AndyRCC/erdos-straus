#!/usr/bin/env python3
"""
Exploración de los primos difíciles hasta 10^8: m(p), cortes con a = 1 (tipo I y tipo II) y
número de desplazamientos primos c = q probados (nota, §4).
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import divisors, factorint, legendre_symbol, primerange

sys.path.insert(0, "herramientas")
from verificar import es_solucion, perfil, tipos_en

HARD_CLASSES = (1, 121, 169, 289, 361, 529)
HARD_SET = set(HARD_CLASSES)
PRIMES_3MOD4 = [q for q in primerange(3, 50000) if q % 4 == 3]


def sieve_primes_mod(limit, mod, allowed_set):
    """Criba rápida de Eratóstenes con bytearray en numpy; devuelve lista de primos p <= limit con p % mod in allowed_set."""
    is_p = np.ones(limit + 1, dtype=np.bool_)
    is_p[:2] = False
    is_p[4::2] = False
    r = math.isqrt(limit)
    for i in range(3, r + 1, 2):
        if is_p[i]:
            is_p[i * i :: 2 * i] = False
    all_p = np.flatnonzero(is_p)
    mask = np.isin(all_p % mod, list(allowed_set))
    return all_p[mask].tolist()


def find_e_min_I1(p, fast_cap=4000):
    """Menor e ≡ 3 (mod 4) con B | (ep+1)/4 y B ≡ -1 (mod e). Exhaustiva hasta (p+4)/3 si supera fast_cap."""
    emax = (p + 4) // 3
    for e in range(3, min(fast_cap, emax) + 1, 4):
        Me = (e * p + 1) // 4
        if e == 3:
            if any(q % 3 == 2 for q in factorint(Me)):
                return e
        else:
            if any(d % e == e - 1 for d in divisors(Me)):
                return e
    # Búsqueda exhaustiva si superara fast_cap
    for e in range(fast_cap + (3 - fast_cap % 4) % 4, emax + 1, 4):
        Me = (e * p + 1) // 4
        if any(d % e == e - 1 for d in divisors(Me)):
            return e
    return -1


def find_u_min_II1(p):
    """
    Menor u >= 1 tal que p + u tiene un divisor s ≡ -1 (mod 4u) (Dahan Alg. 2, p = 4*alpha*u*v - u - v).
    Por simetría en (u, v), si existe solución entonces min(u, v) <= floor((1 + sqrt(4p + 1)) / 4).
    Por tanto la búsqueda hasta umax es 100% EXHAUSTIVA. Devuelve -1 si no existe (primo wild).
    """
    umax = (1 + math.isqrt(4 * p + 1)) // 4
    for u in range(1, umax + 1):
        mod4u = 4 * u
        target = mod4u - 1
        if u == 1:
            # p + 1 tiene divisor ≡ 3 (mod 4) <=> tiene factor primo ≡ 3 (mod 4)
            if any(q % 4 == 3 for q in factorint(p + 1)):
                return 1
        else:
            if any(s % mod4u == target for s in divisors(p + u)):
                return u
    return -1


def find_e_min_II_aET1(p, fast_cap=4000):
    """Menor e ≡ 3 (mod 4) tal que K_e = (p+e)/4 tiene un divisor b ≡ -1 (mod e) (Aigner Tipo II a_ET=1)."""
    emax = 3 * p - 1
    for e in range(3, min(fast_cap, emax) + 1, 4):
        Ke = (p + e) // 4
        if e == 3:
            if any(q % 3 == 2 for q in factorint(Ke)):
                return e
        else:
            if any(b % e == e - 1 for b in divisors(Ke)):
                return e
    for e in range(fast_cap + (3 - fast_cap % 4) % 4, emax + 1, 4):
        Ke = (p + e) // 4
        if any(b % e == e - 1 for b in divisors(Ke)):
            return e
    return -1


def find_k_q34(p):
    """Cuenta cuántos primos q ≡ 3 (mod 4) con (p/q) = -1 hay que probar hasta hallar solución en c = q."""
    k = 0
    for q in PRIMES_3MOD4:
        if legendre_symbol(p, q) == -1:
            k += 1
            t1, t2 = tipos_en(p, q)
            if t1 or t2:
                return k, q
    return -1, -1


def worker_hard(chunk):
    out = []
    for p in chunk:
        m, (t1, t2), c1, c2 = perfil(p, cap=100_000)
        e1 = find_e_min_I1(p)
        u2 = find_u_min_II1(p)
        e2a = find_e_min_II_aET1(p)
        kq, qhit = find_k_q34(p)
        out.append((p, p % 840, m, c1, c2, e1, u2, e2a, kq, qhit))
    return out


def worker_wild_check(chunk):
    wilds = []
    for p in chunk:
        if find_u_min_II1(p) == -1:
            wilds.append(p)
    return wilds


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100_000_000)
    ap.add_argument("--csv", default="datos/clases_dificiles_1e8.csv")
    ap.add_argument("--plot", default="datos/m_crecimiento_1e8.png")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    t_start = time.time()

    # 1. Validación previa de u_min_II1 sobre TODOS los primos p <= 10^7, p ≡ 1 (mod 24)
    t_w0 = time.time()
    p_1mod24_1e7 = sieve_primes_mod(10_000_000, 24, {1})
    chunk_w = max(1, len(p_1mod24_1e7) // (args.workers * 8))
    chunks_w = [p_1mod24_1e7[i : i + chunk_w] for i in range(0, len(p_1mod24_1e7), chunk_w)]
    wilds_1e7 = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(worker_wild_check, chunks_w):
            wilds_1e7.extend(res)
    wilds_1e7.sort()
    expected_wild14 = [
        409, 577, 5569, 9601, 23929, 83449, 102001, 329617, 712321,
        1134241, 1724209, 1726201, 5212561, 8813281,
    ]
    print(f"=== Validación u_min_II1 (Dahan Alg. 2) en TODOS los primos p <= 10^7, p ≡ 1 (mod 24) ({len(p_1mod24_1e7):,} primos) ===")
    print(f"  Tiempo: {time.time() - t_w0:.2f} s")
    print(f"  Primos wild encontrados ({len(wilds_1e7)}): {wilds_1e7}")
    print(f"  Coincide EXACTAMENTE con los 14 primos wild de Dahan Prop. 3.8: {wilds_1e7 == expected_wild14}")
    assert wilds_1e7 == expected_wild14, "Discrepancia en lista de primos wild <= 10^7!"

    # 2. Criba de primos difíciles hasta limit (10^8)
    t_s0 = time.time()
    primos = sieve_primes_mod(args.limit, 840, HARD_SET)
    print(f"\n=== Criba de primos difíciles mod 840 hasta {args.limit:,} ===")
    print(f"  Total primos difíciles: {len(primos):,} (criba en {time.time() - t_s0:.2f} s)")

    # 3. Procesamiento paralelo
    t_c0 = time.time()
    chunk_sz = max(1, len(primos) // (args.workers * 16))
    chunks = [primos[i : i + chunk_sz] for i in range(0, len(primos), chunk_sz)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(worker_hard, chunks):
            rows.extend(res)
    rows.sort(key=lambda r: r[0])
    t_calc = time.time() - t_c0
    print(f"  Cálculo completo en {args.workers} núcleos: {t_calc:.2f} s (total acumulado: {time.time() - t_start:.2f} s)")

    # 4. Guardar CSV
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p", "r_mod_840", "m", "min_c_I", "min_c_II", "e_min_I1", "u_min_II1", "e_min_II_aET1", "k_q34", "q_hit"])
        w.writerows(rows)
    print(f"  CSV guardado en {args.csv} ({len(rows):,} filas).")

    # 5. Consistencia con el CSV hasta 10^7
    if os.path.exists("datos/clases_dificiles_1e7.csv"):
        r1 = list(csv.DictReader(open("datos/clases_dificiles_1e7.csv", encoding="utf-8")))
        r2_sub = [r for r in rows if r[0] <= 10_000_000]
        assert len(r1) == len(r2_sub)
        mism = sum(1 for a, b in zip(r1, r2_sub) if int(a["p"]) != b[0] or int(a["m"]) != b[2])
        print(f"  Comprobación cruzada con los datos (<= 10^7, {len(r1):,} primos): discrepancias = {mism}")

    # --- Análisis , , y extendido a 10^8 ---
    ps = np.array([r[0] for r in rows], dtype=np.int64)
    ms = np.array([r[2] for r in rows], dtype=np.int64)
    c1s = np.array([r[3] for r in rows], dtype=np.int64)
    c2s = np.array([r[4] for r in rows], dtype=np.int64)
    e1s = np.array([r[5] for r in rows], dtype=np.int64)
    u2s = np.array([r[6] for r in rows], dtype=np.int64)
    e2as = np.array([r[7] for r in rows], dtype=np.int64)
    kqs = np.array([r[8] for r in rows], dtype=np.int64)

    print("\n=== Récords de m(p) hasta 10^8 y prueba de la heurística 0.42 (log X)^2 ===")
    rec_p, rec_m = [], []
    cur = -1
    for p_v, m_v in zip(ps, ms):
        if m_v > cur:
            cur = int(m_v)
            rec_p.append(int(p_v))
            rec_m.append(int(m_v))
    print("Secuencia completa de récords de m(p):")
    for rp, rm in zip(rec_p, rec_m):
        lp = math.log(rp)
        row = next(r for r in rows if r[0] == rp)
        print(f"  p={rp:>10d} (mod 840={rp%840:3d}): m={rm:4d} (cI={row[3]:4d}, cII={row[4]:4d}), m/log(p)={rm/lp:.3f}, m/log^2(p)={rm/(lp*lp):.4f}, 0.42*log^2(p)={0.42*lp*lp:.1f}")

    print("\nTop 15 mayores valores de m(p) en p <= 10^8:")
    for r in sorted(rows, key=lambda x: (-x[2], x[0]))[:15]:
        print(f"  p={r[0]:>10d} (mod 840={r[1]:3d}): m={r[2]:4d}, min_c_I={r[3]:4d}, min_c_II={r[4]:4d}, e_min_I1={r[5]:4d}, u_min_II1={r[6]:4d}, k_q34={r[8]}")

    print("\nMáximo y media acumulados de m(p) hasta X:")
    for X in [10**4, 10**5, 10**6, 10**7, 3*10**7, 5*10**7, 10**8]:
        msk = ps <= X
        sub = ms[msk]
        im = int(np.argmax(sub))
        pmax = int(ps[msk][im])
        mmax = int(sub[im])
        lx = math.log(X)
        print(f"  X={X:>11,}: n={msk.sum():>7,}, mean={sub.mean():6.3f}, max={mmax:4d} (en p={pmax:>10d}), max/log^2(X)={mmax/(lx*lx):.4f}, 0.42*log^2(X)={0.42*lx*lx:.1f}")

    print("\nFrecuencias de los mayores m(p) y resumen por década (incl. T1c tasa de fallo en c=3):")
    bounds = [10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
    for a, b in zip(bounds[:-1], bounds[1:]):
        msk = (ps > a) & (ps <= b)
        sub = ms[msk]
        fail3 = np.mean(sub > 3)
        xmid = math.sqrt(a * b)
        print(f"  ({a:>9,}, {b:>10,}]: n={msk.sum():>7,}, mean={sub.mean():6.3f}, p99={np.percentile(sub,99):5.1f}, max={sub.max():4d}, F_3={fail3:.4f}, F_3*sqrt(log x_mid)={fail3*math.sqrt(math.log(xmid)):.4f}")

    print("\n=== Cortes con a = 1 (Tipo I a_ET=1, Tipo II Dahan a_D=1, Tipo II Aigner a_ET=1) ===")
    no_I1 = [r[0] for r in rows if r[5] == -1]
    no_II1_dahan = [r[0] for r in rows if r[6] == -1]
    no_II1_aigner = [r[0] for r in rows if r[7] == -1]
    no_both_I1_II1d = [r[0] for r in rows if r[5] == -1 and r[6] == -1]
    no_both_I1_II1a = [r[0] for r in rows if r[5] == -1 and r[7] == -1]
    print(f"  Sin solución en corte Tipo I (a_ET = 1 = Dahan Thm 3.14 a_D=1): {len(no_I1)} -> {no_I1[:20]}")
    print(f"  Sin solución en corte Tipo II Dahan Alg. 2 (a_D = c_ET = 1):    {len(no_II1_dahan)} -> {no_II1_dahan}")
    print(f"  Sin solución en corte Tipo II Aigner (a_ET = 1, e|1+b,4b|p+e):  {len(no_II1_aigner)} -> {no_II1_aigner[:20]}")
    print(f"  Sin solución en NINGUNO de (Tipo I a=1) ni (Tipo II Dahan):     {len(no_both_I1_II1d)}")
    print(f"  Sin solución en NINGUNO de (Tipo I a=1) ni (Tipo II Aigner):    {len(no_both_I1_II1a)}")

    valid_e1 = e1s[e1s > 0]
    im_e1 = int(np.argmax(e1s))
    print(f"  e_min_I1: mean={valid_e1.mean():.3f}, median={np.median(valid_e1):.1f}, max={int(e1s[im_e1])} (en p={int(ps[im_e1])})")
    top_e1 = sorted(rows, key=lambda r: (-r[5], r[0]))[:10]
    print("  Top 10 mayores e_min_I1:", [(r[0], r[5]) for r in top_e1])

    valid_u2 = u2s[u2s > 0]
    im_u2 = int(np.argmax(u2s))
    print(f"  u_min_II1 (Dahan Alg. 2, excl. wild): mean={valid_u2.mean():.3f}, median={np.median(valid_u2):.1f}, max={int(u2s[im_u2])} (en p={int(ps[im_u2])})")
    top_u2 = sorted([r for r in rows if r[6] > 0], key=lambda r: (-r[6], r[0]))[:10]
    print("  Top 10 mayores u_min_II1 (finitos):", [(r[0], r[6]) for r in top_u2])

    valid_e2a = e2as[e2as > 0]
    im_e2a = int(np.argmax(e2as))
    print(f"  e_min_II_aET1 (Aigner Tipo II a_ET=1): mean={valid_e2a.mean():.3f}, median={np.median(valid_e2a):.1f}, max={int(e2as[im_e2a])} (en p={int(ps[im_e2a])})")

    print("\n=== Número de primos q ≡ 3 (mod 4) con (p/q) = -1 necesarios ===")
    kq_cnt = Counter(int(v) for v in kqs)
    tot = len(rows)
    for k in sorted(kq_cnt):
        print(f"  k_q34 = {k:2d}: {kq_cnt[k]:7d} ({100*kq_cnt[k]/tot:6.3f}%)")
    max_k = int(kqs.max())
    worst_k = [r for r in rows if r[8] == max_k]
    print(f"  Máximo k_q34 en p <= {args.limit:,}: {max_k} (alcanzado en {len(worst_k)} primo(s)):")
    for r in worst_k[:10]:
        print(f"    p={r[0]:>10d} (mod 840={r[1]:3d}): k_q34={r[8]}, q_hit={r[9]}, m(p)={r[2]}")

    # --- Gráfico PNG hasta 10^8 ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)

    ax = axes[0]
    # Submuestreo ligero para scatter si hay > 40k puntos + todos los m >= 35
    hi = ms >= 35
    rng = np.random.default_rng(42)
    lo_idx = np.flatnonzero(~hi)
    sub_idx = rng.choice(lo_idx, size=min(25000, len(lo_idx)), replace=False)
    plot_idx = np.concatenate([sub_idx, np.flatnonzero(hi)])
    ax.scatter(ps[plot_idx], ms[plot_idx], s=3, alpha=0.18, color="#1f77b4", rasterized=True, label="Primos difíciles mod 840 (muestra + cola $m \\geq 35$)")
    ax.step(rec_p, rec_m, where="post", color="#d62728", lw=1.8, label="Récord acumulado $\\max m(p)$")
    ax.scatter(rec_p, rec_m, color="#d62728", s=26, zorder=5)
    for rp, rm in zip(rec_p, rec_m):
        if rm >= 23:
            ax.annotate(f"p={rp:,}\nm={rm}", (rp, rm), xytext=(-6, 6), textcoords="offset points",
                        fontsize=7, color="#8b0000", ha="right" if rp > 15_000_000 else "left")
    x_grid = np.logspace(np.log10(1000), np.log10(args.limit), 300)
    ax.plot(x_grid, 0.42 * (np.log(x_grid) ** 2), color="black", lw=1.2, ls=":", label=r"Heurística R1: $0.42\,(\log p)^2$")
    bins = np.logspace(np.log10(1000), np.log10(args.limit), 28)
    bx, bmean, bp99 = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        msk = (ps >= a) & (ps < b)
        if msk.sum() >= 5:
            bx.append(np.sqrt(a * b))
            bmean.append(ms[msk].mean())
            bp99.append(np.percentile(ms[msk], 99))
    ax.plot(bx, bmean, color="#2ca02c", lw=2.0, label="Media por bin logarítmico")
    ax.plot(bx, bp99, color="#ff7f0e", lw=1.6, ls="--", label="Percentil 99 por bin")
    ax.set_xscale("log")
    ax.set_ylim(0, max(rec_m[-1], int(0.42 * math.log(args.limit) ** 2)) + 22)
    ax.set_xlabel("Primo $p$ (escala logarítmica)")
    ax.set_ylabel(r"$m(p) = 4x_{\min}(p) - p$")
    ax.set_title(r"Crecimiento de $m(p)$ para $p \leq 10^8$ (6 clases difíciles mod 840)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")

    ax2 = axes[1]
    ks_sorted = sorted(kq_cnt)
    ax2.bar(ks_sorted, [kq_cnt[k] for k in ks_sorted], color="#2b6cb0", edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_xticks(ks_sorted)
    ax2.set_xlabel(r"Número de primos $q \equiv 3\ (\mathrm{mod}\ 4)$ con $(p/q)=-1$ probados ($k_{q34}$)")
    ax2.set_ylabel("Número de primos difíciles (escala log)")
    ax2.set_title(f"Distribución de $k_{{q34}}(p)$ para $p \\leq 10^8$ ($N = {tot:,}$, máx = {max_k})")
    ax2.grid(True, axis="y", alpha=0.25)
    for k in ks_sorted:
        ax2.text(k, kq_cnt[k] * 1.15, f"{kq_cnt[k]:,}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(args.plot)
    print(f"\nGráfico guardado en {args.plot}")
    print(f"Tiempo total del script: {time.time() - t_start:.2f} s")


if __name__ == "__main__":
    main()
