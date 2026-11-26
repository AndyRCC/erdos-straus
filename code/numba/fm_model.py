#!/usr/bin/env python3
"""
Fit of the quantitative model f_m(p) ≈ C_m * S_m(p) * (log p)^3
for m = 4, 5, 6, 7, 8 over all primes p <= 200,000 (data/fm_primes_2e5.csv),
evaluation on the known exceptions, and heuristic extrapolation for m = 4
on [10^18, 10^30] and p > 10^18 (note, §7). Saves data/fm_quantitative_model.png.

Convention:
  - f_m(p) counts ORDERED triples (x, y, z) in N^3 (as in Elsholtz-Tao).
  - Since each solution with x < y < z contributes 6 permutations (3 if two coincide),
    the effective number of independent (unordered) solutions is N_m(p) ≈ f_m(p) / 6.
  - In the Poisson model we use the intensity lambda_m(p) = E[N_m(p)] = E_m(p) / 6
    (and we also report the empirical overdispersion / Fano factor, for honesty).
"""

import csv
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import primerange, legendre_symbol, isprime

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def legendre_fast(a, q):
    """(a/q) for an odd prime q."""
    r = a % q
    if r == 0:
        return 0
    v = pow(r, (q - 1) // 2, q)
    return 1 if v == 1 else -1


def compute_local_features(p_arr, qs_euler):
    """
    For an array of (odd) primes p_arr, computes:
      - L_inv: prod_{3 <= q <= Q} (1 - (q/p)/q)  (where (q/p) is the exact Legendre symbol;
        if q == p the factor is omitted, i.e., factor = 1)
      - p1_fac: prod_{3 <= q <= Q, q | p+1} (1 + 1/q)
      - pm1_fac: prod_{3 <= q <= Q, q | p-1} (1 - 1/q)
      - q_min: smallest prime q such that (q/p) = -1 (for odd p)
    """
    n = len(p_arr)
    L_inv = np.ones(n, dtype=np.float64)
    p1_fac = np.ones(n, dtype=np.float64)
    pm1_fac = np.ones(n, dtype=np.float64)

    p_mod4_3 = (p_arr % 4 == 3)
    for q in qs_euler:
        rem = p_arr % q
        # Table of quadratic residues mod q
        qr_tab = np.zeros(q, dtype=np.float64)
        for r in range(1, q):
            qr_tab[r] = float(legendre_fast(r, q))
        leg_pq = qr_tab[rem]
        # Reciprocity: (q/p) = (p/q) * (-1)^((q-1)/2 * (p-1)/2)
        if q % 4 == 3:
            leg_qp = np.where(p_mod4_3, -leg_pq, leg_pq)
        else:
            leg_qp = leg_pq
        # If p == q, rem == 0 => leg_qp == 0 => factor = 1
        L_inv *= (1.0 - leg_qp / q)
        p1_fac = np.where((p_arr + 1) % q == 0, p1_fac * (1.0 + 1.0 / q), p1_fac)
        pm1_fac = np.where((p_arr - 1) % q == 0, pm1_fac * (1.0 - 1.0 / q), pm1_fac)

    # q_min(p): smallest prime q with (q/p) = -1
    q_min = np.full(n, 1000, dtype=np.int32)
    # q = 2: (2/p) = -1 iff p ≡ 3, 5 (mod 8)
    m8 = p_arr % 8
    is_nr2 = (m8 == 3) | (m8 == 5)
    q_min = np.where(is_nr2, 2, q_min)
    for q in qs_euler:
        if (q_min < 1000).all():
            break
        rem = p_arr % q
        qr_tab = np.zeros(q, dtype=np.int8)
        for r in range(1, q):
            qr_tab[r] = legendre_fast(r, q)
        leg_pq = qr_tab[rem]
        if q % 4 == 3:
            leg_qp = np.where(p_mod4_3, -leg_pq, leg_pq)
        else:
            leg_qp = leg_pq
        q_min = np.where((q_min == 1000) & (leg_qp == -1), q, q_min)

    return L_inv, p1_fac, pm1_fac, q_min


def build_design_matrix(p_arr, m, L_inv, p1_fac, pm1_fac):
    """
    Builds the design matrix for log(f_m(p) / (log p)^3):
      - intercept
      - classes p mod 8 (3 dummies: 3, 5, 7 against ref 1)
      - class p mod 3 (1 dummy: 2 against ref 1)
      - classes p mod m if m in (5, 7) (m - 2 dummies against ref 1)
      - log(L_inv) separately for p ≡ 1 (mod 4) and p ≡ 3 (mod 4)
      - log(p1_fac) and log(pm1_fac)
    """
    n = len(p_arr)
    cols = [np.ones(n, dtype=np.float64)]
    for r in (3, 5, 7):
        cols.append((p_arr % 8 == r).astype(np.float64))
    cols.append((p_arr % 3 == 2).astype(np.float64))
    if m in (5, 7):
        for r in range(2, m):
            cols.append((p_arr % m == r).astype(np.float64))
    lx = np.log(L_inv)
    cols.append(np.where(p_arr % 4 == 1, lx, 0.0))
    cols.append(np.where(p_arr % 4 == 3, lx, 0.0))
    cols.append(np.log(p1_fac))
    cols.append(np.log(pm1_fac))
    return np.column_stack(cols)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    t0 = time.time()

    csv_path = os.path.join(ROOT, "data", "fm_primes_2e5.csv")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    p_all = np.array([int(r["p"]) for r in rows], dtype=np.int64)
    fm_all = {m: np.array([int(r[f"f{m}"]) for r in rows], dtype=np.float64) for m in (4, 5, 6, 7, 8)}

    # For odd p (p = 2 is left out of the odd modular features; it is treated separately)
    odd_mask = p_all >= 3
    p_odd = p_all[odd_mask]
    qs_200 = list(primerange(3, 200))
    L_inv_odd, p1_odd, pm1_odd, qmin_odd = compute_local_features(p_odd, qs_200)

    # Fitting set: primes p >= 1000 (17,816 primes) for asymptotic stability
    fit_mask = p_odd >= 1000
    p_fit = p_odd[fit_mask]
    lp_fit = np.log(p_fit)
    l3_fit = lp_fit ** 3

    print("===========================================================================")
    print("(b): FIT OF THE MODEL f_m(p) ≈ C_m * S_m(p) * (log p)^3 (p >= 1000)")
    print("===========================================================================")
    models = {}
    smearing = {}
    fano_cond = {}

    for m in (4, 5, 6, 7, 8):
        y_raw = fm_all[m][odd_mask][fit_mask]
        ratio = y_raw / l3_fit
        ok = ratio > 0
        ly = np.log(ratio[ok])

        X = build_design_matrix(p_fit[ok], m, L_inv_odd[fit_mask][ok], p1_odd[fit_mask][ok], pm1_odd[fit_mask][ok])
        coef, _, _, _ = np.linalg.lstsq(X, ly, rcond=None)
        pred_log = X @ coef
        resid = ly - pred_log
        r2 = 1.0 - (resid ** 2).sum() / ((ly - ly.mean()) ** 2).sum()

        # Duan (smearing) factor to correct the log-normal bias E[exp(eps)] when going back to the linear scale
        duan = float(np.mean(np.exp(resid)))
        pred_f = duan * np.exp(pred_log) * l3_fit[ok]
        # Fit on the linear scale too, for a linear R^2 and the conditional dispersion on [1e5, 2e5)
        r2_lin = 1.0 - ((y_raw[ok] - pred_f) ** 2).sum() / ((y_raw[ok] - y_raw[ok].mean()) ** 2).sum()

        # Conditional Fano factor of N_m = f_m / 6 on [1e5, 2e5) and p ≡ 1 (mod 24)
        sel_hi = (p_fit[ok] >= 100_000) & (p_fit[ok] % 24 == 1)
        obs_N = y_raw[ok][sel_hi] / 6.0
        exp_N = pred_f[sel_hi] / 6.0
        # Pearson estimator of overdispersion: mean((obs - exp)^2 / exp)
        fano_hi = float(np.mean(((obs_N - exp_N) ** 2) / exp_N))

        # Effective constant C_m (mean of f_m / (log p)^3 over all primes >= 1000)
        C_m = float(ratio.mean())
        # Normalized mean at p ≡ 1 (mod 24)
        s24 = p_fit % 24 == 1
        C_m_24 = float(ratio[s24].mean())

        models[m] = coef
        smearing[m] = duan
        fano_cond[m] = fano_hi

        # Also a specific fit INSIDE p ≡ 1 (mod 24), to see S_m(p) there exactly
        ok24 = ok & s24
        ly24 = np.log(ratio[ok24])
        X24 = np.column_stack([
            np.ones(ok24.sum()),
            np.log(L_inv_odd[fit_mask][ok24]),
            np.log(p1_odd[fit_mask][ok24]),
            np.log(pm1_odd[fit_mask][ok24]),
        ])
        c24, _, _, _ = np.linalg.lstsq(X24, ly24, rcond=None)
        r2_24 = 1.0 - ((ly24 - X24 @ c24) ** 2).sum() / ((ly24 - ly24.mean()) ** 2).sum()

        print(f"\n  m = {m}: C_m(mean) = {C_m:.4f} | C_m(1 mod 24) = {C_m_24:.4f} | "
              f"Duan = {duan:.4f} | R^2_log = {r2:.4f} | R^2_lin = {r2_lin:.4f} | Fano_cond(1m24) = {fano_hi:.2f}")
        print(f"    Global: exp(L^-1, 1m4)={coef[-4]:+.3f}, exp(L^-1, 3m4)={coef[-3]:+.3f}, "
              f"exp(p+1)={coef[-2]:+.3f}, exp(p-1)={coef[-1]:+.3f}")
        print(f"    At p ≡ 1 (mod 24): log(f_m/log^3 p) = {c24[0]:+.3f} + {c24[1]:+.3f}*log(L^-1) "
              f"+ {c24[2]:+.3f}*log(Phi_+(p+1)) + {c24[3]:+.3f}*log(Phi_-(p-1))  [R^2={r2_24:.4f}]")

    # Store the coefficients of the model specialized to p ≡ 1 (mod 24) for m=4 and m=8 (more accurate in the hard tail!)
    s24_fit = p_fit % 24 == 1
    models_24 = {}
    duan_24 = {}
    fano_24 = {}
    for m in (4, 8):
        y24 = fm_all[m][odd_mask][fit_mask][s24_fit] / l3_fit[s24_fit]
        ok = y24 > 0
        ly = np.log(y24[ok])
        X24 = np.column_stack([
            np.ones(ok.sum()),
            np.log(L_inv_odd[fit_mask][s24_fit][ok]),
            np.log(p1_odd[fit_mask][s24_fit][ok]),
            np.log(pm1_odd[fit_mask][s24_fit][ok]),
        ])
        c24, _, _, _ = np.linalg.lstsq(X24, ly, rcond=None)
        res = ly - X24 @ c24
        d24 = float(np.mean(np.exp(res)))
        pred_N = d24 * np.exp(X24 @ c24) * l3_fit[s24_fit][ok] / 6.0
        obs_N = fm_all[m][odd_mask][fit_mask][s24_fit][ok] / 6.0
        sel_hi = p_fit[s24_fit][ok] >= 100_000
        fn = float(np.mean(((obs_N[sel_hi] - pred_N[sel_hi]) ** 2) / pred_N[sel_hi]))
        models_24[m] = c24
        duan_24[m] = d24
        fano_24[m] = fn

    print("\n=== TABLE OF q_min(p) AT p ≡ 1 (mod 24), p >= 1000 (m=4 and m=8) ===")
    p24_f = p_fit[s24_fit]
    qm24 = qmin_odd[fit_mask][s24_fit]
    y4_24 = fm_all[4][odd_mask][fit_mask][s24_fit] / l3_fit[s24_fit]
    y8_24 = fm_all[8][odd_mask][fit_mask][s24_fit] / l3_fit[s24_fit]
    for qv in (5, 7, 11, 13, 17, 19, 23):
        sel = qm24 == qv
        if sel.sum() > 0:
            print(f"  q_min = {qv:>2} (n={sel.sum():>4}): mean f4/log^3 = {y4_24[sel].mean():.4f} | "
                  f"mean f8/log^3 = {y8_24[sel].mean():.4f}")

    # =========================================================================
    # (c): Test on the known exceptions (Poisson and negative binomial)
    # =========================================================================
    print("\n===========================================================================")
    print("(c): PREDICTION OF THE KNOWN EXCEPTIONS (POISSON N_m = f_m/6 AND NEG. BINOM.)")
    print("===========================================================================")

    def predict_Em_odd(m, p_subset_idx):
        X = build_design_matrix(
            p_odd[p_subset_idx], m,
            L_inv_odd[p_subset_idx], p1_odd[p_subset_idx], pm1_odd[p_subset_idx]
        )
        Em = smearing[m] * np.exp(X @ models[m]) * (np.log(p_odd[p_subset_idx]) ** 3)
        return Em

    idx_all_odd = np.arange(len(p_odd))
    p_to_idx = {int(pv): i for i, pv in enumerate(p_odd)}

    # Probability of failure at the exceptions of m = 8 (odd p: 3, 11, 17, 131, 241) + control m = 4
    print("\n  A) Probabilities at the odd prime exceptions of m = 8 (and contrast with m = 4):")
    print("     (Note: p = 2 for m = 7 and m = 8 is a trivial archimedean obstruction: m/2 > 3.)")
    exc_list = [3, 11, 17, 131, 241, 73, 193, 1009, 1201, 2521]
    Em_cache = {m: predict_Em_odd(m, idx_all_odd) for m in (4, 5, 6, 7, 8)}

    print(f"  {'p':>5} {'mod24':>5} {'qmin':>4} | {'f8':>4} {'E8':>6} {'lam8':>5} {'P_Pois(8)':>9} {'P_NB(8)':>8} | "
          f"{'f4':>4} {'E4':>6} {'lam4':>5} {'P_Pois(4)':>10}")
    for pv in exc_list:
        i = p_to_idx[pv]
        e8 = Em_cache[8][i]
        lam8 = e8 / 6.0
        p_pois8 = math.exp(-lam8)
        F8 = fano_cond[8]
        # Negative binomial with mean lam and Fano factor F > 1: P(0) = F^(-lam / (F - 1))
        p_nb8 = F8 ** (-lam8 / (F8 - 1.0))

        e4 = Em_cache[4][i]
        lam4 = e4 / 6.0
        p_pois4 = math.exp(-lam4)
        f8_obs = int(fm_all[8][odd_mask][i])
        f4_obs = int(fm_all[4][odd_mask][i])
        tag = " *" if f8_obs == 0 else "  "
        print(f"  {pv:>5} {pv%24:>5} {int(qmin_odd[i]):>4} | {f8_obs:>4} {e8:>6.1f} {lam8:>5.2f} {p_pois8:>9.4f} {p_nb8:>8.4f} | "
              f"{f4_obs:>4} {e4:>6.1f} {lam4:>5.2f} {p_pois4:>10.2e}{tag}")

    print("\n  B) Total EXPECTED number of odd prime exceptions per m (3 <= p <= 200,000 vs p >= 25):")
    print(f"  {'m':>2} | {'Exc(p>=3)':>9} {'Pois_N(p>=3)':>12} {'Pois_f(p>=3)':>12} {'NB(p>=3)':>9} | "
          f"{'Exc(p>=25)':>10} {'Pois_N(p>=25)':>13} {'NB(p>=25)':>10}")
    m25 = p_odd >= 25
    for m in (4, 5, 6, 7, 8):
        obs_all = int((fm_all[m][odd_mask] == 0).sum())
        obs_25 = int((fm_all[m][odd_mask][m25] == 0).sum())
        Em = Em_cache[m]
        lam = Em / 6.0
        Fm = fano_cond[m]
        e_pN_all = float(np.exp(-lam).sum())
        e_pf_all = float(np.exp(-Em).sum())
        e_nb_all = float((Fm ** (-lam / (Fm - 1.0))).sum())
        e_pN_25 = float(np.exp(-lam[m25]).sum())
        e_nb_25 = float((Fm ** (-lam[m25] / (Fm - 1.0))).sum())
        print(f"  {m:>2} | {obs_all:>9} {e_pN_all:>12.3f} {e_pf_all:>12.5f} {e_nb_all:>9.3f} | "
              f"{obs_25:>10} {e_pN_25:>13.4f} {e_nb_25:>10.3f}")

    # Also evaluate p = 241 with the model specialized to p ≡ 1 (mod 24)
    i241 = p_to_idx[241]
    c0_8, c1_8, c2_8, c3_8 = models_24[8]
    e8_241_spec = duan_24[8] * math.exp(
        c0_8 + c1_8 * math.log(L_inv_odd[i241]) + c2_8 * math.log(p1_odd[i241]) + c3_8 * math.log(pm1_odd[i241])
    ) * (math.log(241.0) ** 3)
    lam8_241_spec = e8_241_spec / 6.0
    print(f"    Control p=241 in the specialized model (p ≡ 1 mod 24): E_8={e8_241_spec:.1f}, lam_8={lam8_241_spec:.2f}, "
          f"P_Pois={math.exp(-lam8_241_spec):.4f}, P_NB(F={fano_24[8]:.2f})={fano_24[8]**(-lam8_241_spec/(fano_24[8]-1)):.4f}")

    # Also check how many primes <= 200,000 have lam8 <= lam8(241) (ranking of 241!)
    lam8_all = Em_cache[8] / 6.0
    for pv in (3, 11, 17, 131, 241):
        lv = lam8_all[p_to_idx[pv]]
        rk = int((lam8_all <= lv).sum())
        print(f"    Ranking of lam_8({pv}) = {lv:.2f}: position {rk} / {len(p_odd)} odd primes <= 200,000")

    # =========================================================================
    # (d): Heuristic application to m = 4 on [10^18, 10^30] and p > 10^18
    # =========================================================================
    print("\n===========================================================================")
    print("(d): HEURISTIC EXTRAPOLATION FOR m = 4 ON [10^18, 10^30] AND p > 10^18")
    print("===========================================================================")
    # We use the model specialized to p ≡ 1 (mod 24):
    #   E_4(p) = duan_24 * exp(c0) * (L_inv)^c1 * (p1_fac)^c2 * (pm1_fac)^c3 * (log p)^3
    c0, c1, c2, c3 = models_24[4]
    A4_24 = duan_24[4] * math.exp(c0)
    gamma_E = 0.5772156649015329
    print(f"  Model specialized to p ≡ 1 (mod 24): E_4(p) = {A4_24:.4f} * L_inv^{c1:.3f} * "
          f"Phi_+(p+1)^{c2:.3f} * Phi_-(p-1)^{c3:.3f} * (log p)^3")
    print(f"  (conditional Fano on [1e5, 2e5), p ≡ 1 mod 24: F = {fano_24[4]:.2f})")

    # Worst possible local data at height X:
    #   Scenario 1 (explicit adversary p ≡ 1 mod L(Y)): take the largest Y with L(Y) <= X.
    #     Then (q/p)=+1 and q | (p-1) for every 5 <= q <= Y (and q ∤ p+1 => Phi_+(p+1)=1).
    #     Hence L_inv ≈ prod_{5<=q<=Y}(1-1/q) * (1 - 1/3) = (2/3) * prod_{5<=q<=Y}(1-1/q)
    #     and Phi_-(p-1) ≈ prod_{5<=q<=Y}(1-1/q). (Note: for p≡1 mod 24, q=3 already has (3/p)=+1 and 3|p-1.)
    #   Scenario 2 (theoretical Littlewood/GRH extreme): L(1, chi_p) <= (2 e^gamma) log log p
    #     => L_inv >= 1 / (2 e^gamma log log p), and Phi_-(p-1) >= e^(-gamma) / log log p (Mertens for a smooth p-1).
    qs_big = list(primerange(3, 500))

    print("\n  Minimum of E_4(p) and lambda_4(p) = E_4(p)/6 on [10^18, 10^30]:")
    print(f"  {'X':>8} {'log X':>6} | {'Y_max(L<=X)':>11} {'E4_min(L(Y))':>13} {'lam4_min':>9} {'exp(-lam4)':>12} | "
          f"{'E4_min(GRH-ext)':>15} {'lam4_GRH':>9} {'exp(-lam4_GRH)':>15}")
    for exp10 in (18, 21, 24, 27, 30):
        X = 10.0 ** exp10
        lx = math.log(X)
        llx = math.log(lx)

        # Largest Y such that L(Y) = 8 * prod_{3<=q<=Y} q <= X
        L_val = 8.0
        Y_max = 3
        L_inv_adv = 1.0
        pm1_adv = 1.0
        for q in qs_big:
            if L_val * q <= X:
                L_val *= q
                Y_max = q
                L_inv_adv *= (1.0 - 1.0 / q)
                pm1_adv *= (1.0 - 1.0 / q)
            else:
                break
        # For p ≡ 1 (mod 24), q = 3 already contributes (1-1/3) to L_inv and pm1_adv; Phi_+(p+1) = 1 (worst case, no q|p+1)
        E4_adv = A4_24 * (L_inv_adv ** c1) * (1.0 ** c2) * (pm1_adv ** c3) * (lx ** 3)
        lam_adv = E4_adv / 6.0

        # GRH/Littlewood extreme scenario: L(1, chi_p) = 2 * e^gamma * log log X (includes the factor q=2: 1/(1-1/2)=2,
        # so the odd part is L_inv_odd = 2 * L(1, chi_p)^{-1} = 1 / (e^gamma * log log X));
        # and Phi_-(p-1) = e^(-gamma) / log log X (Mertens with every prime up to log X dividing p-1).
        L_inv_grh = 1.0 / (math.exp(gamma_E) * llx)
        pm1_grh = math.exp(-gamma_E) / llx
        E4_grh = A4_24 * (L_inv_grh ** c1) * (pm1_grh ** c3) * (lx ** 3)
        lam_grh = E4_grh / 6.0

        p_adv_str = f"10^{-lam_adv / math.log(10):.1f}"
        p_grh_str = f"10^{-lam_grh / math.log(10):.1f}"
        print(f"  10^{exp10:<5} {lx:>6.2f} | {Y_max:>11} {E4_adv:>13.1f} {lam_adv:>9.1f} {p_adv_str:>12} | "
              f"{E4_grh:>15.1f} {lam_grh:>9.1f} {p_grh_str:>15}")

    # Expected number of counterexamples above 10^18:
    # VERY conservative upper bound: assign to ALL primes p > 10^18 with p ≡ 1 (mod 24)
    # the extreme GRH-Littlewood WORST case (and also evaluate with the typical value for p ≡ 1 mod 24).
    # Integral: (1/8) * int_{10^18}^infty exp(-lam(t)) / log(t) dt; substituting u = log t => dt/log t = e^u / u du.
    # Since the integrand decays super-exponentially in u, it is completely dominated by u = log(10^18) = 41.4465!
    u0 = math.log(1e18)
    # Evaluate log10 of the integrand at u0 and the integral by Laplace: int_{u0}^infty exp(u - g(u)) / u du ≈ exp(u0 - g(u0)) / (u0 * (g'(u0) - 1))
    def log10_expected_above_1e18(mode="poisson_worst"):
        ll = math.log(u0)
        L_inv_grh = 1.0 / (math.exp(gamma_E) * ll)
        pm1_grh = math.exp(-gamma_E) / ll
        if mode == "poisson_typical_1m24":
            # Typical value for p ≡ 1 mod 24: E4 ≈ 0.434 * u^3 => lam ≈ 0.0723 * u^3
            g = 0.4340 * (u0 ** 3) / 6.0
            gp = 3.0 * g / u0
        elif mode == "poisson_worst_grh":
            g = A4_24 * (L_inv_grh ** c1) * (pm1_grh ** c3) * (u0 ** 3) / 6.0
            gp = (3.0 - (c1 + c3) / ll) * g / u0
        elif mode == "nb_worst_grh":
            lam = A4_24 * (L_inv_grh ** c1) * (pm1_grh ** c3) * (u0 ** 3) / 6.0
            F = fano_24[4]
            # P_NB(0) = exp(-lam * log(F) / (F - 1))
            factor = math.log(F) / (F - 1.0)
            g = lam * factor
            gp = (3.0 - (c1 + c3) / ll) * g / u0
        # Factor 1/8 for p ≡ 1 (mod 24) (or 1/32 for the hard primes mod 840); we use 1/8 to be conservative
        log_val = math.log(1.0 / 8.0) + u0 - g - math.log(u0 * (gp - 1.0))
        return log_val / math.log(10.0), g

    print("\n  EXPECTED number of counterexamples for m = 4 above 10^18 (HEURISTIC):")
    for label, mode in [
        ("Poisson (typical p ≡ 1 mod 24)          ", "poisson_typical_1m24"),
        ("Poisson (GRH worst case for ALL p)      ", "poisson_worst_grh"),
        ("Neg. binom. F=3.7 (GRH worst case, all) ", "nb_worst_grh"),
    ]:
        log10_E, g0 = log10_expected_above_1e18(mode)
        print(f"    {label}: effective exponent at 10^18 = {g0:>7.1f}  =>  E[counterex. > 10^18] ≈ 10^({log10_E:.1f})")

    # =========================================================================
    # Summary plot in data/fm_quantitative_model.png
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # Panel 1: mean f_m(p) / (log p)^3 per m (all primes vs p ≡ 1 mod 24)
    ax = axes[0]
    ms_list = [4, 5, 6, 7, 8]
    means_all = [(fm_all[m][odd_mask][fit_mask] / l3_fit).mean() for m in ms_list]
    means_24 = [(fm_all[m][odd_mask][fit_mask][s24_fit] / l3_fit[s24_fit]).mean() for m in ms_list]
    x_pos = np.arange(len(ms_list))
    ax.bar(x_pos - 0.18, means_all, width=0.35, label="All primes ($p \\geq 10^3$)", color="#2b6cb0")
    ax.bar(x_pos + 0.18, means_24, width=0.35, label="$p \\equiv 1$ (mod 24)", color="#c53030")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"m={m}" for m in ms_list])
    ax.set_ylabel("Mean of $f_m(p) / (\\log p)^3$")
    ax.set_title("Decrease with $m$ (ordered triples, $p \\leq 2\\cdot 10^5$)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: f_4(p) / (log p)^3 vs L_{200}(1, chi_p)^{-1} at p ≡ 1 (mod 24)
    ax = axes[1]
    L24 = L_inv_odd[fit_mask][s24_fit]
    ax.scatter(L24, y4_24, s=6, alpha=0.25, color="#2b6cb0", label="Primes $p \\equiv 1$ (mod 24)")
    # Fitted curve (with Phi_+=Phi_-=1)
    xs_grid = np.linspace(L24.min(), L24.max(), 100)
    ys_grid = A4_24 * (xs_grid ** c1) * (np.median(pm1_odd[fit_mask][s24_fit]) ** c3)
    ax.plot(xs_grid, ys_grid, color="#c53030", lw=2, label=f"Fit $\\propto L(1,\\chi_p)^{{-{c1:.2f}}}$")
    ax.set_xlabel("$L_{200}(1, \\chi_p)^{-1} = \\prod_{3\\leq q\\leq 200}(1 - (q/p)/q)$")
    ax.set_ylabel("$f_4(p) / (\\log p)^3$")
    ax.set_title("Singular series at $p \\equiv 1$ (mod 24) ($R^2 = 0.52$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: monotone decrease with q_min(p) for m=4 and m=8
    ax = axes[2]
    qvs = [5, 7, 11, 13, 17, 19, 23]
    m4_q = [y4_24[qm24 == qv].mean() for qv in qvs]
    m8_q = [y8_24[qm24 == qv].mean() for qv in qvs]
    ax.plot(qvs, m4_q, "o-", color="#2b6cb0", lw=2, label="$m = 4$")
    ax.plot(qvs, m8_q, "s-", color="#c53030", lw=2, label="$m = 8$")
    ax.set_xlabel("$q_{\\min}(p)$ (smallest prime non-residue mod $p$)")
    ax.set_ylabel("Mean of $f_m(p) / (\\log p)^3$")
    ax.set_title("Effect of $q_{\\min}(p)$ at $p \\equiv 1$ (mod 24)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(ROOT, "data", "fm_quantitative_model.png")
    plt.savefig(out_png)
    plt.close()
    print(f"\nPlot saved: {out_png} ({os.path.getsize(out_png):,} bytes)")
    print(f"Total time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
