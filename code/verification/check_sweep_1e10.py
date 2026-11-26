#!/usr/bin/env python3
"""
Independent check (numpy sieve + sympy) of m(p) for all hard primes
up to 10^10 (note, §4).
"""
import os
import sys
import time
from collections import Counter
from math import isqrt, log, sqrt
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify import types_at  # noqa: E402
from sympy import primerange  # noqa: E402

N = 10 ** 10
SEG = 2 * 10 ** 8
DIF = np.array([1, 121, 169, 289, 361, 529])
BASE = np.array(list(primerange(3, isqrt(N) + 1)), dtype=np.int64)


def m_of(p):
    m = 3
    while True:
        t1, t2 = types_at(p, m)
        if t1 or t2:
            return m
        m += 4


def block(ps):
    return [(p, m_of(p)) for p in ps]


def segment_primes(lo, hi):
    """Odd primes in [lo, hi), lo even."""
    s = np.ones((hi - lo) // 2, dtype=bool)          # s[i] <-> lo + 2i + 1
    for q in BASE:
        q = int(q)
        if q * q >= hi:
            break
        st = max(q * q, ((lo + q - 1) // q) * q)
        if st % 2 == 0:
            st += q
        s[(st - lo - 1) // 2::q] = False
    ps = lo + 2 * np.nonzero(s)[0].astype(np.int64) + 1
    if lo == 0:
        ps = ps[ps > 1]
    return ps


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tally = Counter()
    record, records, highs = 0, [], []
    by_decade = Counter()
    m3_decade = Counter()
    total = 0
    with Pool(8) as pool:
        for lo in range(0, N, SEG):
            ps = segment_primes(lo, min(lo + SEG, N + 1))
            ps = ps[np.isin(ps % 840, DIF)]
            pieces = [list(map(int, c)) for c in np.array_split(ps, 64) if len(c)]
            for res in pool.map(block, pieces):
                for p, m in res:
                    total += 1
                    tally[m] += 1
                    k = int(log(p, 10))
                    by_decade[k] += 1
                    m3_decade[k] += m == 3
                    if m > record:
                        record = m
                        records.append((p, m))
                    if m >= 75:
                        highs.append((m, p))
            print(f"  up to {min(lo + SEG, N):,}: {total:,} hard primes, record m = {record}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"\nTOTAL hard primes <= 10^10: {total:,}")
    print("Records:", records)
    print("m >= 75:", sorted(highs, reverse=True))
    print("Distribution:", sorted(tally.items()))
    for k in sorted(by_decade):
        if k >= 3:
            F3 = 1 - m3_decade[k] / by_decade[k]
            print(f"  [10^{k}, 10^{k + 1}): n={by_decade[k]:,}  frac(m=3)={m3_decade[k] / by_decade[k]:.4f}  F3*sqrt(log x_mid)={F3 * sqrt(log(10 ** (k + 0.5))):.4f}")
    print(f"Total time: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
