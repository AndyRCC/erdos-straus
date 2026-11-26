# Erdős–Straus: blind sets and computational exploration

**Andy Casafranca** · November 2026

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24569380.svg)](https://doi.org/10.5281/zenodo.24569380)

**The conjecture (1948):** for every `n ≥ 2` there are positive integers `x, y, z` with `4/n = 1/x + 1/y + 1/z`. It is still open and has been verified up to `10^18`. This repository does **not** prove it.

## Summary

> **Theorem.** For every c ≡ 3 (mod 4), the number of primes n ≤ N for which the shift c fails (no Type I or Type II solution with first denominator x = (n+c)/4) is ≍ N(log N)^{−3/2}. This extends Dahan's Theorem 4.15 (c = 7, arXiv:2608.24035) and settles the case c = 3, which was left open. It follows from Fuchs–Hsu–Rickards–Schindler–Stange (arXiv:2504.20289) applied to the form of discriminant −3.
> **Conditional results.** Dickson's prime k-tuples conjecture implies that the first working shift m(p) is unbounded. Bunyakovsky's conjecture for a single explicit quadratic implies m(p) ≥ 15 infinitely often; that quadratic is built from the congruent-number curve y² = x³ − 231²x.
> **Computations.** We replicate and extend to 10⁹ Dahan's sieve experiments. The Aigner Type II slice with a = 1 has exactly three exceptions below 10⁹ (193, 2521 and 66529). We compute m(p) exactly for all 128 671 219 "hard" primes below 10¹¹. We also run a targeted search above 10¹⁸ among primes that mimic squares.
> **This is not a proof of the conjecture.** Details are in [`note/erdos_straus_note.md`](note/erdos_straus_note.md).

## Results

Everything is in the note [`note/erdos_straus_note.md`](note/erdos_straus_note.md):

1. **Theorem (§5).** For every `c ≡ 3 (mod 4)`, the primes at which the shift `c` fails number `≍ N(log N)^{−3/2}`. This generalizes Dahan's Theorem 4.15 (c = 7) and settles the case `c = 3`.
2. **Is m(p) bounded? (§5.1)**
   - If Dickson's conjecture is true, `m(p)` is unbounded: no finite set of shifts works for every prime.
   - If Bunyakovsky's conjecture holds for a single quadratic, there are infinitely many primes with `m(p) ≥ 11`, or with `m(p) ≥ 15` using the curve E₂₃₁.
   - Up to `10^11` the record is `m(p) = 107`.
3. **Dahan's experiments (§2):** replicated exactly and extended from `2·10^7` to `10^9`.
4. **Aigner's slice (§3):** its only exceptions with `p ≡ 1 (mod 24)` and `p < 10^9` are 193, 2521 and 66529, decided with an exhaustive algorithm of cost `O(√p log p)`.
5. **Local criteria (§4), Bright–Loughran and the numerator barrier (§6), and a targeted search above `10^18` (§7).**

Every numerical result was computed with two independent implementations: one with Numba (`code/numba/`) and another with sympy (`code/verification/`).

## Reproduce

```
pip install -r requirements.txt
python tools/verify.py known                     # self-test: classical and Mordell identities
python tools/verify.py mod840                    # the 6 hard classes modulo 840 and their coverage
python tools/verify.py profile 8803369           # m(p) and which solution types exist
python code/verification/check_blind.py          # theorem of §5, checked up to 10^7
python code/verification/check_m_unbounded.py    # §5.1: E_231 certificate, Dickson primes and records
python tools/download_references.py              # optional: downloads the cited PDFs
```

Run the scripts from the root of the repository. The table in §8 of the note says which script reproduces each result.

## Layout

```
note/                  the note with all the results
tools/                 exact verifier (verify.py) and reference downloader
code/numba/            large computations (segmented sieves with Numba)
code/verification/     independent checks (sympy and numpy)
data/                  aggregated tables, plots and the E_231 certificate
references/README.md   bibliography with links (the PDFs are not included)
```

## How to cite

Andy Casafranca (2026). *Erdős–Straus conjecture: blind shifts, conditional results and computations*. Zenodo. https://doi.org/10.5281/zenodo.24569380

The metadata are in [`CITATION.cff`](CITATION.cff).

## Licence

The code is under the MIT licence, and the note and the data are under CC BY 4.0 (see [`LICENSE`](LICENSE)). The cited papers belong to their authors and are not redistributed.
