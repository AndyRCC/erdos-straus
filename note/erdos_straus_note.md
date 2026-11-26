# Blind sets and computational observations on the Erdős–Straus conjecture

**Andy Casafranca** · November 2026

> **Abstract.** **Theorem (§5).** For every $c \equiv 3 \pmod 4$, the primes at which the shift $c$ fails (no solution of type I or II with $x = (n+c)/4$) number $\asymp_c N(\log N)^{-3/2}$. This generalizes Dahan's Theorem 4.15, which treats $c = 7$, and settles the case $c = 3$, which he left open because of a 2-adic obstruction. The key is to use the form of discriminant $-3$; the theorem is a corollary of Fuchs–Hsu–Rickards–Schindler–Stange. **On $m(p)$**, the first shift that works for $p$, we prove that Dickson's prime $k$-tuples conjecture implies that **$m(p)$ is unbounded**. We also show that Bunyakovsky's conjecture for a single quadratic implies $m(p) \ge 11$ infinitely often, or $m(p) \ge 15$ with a quadratic built from the curve $y^2 = x^3 - 231^2x$. Up to $10^{11}$ the record is $m = 107$.
>
> In addition, for the primes $p \equiv 1 \pmod{24}$ below $10^9$ (6 354 932 primes) we present two computational results.
> **(1)** We replicate exactly, and extend from $2\cdot 10^7$ to $10^9$, Dahan's numerical experiments (arXiv:2608.24035) on the exceptional set $E_1(N;J)$. The local slopes of $\log(|E_1|/\pi)$ against $\log\log n$ come within 5 % of the sieve exponent $-J/2$ of his Theorem 4.3 for $J \le 6$; this could not be seen in his data. On the other hand, the ratio $\beta/\log n$ of his second heuristic regime is **not** constant.
> **(2)** The slice of type II solutions "with $a = 1$" in the Aigner / Elsholtz–Tao parametrization, which is the row $u = 1$ of Dahan's grid, has exactly **three** exceptions below $10^9$: $193$, $2521$ and $66\,529$. We also give an exhaustive procedure of cost $O(\sqrt p \log p)$ to decide that slice.
> We add an explicit example (§6): the surface of $8/241$ has no Brauer–Manin obstruction, and yet $8/241$ is not a sum of three unit fractions. It shows why a proof has to use the numerator 4 quantitatively.
> Finally (§7), we ran a targeted search for counterexamples **above $10^{18}$**, up to $10^{43}$. We ran it on primes built to look like a square: quadratic residues modulo every prime $q \le Y$, with $Y \le 101$. They are measurably harder than random primes, but all 1620 primes tried have a solution, checked in exact arithmetic.
>
> **This is not progress towards a proof.** The conjecture remains open.

---

## 1. Background

The Erdős–Straus conjecture (1948) states that for every $n \ge 2$ there are positive integers $x, y, z$ with
$$\frac{4}{n} = \frac1x + \frac1y + \frac1z .$$
It suffices to prove it for $n = p$ prime. The classical identities and Mordell's identities settle it except for $p \bmod 840 \in \{1, 121, 169, 289, 361, 529\}$, which are the invertible squares modulo 840. It has been verified up to $10^{17}$ (Salez, 2014) and up to $10^{18}$ (Mihnea–Dumitru, 2025, preprint). No polynomial identity covers a class that is a square (Mordell, Schinzel; see Elsholtz–Tao 2013, §1). Bright and Loughran (2020) reinterpret that obstruction as a Brauer–Manin obstruction to strong approximation, and prove that there is no Brauer–Manin obstruction to the conjecture itself. Vaughan (1970) proved that the number of exceptions up to $N$ is $\ll N\exp(-c(\log N)^{2/3})$. For large numerators $m$, Schinzel's analogous conjecture ($m/n$ instead of $4/n$) has very large exceptions (Pomerance–Weingartner).

For an odd prime $p$, every solution is of **type I** ($p$ divides exactly one of the denominators) or of **type II** (it divides exactly two).

**Notation.** For $c \equiv 3 \pmod 4$ we set $K_c = (p+c)/4$. By Dahan (Thm. 3.9), $4/p$ has a type II solution if and only if there are $u, a \ge 1$ and a divisor $s$ of $ap + u$ with $s \equiv -1 \pmod{4ua}$. In terms of $x = K_c$, this amounts to the existence of divisors $u, v$ of $K_c$ with $uv \mid K_c$ and $u + v = a\,c$. We call the set of pairs $(u, a)$ **Dahan's grid**.
- The **column $a = 1$** ($u + v = c$, Dahan's Algorithm 2) is the "hyperbolic" family $p = 4\alpha uv - u - v$.
- The **row $u = 1$** (one of the divisors equals 1) is the slice $a_{\mathrm{ET}} = 1$ of the type II parametrization of Elsholtz–Tao, Prop. 2.7 (Aigner, Rosati).

In the variables of Elsholtz–Tao, Dahan's "$a$" is $c_{\mathrm{ET}}$ and his "$u$" is $a_{\mathrm{ET}}$.

## 2. Result 1: Dahan's experiments, replicated and extended

Let $\mathcal N(N)$ be the set of primes $n \le N$ with $n \equiv 1 \pmod{24}$, and let $E_1(N;J) \subseteq \mathcal N(N)$ be the subset of those that have no admissible $u \le J$ in the column $a = 1$ (Dahan, Def. 4.1). Dahan proves (Thm. 4.3) that
$$|E_1(N;J)| \ll_J \frac{N}{(\log N)^{1+J/2}},$$
and in his §6(11) he measures two things up to $2\cdot 10^7$:
- the slope of $\log\big(|E_1|/|\mathcal N|\big)$ against $\log\log n$, in 6 windows and for $J = 1, \dots, 6$;
- the exponent $\beta$ of a fit $J^{-\beta}$ for $J = 1, \dots, 20$ in each window (Table 2), which according to his second heuristic regime should satisfy $\beta \approx c_2 \log n$.

**Replication.** We computed $u_{\min}(n) = \min\{u \le 20 \text{ admissible}\}$ for every $n \in \mathcal N(10^9)$, with two independent implementations: an inverted sieve compiled with Numba and a direct factorization with sympy. Both give identical results. Below $2\cdot 10^7$ we reproduce Dahan's numbers exactly: there are 158 595 primes, the values of $\beta$ in Table 2, the fit $\beta = 0.1433 \log n + 0.043$ ($R^2 = 0.967$) and the slopes $-0.39, -0.50, -0.84, -1.24, -1.51, -1.59$.

**Extension.** The four new windows appear below the horizontal rule:

| window | primes | $\beta$ | $\beta/\log n_{\rm mid}$ | $R^2$ |
|---|---:|---:|---:|---:|
| $[10^3, 10^4)$ | 129 | 1.134 | 0.1407 | 0.946 |
| $[10^4, 10^5)$ | 1 038 | 1.632 | 0.1575 | 0.992 |
| $[10^5, 10^6)$ | 8 551 | 1.905 | 0.1504 | 0.993 |
| $[10^6, 3\cdot10^6)$ | 17 251 | 2.008 | 0.1398 | 0.994 |
| $[3\cdot10^6, 10^7)$ | 55 904 | 2.200 | 0.1418 | 0.992 |
| $[10^7, 2\cdot10^7)$ | 75 708 | 2.480 | 0.1506 | 0.983 |
| — | — | — | — | — |
| $[2\cdot10^7, 5\cdot10^7)$ | 216 307 | 2.498 | 0.1446 | 0.985 |
| $[5\cdot10^7, 10^8)$ | 344 879 | 2.582 | 0.1429 | 0.985 |
| $[10^8, 3\cdot10^8)$ | 1 310 830 | 2.682 | 0.1414 | 0.985 |
| $[3\cdot10^8, 10^9)$ | 4 324 321 | 2.800 | 0.1392 | 0.984 |

Slopes of $\log(|E_1|/|\mathcal N|)$ against $\log\log n_{\rm mid}$, where $n_{\rm mid}$ is the geometric midpoint of each window:

| windows used | $J=1$ | $2$ | $3$ | $4$ | $5$ | $6$ |
|---|---:|---:|---:|---:|---:|---:|
| prediction $-J/2$ | −0.50 | −1.00 | −1.50 | −2.00 | −2.50 | −3.00 |
| Dahan's 6 ($<2\cdot10^7$) | −0.387 | −0.499 | −0.838 | −1.235 | −1.509 | −1.587 |
| all 10 ($<10^9$) | −0.412 | −0.645 | −1.013 | −1.406 | −1.721 | −1.888 |
| the 9 without $[10^3,10^4)$ | −0.454 | −0.901 | −1.283 | −1.663 | −1.968 | −2.236 |
| the 4 new ones ($[2\cdot10^7, 10^9)$) | **−0.507** | **−1.021** | **−1.464** | **−1.906** | **−2.488** | **−2.929** |

**Interpretation.**
1. Between $2\cdot 10^7$ and $10^9$, the local slopes are within 5 % of $-J/2$ for every $J \le 6$. There are millions of primes per window, so the statistical error is small. But the range is narrow: $\Delta \log\log n = 0.15$, and lower-order terms could still matter. We take this as **empirical support** for the exponent $J/2$ of Theorem 4.3 being the true one, **not** as a proof of a lower bound.
2. The deficit Dahan observes is mainly due to height: removing the small window $[10^3, 10^4)$ only reduces it in part.
3. $\beta/\log n$ **decreases monotonically** above $10^7$ ($0.1506 \to 0.1392$). In the 4 new windows, $\beta = 0.1065 \log n + 0.658$. So the constant $c_2 \approx 0.143$ of the second regime of Dahan's eq. (16) is not stable at these heights. This is compatible with the first regime starting to dominate for $J \le 20 \lesssim \log n$. But that first regime is not observed in pure form either: an exponential fit $e^{-\gamma J}$ remains worse than the power-law fit in every window (in $[3\cdot10^8, 10^9)$, $R^2 = 0.930$ against $0.984$).

## 3. Result 2: exceptions of the row $u = 1$ (Aigner's slice)

**Lemma.** Let $p \equiv 1 \pmod 4$ be prime. The following are equivalent:
- (i) $4/p$ has a type II solution in the row $u = 1$;
- (ii) there are integers $b, e \ge 1$ with $4 \mid p + e$, $e \mid b + 1$ and $b \mid (p+e)/4$ (Aigner, with $a_{\mathrm{ET}} = 1$; Elsholtz–Tao, Prop. 2.7);
- (iii) there are integers $c, d, e \ge 1$ with
$$e\,(4cd - 1) = p + 4d .$$

In that case $e \equiv 3 \pmod 4$ and $e \le (p+4)/3$ hold automatically, and the solution is $4/p = 1/(bd) + 1/(cdp) + 1/(bcdp)$ with $b = ce - 1$.

*Proof.* From $4bd = p + e$ and $ce = b + 1$ we get (iii). Conversely, (iii) gives $b := ce - 1 \ge 1$ and $4bd = p + e$. The bound $e \le (p+4)/3$ follows from $e \le (p+4d)/(4d-1)$, and the congruence from reducing (iii) modulo 4. For $p \ge 5$ we have $e \le (p+4)/3 < p$, so $p \nmid e$ and $p \nmid 4bd = p + e$. Hence the denominator $bd$ is coprime to $p$ and the other two are multiples of $p$: the solution is of type II. $\square$

**Exhaustive procedure in $O(\sqrt p \log p)$, without factoring.** Let $S = \lfloor \sqrt p \rfloor$. There are two cases:
- **If $e \le S$.** The numbers $K_e = (p+e)/4$, with $e \equiv 3 \pmod 4$, form a block of at most $\sqrt p / 4$ consecutive integers. We invert the loop: we run through the possible small divisors $q \le \sqrt{K_{\max}} \approx \sqrt p/2$ and, for each one, only through its multiples $K$ inside the block. For each $K$ we set $e = 4K - p$ and check whether $q$ or $K/q$ is $\equiv -1 \pmod e$. This suffices, because $\min(b, d) \le \sqrt{K_e}$. The cost is $\sum_{q \le \sqrt p/2} (1 + \sqrt p/(4q)) = O(\sqrt p \log p)$.
- **If $e > S$.** Dividing (iii) by $e$ and using $e \ge 3$ and $c \ge 1$, we get $\tfrac83\,cd - 1 < \sqrt p$, that is, $cd \le \lfloor 3(S+2)/8 \rfloor$. There are $O(\sqrt p \log p)$ such pairs $(c, d)$, and for each one it suffices to check whether $(4cd-1) \mid p + 4d$.

No solution is missed, and there are no false positives. We validated this against the brute-force search $e \le (p+4)/3$ on the 2212 primes $p \equiv 1 \pmod{24}$ below $2\cdot10^5$, with 0 discrepancies.

**Result.** *For $p \equiv 1 \pmod{24}$ prime with $p < 10^9$, the only exceptions of the row $u = 1$ are*
$$p \in \{193,\ 2521,\ 66\,529\}.$$
We obtained this with two independent implementations. Each uses its own filtering phase (with different thresholds: $cd \le 600$ and $cd \le 1000$) followed by an exhaustive decision for the survivors. The three exceptions do have solutions in the column $a = 1$ and in the type I slice with $a = 1$:

| $p$ | column $a = 1$ (type II) | type I, $a = 1$ |
|---|---|---|
| 193 | $1/52 + 1/772 + 1/5018$ | $1/58 + 1/290 + 1/27985$ |
| 2521 | $1/638 + 1/55462 + 1/804199$ | $1/652 + 1/18908 + 1/23833534$ |
| 66529 | $1/17060 + 1/665290 + 1/226996948$ | $1/17038 + 1/698558 + 1/3319597513$ |

**Comparison and corollary.**
- The column $a = 1$ has 14 exceptions below $5\cdot 10^9$, Dahan's "wild" primes (Prop. 3.8). The two exceptional sets are **disjoint**, so the union of the row $u = 1$ and the column $a = 1$ covers every $p \equiv 1 \pmod{24}$ below $10^9$.
- Since Dahan's Theorem 4.3 holds for every finite set $P$ of coprime pairs, taking $P = \{(1, a) : a \le J\}$ shows that the number of exceptions of the row $u = 1$ up to $N$ is $\ll_A N(\log N)^{-A}$ for every $A$. This is a density bound, not a finiteness statement. Whether the set is finite is an open question, and the data suggest it is.

## 4. Complementary elementary observations

For $p$ prime in the six hard classes (in particular $p \equiv 1 \pmod{24}$), with the condition on $x = K_c$:
- **$c = 3$:** there is a solution $\iff$ $K_3$ has a prime factor $\equiv 2 \pmod 3$. Types I and II exist or fail together. The "fails" direction is in Dahan, Thm. 4.8(a).
- **$c = 7$:** since $2 \mid K_7$, there is type I $\iff$ there is type II $\iff$ $K_7$ has a prime factor that is a non-residue modulo 7.
- **$c = 15$:** there is type I $\iff$ there is type II $\iff$ $K_{15}$ has a prime factor with Jacobi symbol $(q/15) = -1$.
- For the primes $c \ge 23$ with $c \equiv 7 \pmod 8$, the analogue **fails**: there are residue classes where it is not enough, and counterexamples with actual primes, checked for $c \le 79$.

All of this was checked without discrepancies on the 20 513 hard primes below $10^7$. For $c = 7$ and $c = 15$ we also checked it exhaustively at the level of residues, over all multisets of up to 8 prime factors.

**Auxiliary data.** We computed $m(p) = 4x_{\min} - p$ for the **14 215 707 hard primes below $10^{10}$**, with two independent implementations (Numba and sympy) that give identical results.
- The record is still $m = 107$, at $p = 8\,803\,369$: no new one appears in three full decades. The next value is $m = 83$, reached by exactly five primes (the smallest being $287\,567\,281$).
- The proportion of primes **not** solved at the first shift ($c = 3$), multiplied by $\sqrt{\log x}$, stays at $2.23$ between $10^5$ and $10^{10}$. None of these primes is an exception of the type I slice with $a = 1$, which agrees with the verification up to $10^{14}$ by Bello-Hernández, Benito and Fernández cited by Elsholtz–Tao in Remark 2.5.

## 5. Theorem: the blind sets of each shift

For $c \equiv 3 \pmod 4$, we say that a prime $n \equiv 1 \pmod{24}$, with $n > c$, is **blind at $c$** if there is no solution of type I or of type II with first denominator $x = K_c = (n+c)/4$. That is, if the shift $c$ does not work for that prime. Dahan proved (Thm. 4.15) that there are $\asymp N(\log N)^{-3/2}$ primes blind at $c = 7$. For $c = 3$ he only obtained the upper bound (Prop. 4.14) and pointed out a 2-adic obstruction (pp. 27 and 44–45).

**Theorem.** *For every integer $c \equiv 3 \pmod 4$,*
$$\#\{n \le N \text{ prime, blind at } c\} \asymp_c \frac{N}{(\log N)^{3/2}}.$$

*Proof (sketch).*

**Lower bound.** It is a corollary of Theorem 1.1 of Fuchs, Hsu, Rickards, Schindler and Stange [FHRSS], which gives $\gg N(\log N)^{-3/2}$ primes primitively represented by $B\,Q(x,y) + A$, with $A = -c$. Let $c_0$ be the squarefree part of $c$; then $c_0 \equiv c \pmod 8$. We choose a form $Q$ of discriminant $-c_0$ and a value of $B$ according to the class of $c_0$ modulo 24:

| class | $Q$ | $B$ |
|---|---|---|
| $c_0 = 3$ | $x^2 + xy + y^2$ | 4 |
| $c_0 \equiv 19$ | $x^2 + xy + \frac{c_0+1}{4}y^2$ | 4 |
| $c_0 \equiv 11$ | $3u^2 + uv + \frac{c_0+1}{12}v^2$ | 12 |
| $c_0 \equiv 7$ | $x^2 + xy + \frac{c_0+1}{4}y^2$ | 16 |
| $c_0 \equiv 23$ | $3u^2 + uv + \frac{c_0+1}{12}v^2$ | 48 |

If $3 \mid c$ but $3 \nmid c_0$, part (2) of the theorem is used with modulus 3. In each case its hypotheses are checked: $(a, 2D) = 1$, $\gcd(A, B) = 1$, and $2 \mid AB$ or $D \not\equiv 5 \pmod 8$. Every prime obtained this way satisfies $n \equiv 1 \pmod{24}$, and all the prime factors of $K_c$ lie in the kernel of the odd character $\chi_{c_0}$. By Dahan (Prop. 4.9), this blocks both branches.

**The case $c = 3$.** The key is to use $x^2 + xy + y^2$, of discriminant $-3$, instead of Dahan's $U^2 + 3V^2$. Since $-3 \equiv 5 \pmod 8$, 2 is inert in $\mathbb Z[\omega]$ and **all primitive values are odd**, so no parity selection is needed. The primes blind at $c = 3$ are *exactly* the primes of the form $4(x^2+xy+y^2) - 3$ with $\gcd(x, y) = 1$. Up to $10^9$ there are 2 399 781 of them, and the ratio to $N(\log N)^{-3/2}$ stays between 0.235 and 0.226.

**Upper bound.** If there is no type II solution, the residues modulo $c$ of the prime factors of $K_c$, together with 1, contain no pair $\{x, -x\}$. Hence $K_c$ avoids at least $\varphi(c)/2$ classes, and an upper sieve of dimension $1/2$ (as in Dahan, Prop. 4.14), summed over the finitely many possible cases, gives $\ll_c N(\log N)^{-3/2}$. $\square$

**Numerical check.** Two independent implementations agree: up to $10^9$ with $c \in \{3, 7, 11, 19, 23, 31, 43, 15, 27, 35, 63, 99\}$, and up to $10^7$ with $c \in \{3, 7, 11, 19, 23\}$. For $c = 7$ Dahan's numbers are reproduced (18 732 and 9 590 up to $10^7$).

**The frontier: blind at $c = 3$ and at $c = 7$ simultaneously.** This is equivalent to $m(p) \ge 11$. Up to $10^5$ there are 131 such primes, but it is not known how to prove that there are infinitely many.
- There is no linear two-parameter family producing them. The quaternary form $b\,g_7 - a\,g_3$ has determinant $21 \cdot \square$, which is not a square, and hence Witt index $\le 1$.
- There is a one-parameter family: $(u, v) = (t+5, 19)$ in $u^2+uv+2v^2$ and $(x, y) = (t, 29)$ in $x^2+xy+y^2$ differ by exactly 1. This gives
$$p(t) = 4t^2 + 116t + 3361 = (2t+29)^2 + 2520,$$
and whenever $p(t)$ is prime (with $\gcd(t, 29) = \gcd(t+5, 19) = 1$) it is blind at $c = 3$ and at $c = 7$.
- **Conditional corollary.** *If Bunyakovsky's conjecture holds for $4t^2 + 116t + 3361$, there are infinitely many primes $p$ with $m(p) \ge 11$.*

### 5.1. Is $m(p)$ bounded? Conditional results

Let $m(p) = 4x_{\min}(p) - p$ be the first shift that works for $p$. If $m(p)$ were bounded, a finite number of shifts would suffice for every prime.

**Lemma.** Suppose that $p > C$ and that every prime factor of every $K_c$ with $c \equiv 3 \pmod 4$, $c \le C$, is a quadratic residue modulo $p$. Then $m(p) > C$. This follows from the reciprocity law $(q/c) = (q/p)$ for $q \mid K_c$ and from Dahan, Prop. 4.9.

**Theorem (conditional on Dickson's conjecture).** Let $L = 4\,\mathrm{lcm}(1, \dots, J)\prod_{q \le 4J-1} q$ and $p = Lt + 1$. Then
$$K_{4j-1} = j \cdot q_j(t), \qquad q_j(t) = 1 + \tfrac{L}{4j}\,t \quad (1 \le j \le J),$$
and if $p, q_1, \dots, q_J$ are all prime, then $m(p) > 4J - 1$. The $(J+1)$-tuple is admissible, so **Dickson's conjecture implies that $m(p)$ is unbounded**. We have not found this statement in the literature.

*Explicit examples:* with $J = 9$ (ten simultaneous prime forms), $p = 2\,706\,777\,030\,078\,416\,655\,361$ has $m(p) = 59$ exactly. Over all the searches, Bateman–Horn predicts 30.75 tuples and 29 were found.

**With a single quadratic (Bunyakovsky):**
- $m(p) \ge 11$ with $4t^2 + 116t + 3361$ (§5).
- $m(p) \ge 15$ for $c = 3, 7, 11$ simultaneously. The corresponding genus-1 curve is birational to $E_{231} : y^2 = x^3 - 231^2 x$, which has rank $\ge 1$ (the point $P_0 = (-63, 1764)$ has infinite order). The point $5P_0$ has a denominator $D$ all of whose prime factors are $\equiv 1 \pmod 3$, and this allows us to build an explicit quadratic polynomial $P(u)$, irreducible and without fixed prime divisors, all of whose prime values have $m(p) \ge 15$. For example, $P(0)$ is a 120-digit prime. The checked certificate is in `data/e231_certificate.json`.
- For $c = 3, \dots, 15$ the curve has genus 5: by Faltings, it has only finitely many rational points.

**Unconditionally**, it is not even known that there are infinitely many primes with $m(p) \ge 11$. It would require a sieve of dimension 1 over primes, which runs into the parity barrier. The closest result in the literature is Nath–Xie (2026): primes with $p - 1$ a sum of two squares and $p + 2$ with at most 9 prime factors.

**Records.** We computed the exact $m(p)$ for the 128 671 219 hard primes up to $10^{11}$; up to $10^{10}$, with two independent implementations. The record is still $m = 107$ ($p = 8\,803\,369$), followed by $m = 103$ ($p = 45\,195\,081\,409$) and $m = 99$ ($p = 92\,159\,707\,201$). All values $3 \le m \le 107$ already occur.

## 6. Observation: the numerator barrier

We checked the results of Bright–Loughran numerically, with two independent implementations of the Hilbert symbol:
- Thm. 1.2, $\prod_{p \mid n} (-u_1/u_3, -u_2/u_3)_p = -1$ for every positive solution with $n$ odd: on all solutions of every odd $n \le 2000$ with one implementation, and $\le 151$ with the other;
- Thm. 1.5 (sign $+1$ for non-positive integer solutions), Cor. 1.3 (primes) and Cor. 1.4 (odd squares).

**An explicit example of why Brauer–Manin is not enough.** Let $U_{m,n} : m\,u_1u_2u_3 = n(u_1u_2 + u_1u_3 + u_2u_3)$. All these surfaces are isomorphic over $\mathbb Q$, and according to Bright–Loughran (Remark 1.10) the version of their Thm. 1.8 also holds for $m \ne 4$: there is no Brauer–Manin obstruction. For $m = 8$ and $n = 241$ we checked this directly:
- The integral point $(30, -14460, -14460) \in U_{8,241}(\mathbb Z)$ has local invariant $+1$ at every place.
- The positive real points have invariant $-1$.
- There are $241$-adically integral points with invariant $-1$.

Combining them gives an adelic point in the Brauer–Manin set. **However, $8/241$ is not a sum of three unit fractions** (exhaustive search, in agreement with the table of Pomerance–Weingartner). So any argument that worked equally well with 8 in place of 4 would "prove" something false: a proof of Erdős–Straus has to use quantitatively that the numerator is 4. The experts probably know this already; we include it as an explicit illustration.

Along the same lines, the squarefree composites $105$, $345$ and $561$ have hundreds of solutions, but none of type I or of type II. Not being a square is not enough for solutions of these two types to exist.

## 7. Where would a counterexample be? "Almost square" primes above $10^{18}$

Odd squares have no solutions of type I or of type II, and for a prime these are the only possible ones (§1, §6). The natural family in which to look for a counterexample is therefore the primes that "imitate" a square: the $p$ with $(q/p) = +1$ for every prime $q \le Y$, that is, with $q_{\min}(p) > Y$. They make up a proportion of the primes of order $2^{-\pi(Y)}$, so they are also plentiful below $10^{18}$, where everything has already been verified. The subfamily built by the Chinese remainder theorem, $p \equiv r \pmod{L(Y)}$ with $L(Y) = 8\prod_{3 \le q \le Y} q$, on the other hand, makes it possible to produce them **above $10^{18}$** (for $Y \ge 47$, $L(Y) > 2\cdot 10^{18}$), in the unverified range.

**Starting data ($p \le 2\cdot10^5$).** The number of solutions of $4/p$ clearly decreases as $q_{\min}(p)$ grows. In the class $p \equiv 1 \pmod{24}$, the mean of $f_4(p)/(\log p)^3$ is $0.498$, $0.411$, $0.343$, $0.306$, $0.280$, $0.257$ and $0.248$ for $q_{\min} = 5, 7, 11, 13, 17, 19, 23$.

**Search.** We take three families with $Y \in \{47, 53, \dots, 101\}$ and $p$ between $10^{19}$ and $10^{43}$:
- the primes $p \equiv 1 \pmod{L(Y)}$, with $L(Y) = 8\prod_{3 \le q \le Y} q$;
- primes with random quadratic residues modulo each $q \le Y$;
- a control group of random hard primes of the same size.

That makes 1620 primes in total. For each one we look for a solution using a partial factorization of the $K_c$, so the $m$ obtained is an upper bound for $m(p)$.

**Result.** All 1620 have a solution, and every triple is checked in exact integer arithmetic. The adversarial families are clearly harder. With the exact $m(p)$, the family $p \equiv 1 \pmod{L(47)}$ (its first 100 primes, of about 20 digits) gives:

| | mean of $m$ | $P(m > 11)$ | $P(m > 23)$ | max. |
|---|---:|---:|---:|---:|
| family $p \equiv 1 \pmod{L(47)}$ | 8.28 | 17 % | 8 % | 39 |
| control (two independent samples) | 5.1 | 1–2 % | 0 % | 15–23 |

Even so, no prime in these families comes close to having no solution: the largest exact $m$ we obtained is 39.

**Caveat.** Above $10^{18}$ a counterexample **cannot be certified**: that would require an infeasible exhaustive search. This search can only find solutions. Nor is it a verification of the whole range; it is a targeted sample.

**Heuristic (not a proof).** We fitted a Poisson model with a local factor that depends on $L(1, \chi_p)$ and on the small divisors of $p - 1$. For the worst possible prime above $10^{18}$, the model predicts thousands of solutions, and an astronomically small expected number of counterexamples with every variant we tried. The model is crude: it describes very small primes poorly, and for $m = 8$ it explains the exceptions $131$ and $241$ only if overdispersion is added. The specific exponents mean nothing, but the qualitative conclusion is robust: the only obstacle to a proof is that no one currently knows how to bound from below, **for each prime**, a sum of polylogarithmic size.

## 8. Reproducibility

| What | Script |
|---|---|
| Shared exact verifier | `tools/verify.py` |
| Result 1 (Numba) | `code/numba/dahan_E1_1e9.py` |
| Result 2 (Numba) | `code/numba/aigner_exceptions_1e9.py` |
| Results 1 and 2 and §4 (independent implementation, sympy and numpy) | `code/verification/check_dahan_aigner_1e9.py` |
| §4 | `code/numba/local_criteria.py`, `code/numba/shifts_7mod8.py`, `code/numba/explore_1e7.py`, `code/numba/explore_1e8.py`, `code/verification/check_1e8.py` |
| §5 | `code/numba/blind_1e9.py`, `code/numba/frontier_3_7.py`, `code/verification/check_blind.py` |
| §5.1 | `code/numba/dickson.py`, `code/numba/curve_E231.py`, `code/numba/records_1e11.py`, `code/verification/check_m_unbounded.py` |
| §6 | `code/numba/bright_loughran.py`, `code/numba/composites_and_forms.py`, `code/verification/check_bright_loughran.py` |
| §7 and records up to $10^{10}$ | `code/numba/fm_exact.py`, `code/numba/fm_model.py`, `code/numba/records_1e10.py`, `code/numba/search_1e18.py`, `code/verification/check_adversarial.py`, `code/verification/check_sweep_1e10.py` |
| Aggregated data | `data/dahan_E1_1e9.csv` (and its plot `.png`), `data/aigner_II_exceptions_1e9.csv` |

Timings on an Intel i7-11370H (8 threads):
- Results 1 and 2 with Numba: about 46 s in total (29 s + 17 s, including validations and compilation).
- With sympy: about 270 s, for a single script that does both results and the checks of §4.

## 9. Limitations

Everything above is numerical verification up to the stated bounds, except for the lemmas and theorems, which come with their proofs. Every numerical result was computed twice, with independent implementations, and the citations were checked against the original texts. Any remaining errors are the author's responsibility.

## References

- B. Dahan, *Sieve dimension and search depth for the Erdős–Straus conjecture, $n \equiv 1 \pmod{24}$*, arXiv:2608.24035 (v2, Nov. 2026).
- C. Elsholtz, T. Tao, *Counting the number of solutions to the Erdős–Straus equation on unit fractions*, J. Aust. Math. Soc. 94 (2013); arXiv:1107.1010.
- S. E. Salez, *The Erdős–Straus conjecture: new modular equations and checking up to $N = 10^{17}$*, arXiv:1406.6307 (2014).
- S. Mihnea, B. C. Dumitru, *Further verification and empirical evidence for the Erdős–Straus conjecture*, arXiv:2509.00128 (2025).
- R. C. Vaughan, *On a problem of Erdős, Straus and Schinzel*, Mathematika 17 (1970).
- C. Pomerance, A. Weingartner, *Exceptions to the Erdős–Straus–Schinzel conjecture*, arXiv:2511.16817.
- E. Fuchs, C. Hsu, J. Rickards, D. Schindler, K. E. Stange, *Primes represented by shifted quadratic forms: on primitivity and congruence classes*, arXiv:2504.20289 (2025). [FHRSS]
- K. Nath, L. Xie, *Almost primes and primes that are sums of two squares plus one*, Acta Arith. (2026); arXiv:2501.16723.
- M. Bright, D. Loughran, *Brauer–Manin obstruction for Erdős–Straus surfaces*, Bull. London Math. Soc. 52 (2020); arXiv:1908.02526.
