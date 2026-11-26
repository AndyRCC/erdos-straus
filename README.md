# Erdős–Straus: conjuntos ciegos y exploración computacional

**Andy Casafranca** · noviembre de 2026

> **English summary.** Notes on the (still open) Erdős–Straus conjecture 4/n = 1/x + 1/y + 1/z.
> **Theorem.** For every c ≡ 3 (mod 4), the number of primes n ≤ N for which the shift c fails (no Type I or Type II solution with first denominator x = (n+c)/4) is ≍ N(log N)^{−3/2}. This extends Dahan's Theorem 4.15 (c = 7, arXiv:2608.24035) and settles the case c = 3, which was left open. It follows from Fuchs–Hsu–Rickards–Schindler–Stange (arXiv:2504.20289) applied to the form of discriminant −3.
> **Conditional results.** Dickson's prime k-tuples conjecture implies that the first working shift m(p) is unbounded. Bunyakovsky's conjecture for a single explicit quadratic implies m(p) ≥ 15 infinitely often; that quadratic is built from the congruent-number curve y² = x³ − 231²x.
> **Computations.** We replicate and extend to 10⁹ Dahan's sieve experiments. The Aigner Type II slice with a = 1 has exactly three exceptions below 10⁹ (193, 2521 and 66529). We compute m(p) exactly for all 128 671 219 "hard" primes below 10¹¹. We also run a targeted search above 10¹⁸ among primes that mimic squares.
> **This is not a proof of the conjecture.** Details are in [`nota/nota_erdos_straus.md`](nota/nota_erdos_straus.md) (in Spanish).

**La conjetura (1948):** para todo `n ≥ 2` existen enteros positivos `x, y, z` con `4/n = 1/x + 1/y + 1/z`. Sigue abierta y está verificada hasta `10^18`. Este repositorio **no** la demuestra.

## Resultados

Todo está en la nota [`nota/nota_erdos_straus.md`](nota/nota_erdos_straus.md):

1. **Teorema (§5).** Para todo `c ≡ 3 (mod 4)`, los primos en los que falla el desplazamiento `c` son `≍ N(log N)^{−3/2}`. Generaliza el Teorema 4.15 de Dahan (c = 7) y cierra el caso `c = 3`.
2. **¿Está acotado m(p)? (§5.1)**
   - Si la conjetura de Dickson es cierta, `m(p)` no está acotado: ningún número finito de desplazamientos basta para todos los primos.
   - Si la de Bunyakovsky vale para una sola cuadrática, hay infinitos primos con `m(p) ≥ 11`, o con `m(p) ≥ 15` usando la curva E₂₃₁.
   - Hasta `10^11` el récord es `m(p) = 107`.
3. **Experimentos de Dahan (§2):** replicados exactamente y ampliados de `2·10^7` a `10^9`.
4. **Corte de Aigner (§3):** sus únicas excepciones con `p ≡ 1 (mod 24)` y `p < 10^9` son 193, 2521 y 66529, decididas con un algoritmo exhaustivo de coste `O(√p log p)`.
5. **Criterios locales (§4), Bright–Loughran y la barrera del numerador (§6), y búsqueda dirigida por encima de `10^18` (§7).**

Cada resultado numérico se calculó con dos implementaciones independientes: una con Numba (`codigo/numba/`) y otra con sympy (`codigo/verificacion/`).

## Reproducir

```
pip install -r requirements.txt
python herramientas/verificar.py conocidas        # autotest: identidades clásicas y de Mordell
python herramientas/verificar.py mod840           # las 6 clases difíciles módulo 840 y su cobertura
python herramientas/verificar.py perfil 8803369   # m(p) y qué tipos de solución existen
python codigo/verificacion/verif_ciegos.py        # teorema de §5, comprobado hasta 10^7
python codigo/verificacion/verif_m_no_acotado.py  # §5.1: certificado E_231, primos de Dickson y récords
python herramientas/descargar_referencias.py      # opcional: descarga los PDFs citados
```

Los scripts se ejecutan desde la raíz del repositorio. La tabla de la nota, §8, indica qué script reproduce cada resultado.

## Estructura

```
nota/                  la nota con todos los resultados
herramientas/          verificador exacto (verificar.py) y descarga de referencias
codigo/numba/          cálculos grandes (criba segmentada con Numba)
codigo/verificacion/   comprobaciones independientes (sympy y numpy)
datos/                 tablas agregadas, gráficos y el certificado de E_231
referencias/README.md  bibliografía con enlaces (los PDFs no se incluyen)
```

## Licencia

El código está bajo licencia MIT, y la nota y los datos bajo CC BY 4.0 (ver [`LICENSE`](LICENSE)). Los artículos citados pertenecen a sus autores y no se redistribuyen.
