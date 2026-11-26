# Conjuntos ciegos y observaciones computacionales sobre la conjetura de Erdős–Straus

**Andy Casafranca** · noviembre de 2026

> **Resumen.** **Teorema (§5).** Para todo $c \equiv 3 \pmod 4$, los primos en los que falla el desplazamiento $c$ (sin soluciones de tipo I ni II con $x = (n+c)/4$) son $\asymp_c N(\log N)^{-3/2}$. Esto generaliza el Teorema 4.15 de Dahan, que trata $c = 7$, y cierra el caso $c = 3$, que él dejaba abierto por una obstrucción 2-ádica. La clave es usar la forma de discriminante $-3$; el teorema es un corolario de Fuchs–Hsu–Rickards–Schindler–Stange. **Sobre $m(p)$**, el primer desplazamiento que funciona para $p$, probamos que la conjetura de Dickson sobre $k$-uplas de primos implica que **$m(p)$ no está acotado**. También que la conjetura de Bunyakovsky para una sola cuadrática implica $m(p) \ge 11$, o $m(p) \ge 15$ con una cuadrática construida a partir de la curva $y^2 = x^3 - 231^2x$, infinitas veces. Hasta $10^{11}$ el récord es $m = 107$.
>
> Además, para los primos $p \equiv 1 \pmod{24}$ menores que $10^9$ (6 354 932 primos) presentamos dos resultados computacionales.
> **(1)** Replicamos de forma exacta y ampliamos de $2\cdot 10^7$ a $10^9$ los experimentos numéricos de Dahan (arXiv:2608.24035) sobre el conjunto excepcional $E_1(N;J)$. Las pendientes locales de $\log(|E_1|/\pi)$ frente a $\log\log n$ quedan a menos del 5 % del exponente de criba $-J/2$ de su Teorema 4.3 para $J \le 6$; con sus datos esto no se observaba. En cambio, el cociente $\beta/\log n$ de su segundo régimen heurístico **no** es constante.
> **(2)** El corte de las soluciones de tipo II "con $a = 1$" en la parametrización de Aigner / Elsholtz–Tao, que es la fila $u = 1$ de la rejilla de Dahan, tiene exactamente **tres** excepciones por debajo de $10^9$: $193$, $2521$ y $66\,529$. Damos además un procedimiento exhaustivo de coste $O(\sqrt p \log p)$ para decidir ese corte.
> Añadimos un ejemplo explícito (§6): la superficie de $8/241$ no tiene obstrucción de Brauer–Manin y, sin embargo, $8/241$ no es suma de tres fracciones unitarias. Muestra por qué una demostración tiene que usar cuantitativamente el numerador 4.
> Además (§7), hicimos una búsqueda dirigida de contraejemplos **por encima de $10^{18}$**, hasta $10^{43}$. La hicimos en primos construidos para parecerse a un cuadrado: residuos cuadráticos módulo todo primo $q \le Y$, con $Y \le 101$. Son mediblemente más difíciles que los primos aleatorios, pero los 1620 probados tienen solución, verificada en aritmética exacta.
>
> **No es un avance hacia una demostración.** La conjetura sigue abierta.

---

## 1. Contexto

La conjetura de Erdős–Straus (1948) afirma que para todo $n \ge 2$ existen enteros positivos $x, y, z$ con
$$\frac{4}{n} = \frac1x + \frac1y + \frac1z .$$
Basta probarla para $n = p$ primo. Las identidades clásicas y las de Mordell la resuelven salvo para $p \bmod 840 \in \{1, 121, 169, 289, 361, 529\}$, que son los cuadrados invertibles módulo 840. Está verificada hasta $10^{17}$ (Salez, 2014) y hasta $10^{18}$ (Mihnea–Dumitru, 2025, preprint). Ninguna identidad polinomial cubre una clase que sea un cuadrado (Mordell, Schinzel; ver Elsholtz–Tao 2013, §1). Bright y Loughran (2020) reinterpretan esa obstrucción como una obstrucción de Brauer–Manin a la aproximación fuerte, y prueban que no hay obstrucción de Brauer–Manin a la conjetura misma. Vaughan (1970) probó que el número de excepciones hasta $N$ es $\ll N\exp(-c(\log N)^{2/3})$. Para numeradores $m$ grandes, la conjetura análoga de Schinzel ($m/n$ en lugar de $4/n$) tiene excepciones muy grandes (Pomerance–Weingartner).

Para un primo impar $p$, toda solución es de **tipo I** ($p$ divide exactamente uno de los denominadores) o de **tipo II** (divide exactamente dos).

**Notación.** Para $c \equiv 3 \pmod 4$ ponemos $K_c = (p+c)/4$. Por Dahan (Thm. 3.9), $4/p$ tiene solución de tipo II si y solo si existen $u, a \ge 1$ y un divisor $s$ de $ap + u$ con $s \equiv -1 \pmod{4ua}$. En términos de $x = K_c$, esto equivale a que existan divisores $u, v$ de $K_c$ con $uv \mid K_c$ y $u + v = a\,c$. Al conjunto de pares $(u, a)$ lo llamamos la **rejilla de Dahan**.
- La **columna $a = 1$** ($u + v = c$, el Algoritmo 2 de Dahan) es la familia "hiperbólica" $p = 4\alpha uv - u - v$.
- La **fila $u = 1$** (uno de los divisores vale 1) es el corte $a_{\mathrm{ET}} = 1$ de la parametrización de tipo II de Elsholtz–Tao, Prop. 2.7 (Aigner, Rosati).

En las variables de Elsholtz–Tao, el "$a$" de Dahan es el $c_{\mathrm{ET}}$ y su "$u$" es el $a_{\mathrm{ET}}$.

## 2. Resultado 1: los experimentos de Dahan, replicados y ampliados

Sea $\mathcal N(N)$ el conjunto de primos $n \le N$ con $n \equiv 1 \pmod{24}$, y sea $E_1(N;J) \subseteq \mathcal N(N)$ el subconjunto de los que no tienen ningún $u \le J$ admisible en la columna $a = 1$ (Dahan, Def. 4.1). Dahan demuestra (Thm. 4.3) que
$$|E_1(N;J)| \ll_J \frac{N}{(\log N)^{1+J/2}},$$
y en su §6(11) mide, hasta $2\cdot 10^7$, dos cosas:
- la pendiente de $\log\big(|E_1|/|\mathcal N|\big)$ frente a $\log\log n$, en 6 ventanas y para $J = 1, \dots, 6$;
- el exponente $\beta$ de un ajuste $J^{-\beta}$ para $J = 1, \dots, 20$ en cada ventana (Tabla 2), que según su segundo régimen heurístico debería cumplir $\beta \approx c_2 \log n$.

**Replicación.** Calculamos $u_{\min}(n) = \min\{u \le 20 \text{ admisible}\}$ para todos los $n \in \mathcal N(10^9)$, con dos implementaciones independientes: una criba invertida compilada con Numba y una factorización directa con sympy. Ambas dan resultados idénticos. Por debajo de $2\cdot 10^7$ reproducimos exactamente los números de Dahan: hay 158 595 primos, los valores de $\beta$ de la Tabla 2, el ajuste $\beta = 0.1433 \log n + 0.043$ ($R^2 = 0.967$) y las pendientes $-0.39, -0.50, -0.84, -1.24, -1.51, -1.59$.

**Ampliación.** Las cuatro ventanas nuevas aparecen después de la línea horizontal:

| ventana | primos | $\beta$ | $\beta/\log n_{\rm mid}$ | $R^2$ |
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

Pendientes de $\log(|E_1|/|\mathcal N|)$ frente a $\log\log n_{\rm mid}$, donde $n_{\rm mid}$ es el punto medio geométrico de cada ventana:

| ventanas usadas | $J=1$ | $2$ | $3$ | $4$ | $5$ | $6$ |
|---|---:|---:|---:|---:|---:|---:|
| predicción $-J/2$ | −0.50 | −1.00 | −1.50 | −2.00 | −2.50 | −3.00 |
| las 6 de Dahan ($<2\cdot10^7$) | −0.387 | −0.499 | −0.838 | −1.235 | −1.509 | −1.587 |
| las 10 ($<10^9$) | −0.412 | −0.645 | −1.013 | −1.406 | −1.721 | −1.888 |
| las 9 sin $[10^3,10^4)$ | −0.454 | −0.901 | −1.283 | −1.663 | −1.968 | −2.236 |
| las 4 nuevas ($[2\cdot10^7, 10^9)$) | **−0.507** | **−1.021** | **−1.464** | **−1.906** | **−2.488** | **−2.929** |

**Interpretación.**
1. Entre $2\cdot 10^7$ y $10^9$, las pendientes locales están a menos del 5 % de $-J/2$ para todo $J \le 6$. Hay millones de primos por ventana, así que el error estadístico es pequeño. Pero el rango es estrecho: $\Delta \log\log n = 0.15$, y los términos de orden inferior podrían seguir pesando. Lo tomamos como **apoyo empírico** a que el exponente $J/2$ del Teorema 4.3 sea el verdadero, **no** como una prueba de una cota inferior.
2. El déficit que observa Dahan se debe sobre todo a la altura: quitar la ventana pequeña $[10^3, 10^4)$ solo lo reduce en parte.
3. $\beta/\log n$ **baja de forma monótona** por encima de $10^7$ ($0.1506 \to 0.1392$). En las 4 ventanas nuevas, $\beta = 0.1065 \log n + 0.658$. Así que la constante $c_2 \approx 0.143$ del segundo régimen de la ec. (16) de Dahan no es estable a estas alturas. Es compatible con que, para $J \le 20 \lesssim \log n$, empiece a mandar el primer régimen. Pero tampoco se observa ese primer régimen puro: un ajuste exponencial $e^{-\gamma J}$ sigue siendo peor que el potencial en todas las ventanas (en $[3\cdot10^8, 10^9)$, $R^2 = 0.930$ frente a $0.984$).

## 3. Resultado 2: excepciones de la fila $u = 1$ (corte de Aigner)

**Lema.** Sea $p \equiv 1 \pmod 4$ primo. Las condiciones siguientes son equivalentes:
- (i) $4/p$ tiene una solución de tipo II en la fila $u = 1$;
- (ii) existen enteros $b, e \ge 1$ con $4 \mid p + e$, $e \mid b + 1$ y $b \mid (p+e)/4$ (Aigner, con $a_{\mathrm{ET}} = 1$; Elsholtz–Tao, Prop. 2.7);
- (iii) existen enteros $c, d, e \ge 1$ con
$$e\,(4cd - 1) = p + 4d .$$

En ese caso se cumple automáticamente $e \equiv 3 \pmod 4$ y $e \le (p+4)/3$, y la solución es $4/p = 1/(bd) + 1/(cdp) + 1/(bcdp)$ con $b = ce - 1$.

*Prueba.* De $4bd = p + e$ y $ce = b + 1$ se obtiene (iii). Recíprocamente, (iii) da $b := ce - 1 \ge 1$ y $4bd = p + e$. Que $e \le (p+4)/3$ sale de $e \le (p+4d)/(4d-1)$, y la congruencia de reducir (iii) módulo 4. Para $p \ge 5$ se tiene $e \le (p+4)/3 < p$, luego $p \nmid e$ y $p \nmid 4bd = p + e$. Así que el denominador $bd$ es coprimo con $p$ y los otros dos son múltiplos de $p$: la solución es de tipo II. $\square$

**Procedimiento exhaustivo en $O(\sqrt p \log p)$, sin factorizar.** Sea $S = \lfloor \sqrt p \rfloor$. Se distinguen dos casos:
- **Si $e \le S$.** Los números $K_e = (p+e)/4$, con $e \equiv 3 \pmod 4$, forman un bloque de a lo sumo $\sqrt p / 4$ enteros consecutivos. Invertimos el bucle: recorremos los posibles divisores pequeños $q \le \sqrt{K_{\max}} \approx \sqrt p/2$ y, para cada uno, solo sus múltiplos $K$ dentro del bloque. Para cada $K$ ponemos $e = 4K - p$ y comprobamos si $q$ o $K/q$ es $\equiv -1 \pmod e$. Basta, porque $\min(b, d) \le \sqrt{K_e}$. El coste es $\sum_{q \le \sqrt p/2} (1 + \sqrt p/(4q)) = O(\sqrt p \log p)$.
- **Si $e > S$.** Dividiendo (iii) por $e$ y usando $e \ge 3$ y $c \ge 1$, se llega a $\tfrac83\,cd - 1 < \sqrt p$, es decir, $cd \le \lfloor 3(S+2)/8 \rfloor$. Hay $O(\sqrt p \log p)$ pares $(c, d)$ así, y para cada uno basta comprobar si $(4cd-1) \mid p + 4d$.

Ninguna solución se escapa, y no hay falsos positivos. Lo validamos contra la búsqueda bruta $e \le (p+4)/3$ en los 2212 primos $p \equiv 1 \pmod{24}$ menores que $2\cdot10^5$, con 0 discrepancias.

**Resultado.** *Para $p \equiv 1 \pmod{24}$ primo con $p < 10^9$, las únicas excepciones de la fila $u = 1$ son*
$$p \in \{193,\ 2521,\ 66\,529\}.$$
Lo obtuvimos con dos implementaciones independientes. Cada una usa una fase de filtrado propia (con umbrales distintos: $cd \le 600$ y $cd \le 1000$) seguida de una decisión exhaustiva para los supervivientes. Las tres excepciones sí tienen solución en la columna $a = 1$ y en el corte de tipo I con $a = 1$:

| $p$ | columna $a = 1$ (tipo II) | tipo I, $a = 1$ |
|---|---|---|
| 193 | $1/52 + 1/772 + 1/5018$ | $1/58 + 1/290 + 1/27985$ |
| 2521 | $1/638 + 1/55462 + 1/804199$ | $1/652 + 1/18908 + 1/23833534$ |
| 66529 | $1/17060 + 1/665290 + 1/226996948$ | $1/17038 + 1/698558 + 1/3319597513$ |

**Comparación y corolario.**
- La columna $a = 1$ tiene 14 excepciones por debajo de $5\cdot 10^9$, los primos "wild" de Dahan (Prop. 3.8). Los dos conjuntos excepcionales son **disjuntos**, de modo que la unión de la fila $u = 1$ y la columna $a = 1$ cubre todos los $p \equiv 1 \pmod{24}$ menores que $10^9$.
- Como el Teorema 4.3 de Dahan vale para todo conjunto finito $P$ de pares coprimos, tomando $P = \{(1, a) : a \le J\}$ se obtiene que el número de excepciones de la fila $u = 1$ hasta $N$ es $\ll_A N(\log N)^{-A}$ para todo $A$. Es una cota de densidad, no de finitud. Que el conjunto sea finito es una pregunta abierta, y los datos lo sugieren.

## 4. Observaciones elementales complementarias

Para $p$ primo en las seis clases difíciles (en particular $p \equiv 1 \pmod{24}$), con la condición sobre $x = K_c$:
- **$c = 3$:** hay solución $\iff$ $K_3$ tiene un factor primo $\equiv 2 \pmod 3$. Los tipos I y II existen o fallan a la vez. La dirección "falla" está en Dahan, Thm. 4.8(a).
- **$c = 7$:** como $2 \mid K_7$, hay tipo I $\iff$ hay tipo II $\iff$ $K_7$ tiene un factor primo no residuo módulo 7.
- **$c = 15$:** hay tipo I $\iff$ hay tipo II $\iff$ $K_{15}$ tiene un factor primo con símbolo de Jacobi $(q/15) = -1$.
- Para los primos $c \ge 23$ con $c \equiv 7 \pmod 8$, el análogo **falla**: hay clases de restos en las que no basta, y contraejemplos con primos reales comprobados para $c \le 79$.

Todo esto está comprobado sin discrepancias en los 20 513 primos difíciles menores que $10^7$. Para $c = 7$ y $c = 15$, además, lo comprobamos de forma exhaustiva a nivel de restos, sobre todos los multiconjuntos de hasta 8 factores primos.

**Datos auxiliares.** Calculamos $m(p) = 4x_{\min} - p$ para los **14 215 707 primos difíciles menores que $10^{10}$**, con dos implementaciones independientes (Numba y sympy) que dan resultados idénticos.
- El récord sigue siendo $m = 107$, en $p = 8\,803\,369$: en tres décadas completas no aparece ninguno nuevo. El siguiente valor es $m = 83$, que alcanzan exactamente cinco primos (el menor, $287\,567\,281$).
- La proporción de primos que **no** se resuelven en el primer desplazamiento ($c = 3$), multiplicada por $\sqrt{\log x}$, se mantiene en $2.23$ entre $10^5$ y $10^{10}$. Ninguno de esos primos es excepción del corte de tipo I con $a = 1$, lo cual concuerda con la verificación hasta $10^{14}$ de Bello-Hernández, Benito y Fernández citada por Elsholtz–Tao en el Remark 2.5.

## 5. Teorema: los conjuntos ciegos de cada desplazamiento

Para $c \equiv 3 \pmod 4$, decimos que un primo $n \equiv 1 \pmod{24}$, con $n > c$, es **ciego en $c$** si no hay ninguna solución de tipo I ni de tipo II con primer denominador $x = K_c = (n+c)/4$. Es decir, si el desplazamiento $c$ no sirve para ese primo. Dahan demostró (Thm. 4.15) que hay $\asymp N(\log N)^{-3/2}$ primos ciegos en $c = 7$. Para $c = 3$ solo obtuvo la cota superior (Prop. 4.14) y señaló una obstrucción 2-ádica (pp. 27 y 44–45).

**Teorema.** *Para todo entero $c \equiv 3 \pmod 4$,*
$$\#\{n \le N \text{ primo ciego en } c\} \asymp_c \frac{N}{(\log N)^{3/2}}.$$

*Prueba (esquema).*

**Cota inferior.** Es un corolario del Teorema 1.1 de Fuchs, Hsu, Rickards, Schindler y Stange [FHRSS], que da $\gg N(\log N)^{-3/2}$ primos representados primitivamente por $B\,Q(x,y) + A$, con $A = -c$. Sea $c_0$ la parte libre de cuadrados de $c$; se cumple $c_0 \equiv c \pmod 8$. Se elige una forma $Q$ de discriminante $-c_0$ y un valor de $B$ según la clase de $c_0$ módulo 24:

| clase | $Q$ | $B$ |
|---|---|---|
| $c_0 = 3$ | $x^2 + xy + y^2$ | 4 |
| $c_0 \equiv 19$ | $x^2 + xy + \frac{c_0+1}{4}y^2$ | 4 |
| $c_0 \equiv 11$ | $3u^2 + uv + \frac{c_0+1}{12}v^2$ | 12 |
| $c_0 \equiv 7$ | $x^2 + xy + \frac{c_0+1}{4}y^2$ | 16 |
| $c_0 \equiv 23$ | $3u^2 + uv + \frac{c_0+1}{12}v^2$ | 48 |

Si $3 \mid c$ pero $3 \nmid c_0$, se usa la parte (2) del teorema con módulo 3. En cada caso se comprueban sus hipótesis: $(a, 2D) = 1$, $\gcd(A, B) = 1$, y $2 \mid AB$ o $D \not\equiv 5 \pmod 8$. Todo primo así obtenido cumple $n \equiv 1 \pmod{24}$, y todos los factores primos de $K_c$ están en el núcleo del carácter impar $\chi_{c_0}$. Por Dahan (Prop. 4.9), eso bloquea las dos ramas.

**El caso $c = 3$.** La clave es usar $x^2 + xy + y^2$, de discriminante $-3$, en lugar de la $U^2 + 3V^2$ de Dahan. Como $-3 \equiv 5 \pmod 8$, el 2 es inerte en $\mathbb Z[\omega]$ y **todos los valores primitivos son impares**, así que no hace falta seleccionar la paridad. Los primos ciegos en $c = 3$ son *exactamente* los primos de la forma $4(x^2+xy+y^2) - 3$ con $\gcd(x, y) = 1$. Hasta $10^9$ hay 2 399 781, y el cociente frente a $N(\log N)^{-3/2}$ se mantiene entre 0.235 y 0.226.

**Cota superior.** Si no hay tipo II, los restos módulo $c$ de los factores primos de $K_c$, junto con el 1, no contienen ningún par $\{x, -x\}$. Por tanto $K_c$ evita al menos $\varphi(c)/2$ clases, y una criba superior de dimensión $1/2$ (como en Dahan, Prop. 4.14), sumada sobre los finitos casos posibles, da $\ll_c N(\log N)^{-3/2}$. $\square$

**Comprobación numérica.** Dos implementaciones independientes coinciden: hasta $10^9$ con $c \in \{3, 7, 11, 19, 23, 31, 43, 15, 27, 35, 63, 99\}$, y hasta $10^7$ con $c \in \{3, 7, 11, 19, 23\}$. Para $c = 7$ se reproducen los números de Dahan (18 732 y 9 590 hasta $10^7$).

**La frontera: ciegos en $c = 3$ y en $c = 7$ a la vez.** Esto equivale a $m(p) \ge 11$. Hasta $10^5$ hay 131 de estos primos, pero no se sabe demostrar que haya infinitos.
- No existe ninguna familia lineal en dos parámetros que los produzca. La forma cuaternaria $b\,g_7 - a\,g_3$ tiene determinante $21 \cdot \square$, que no es un cuadrado, y por tanto índice de Witt $\le 1$.
- Sí existe una familia de un parámetro: $(u, v) = (t+5, 19)$ en $u^2+uv+2v^2$ y $(x, y) = (t, 29)$ en $x^2+xy+y^2$ difieren exactamente en 1. Así se obtiene
$$p(t) = 4t^2 + 116t + 3361 = (2t+29)^2 + 2520,$$
y cuando $p(t)$ es primo (con $\gcd(t, 29) = \gcd(t+5, 19) = 1$) es ciego en $c = 3$ y en $c = 7$.
- **Corolario condicional.** *Si la conjetura de Bunyakovsky vale para $4t^2 + 116t + 3361$, hay infinitos primos $p$ con $m(p) \ge 11$.*

### 5.1. ¿Está acotado $m(p)$? Resultados condicionales

Sea $m(p) = 4x_{\min}(p) - p$ el primer desplazamiento que funciona para $p$. Si $m(p)$ estuviera acotado, un número finito de desplazamientos bastaría para todos los primos.

**Lema.** Supongamos que $p > C$ y que todo factor primo de todo $K_c$ con $c \equiv 3 \pmod 4$, $c \le C$, es un residuo cuadrático módulo $p$. Entonces $m(p) > C$. Se sigue de la ley de reciprocidad $(q/c) = (q/p)$ para $q \mid K_c$ y de Dahan, Prop. 4.9.

**Teorema (condicional a la conjetura de Dickson).** Sean $L = 4\,\mathrm{mcm}(1, \dots, J)\prod_{q \le 4J-1} q$ y $p = Lt + 1$. Entonces
$$K_{4j-1} = j \cdot q_j(t), \qquad q_j(t) = 1 + \tfrac{L}{4j}\,t \quad (1 \le j \le J),$$
y si $p, q_1, \dots, q_J$ son todos primos, se cumple $m(p) > 4J - 1$. La $(J+1)$-upla es admisible, así que **la conjetura de Dickson implica que $m(p)$ no está acotado**. No hemos encontrado este enunciado en la literatura.

*Ejemplos explícitos:* con $J = 9$ (diez formas primas a la vez), $p = 2\,706\,777\,030\,078\,416\,655\,361$ tiene $m(p) = 59$ exacto. En todas las búsquedas, Bateman–Horn predice 30.75 tuplas y se encontraron 29.

**Con una sola cuadrática (Bunyakovsky):**
- $m(p) \ge 11$ con $4t^2 + 116t + 3361$ (§5).
- $m(p) \ge 15$ para $c = 3, 7, 11$ a la vez. La curva de género 1 correspondiente es birracional a $E_{231} : y^2 = x^3 - 231^2 x$, que tiene rango $\ge 1$ (el punto $P_0 = (-63, 1764)$ es de orden infinito). El punto $5P_0$ tiene denominador $D$ con todos sus factores primos $\equiv 1 \pmod 3$, y eso permite construir un polinomio cuadrático explícito $P(u)$, irreducible y sin divisores primos fijos, cuyos valores primos tienen todos $m(p) \ge 15$. Por ejemplo, $P(0)$ es un primo de 120 cifras. El certificado comprobado está en `datos/e231_certificado.json`.
- Para $c = 3, \dots, 15$ la curva tiene género 5: por Faltings, solo finitos puntos racionales.

**Incondicionalmente**, ni siquiera se sabe que haya infinitos primos con $m(p) \ge 11$. Haría falta una criba de dimensión 1 sobre primos, que choca con la barrera de paridad. Lo más cercano en la literatura es Nath–Xie (2026): primos con $p - 1$ suma de dos cuadrados y $p + 2$ con a lo sumo 9 factores primos.

**Récords.** Calculamos $m(p)$ exacto para los 128 671 219 primos difíciles hasta $10^{11}$; hasta $10^{10}$, con dos implementaciones independientes. El récord sigue siendo $m = 107$ ($p = 8\,803\,369$), seguido de $m = 103$ ($p = 45\,195\,081\,409$) y $m = 99$ ($p = 92\,159\,707\,201$). Ya aparecen todos los valores $3 \le m \le 107$.

## 6. Observación: la barrera del numerador

Los resultados de Bright–Loughran los comprobamos numéricamente, con dos implementaciones independientes del símbolo de Hilbert:
- el Thm. 1.2, $\prod_{p \mid n} (-u_1/u_3, -u_2/u_3)_p = -1$ para toda solución positiva con $n$ impar: en todas las soluciones de todo $n$ impar $\le 2000$ con una implementación, y $\le 151$ con la otra;
- el Thm. 1.5 (signo $+1$ para soluciones enteras no positivas), el Cor. 1.3 (primos) y el Cor. 1.4 (cuadrados impares).

**Un ejemplo explícito de por qué Brauer–Manin no basta.** Sea $U_{m,n} : m\,u_1u_2u_3 = n(u_1u_2 + u_1u_3 + u_2u_3)$. Todas estas superficies son isomorfas sobre $\mathbb Q$, y según Bright–Loughran (Remark 1.10) también para $m \ne 4$ vale la versión de su Thm. 1.8: no hay obstrucción de Brauer–Manin. Para $m = 8$ y $n = 241$ lo comprobamos directamente:
- El punto entero $(30, -14460, -14460) \in U_{8,241}(\mathbb Z)$ tiene invariante local $+1$ en todos los lugares.
- Los puntos positivos reales tienen invariante $-1$.
- Hay puntos $241$-ádicamente enteros con invariante $-1$.

Combinándolos se obtiene un punto adélico en el conjunto de Brauer–Manin. **Sin embargo, $8/241$ no es suma de tres fracciones unitarias** (búsqueda exhaustiva, de acuerdo con la tabla de Pomerance–Weingartner). Así que cualquier argumento que funcionara igual con 8 en lugar de 4 "probaría" algo falso: una demostración de Erdős–Straus tiene que usar cuantitativamente que el numerador es 4. Probablemente los expertos ya lo saben; lo incluimos como ilustración explícita.

En la misma línea, los compuestos libres de cuadrados $105$, $345$ y $561$ tienen cientos de soluciones, pero ninguna de tipo I ni de tipo II. No ser un cuadrado no basta para que existan soluciones de esos dos tipos.

## 7. ¿Dónde estaría un contraejemplo? Primos "casi cuadrados" por encima de $10^{18}$

Los cuadrados impares no tienen soluciones de tipo I ni de tipo II, y para un primo esas son las únicas posibles (§1, §6). La familia natural donde buscar un contraejemplo son, por tanto, los primos que "imitan" a un cuadrado: los $p$ con $(q/p) = +1$ para todo primo $q \le Y$, es decir, con $q_{\min}(p) > Y$. Son una proporción de los primos del orden de $2^{-\pi(Y)}$, así que también abundan por debajo de $10^{18}$, donde ya están verificados. La subfamilia construida por el teorema chino de los restos, $p \equiv r \pmod{L(Y)}$ con $L(Y) = 8\prod_{3 \le q \le Y} q$, permite en cambio fabricarlos **por encima de $10^{18}$** (para $Y \ge 47$, $L(Y) > 2\cdot 10^{18}$), en la zona no verificada.

**Datos de partida ($p \le 2\cdot10^5$).** El número de soluciones de $4/p$ baja de forma clara al crecer $q_{\min}(p)$. En la clase $p \equiv 1 \pmod{24}$, la media de $f_4(p)/(\log p)^3$ vale $0.498$, $0.411$, $0.343$, $0.306$, $0.280$, $0.257$ y $0.248$ para $q_{\min} = 5, 7, 11, 13, 17, 19, 23$.

**Búsqueda.** Tomamos tres familias con $Y \in \{47, 53, \dots, 101\}$ y $p$ entre $10^{19}$ y $10^{43}$:
- los primos $p \equiv 1 \pmod{L(Y)}$, con $L(Y) = 8\prod_{3 \le q \le Y} q$;
- primos con restos cuadráticos aleatorios módulo cada $q \le Y$;
- un grupo de control de primos difíciles aleatorios del mismo tamaño.

En total son 1620 primos. Para cada uno buscamos una solución con factorización parcial de los $K_c$, de modo que el $m$ obtenido es una cota superior de $m(p)$.

**Resultado.** Los 1620 tienen solución, y cada terna está comprobada con aritmética entera exacta. Las familias adversarias son claramente más difíciles. Con $m(p)$ exacto, en la familia $p \equiv 1 \pmod{L(47)}$ (los 100 primeros primos, de unas 20 cifras) se obtiene:

| | media de $m$ | $P(m > 11)$ | $P(m > 23)$ | máx. |
|---|---:|---:|---:|---:|
| familia $p \equiv 1 \pmod{L(47)}$ | 8.28 | 17 % | 8 % | 39 |
| control (dos muestras independientes) | 5.1 | 1–2 % | 0 % | 15–23 |

Aun así, ningún primo de estas familias está cerca de no tener solución: el mayor $m$ exacto que obtuvimos es 39.

**Advertencia.** Por encima de $10^{18}$ un contraejemplo **no se puede certificar**: haría falta una búsqueda exhaustiva inviable. Esta búsqueda solo puede encontrar soluciones. Tampoco es una verificación del rango completo, sino una muestra dirigida.

**Heurística (no demostración).** Hemos ajustado un modelo de Poisson con un factor local que depende de $L(1, \chi_p)$ y de los divisores pequeños de $p - 1$. Para el peor primo posible por encima de $10^{18}$, el modelo predice miles de soluciones, y un número esperado de contraejemplos astronómicamente pequeño con cualquiera de las variantes que probamos. El modelo es tosco: describe mal los primos muy pequeños, y para $m = 8$ explica las excepciones $131$ y $241$ solo si se le añade sobredispersión. Los exponentes concretos no significan nada, pero la conclusión cualitativa es robusta: el único obstáculo para una demostración es que hoy no se sabe acotar inferiormente, **para cada primo**, una suma de tamaño polilogarítmico.

## 8. Reproducibilidad

| Qué | Script |
|---|---|
| Verificador exacto común | `herramientas/verificar.py` |
| Resultado 1 (Numba) | `codigo/numba/dahan_E1_1e9.py` |
| Resultado 2 (Numba) | `codigo/numba/aigner_excepciones_1e9.py` |
| Resultados 1 y 2 y §4 (implementación independiente, sympy y numpy) | `codigo/verificacion/verif_dahan_aigner_1e9.py` |
| §4 | `codigo/numba/criterios_locales.py`, `codigo/numba/desplazamientos_7mod8.py`, `codigo/numba/explorar_1e8.py`, `codigo/verificacion/verif_1e8.py` |
| §5 | `codigo/numba/ciegos_1e9.py`, `codigo/numba/frontera_3_7.py`, `codigo/verificacion/verif_ciegos.py` |
| §5.1 | `codigo/numba/dickson.py`, `codigo/numba/curva_E231.py`, `codigo/numba/records_1e11.py`, `codigo/verificacion/verif_m_no_acotado.py` |
| §6 | `codigo/numba/bright_loughran.py`, `codigo/numba/compuestos_y_formas.py`, `codigo/verificacion/verif_bright_loughran.py` |
| §7 y récords hasta $10^{10}$ | `codigo/numba/fm_exacto.py`, `codigo/numba/records_1e10.py`, `codigo/numba/busqueda_1e18.py`, `codigo/verificacion/verif_adversarios.py`, `codigo/verificacion/verif_barrido_1e10.py` |
| Datos agregados | `datos/dahan_E1_1e9.csv` (y su gráfico `.png`), `datos/excepciones_aigner_II_1e9.csv` |

Tiempos en un Intel i7-11370H (8 hilos):
- Resultados 1 y 2 con Numba: unos 46 s en total (29 s + 17 s, contando validaciones y compilación).
- Con sympy: unos 270 s, para un solo script que hace ambos resultados y las comprobaciones de §4.

## 9. Limitaciones

Todo lo anterior es verificación numérica hasta las cotas indicadas, salvo los lemas y teoremas, que llevan su prueba. Cada resultado numérico se calculó dos veces, con implementaciones independientes, y las citas se contrastaron con los textos originales. Los errores que queden son responsabilidad del autor.

## Referencias

- B. Dahan, *Sieve dimension and search depth for the Erdős–Straus conjecture, $n \equiv 1 \pmod{24}$*, arXiv:2608.24035 (v2, nov. 2026).
- C. Elsholtz, T. Tao, *Counting the number of solutions to the Erdős–Straus equation on unit fractions*, J. Aust. Math. Soc. 94 (2013); arXiv:1107.1010.
- S. E. Salez, *The Erdős–Straus conjecture: new modular equations and checking up to $N = 10^{17}$*, arXiv:1406.6307 (2014).
- S. Mihnea, B. C. Dumitru, *Further verification and empirical evidence for the Erdős–Straus conjecture*, arXiv:2509.00128 (2025).
- R. C. Vaughan, *On a problem of Erdős, Straus and Schinzel*, Mathematika 17 (1970).
- C. Pomerance, A. Weingartner, *Exceptions to the Erdős–Straus–Schinzel conjecture*, arXiv:2511.16817.
- E. Fuchs, C. Hsu, J. Rickards, D. Schindler, K. E. Stange, *Primes represented by shifted quadratic forms: on primitivity and congruence classes*, arXiv:2504.20289 (2025). [FHRSS]
- K. Nath, L. Xie, *Almost primes and primes that are sums of two squares plus one*, Acta Arith. (2026); arXiv:2501.16723.
- M. Bright, D. Loughran, *Brauer–Manin obstruction for Erdős–Straus surfaces*, Bull. London Math. Soc. 52 (2020); arXiv:1908.02526.
