---
title: 3. Asymptotic Equipartition Property
description: How the law of large numbers creates typical sets, why entropy determines compression rates, and how high-probability sets acquire an exponential size.
date: 2026-09-08
lastmod: 2026-09-08
draft: false
tags:
  - information-theory
  - probability
  - coding
  - notes
---

Entropy measures the average surprise of one observation. **What does it tell us about a long sequence of observations?** The asymptotic equipartition property (AEP) gives the bridge: for an independent source, the surprise per symbol concentrates near entropy. A relatively small collection of sequences then carries almost all the probability, which makes compression possible.

This note follows Chapter 3 of Cover and Thomas, *Elements of Information Theory*. It continues [[entropy-relative-entropy-mutual-information|Entropy and Mutual Information]] and [[information-inequalities-data-processing-fano|Inequalities and Their Consequences]]. The explanations and worked calculations below are written for these notes.

> [!note] Assumptions and notation
> Let $X_1,X_2,\ldots$ be independent and identically distributed (i.i.d.) with PMF $p(x)$ on a finite alphabet $\mathcal{X}$. Write $X^n=(X_1,\ldots,X_n)$ for the random block and $x^n$ for a realization. All logarithms are base 2, so entropy and surprise are measured in bits. We restrict attention to symbols with positive probability.

---

## 1. From Average Surprise to the AEP

The probability of a particular sequence factors as

$$
p(x^n)=\prod_{i=1}^n p(x_i).
$$

Taking a negative logarithm turns this product into a sum:

$$
-\frac1n\log_2 p(X^n)
=\frac1n\sum_{i=1}^n\bigl[-\log_2 p(X_i)\bigr].
$$

The expression on the left is random: we first draw a sequence, then evaluate the probability assigned to that sequence. On the right, we average the surprise of its individual symbols.

Define $Z_i=-\log_2p(X_i)$. The $Z_i$ are i.i.d., with

$$
\mathbb{E}[Z_i]
=-\sum_xp(x)\log_2p(x)
=H(X).
$$

The weak law of large numbers therefore gives the **asymptotic equipartition property**:

$$
\boxed{-\frac1n\log_2p(X^n)\xrightarrow{\Pr}H(X).}
$$

Explicitly, for every fixed $\epsilon>0$,

$$
\Pr\left(\left|-\frac1n\log_2p(X^n)-H(X)\right|>\epsilon\right)
\longrightarrow0.
$$

The proof is short because the main idea is the change of variables: apply the law of large numbers to **surprise**, rather than to the source symbols themselves. This is Theorem 3.1.1 in the book.

> [!note]- A finite-block bound
> Let $V=\operatorname{Var}[-\log_2p(X)]$, which is finite under our finite-alphabet assumption. Independence gives
>
> $$
> \operatorname{Var}\left(\frac1n\sum_{i=1}^n Z_i\right)=\frac{V}{n}.
> $$
>
> Chebyshev's inequality then yields
>
> $$
> \Pr\left(\left|-\frac1n\log_2p(X^n)-H(X)\right|>\epsilon\right)
> \leq\frac{V}{n\epsilon^2}.
> $$
>
> This bound can be loose, but it makes the tradeoff explicit: a narrower tolerance generally requires a longer block.

## 2. The Typical Set

For $\epsilon>0$, define the **weakly typical set**

$$
A_\epsilon^{(n)}
=\left\{x^n:\left|-\frac1n\log_2p(x^n)-H(X)\right|\leq\epsilon\right\}.
$$

Equivalently, a sequence is typical when

$$
\boxed{2^{-n(H(X)+\epsilon)}\leq p(x^n)\leq2^{-n(H(X)-\epsilon)}.}
$$

The AEP says $\Pr(X^n\in A_\epsilon^{(n)})\to1$. It does **not** say that most sequences are typical when we count every possible sequence equally. Probability mass and the number of sequences are different quantities.

### 2.1 Why typical sequences are “almost equiprobable”

Every typical sequence has normalized surprise within $\epsilon$ of $H(X)$. This motivates the shorthand

$$
p(x^n)\approx2^{-nH(X)}.
$$

Here the approximation is on the **exponential scale**. For fixed $\epsilon$, the ratio between the largest and smallest allowed probabilities can be as large as $2^{2n\epsilon}$. AEP does not assert that these probability ratios approach one, or that the conditional distribution on the typical set becomes uniform.

### 2.2 How many typical sequences are there?

Fix an error tolerance $0<\delta<1$. For sufficiently large $n$, AEP gives $\Pr(A_\epsilon^{(n)})\geq1-\delta$. Combining this with the probability bounds gives

$$
\boxed{
(1-\delta)2^{n(H(X)-\epsilon)}
\leq\lvert A_\epsilon^{(n)}\rvert
\leq2^{n(H(X)+\epsilon)}.
}
$$

> [!note]- Proof of the cardinality bounds
> Each typical sequence has probability at least $2^{-n(H(X)+\epsilon)}$. Thus
>
> $$
> 1\geq\sum_{x^n\in A_\epsilon^{(n)}}p(x^n)
> \geq\lvert A_\epsilon^{(n)}\rvert2^{-n(H(X)+\epsilon)},
> $$
>
> proving the upper bound. For the lower bound, use the upper probability bound for each sequence:
>
> $$
> 1-\delta\leq\sum_{x^n\in A_\epsilon^{(n)}}p(x^n)
> \leq\lvert A_\epsilon^{(n)}\rvert2^{-n(H(X)-\epsilon)}.
> $$

These are the properties in Theorem 3.1.2, with separate symbols for the surprise tolerance $\epsilon$ and the missing probability $\delta$.

There are $\lvert\mathcal{X}\rvert^n$ possible blocks, but approximately $2^{nH(X)}$ relevant ones at exponential resolution. In fact,

$$
\frac{\lvert A_\epsilon^{(n)}\rvert}{\lvert\mathcal{X}\rvert^n}
\leq2^{-n(\log_2\lvert\mathcal{X}\rvert-H(X)-\epsilon)}.
$$

For a nonuniform source and $\epsilon<\log_2\lvert\mathcal{X}\rvert-H(X)$, this fraction vanishes even while the typical set's probability tends to one.

## 3. A Biased-Coin Example

Let $X_i\sim\operatorname{Bernoulli}(t)$ with $t=0.9$. A particular block containing $k$ ones has probability

$$
p(x^n)=0.9^k0.1^{n-k},
$$

and entropy per symbol

$$
H(X)=H(0.9)=-0.9\log_2(0.9)-0.1\log_2(0.1)
\approx0.4690\text{ bits}.
$$

Writing $r=k/n$, we obtain an exact identity:

$$
-\frac1n\log_2p(x^n)-H(t)
=(r-t)\log_2\frac{1-t}{t}.
$$

Thus, for $t=0.9$, weak typicality is equivalent to

$$
\left|\frac{k}{n}-0.9\right|
\leq\frac{\epsilon}{\log_2 9}.
$$

For $n=100$ and $\epsilon=0.1$, this means **87 through 93 ones**. Summing the binomial probabilities gives

$$
\Pr(A_{0.1}^{(100)})
=\sum_{k=87}^{93}\binom{100}{k}0.9^k0.1^{100-k}
\approx0.75897.
$$

Even a block of 100 symbols does not make this particular typical set overwhelmingly likely. AEP is a limit theorem; finite-block coverage depends on the source and the tolerance.

### 3.1 The most likely sequence can be atypical

The most likely individual block is the all-ones sequence. Its normalized surprise is

$$
-\frac1n\log_2(0.9^n)=-\log_2(0.9)\approx0.1520,
$$

which remains far from $H(0.9)\approx0.4690$. It is atypical whenever $\epsilon<0.3170$, regardless of $n$.

At $n=100$, the contrast is:

| Quantity | Value |
| --- | --- |
| Probability of the single all-ones block | $0.9^{100}\approx2.656\times10^{-5}$ |
| Probability of one particular block with 90 ones | $0.9^{90}0.1^{10}\approx7.618\times10^{-15}$ |
| Number of blocks with exactly 90 ones | $\binom{100}{90}=17{,}310{,}309{,}456{,}440$ |
| Total probability of all blocks with exactly 90 ones | $\binom{100}{90}0.9^{90}0.1^{10}\approx0.13187$ |

One all-ones block is much more probable than one block with 90 ones. But the enormous number of blocks with 90 ones makes their combined probability much larger.

> [!note] The fair-coin boundary case
> If $t=1/2$, every length-$n$ sequence has probability $2^{-n}$ and normalized surprise exactly 1. Every sequence is weakly typical, including all zeros. Weak typicality constrains average surprise; it does not generally require each symbol's empirical frequency to be close to its true probability. That stronger requirement belongs to frequency-based, or strong, typicality.

## 4. Why AEP Enables Compression

If both encoder and decoder know $p$, $n$, and $\epsilon$, they can agree on an ordering of the typical set. A typical sequence can then be represented by its index, requiring at most

$$
\left\lceil n(H(X)+\epsilon)\right\rceil
$$

bits. This is the operational meaning of the typical-set size bound.

### 4.1 Fixed-length coding with a small error probability

Choose a rate $R>H(X)$ and select $0<\epsilon<R-H(X)$. For sufficiently large $n$, $\lfloor nR\rfloor$ bits provide enough distinct indices for every typical sequence. Decode those indices correctly and allow an error on atypical inputs. Then

$$
P_e^{(n)}\leq\Pr(X^n\notin A_\epsilon^{(n)})\longrightarrow0.
$$

This scheme compresses below the raw alphabet rate while allowing a vanishing probability of reconstruction error. It does not represent every possible block without error.

### 4.2 Lossless coding with variable length

Section 3.2 of the book gives a construction that also handles atypical sequences. Use a leading flag bit to distinguish two cases:

| Input block | Encoding |
| --- | --- |
| Typical | `0` followed by its fixed-width typical-set index |
| Atypical | `1` followed by a fixed-width index among all $\lvert\mathcal{X}\rvert^n$ blocks |

The decoder reads the flag, then knows how many bits to read. The resulting block code is prefix-free and reconstructs every block exactly.

Let $\delta_n=\Pr(X^n\notin A_\epsilon^{(n)})$. Allowing for the flag and rounding, typical codewords have length at most $n(H(X)+\epsilon)+2$; atypical codewords have length at most $n\log_2\lvert\mathcal{X}\rvert+2$. Consequently,

$$
\begin{aligned}
\frac{\mathbb{E}[\ell(X^n)]}{n}
&\leq(1-\delta_n)(H(X)+\epsilon)
+\delta_n\log_2\lvert\mathcal{X}\rvert+\frac2n\\
&\leq H(X)+\epsilon
+\delta_n\log_2\lvert\mathcal{X}\rvert+\frac2n.
\end{aligned}
$$

First choose $\epsilon$ small, then choose $n$ large. The expected rate can be made arbitrarily close to $H(X)$ from above, as in Theorem 3.2.1.

The construction proves existence. Enumerating an exponentially large typical set is generally impractical; an efficient implementation needs a more structured coding algorithm.

## 5. Could a Much Smaller Set Carry the Same Probability?

The typical set is one way to capture most of the probability, but it need not be the smallest such set. Section 3.3 asks whether a better choice could change the exponential size.

Fix $0<\delta<1$ and let $B_n$ be any set with $\Pr(X^n\in B_n)\geq1-\delta$. Write $\eta_n=\Pr(X^n\notin A_\epsilon^{(n)})$, which tends to zero. The intersection obeys

$$
\Pr(X^n\in B_n\cap A_\epsilon^{(n)})\geq1-\delta-\eta_n.
$$

Every sequence in that intersection has probability at most $2^{-n(H(X)-\epsilon)}$, so

$$
\boxed{\lvert B_n\rvert\geq(1-\delta-\eta_n)2^{n(H(X)-\epsilon)}}
$$

for sufficiently large $n$ that the prefactor is positive.

Let $M_n(\delta)$ be the minimum cardinality of a set having probability at least $1-\delta$. The intersection bound gives a lower bound, while the typical set supplies an upper bound. Since $\epsilon>0$ can be arbitrarily small,

$$
\boxed{\lim_{n\to\infty}\frac1n\log_2 M_n(\delta)=H(X).}
$$

Thus $M_n(\delta)=2^{nH(X)+o(n)}$: entropy determines the exponential number of sequences required to capture any fixed probability strictly between zero and one.

For the biased coin, a smallest high-probability set can be formed by taking individual sequences in decreasing probability order, starting with all ones. The typical set instead selects a band of normalized surprise. They can differ in membership and still have the same limiting exponent when the typicality tolerance is taken sufficiently small.

### 5.1 A compression converse

A fixed-length code with at most $2^{nR}$ messages can correctly reconstruct at most $2^{nR}$ distinct blocks. Call their set $C_n$. Splitting it into typical and atypical parts gives

$$
\begin{aligned}
\Pr(X^n\in C_n)
&\leq\eta_n+\lvert C_n\rvert2^{-n(H(X)-\epsilon)}\\
&\leq\eta_n+2^{-n(H(X)-\epsilon-R)}.
\end{aligned}
$$

If $R<H(X)$, choose $0<\epsilon<H(X)-R$. Both terms tend to zero. The probability of correct reconstruction then vanishes, so the error probability tends to one. This is the fixed-length strong converse for the source considered here, derived from the same typical-set bounds.

## 6. What the Limit Does—and Does Not—Promise

The order of limits matters. AEP gives high probability for each **fixed** $\epsilon>0$ as $n$ grows. It does not guarantee high probability for an arbitrarily fast shrinking $\epsilon_n$. The Chebyshev bound gives one sufficient condition: $n\epsilon_n^2\to\infty$ when $V$ is finite.

Similarly, for fixed $\epsilon$, the cardinality bounds place the typical set's normalized log-size between approximately $H(X)-\epsilon$ and $H(X)+\epsilon$. To conclude an exact exponent $H(X)$ for a sequence of typical sets, the tolerance must shrink in a way that preserves high probability.

Our proof uses independence to factor block probabilities and identical distributions to apply the law of large numbers. Dependent sources require additional assumptions and a more general theorem; the i.i.d. proof cannot simply be reused unchanged.

The useful connection is now concrete: **entropy controls the surprise of a long random block, the size of sets that capture its probability, and the number of bits needed to describe it.**

---

## Reference

Thomas M. Cover and Joy A. Thomas. *Elements of Information Theory*, 2nd ed. Wiley, 2006. Chapter 3, “Asymptotic Equipartition Property,” especially Sections 3.1–3.3 and Theorems 3.1.1–3.3.1. Reference consulted: the supplied EPUB, ISBN 9781118585771. The biased-coin discussion develops the chapter's Bernoulli example; the finite-block values above are calculated directly.
