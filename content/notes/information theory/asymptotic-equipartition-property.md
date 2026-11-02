---
title: 3. Asymptotic Equipartition Property
description: How the law of large numbers creates typical sets, why entropy determines compression rates, and how high-probability sets acquire an exponential size.
date: 2026-09-08
lastmod: 2026-10-10
draft: false
tags:
  - information-theory
  - probability
  - coding
  - notes
---

Entropy measures the average surprise of one observation. **What does it tell us about a long sequence of observations?** Suppose a coin lands heads 90% of the time. The all-heads sequence is the most likely individual outcome, yet a long run usually contains some tails. To understand what we will actually observe, we need to consider whole collections of sequences and their combined probability.

The **asymptotic equipartition property (AEP)** explains what happens when we draw many independent observations from the same distribution. For a long sequence of $n$ observations, the total surprise $-\log p(X^n)$ is usually close to $nH(X)$. Equivalently, the probability of the sequence we observe is usually around $2^{-nH(X)}$.

These sequences form the **typical set**, which contains almost all the probability. Roughly speaking, if each sequence contributes probability $2^{-nH(X)}$, we need about $2^{nH(X)}$ of them for their probabilities to add up to one. To encode a typical sequence, we can send its index in this set. Choosing among about $2^{nH(X)}$ indices takes about $\log(2^{nH(X)})=nH(X)$ bits. The bounds below make these approximations precise.

We will derive the AEP, count typical sequences, and use that count to explain compression and the size of high-probability sets.

> [!note] Assumptions and notation
> Let $X_1,X_2,\ldots$ be independent and identically distributed (i.i.d.) with PMF $p(x)$ on a finite alphabet $\mathcal{X}$. Write $X^n=(X_1,\ldots,X_n)$ for the random block and $x^n$ for a realization. All logarithms are base 2, so $\log$ means $\log_2$ and entropy and surprise are measured in bits. We restrict attention to symbols with positive probability.

---

## 1. From Average Surprise to the AEP

The probability of a particular sequence factors as

$$
p(x^n)=\prod_{i=1}^n p(x_i).
$$

Taking a negative logarithm turns this product into a sum:

$$
-\frac1n\log p(X^n)
=\frac1n\sum_{i=1}^n\bigl[-\log p(X_i)\bigr].
$$

The expression on the left is random: we first draw a sequence, then evaluate the probability assigned to that sequence. On the right, we average the surprise of its individual symbols.

Define $Z_i=-\log p(X_i)$. The $Z_i$ are i.i.d., with

$$
\mathbb{E}[Z_i]
=-\sum_xp(x)\log p(x)
=H(X).
$$

The weak law of large numbers therefore gives the **asymptotic equipartition property**:

$$
\boxed{-\frac1n\log p(X^n)\xrightarrow{\Pr}H(X).}
$$

Explicitly, for every fixed $\epsilon>0$,

$$
\Pr\left(\left|-\frac1n\log p(X^n)-H(X)\right|>\epsilon\right)
\longrightarrow0.
$$

The proof is short because the main idea is the change of variables: apply the law of large numbers to **surprise**, rather than to the source symbols themselves.

The name now has a precise meaning: **asymptotic** refers to long blocks, and **equipartition** refers to the nearly equal surprise per symbol of blocks that carry most of the probability. It does not make every possible block equally likely.

> [!example]- Problem 1: AEP-like limit
> **Problem.** Let $X_1,X_2,\ldots$ be i.i.d. drawn according to probability mass function $p(x)$. Does the following sequence converge in probability as $n\to\infty$? If so, what does it converge to?
>
> $$
> \bigl(p(X_1,X_2,\ldots,X_n)\bigr)^{1/n}.
> $$
>
> **Hint.** You may use the Continuous Mapping Theorem in this and subsequent problems.
>
> **Solution.**
>
> The AEP gives $-\frac1n\log p(X^n)\xrightarrow{\Pr}H(X)$. Apply the continuous function $z\mapsto2^{-z}$:
>
> $$
> p(X^n)^{1/n}=2^{\frac1n\log p(X^n)}\xrightarrow{\Pr}2^{-H(X)}.
> $$
>
> This is convergence of the probability's $n$th root; the probability of a whole block generally becomes small as its length increases.

> [!example]- Problem 2: Repeated cake cuts
> **Problem.** A cake is sliced roughly in half, the largest piece being chosen each time and the other pieces discarded. Assume that a random cut creates pieces of proportions
>
> $$
> P=\begin{cases}
> (2/3,1/3),&\text{with probability }1/4,\\
> (2/5,3/5),&\text{with probability }3/4.
> \end{cases}
> $$
>
> For example, the first cut and choice of the largest piece may result in a piece of size $3/5$. Cutting and choosing from this piece might reduce its size to $(3/5)(2/3)$ at time 2, and so on. Roughly how large is the piece of cake after $n$ cuts?
>
> **Solution.**
>
> Assume successive cuts are independent, and let $R_i$ be the fraction retained at cut $i$: $2/3$ with probability $1/4$ and $3/5$ with probability $3/4$. If the starting size is one, $C_n=\prod_{i=1}^nR_i$. Taking logarithms lets us use the law of large numbers:
>
> $$
> \frac1n\log C_n\xrightarrow{\Pr}\frac14\log\frac23+\frac34\log\frac35.
> $$
>
> Thus $C_n^{1/n}\xrightarrow{\Pr}(2/3)^{1/4}(3/5)^{3/4}\approx0.6160$. The typical size decays like $(0.6160)^n$ on the logarithmic scale. This does not claim that $C_n/(0.6160)^n$ converges to one.

> [!example]- Problem 3: Geometric mean of a product
> **Problem.** Let
>
> $$
> X=\begin{cases}
> 1,&\text{with probability }1/2,\\
> 2,&\text{with probability }1/4,\\
> 3,&\text{with probability }1/4.
> \end{cases}
> $$
>
> Suppose $X_1,X_2,\ldots$ are drawn i.i.d. according to this distribution. Find the limiting behavior in probability of the product
>
> $$
> (X_1X_2\cdots X_n)^{1/n}.
> $$
>
> **Solution.**
>
> Average the logarithms, then exponentiate:
>
> $$
> \frac1n\log(X_1\cdots X_n)\xrightarrow{\Pr}\mathbb{E}[\log X]=\frac14\log2+\frac14\log3.
> $$
>
> Therefore $(X_1\cdots X_n)^{1/n}\xrightarrow{\Pr}6^{1/4}\approx1.5651$. The geometric mean appears because the quantity being averaged is $\log X$, rather than $X$.

> [!example]- Problem 4: AEP with two probability models
> **Problem.** Let $X_1,X_2,\ldots$ be independent and identically distributed random variables drawn according to the probability mass function $p(x)$, where $x\in\{1,2,\ldots,m\}$. Thus
>
> $$
> p(x_1,x_2,\ldots,x_n)=\prod_{i=1}^np(x_i).
> $$
>
> We know that $-\frac1n\log p(X_1,X_2,\ldots,X_n)\to H(X)$ in probability. Let
>
> $$
> q(x_1,x_2,\ldots,x_n)=\prod_{i=1}^nq(x_i),
> $$
>
> where $q$ is another probability mass function on $\{1,2,\ldots,m\}$.
>
> **(a)** Evaluate the limit in probability of $-\frac1n\log q(X_1,X_2,\ldots,X_n)$, where $X_1,X_2,\ldots$ are i.i.d. $\sim p(x)$.
>
> **(b)** Evaluate the limit in probability of the log-likelihood ratio
>
> $$
> \frac1n\log\frac{q(X_1,\ldots,X_n)}{p(X_1,\ldots,X_n)},
> $$
>
> when $X_1,X_2,\ldots$ are i.i.d. $\sim p(x)$.
>
> **Solution.**
>
> If $q(x)>0$ whenever $p(x)>0$, the law of large numbers gives
>
> $$
> -\frac1n\log q(X^n)\xrightarrow{\Pr}-\sum_xp(x)\log q(x)=H(X)+D(p\Vert q).
> $$
>
> Subtracting the AEP limit for $p$ gives
>
> $$
> \frac1n\log\frac{q(X^n)}{p(X^n)}\xrightarrow{\Pr}-D(p\Vert q).
> $$
>
> Here $D(p\Vert q)=\sum_xp(x)\log[p(x)/q(x)]$. The true distribution $p$ determines the averaging weights. See [[entropy-relative-entropy-mutual-information#32-coding-interpretation-the-cost-of-the-wrong-model|the coding interpretation of relative entropy]].
>
> If $p(x)>0$ but $q(x)=0$ for some symbol, a long block contains such a symbol with probability tending to one. The limits are then $+\infty$ and $-\infty$, respectively, in the extended sense.

## 2. The Typical Set

For $\epsilon>0$, define the **typical set**

$$
A^{(n)}_\epsilon
=\left\{x^n:\left|-\frac1n\log p(x^n)-H(X)\right|\leq\epsilon\right\}.
$$

Equivalently, a sequence is typical when

$$
\boxed{2^{-n(H(X)+\epsilon)}\leq p(x^n)\leq2^{-n(H(X)-\epsilon)}.}
$$

The AEP says $\Pr(X^n\in A^{(n)}_\epsilon)\to1$. It does **not** say that most sequences are typical when we count every possible sequence equally. Probability mass and the number of sequences are different quantities.

### 2.1 Why typical sequences are “almost equiprobable”

Every typical sequence has normalized surprise within $\epsilon$ of $H(X)$. This motivates the shorthand

$$
p(x^n)\approx2^{-nH(X)}.
$$

The probability bounds specify what this approximation means: the normalized surprise of every typical sequence lies between $H(X)-\epsilon$ and $H(X)+\epsilon$.

### 2.2 How many typical sequences are there?

For sufficiently large $n$, the AEP gives $\Pr(A^{(n)}_\epsilon)>1-\epsilon$. Combining this with the probability bounds gives

$$
\boxed{
(1-\epsilon)2^{n(H(X)-\epsilon)}
\leq\lvert A^{(n)}_\epsilon\rvert
\leq2^{n(H(X)+\epsilon)}.
}
$$

> [!note]- Proof of the cardinality bounds
> Each typical sequence has probability at least $2^{-n(H(X)+\epsilon)}$. Thus
>
> $$
> 1\geq\sum_{x^n\in A^{(n)}_\epsilon}p(x^n)
> \geq\lvert A^{(n)}_\epsilon\rvert2^{-n(H(X)+\epsilon)},
> $$
>
> proving the upper bound. For the lower bound, use the upper probability bound for each sequence:
>
> $$
> 1-\epsilon\leq\sum_{x^n\in A^{(n)}_\epsilon}p(x^n)
> \leq\lvert A^{(n)}_\epsilon\rvert2^{-n(H(X)-\epsilon)}.
> $$

The upper bound holds for every $n$; the lower bound requires sufficiently large $n$.

The typical set has probability close to one, its sequences have probabilities close to $2^{-nH(X)}$ on the exponential scale, and its size is bounded by exponents near $nH(X)$. These three properties make it useful for coding.

> [!example]- Problem 5: Counting a high-probability set
> **Problem.** A discrete memoryless source emits a sequence of statistically independent binary digits with probabilities $p(1)=0.005$ and $p(0)=0.995$. The digits are taken 100 at a time, and a binary codeword is provided for every sequence of 100 digits containing three or fewer ones.
>
> **(a)** Assuming that all codewords are the same length, find the minimum length required to provide codewords for all sequences with three or fewer ones.
>
> **(b)** Calculate the probability of observing a source sequence for which no codeword has been assigned.
>
>
> **Solution.**
>
> A block with exactly $k$ ones can place those ones in $\binom{100}{k}$ positions. Thus the selected set $S=\{x^{100}:\text{at most three ones}\}$ has size
>
> $$
> |S|=\sum_{k=0}^3\binom{100}{k}=1+100+4950+161700=166751.
> $$
>
> For part (a), distinct labels of length $l$ require $2^l\geq|S|$. Hence $l=\lceil\log166751\rceil=18$ bits.
>
> For part (b), the number of ones is $K\sim\operatorname{Binomial}(100,0.005)$, so
>
> $$
> \Pr(K>3)=1-\sum_{k=0}^3\binom{100}{k}(0.005)^k(0.995)^{100-k}\approx0.00167327.
> $$
>
> This illustrates the difference between cardinality and probability: 166751 blocks out of $2^{100}$ possibilities carry about 99.833% of the probability. $S$ is a high-probability set defined by a count cutoff, rather than the two-sided typicality criterion.

## 3. A Biased-Coin Example

Consider a Bernoulli source. Let $X_i\sim\operatorname{Bernoulli}(p)$ and write $q=1-p$. A particular block containing $k$ ones has probability

$$
p(x^n)=p^kq^{n-k}.
$$

Its probability depends on the number of ones, not their positions. With high probability, the proportion of ones is close to $p$. The normalized surprise is then close to the binary entropy

$$
H(p)=-p\log p-q\log q,
$$

and these typical sequences have probabilities near $2^{-nH(p)}$.

Now take $p=0.9$. The typical sequences have a proportion of ones close to 0.9. The most likely individual sequence, however, is the all-ones sequence. It falls outside a sufficiently narrow typical band.

This distinction matters: the most likely individual outcome need not belong to the collection of outcomes that carries most of the probability. We will use this distinction to compare typical sets with the smallest high-probability sets.

> [!example]- Problem 5: Probability of a biased-coin count range
> **Problem.** A discrete memoryless source emits a sequence of statistically independent binary digits with probabilities $p(1)=0.005$ and $p(0)=0.995$. The digits are taken 100 at a time, and a binary codeword is provided for every sequence of 100 digits containing three or fewer ones.
>
> **(a)** Assuming that all codewords are the same length, find the minimum length required to provide codewords for all sequences with three or fewer ones.
>
> **(b)** Calculate the probability of observing a source sequence for which no codeword has been assigned.
>
>
> **Solution.**
>
> For a particular block with $k$ ones,
>
> $$
> p(x^{100})=(0.005)^k(0.995)^{100-k}.
> $$
>
> There are $\binom{100}{k}$ such blocks. To find the total probability of a count, multiply the probability of one block by the number of blocks with that count.
>
> For part (a), $\sum_{k=0}^3\binom{100}{k}=166751$, so the minimum length is $\lceil\log166751\rceil=18$ bits.
>
> For part (b), add the probabilities of all excluded counts:
>
> $$
> \Pr(K>3)=\sum_{k=4}^{100}\binom{100}{k}(0.005)^k(0.995)^{100-k}\approx0.00167327.
> $$
>
> Here $K$ is binomial with parameters 100 and 0.005. The selected count range has probability about 0.99832673. This calculation shows why the probability of one sequence and the probability of a whole collection must be distinguished.

## 4. Why AEP Enables Compression

If both encoder and decoder know $p$, $n$, and $\epsilon$, they can agree on an ordering of the typical set. A typical sequence can then be represented by its index, requiring at most

$$
\left\lceil n(H(X)+\epsilon)\right\rceil
$$

bits. This is the operational meaning of the typical-set size bound.

To encode every possible block, we need to handle both typical and atypical sequences. Use a leading flag bit to distinguish two cases:

| Input block | Encoding |
| --- | --- |
| Typical | `0` followed by its fixed-width typical-set index |
| Atypical | `1` followed by a fixed-width index in the atypical set |

The decoder reads the flag, then knows how many bits to read. The mapping is one-to-one and reconstructs every block exactly.

Allowing for the flag and rounding, typical codewords have length at most $n(H(X)+\epsilon)+2$. Atypical codewords have length at most $n\log\lvert\mathcal{X}\rvert+2$, because there are no more than $\lvert\mathcal{X}\rvert^n$ atypical blocks.

For sufficiently large $n$, $\Pr(X^n\notin A^{(n)}_\epsilon)\leq\epsilon$. Splitting the expected codeword length between the two sets gives

$$
\begin{aligned}
\mathbb{E}[l(X^n)]
&\leq\Pr(A^{(n)}_\epsilon)\bigl[n(H(X)+\epsilon)+2\bigr]\\
&\quad+\Pr((A^{(n)}_\epsilon)^c)\bigl[n\log\lvert\mathcal{X}\rvert+2\bigr]\\
&\leq n(H(X)+\epsilon)+\epsilon n\log\lvert\mathcal{X}\rvert+2.
\end{aligned}
$$

Thus

$$
\frac{\mathbb{E}[l(X^n)]}{n}
\leq H(X)+\epsilon+\epsilon\log\lvert\mathcal{X}\rvert+\frac2n.
$$

First choose $\epsilon$ small, then choose $n$ large. The terms beyond $H(X)$ can be made arbitrarily small. Consequently, for any positive rate tolerance, sufficiently long i.i.d. blocks have a one-to-one binary code whose expected bits per symbol are at most entropy plus that tolerance.

The essential idea is that typical sequences receive short descriptions of approximately $nH(X)$ bits. Atypical sequences may receive longer descriptions, but their small total probability keeps their contribution to the average small.

> [!example]- Problem 5: Encoding blocks with at most three ones
> **Problem.** A discrete memoryless source emits a sequence of statistically independent binary digits with probabilities $p(1)=0.005$ and $p(0)=0.995$. The digits are taken 100 at a time, and a binary codeword is provided for every sequence of 100 digits containing three or fewer ones.
>
> **(a)** Assuming that all codewords are the same length, find the minimum length required to provide codewords for all sequences with three or fewer ones.
>
> **(b)** Calculate the probability of observing a source sequence for which no codeword has been assigned.
>
> **Solution.**
>
> There are
>
> $$
> M=\sum_{k=0}^3\binom{100}{k}=1+100+4950+161700=166751
> $$
>
> eligible blocks. Distinct fixed-length binary labels require $2^l\geq M$, so $l=\lceil\log M\rceil=18$ bits.
>
> If $K$ counts ones, then $K\sim\operatorname{Binomial}(100,0.005)$. The probability of having no assigned codeword is
>
> $$
> \Pr(K>3)=1-\sum_{k=0}^3\binom{100}{k}(0.005)^k(0.995)^{100-k}\approx0.00167327.
> $$
>
> Thus the selected set covers about 99.833% of the probability. It is a high-probability set selected by a count cutoff; it is not defined by the two-sided typical-set criterion. Unlike the flag construction, this code leaves some blocks unencoded.

## 5. Could a Much Smaller Set Carry the Same Probability?

The typical set is one way to capture most of the probability, but it need not be the smallest such set. Could a better choice change the exponential size?

Let $B^{(n)}_\delta$ be a smallest set of blocks satisfying

$$
\Pr(X^n\in B^{(n)}_\delta)\geq1-\delta.
$$

The lower bound is as follows: for $0<\delta<1/2$ and any $\delta'>0$, a set with probability greater than $1-\delta$ satisfies

$$
\boxed{\frac1n\log\lvert B^{(n)}_\delta\rvert>H(X)-\delta'}
$$

for sufficiently large $n$. To see why, note that the typical set has probability approaching one. Any set with probability greater than $1-\delta$ must therefore share substantial probability with it. Each sequence in that intersection has probability at most $2^{-n(H(X)-\epsilon)}$, so carrying that shared probability requires exponentially many sequences.

The typical set has size between exponents near $nH(X)$, and a smallest high-probability set cannot have a substantially smaller exponent. We express this using **equality to first order in the exponent**:

$$
a_n\doteq b_n
\quad\Longleftrightarrow\quad
\lim_{n\to\infty}\frac1n\log\frac{a_n}{b_n}=0.
$$

For a fixed $0<\delta<1/2$, the smallest-set result is

$$
\boxed{\lvert B^{(n)}_\delta\rvert\doteq2^{nH(X)}.}
$$

The two sets can nevertheless differ in membership. For the Bernoulli source with parameter 0.9, a smallest high-probability set is obtained by taking the most probable individual sequences first, including all ones. The typical set instead selects sequences whose proportion of ones is close to 0.9. Their different selection rules still lead to the same leading entropy exponent as the typicality tolerance is made small.

Entropy therefore connects three quantities: the surprise per symbol of a long random block, the exponential size of a high-probability set, and the average number of bits needed to describe a block.

> [!example]- Problem 5: Selecting the most probable blocks
> **Problem.** A discrete memoryless source emits a sequence of statistically independent binary digits with probabilities $p(1)=0.005$ and $p(0)=0.995$. The digits are taken 100 at a time, and a binary codeword is provided for every sequence of 100 digits containing three or fewer ones.
>
> **(a)** Assuming that all codewords are the same length, find the minimum length required to provide codewords for all sequences with three or fewer ones.
>
> **(b)** Calculate the probability of observing a source sequence for which no codeword has been assigned.
>
>
> **Solution.**
>
> For part (a), the selected set $S$ has $\sum_{k=0}^3\binom{100}{k}=166751$ blocks, requiring $\lceil\log166751\rceil=18$ bits per label.
>
> For part (b), its missing probability is
>
> $$
> \delta=1-\sum_{k=0}^3\binom{100}{k}(0.005)^k(0.995)^{100-k}\approx0.00167327,
> $$
>
> so $\Pr(S)=1-\delta\approx0.99832673$.
>
> The probability of an individual block decreases with its number of ones: increasing that count by one multiplies its probability by $0.005/0.995<1$. Thus $S$ contains the most probable individual blocks, down to the cutoff of three ones.
>
> This makes $S$ a smallest-cardinality set for this exact coverage target. Any set with fewer than $|S|$ elements has total probability at most that of the $|S|-1$ most probable blocks, which is strictly below $\Pr(S)$. This is a finite-block illustration of selecting a smallest high-probability set; the asymptotic result above describes its leading size as block length grows with a fixed probability target.

## 6. Takeaways

- **Average surprise becomes predictable.** For i.i.d. symbols, $-\frac1n\log p(X^n)$ converges in probability to $H(X)$.
- **Typicality concerns total probability.** The typical set carries almost all the probability, even though a particular typical sequence can be very unlikely. The most likely individual sequence need not be typical.
- **Entropy controls the number of descriptions.** Typical-set size is bounded by exponents near $nH(X)$, so a typical block needs about $nH(X)$ bits to identify.
- **Compression guarantees matter.** The flag construction recovers every block exactly and approaches entropy in expected bits per symbol. Problem 5 instead uses fixed-length labels for a selected set and leaves a small probability of unencoded blocks.
- **For products, take logarithms first.** Problems 1–4 turn multiplicative quantities into averages, apply the law of large numbers, and transform back when needed.
