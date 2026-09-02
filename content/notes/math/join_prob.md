---
title: 2. Joint Probability Distributions
description: An introduction to joint, marginal, and conditional probability distributions, covering the chain rule, independence, and Bayes' theorem with worked examples.
date: 2025-09-14
lastmod: 2026-08-30
draft: false
tags:
  - math
  - notes
---

Joint probability distributions are fundamental to understanding how multiple random variables behave together.

---

## Joint Distribution

A joint probability distribution describes the probability of events involving multiple random variables simultaneously. Think of it as extending our understanding from single-variable probability to multi-variable scenarios where we can capture complex relationships and dependencies.

> Joint distributions are high-dimensional PDFs (continuous variables) or PMFs (discrete variables).

### Mathematical Formulation

For two discrete random variables $X$ and $Y$, the joint PMF is written as $p(x, y)$, where:

$$
p(x, y) = p(X = x, Y = y)
$$

For continuous variables, PDF are computed by integrating:

$$
p(a \leq X \leq b,\; c \leq Y \leq d) = \int_a^b \int_c^d p(x,y) \, dx \, dy
$$

As we add more variables, the dimensionality grows naturally:

- 1D: $p(x)$
- 2D: $p(x,y)$
- 3D: $p(x,y,z)$
- $n$D: $p(\mathbf{x})$ or $f(\mathbf{x})$ where $\mathbf{x} = (x_1, x_2, \ldots, x_n), \quad \mathbf{x}\in \mathbb{R}^n$

### Essential Properties

Every distributbion must satisfy these fundamental properties:

1. **Non-negativity**: $p(\mathbf{x}) \geq 0$ for all $\mathbf{x} = (x_1, x_2, ..., x_n)$
2. **Normalization**: $\sum p(\mathbf{x}) = 1$ (discrete) or $\int\_{-\infty}^{\infty} f(\mathbf{x}) d \mathbf{x} = 1 $ (continuous)

These properties ensure that joint distributions are valid probability measures.

---

## Conditional Distributions

Conditional probability answers: "Given that $X$ has occurred, what's the probability that $Y$ also occurs?" It's like updating our beliefs based on new information.

$$
p(Y \mid X) = \frac{p(X, Y)}{p(X)}
\Rightarrow p(Y=y \mid X=x) = \frac{p(X=x, Y=y)}{p(X=x)}, \forall x,y
$$

You might wonder: "what is the difference between this and the joint distribution $p(x,y)$?" You can think of conditional probability as focusing on the "world" in which $X = x$ has occurred, and asking: "Within that restricted world, what's the likelihood that $Y = y$ also occurs?". And like the any distribution, conditional distributions also satisfy the same fundamental properties like non-negativity and Normalization

### The Chain Rule

The fundamental relationship connecting joint and conditional distributions is the **chain rule**:

$$
p(x, y) = p(x \mid y) \cdot p(y) = p(y \mid x) \cdot p(x),\quad  \forall x,y,z
$$

The chain rule decomposes a joint probability into a sequence of conditional probabilities. Each factor represents the probability of one variable given all the previous variables in the sequence:

The same factorization becomes the [[entropy-relative-entropy-mutual-information#51-chain-rule-for-entropy|entropy chain rule]] after taking logarithms and expectations.

$$
p(x,y,z) = p(x \mid y, z)  p(y \mid z) \cdot p(z),\quad \forall x,y,z
$$

For $n$ variables $X_1, X_2, \ldots, X_n$:

$$
p(\mathbf{x})=p(x_1, x_2, \ldots, x_n) = p(x_1 \mid x_2, \ldots, x_n)  \cdots p(x_{n-1} \mid x_n) \cdot p(x_n),\quad  \forall \mathbf{x}
$$

> [!example]- Multi-Variable Chain Rule
>
> **Scenario**: Consider three variables:
>
> - $W$: Weather (sunny, rainy)
> - $T$: Traffic (light, heavy)
> - $M$: Meeting attendance (attend, skip)
>
> **Given probabilities**:
>
> - $p(W = \text{sunny}) = 0.7$
> - $p(T = \text{light} \mid W = \text{sunny}) = 0.8$
> - $p(T = \text{light} \mid W = \text{rainy}) = 0.3$
> - $p(M = \text{attend} \mid W = \text{sunny}, T = \text{light}) = 0.9$
> - $p(M = \text{attend} \mid W = \text{sunny}, T = \text{heavy}) = 0.6$
> - $p(M = \text{attend} \mid W = \text{rainy}, T = \text{light}) = 0.7$
> - $p(M = \text{attend} \mid W = \text{rainy}, T = \text{heavy}) = 0.2$
>
> **Question**: What's the probability of attending a meeting on a sunny day with light traffic?
>
> **Solution using chain rule**:
> $$p(W = \text{sunny}, T = \text{light}, M = \text{attend}) = p(M = \text{attend} \mid W = \text{sunny}, T = \text{light}) \cdot p(T = \text{light} \mid W = \text{sunny}) \cdot p(W = \text{sunny})$$
>
> $$= 0.9 \times 0.8 \times 0.7 = 0.504$$
>
> **Interpretation**: There's a 50.4% chance of attending a meeting on a sunny day with light traffic.

### Independence

Two random variables $X$ and $Y$ are independent if and only if:

$$
p(x, y) = p(x) \cdot p(y)
$$

Independence means that knowing the value of one variable doesn't change our beliefs about the other. The joint probability factors into the product of individual probabilities.

> [!example]- Coin Toss Independence
>
> **Scenario**: Consider two fair coin tosses:
>
> - $X$: First coin (H = 1, T = 0)
> - $Y$: Second coin (H = 1, T = 0)
>
> **Joint probability table**:
>
> | $x$ | $y$ | $p(x, y)$ |
> | --- | --- | --------- |
> | 0   | 0   | 0.25      |
> | 0   | 1   | 0.25      |
> | 1   | 0   | 0.25      |
> | 1   | 1   | 0.25      |
>
> **Test for independence**:
>
> **Step 1: Calculate marginal probabilities**
>
> - $p(X = 0) = 0.25 + 0.25 = 0.5$
> - $p(X = 1) = 0.25 + 0.25 = 0.5$
> - $p(Y = 0) = 0.25 + 0.25 = 0.5$
> - $p(Y = 1) = 0.25 + 0.25 = 0.5$
>
> **Step 2: Check independence condition**
>
> - $p(X = 0, Y = 0) = 0.25$
> - $p(X = 0) \cdot p(Y = 0) = 0.5 \times 0.5 = 0.25$ ✓
> - $p(X = 0, Y = 1) = 0.25$
> - $p(X = 0) \cdot p(Y = 1) = 0.5 \times 0.5 = 0.25$ ✓
> - $p(X = 1, Y = 0) = 0.25$
> - $p(X = 1) \cdot p(Y = 0) = 0.5 \times 0.5 = 0.25$ ✓
> - $p(X = 1, Y = 1) = 0.25$
> - $p(X = 1) \cdot p(Y = 1) = 0.5 \times 0.5 = 0.25$ ✓
>
> **Conclusion**: All conditions hold, so $X$ and $Y$ are **independent**.

> [!example]- Dependent Coin Tosses
>
> **Scenario**: Consider a modified experiment where the second coin is biased based on the first:
>
> - If first coin is H, second coin has 0.8 probability of H
> - If first coin is T, second coin has 0.3 probability of H
>
> **Joint probability table**:
>
> | $x$ | $y$ | $p(x, y)$ |
> | --- | --- | --------- |
> | 0   | 0   | 0.35      |
> | 0   | 1   | 0.15      |
> | 1   | 0   | 0.10      |
> | 1   | 1   | 0.40      |
>
> **Test for independence**:
>
> **Step 1: Calculate marginal probabilities**
>
> - $p(X = 0) = 0.35 + 0.15 = 0.5$
> - $p(X = 1) = 0.10 + 0.40 = 0.5$
> - $p(Y = 0) = 0.35 + 0.10 = 0.45$
> - $p(Y = 1) = 0.15 + 0.40 = 0.55$
>
> **Step 2: Check independence condition**
>
> - $p(X = 0, Y = 0) = 0.35$
> - $p(X = 0) \cdot p(Y = 0) = 0.5 \times 0.45 = 0.225$ ✗
>
> **Conclusion**: Since $0.35 \neq 0.225$, $X$ and $Y$ are **not independent**.

---

## Marginal Distributions

From a joint distribution, we can derive marginal distributions for individual variables by "summing out" or "integrating out" the other variables:

**Discrete case:**

$$
p(x) = \sum_y p(x, y)
$$

$$
p(y) = \sum_x p(x, y)
$$

**Continuous case:**

$$
f(x) = \int_{-\infty}^{\infty} f(x,y) \, dy
$$

$$
f(y) = \int_{-\infty}^{\infty} f(x,y) \, dx
$$

The marginal distribution tells us about individual variables when we ignore the others.

> [!example]- Example
> Say we have a bakery that tracks the joint distribution of bread ($B$) and coffee ($C$) sales $p(b, c)$. The marginal $p(b) = \sum_c p(b, c)$ answers "how often do people buy bread - regardless of whether they also buy coffee?", focusing just on bread sales alone.

### Bayes' Theorem

Rearranging the chain rule gives us **Bayes' theorem**:

$$
p(y \mid x) = \frac{p(x \mid y) \cdot p(y)}{p(x)}
$$

Bayes' rule is a fundamental principle for **updating beliefs** based on new evidence. It tells us how to revise our initial beliefs when we observe new data. This is extremely important when we want to **experiment and observe new data in an unknown world** - it provides a principled framework for learning from experience and adapting our understanding as we gather more information. I will try to cover this aspect in future blog posts on [Maximum Likelihood Estimation](MLE) and [Maximum A Posteriori](MAP).

Components of Bayes' Theorem:

- **$p(y \mid x)$**: Posterior probability
- **$p(x \mid y)$**: Likelihood (how likely is $x$ given $y$?)
- **$p(y)$**: Prior probability (our initial belief about $Y$)
- **$p(x)$**: Evidence (probability of observing $x$)

> [!example]- Medical Test Interpretation
>
> **Scenario**: A disease affects 1% of the population. A test is 95% accurate (95% of sick people test positive, 95% of healthy people test negative).
>
> **Question**: If someone tests positive, what's the probability they actually have the disease?
>
> **Solution using Bayes' rule**:
>
> - $p(\text{disease}) = 0.01$ (prior: 1% of population)
> - $p(\text{positive} \mid \text{disease}) = 0.95$ (likelihood: test accuracy)
> - $p(\text{positive}) = p(\text{positive} \mid \text{disease}) \cdot p(\text{disease}) + p(\text{positive} \mid \text{healthy}) \cdot p(\text{healthy})$
> - $p(\text{positive}) = 0.95 \times 0.01 + 0.05 \times 0.99 = 0.059$
>
> **Bayes' rule**:
> $$p(\text{disease} \mid \text{positive}) = \frac{p(\text{positive} \mid \text{disease}) \cdot p(\text{disease})}{p(\text{positive})}$$
>
> $$p(\text{disease} \mid \text{positive}) = \frac{0.95 \times 0.01}{0.059} \approx 0.161$$
>
> **Surprising result**: Even with a positive test, there's only a 16.1% chance of having the disease! This is because the disease is rare (1%) and false positives are common when testing a large healthy population.

> [!example]- The Monty Hall Problem
> The Monty Hall problem is one of the most famous probability puzzles that often challenges our intuition. Named after the host of the game show "Let's Make a Deal," this problem demonstrates how Bayesian reasoning can help us understand counterintuitive probability results.
>
> **Problem Setup**: You are a contestant on a game show with three doors. Behind one door is a valuable prize (like a car), and behind the other two doors are less desirable prizes (like goats). The game proceeds as follows:
>
> - You choose one of the three doors (but don't open it yet)
> - The host, who knows what's behind all doors, opens one of the remaining doors that contains a goat
> - The host then offers you a choice: stick with your original door or switch to the other unopened door
>
> Question: Should you stick with your original choice or switch doors?
>
> I highly recommend that you try to work out a decision for this question on your own. I'll be sharing my explanation in a future blog post about [MLE and MAP](MLE).

---

## Key Takeaways

### Fundamental Concepts

- **Joint distributions**: high-dimensional PDFs (continuous variables) or PMFs (discrete variables).
- **Marginal distributions**: can be derived by "summing out" other variables from joint distributions.
- **Conditional distributions**: describe how one variable behaves given knowledge of another.

**Independence** means variables don't influence each other:

$$
p(x, y) = p(x) \cdot p(y)  \quad \forall x,y
$$

**Bayes' theorem**:

$$
p(y \mid x) = \frac{p(x \mid y) \cdot p(y)}{p(x)}, \quad \forall x,y
$$

**Chain rule**:

$$
p(\mathbf{x})=p(x_1, x_2, \ldots, x_n) = p(x_1 \mid x_2, \ldots, x_n)  \cdots p(x_{n-1} \mid x_n) \cdot p(x_n),\quad  \forall \mathbf{x}
$$

### Essential Properties

All kind of distribution must satisfy:

1. **Non-negativity**: $p(\mathbf{x}) \geq 0$ for all $\mathbf{x} = (x_1, x_2, ..., x_n)$
2. **Normalization**: $\sum p(\mathbf{x}) = 1$ (discrete) or $\int\_{-\infty}^{\infty} f(\mathbf{x}) d \mathbf{x} = 1 $ (continuous)

---

> [!note]- Notations
> See the [notation reference](notation.md) for a full summary of symbols used across all notes. In brief: capital letters ($X, Y$) denote random variables, lowercase ($x, y$) their values, and bold lowercase ($\mathbf{x}$) vectors.
