---
title: Joint Probability Distributions 
draft: false
tags:
  - Probabilities and Statistic 
  - notes
---


Joint probability distributions are fundamental to understanding how multiple random variables behave together.



## Joint Distribution

A joint probability distribution describes the probability of events involving multiple random variables simultaneously. Think of it as extending our understanding from single-variable probability to multi-variable scenarios where we can capture complex relationships and dependencies.
>  Joint distributions are high-dimensional PDFs (continuous variables) or PMFs (discrete variables).

> [!note]- Notation Convention
> To avoid confusion, I'll use capital letters ($X, Y$) for random variables and lowercase letters ($x, y$) for their specific values. For example:
> - $P(X)$ represents the probability distribution of $X$
> - $p(x)$ or $p_X(x) = P(X=x)$ represents the PMF of event $X=x$ occurs

### Mathematical Formulation

For two discrete random variables $X$ and $Y$, the joint probability mass function (PMF) is:
$$
P(X = x, Y = y) = p_{X,Y}(x,y)
$$

For continuous variables, we have the joint probability density function (PDF):
$$
P(a \leq X \leq b, c \leq Y \leq d) = \int_a^b \int_c^d f_{X,Y}(x,y) \, dx \, dy
$$

As we add more variables, the dimensionality grows naturally:
- 1D: $p_X(x)$ or $f_X(x)$
- 2D: $p_{X,Y}(x,y)$ or $f_{X,Y}(x,y)$  
- 3D: $p_{X,Y,Z}(x,y,z)$ or $f_{X,Y,Z}(x,y,z)$
- nD: $p_{\mathbf{X}}(\mathbf{x})$ or $f_{\mathbf{X}}(\mathbf{x})$ where $\mathbf{X} = (X_1, X_2, ..., X_n)$

### Key Properties

Every joint distribution must satisfy these fundamental properties:

1. **Non-negativity**: $p_{X,Y}(x,y) \geq 0$ for all $(x,y)$
2. **Normalization**: $\sum_x \sum_y p_{X,Y}(x,y) = 1$ (discrete) or $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dx \, dy = 1$ (continuous)

These properties ensure that joint distributions are valid probability measures.

## Marginal Distributions

From a joint distribution, we can derive marginal distributions for individual variables by "summing out" or "integrating out" the other variables:

**Discrete case:**
$$
p_X( x) = \sum_y p_{X,Y}(x,  y)
$$
$$
p_Y(y) = \sum_x p_{X,Y}( x,  y)
$$

**Continuous case:**
$$
f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dy
$$
$$
f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dx
$$

The marginal distribution tells us about individual variables when we ignore the others. 

> [!example]- Example
> Let say we have a bakery that tracks join distribution of bread ($B$) and coffee ($C$) sales $P(B,C)$. The marginal probability $P_B(b)=\sum_c p_{B,C}(b,c)$ answers “how often do people buy bread?—regardless of whether they also buy coffee or not", focusing just on bread sales alone.


## Conditional Distributions

Conditional probability answers: "Given that $X$ has occurred, what's the probability that $Y$ also occurs?" It's like updating our beliefs based on new information.

$$
P(Y | X) = \frac{P(X, Y)}{P(X)} 
$$

You might wonder: "what is the different between this and the joint distribution P(X,Y)". To advoid confusion, You can think of conditional probability as focusing on the “world” in which $X$ has occurred, and asking: "Within that restricted world, what’s the likelihood that $Y$ also occurs?"

### Essential Properties
Like join probability distribution, conditional distribution also satisfy the same fundamental properties
 1. **Non-negativity**: $p_{X,Y}(y|x) \geq 0$ for all $y$ and $x$
 2. **Normalization**: $\sum_y p_{X,Y}(y| x) = 1$ or $\int_y p_{X,Y}(y| x) = 1$ in a world that $X=x$ happended


### The Chain Rule

The fundamental relationship connecting joint and conditional distributions is the **chain rule**:

$$
P(X, Y) = P(X|Y) \cdot P(Y) = P(Y|X) \cdot P(X)
$$


Basically, the chain rule decomposes a joint probability into a sequence of conditional probabilities. Each factor represents the probability of one variable given all the previous variables in the sequence.

$$
P(X, Y, Z) = P(X|Y, Z) \cdot P(Y|Z) \cdot P(Z)
$$

For $n$ variables $X_1, X_2, \ldots, X_n$:

$$
P(X_1, X_2, \ldots, X_n) = P(X_1|X_2, \ldots, X_n) \cdot P(X_2|X_3, \ldots, X_n) \cdots P(X_{n-1}|X_n) \cdot P(X_n)
$$



> [!example]- Multi-Variable Chain Rule
> 
> **Scenario**: Consider three variables:
> - $W$: Weather (sunny, rainy)
> - $T$: Traffic (light, heavy) 
> - $M$: Meeting attendance (attend, skip)
> 
> **Given probabilities**:
> - $P(W = \text{sunny}) = 0.7$
> - $P(T = \text{light} | W = \text{sunny}) = 0.8$
> - $P(T = \text{light} | W = \text{rainy}) = 0.3$
> - $P(M = \text{attend} | W = \text{sunny}, T = \text{light}) = 0.9$
> - $P(M = \text{attend} | W = \text{sunny}, T = \text{heavy}) = 0.6$
> - $P(M = \text{attend} | W = \text{rainy}, T = \text{light}) = 0.7$
> - $P(M = \text{attend} | W = \text{rainy}, T = \text{heavy}) = 0.2$
> 
> **Question**: What's the probability of attending a meeting on a sunny day with light traffic?
> 
> **Solution using chain rule**:
> $$P(W = \text{sunny}, T = \text{light}, M = \text{attend}) = P(M = \text{attend} | W = \text{sunny}, T = \text{light}) \cdot P(T = \text{light} | W = \text{sunny}) \cdot P(W = \text{sunny})$$
> 
> $$= 0.9 \times 0.8 \times 0.7 = 0.504$$
> 
> **Interpretation**: There's a 50.4% chance of attending a meeting on a sunny day with light traffic.


### Independence

Two random variables $X$ and $Y$ are independent if and only if:

$$
P(X, Y) = P(X) \cdot P(Y)
$$
Independence means that knowing the value of one variable doesn't change our beliefs about the other. The joint probability factors into the product of individual probabilities.


> [!example]- Coin Toss Independence
> 
> **Scenario**: Consider two fair coin tosses:
> - $X$: First coin (H = 1, T = 0)
> - $Y$: Second coin (H = 1, T = 0)
> 
> **Joint probability table**:
> | $X$ | $Y$ | $P(X, Y)$ |
> |-----|-----|-----------|
> | 0   | 0   | 0.25      |
> | 0   | 1   | 0.25      |
> | 1   | 0   | 0.25      |
> | 1   | 1   | 0.25      |
> 
> **Test for independence**:
> 
> **Step 1: Calculate marginal probabilities**
> - $P(X = 0) = 0.25 + 0.25 = 0.5$
> - $P(X = 1) = 0.25 + 0.25 = 0.5$
> - $P(Y = 0) = 0.25 + 0.25 = 0.5$
> - $P(Y = 1) = 0.25 + 0.25 = 0.5$
> 
> **Step 2: Check independence condition**
> - $P(X = 0, Y = 0) = 0.25$
> - $P(X = 0) \cdot P(Y = 0) = 0.5 \times 0.5 = 0.25$ ✓
> - $P(X = 0, Y = 1) = 0.25$
> - $P(X = 0) \cdot P(Y = 1) = 0.5 \times 0.5 = 0.25$ ✓
> - $P(X = 1, Y = 0) = 0.25$
> - $P(X = 1) \cdot P(Y = 0) = 0.5 \times 0.5 = 0.25$ ✓
> - $P(X = 1, Y = 1) = 0.25$
> - $P(X = 1) \cdot P(Y = 1) = 0.5 \times 0.5 = 0.25$ ✓
> 
> **Conclusion**: All conditions hold, so $X$ and $Y$ are **independent**.


> [!example]- Dependent Coin Tosses
> 
> **Scenario**: Consider a modified experiment where the second coin is biased based on the first:
> - If first coin is H, second coin has 0.8 probability of H
> - If first coin is T, second coin has 0.3 probability of H
> 
> **Joint probability table**:
> | $X$ | $Y$ | $P(X, Y)$ |
> |-----|-----|-----------|
> | 0   | 0   | 0.35      |
> | 0   | 1   | 0.15      |
> | 1   | 0   | 0.10      |
> | 1   | 1   | 0.40      |
> 
> **Test for independence**:
> 
> **Step 1: Calculate marginal probabilities**
> - $P(X = 0) = 0.35 + 0.15 = 0.5$
> - $P(X = 1) = 0.10 + 0.40 = 0.5$
> - $P(Y = 0) = 0.35 + 0.10 = 0.45$
> - $P(Y = 1) = 0.15 + 0.40 = 0.55$
> 
> **Step 2: Check independence condition**
> - $P(X = 0, Y = 0) = 0.35$
> - $P(X = 0) \cdot P(Y = 0) = 0.5 \times 0.45 = 0.225$ ✗
> 
> **Conclusion**: Since $0.35 ≠ 0.225$, $X$ and $Y$ are **not independent**.


### Bayes' Theorem

Rearranging the chain rule gives us **Bayes' theorem**:

$$
P(Y | X) = \frac{P(X | Y) \cdot P(Y)}{P(X)}
$$

Bayes' rule is a fundamental principle for **updating beliefs** based on new evidence. It tells us how to revise our initial beliefs when we observe new data. This is extremely important when we want to **experiment and observe new data in an unknown world** - it provides a principled framework for learning from experience and adapting our understanding as we gather more information. I will try to cover this aspect in future blog posts on [Maximum Likelihood Estimation](MLE) and [Maximum A Postiriories](MAP)   

Components of Bayes' Theorem
 - **$P(Y|X)$**: Posterior probability 
 - **$P(X|Y)$**: Likelihood (how likely is $X$ given $Y$?)
 - **$P(Y)$**: Prior probability (our initial belief about $Y$)
 - **$P(X)$**: Evidence (probability of observing $X$)

> [!example]- Medical Test Interpretation
> 
> **Scenario**: A disease affects 1% of the population. A test is 95% accurate (95% of sick people test positive, 95% of healthy people test negative).
> 
> **Question**: If someone tests positive, what's the probability they actually have the disease?
> 
> **Solution using Bayes' rule**:
> - $P(\text{disease}) = 0.01$ (prior: 1% of population)
> - $P(\text{positive}|\text{disease}) = 0.95$ (likelihood: test accuracy)
> - $P(\text{positive}) = P(\text{positive}|\text{disease}) \cdot P(\text{disease}) + P(\text{positive}|\text{healthy}) \cdot P(\text{healthy})$
> - $P(\text{positive}) = 0.95 \times 0.01 + 0.05 \times 0.99 = 0.059$
> 
> **Bayes' rule**:
> $$P(\text{disease}|\text{positive}) = \frac{P(\text{positive}|\text{disease}) \cdot P(\text{disease})}{P(\text{positive})}$$
> 
> $$P(\text{disease}|\text{positive}) = \frac{0.95 \times 0.01}{0.059} ≈ 0.161$$
> 
> **Surprising result**: Even with a positive test, there's only a 16.1% chance of having the disease! This is because the disease is rare (1%) and false positives are common when testing a large healthy population.



> [!example]- Spam Filter
> 
> **Scenario**: 20% of emails are spam. A spam filter correctly identifies 90% of spam emails and 95% of legitimate emails.
> 
> **Question**: If an email is flagged as spam, what's the probability it's actually spam?
> 
> **Solution**:
> - $P(\text{spam}) = 0.20$ (prior)
> - $P(\text{flagged}|\text{spam}) = 0.90$ (likelihood)
> - $P(\text{flagged}) = 0.90 \times 0.20 + 0.05 \times 0.80 = 0.22$
> 
> **Bayes' rule**:
> $$P(\text{spam}|\text{flagged}) = \frac{0.90 \times 0.20}{0.22} ≈ 0.818$$
> 
> **Result**: 81.8% of flagged emails are actually spam.








## Key Takeaways
 
 ### Fundamental Concepts
 - **Joint distributions**: high-dimensional PDFs (continuous variables) or PMFs (discrete variables).
 - **Marginal distributions**: can be derived by "summing out" other variables from joint distributions
 - **Conditional distributions**: describe how one variable behaves given knowledge of another
 - **Independence** means variables don't influence each other: 
$$
 P(X,Y) = P(X) \cdot P(Y)
 $$
 - **Chain rule**: 
  $$
  P(X,Y) = P(X|Y) \cdot P(Y) = P(Y|X) \cdot P(X)
  $$
 - **Bayes' theorem**: 
 $$
 P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}
 $$

 


