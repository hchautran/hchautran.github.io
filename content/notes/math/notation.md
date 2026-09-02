---
title: 1. Notation Reference
description: Notation conventions used across all math and ML notes - covering random variables, probability distributions, parametric models, expectations, and divergences.
date: 2026-03-24
lastmod: 2026-08-30
draft: false
tags:
  - math
  - notes
---

## Variables and Values

| Symbol                     | Meaning                                                                                |
| -------------------------- | -------------------------------------------------------------------------------------- |
| $X, Y, Z$                  | Random variables (capital letters)                                                     |
| $x, y, z$                  | Specific realized values of the corresponding random variables                         |
| $\mathcal{X}, \mathcal{Y}$ | Alphabets or supports of random variables; for example, $x\in\mathcal{X}$              |
| $\mathbf{x}, \mathbf{y}$   | Vectors (bold lowercase); e.g. $\mathbf{x} \in \mathbb{R}^d$                           |
| $\mathbf{X}$               | A matrix (bold uppercase); e.g. design matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$ |

## Probability

| Symbol                               | Meaning                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------ |
| $P_X$                                | Probability distribution, or law, of the random variable $X$             |
| $p_X(x)=\Pr(X=x)$                    | PMF of a discrete random variable $X$, evaluated at the realization $x$  |
| $f_X(x)$                             | PDF of a continuous random variable $X$, evaluated at $x$                |
| $p_{X,Y}(x,y)$                       | Joint PMF of discrete random variables $(X,Y)$, evaluated at $(x,y)$     |
| $p_{Y\mid X}(y\mid x)$               | Conditional PMF of $Y=y$ given the event $X=x$                           |
| $P_{Y\mid X}$                        | Conditional distribution of the random variable $Y$ given $X$            |
| $p_{\mathbf{X}}(\mathbf{x})$         | Joint PMF/PDF of a random vector $\mathbf{X}$, evaluated at $\mathbf{x}$ |
| $p_\theta(\mathbf{x})$               | Parametric PMF/PDF evaluated at $\mathbf{x}$, with parameter $\theta$    |
| $p_\theta(\mathbf{x}\mid\mathbf{y})$ | Parametric conditional PMF/PDF evaluated at $(\mathbf{x},\mathbf{y})$    |

When the associated random variables are obvious, subscripts may be suppressed for readability: $p_X(x)$ becomes $p(x)$ and $p_{Y\mid X}(y\mid x)$ becomes $p(y\mid x)$. Uppercase arguments such as $p_X(X)$ denote the PMF evaluated at the random outcome $X$ and therefore produce a random variable.

## Datasets and Expectations

| Symbol                                                          | Meaning                                                                     |
| --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| $\mathcal{D}=\{\mathbf{x}^{(1)},\dots,\mathbf{x}^{(N)}\}$       | Dataset containing $N$ realized samples                                     |
| $\mathbf{X}^{(i)}\overset{\mathrm{i.i.d.}}{\sim}P_{\mathbf{X}}$ | Random samples drawn independently from the same distribution               |
| $\mathbb{E}_{X\sim P_X}[g(X)]$                                  | Expectation of the random variable $g(X)$ under $P_X$                       |
| $\mathbb{E}_{x\sim p_{\mathrm{data}}}[g(x)]$                    | Common machine-learning shorthand for an expectation over data realizations |

## Common Distributions

| Symbol                                               | Meaning                                                                                 |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| $\mathcal{N}(\mu, \sigma^2)$                         | Univariate Gaussian with mean $\mu$ and variance $\sigma^2$                             |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Multivariate Gaussian with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ |
| $\mathcal{N}(\mathbf{0}, \mathbf{I})$                | Standard multivariate Gaussian (zero mean, identity covariance)                         |
| $\text{Bernoulli}(p)$                                | Bernoulli distribution with success probability $p$                                     |
