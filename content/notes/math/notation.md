---
title: Notations
draft: false
tags:
  - math
  - notes
---


## Variables and Values

| Symbol | Meaning |
|--------|---------|
| $X, Y, Z$ | Random variables (capital letters) |
| $x, y, z$ | Specific realized values of the corresponding random variables |
| $\mathbf{x}, \mathbf{y}$ | Vectors (bold lowercase); e.g. $\mathbf{x} \in \mathbb{R}^d$ |
| $\mathbf{X}$ | A matrix (bold uppercase); e.g. design matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$ |

## Probability

| Symbol | Meaning |
|--------|---------|
| $p(x)$ | Probability mass function (PMF) or Probability density function (PDF) evaluated at $x$ — for discrete/continuous variables |
| $p(X)$ | Probability distribution of random variable $X$ |
| $p(y \mid x)$ | Conditional PMF/PDF of $Y=y$ given $X=x$ |
| $p(Y \mid X)$ | Conditional distribution of $Y$ given $X$ |
| $p(x, y)$ | Join PDF/PMF of $p(X,Y)$ at $X=x$ and $Y=y$ |
| $p(X, Y)$ | Join distribution of $X$ and $Y$ |
| $p(\mathbf{x})$ | join PDF/PMF over vector $\mathbf{x}$ |
| $p_\theta(\mathbf{x})$ | join PDF/PMF over vector $\mathbf{x}$, parametrized by $\theta$ |
| $p_\theta(\mathbf{x} \mid \mathbf{y})$ | join PDF/PMF over vector $\mathbf{x}$ given $\mathbf{y}$,  parametrized by $\theta$  |
| $p(\mathcal{X})$ | Join distribution of all possible $\mathbf{x} \in \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\}$  |

## Datasets and Expectations

| Symbol | Meaning |
|--------|---------|
| $\mathcal{X} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\}$ | Dataset of $N$ i.i.d. samples |
| $\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})}[f(\mathbf{x})]$ | Expectation of $f(\mathbf{x})$ under distribution $p$ |
| $\mathbf{x} \overset{\text{i.i.d.}}{\sim} p(\mathbf{x})$ | Samples drawn independently and identically from $p(\mathcal{X})$ |

## Common Distributions

| Symbol | Meaning |
|--------|---------|
| $\mathcal{N}(\mu, \sigma^2)$ | Univariate Gaussian with mean $\mu$ and variance $\sigma^2$ |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Multivariate Gaussian with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ |
| $\mathcal{N}(\mathbf{0}, \mathbf{I})$ | Standard multivariate Gaussian (zero mean, identity covariance) |
| $\text{Bernoulli}(p)$ | Bernoulli distribution with success probability $p$ |
