---
title: Maximum Likehood Estimation
draft: false
tags:
  - math
  - notes
---

## Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation (MLE) is one of the most fundamental methods for fitting a parametric model to observed data. The core idea is simple: given a dataset, find the parameters $\theta$ that make the observed data **most probable** under the model.

## Setup

Suppose we observe a dataset $\mathcal{X} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\}$ drawn i.i.d. from an unknown distribution $p_{data}(\mathbf{x})$. We posit a parametric family of [joint distributions](join_prob.md) $p_\theta(\mathbf{x})$ and want to find the $\theta$ that best explains the data — assigning high probability to frequently occurring data and low probability to rare observations.

Since the data are i.i.d., the joint likelihood factorizes into a product, we can write the **likelihood** of the dataset under the model as:

$$
p_\theta(\mathcal{X}) = \prod_{i=1}^{N} p_\theta(\mathbf{x}^{(i)}) \tag{1}
$$


Thus the MLE objective is:
$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\arg\max} \prod_{i=1}^{N} p_\theta(\mathbf{x}^{(i)}) \tag{2}
$$

## Log-Likelihood

Products are numerically unstable and hard to differentiate. Taking the logarithm — a monotone transformation — converts the product into a sum without changing the argmax:

$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\arg\max} \sum_{i=1}^{N} \log p_\theta(\mathbf{x}^{(i)}) \tag{3}
$$

In practice, we minimize the **negative log-likelihood (NLL)**:

$$
\hat{\mathcal{L}}_{\text{MLE}}(\theta) := -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(\mathbf{x}^{(i)}) \tag{4}
$$

The $\frac{1}{N}$ factor normalizes the loss so it does not scale with dataset size, making it a Monte Carlo estimate of the population objective:

$$
\mathcal{L}_{\text{MLE}}(\theta) = -\mathbb{E}_{\mathbf{x} \sim p_{data}}\left[\log p_\theta(\mathbf{x})\right] \tag{5}
$$


## Example: Gaussian MLE

Suppose we model the data as $x^{(i)} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$, with parameters $\theta = (\mu, \sigma^2)$. The density of a single sample is:

$$
p_\theta(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \tag{7}
$$

Taking the log, the log-likelihood of a single sample is:

$$
\log p_\theta(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2} \tag{8}
$$

Summing over the dataset, the total log-likelihood is:

$$
\ell(\mu, \sigma^2) = \sum_{i=1}^{N} \log p_\theta(x^{(i)}) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(x^{(i)} - \mu)^2 \tag{9}
$$

**Solving for $\hat{\mu}$:** Taking the derivative with respect to $\mu$ and setting it to zero:

$$
\begin{align}
\frac{\partial \ell}{\partial \mu} &= \frac{1}{\sigma^2} \sum_{i=1}^{N}(x^{(i)} - \mu) = 0 \notag \\
&\Rightarrow \hat{\mu} = \frac{1}{N}\sum_{i=1}^{N} x^{(i)} \tag{10}
\end{align}
$$

**Solving for $\hat{\sigma}^2$:** Taking the derivative with respect to $\sigma^2$ and setting it to zero:

$$
\begin{align}
\frac{\partial \ell}{\partial \sigma^2} &= -\frac{N}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{N}(x^{(i)} - \mu)^2 = 0 \notag \\
&\Rightarrow \hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^{N}(x^{(i)} - \hat{\mu})^2 \tag{11}
\end{align}
$$

These are simply the **sample mean** and **sample variance** — the MLE recovers the intuitive estimators from first principles.

## Example: Coin Flipping (Bernoulli MLE)

Suppose we flip a coin $N$ times and observe outcomes $x^{(i)} \in \{0, 1\}$, where $1$ = heads and $0$ = tails. We model each flip as $x^{(i)} \overset{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$, with the single unknown parameter $\theta = p \in [0,1]$.

The probability of a single outcome is:

$$
p_\theta(x) = p^x (1-p)^{1-x} \tag{12}
$$

The log-likelihood of a single sample is:

$$
\log p_\theta(x) = x \log p + (1-x) \log(1-p) \tag{13}
$$

Summing over all $N$ flips, let $H = \sum_{i=1}^N x^{(i)}$ denote the number of heads. The total log-likelihood is:

$$
\ell(p) = \sum_{i=1}^{N} \log p_\theta(x^{(i)}) = H \log p + (N - H) \log(1-p) \tag{14}
$$

**Solving for $\hat{p}$:** Taking the derivative with respect to $p$ and setting it to zero:

$$
\begin{align}
\frac{\partial \ell}{\partial p} &= \frac{H}{p} - \frac{N-H}{1-p} = 0 \notag \\
&\Rightarrow H(1-p) = (N-H)p \notag \\
&\Rightarrow \hat{p} = \frac{H}{N} \tag{15}
\end{align}
$$

The MLE estimate is simply the **empirical fraction of heads** — exactly what intuition suggests.

## Example: Linear Regression (Gaussian noise MLE)

In linear regression, we observe pairs $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$ and model the output as:

$$
y^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + \varepsilon^{(i)}, \quad \varepsilon^{(i)} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2) \tag{16}
$$

This means the conditional distribution of $y^{(i)}$ given $\mathbf{x}^{(i)}$ is:

$$
p_\mathbf{w}(y^{(i)} \mid \mathbf{x}^{(i)}) = \mathcal{N}(y^{(i)};\, \mathbf{w}^\top \mathbf{x}^{(i)},\, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})^2}{2\sigma^2}\right) \tag{17}
$$

The log-likelihood of a single pair is:

$$
\log p_\mathbf{w}(y^{(i)} \mid \mathbf{x}^{(i)}) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})^2}{2\sigma^2} \tag{18}
$$

Summing over the dataset and dropping the constant term (which does not depend on $\mathbf{w}$), the total log-likelihood is:

$$
\ell(\mathbf{w}) = -\frac{1}{2\sigma^2} \sum_{i=1}^{N} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)}\right)^2 \tag{19}
$$

Maximizing $\ell(\mathbf{w})$ over $\mathbf{w}$ is equivalent to minimizing the **sum of squared residuals**:

$$
\hat{\mathbf{w}}_{\text{MLE}} = \underset{\mathbf{w}}{\arg\min} \sum_{i=1}^{N} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)}\right)^2 \tag{20}
$$

**Solving for $\hat{\mathbf{w}}$:** In matrix form, let $\mathbf{X} \in \mathbb{R}^{N \times d}$ be the design matrix and $\mathbf{y} \in \mathbb{R}^N$ the target vector. The objective becomes $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$. Taking the gradient and setting it to zero:

$$
\begin{align}
\nabla_\mathbf{w} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 &= -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) = \mathbf{0} \notag \\
&\Rightarrow \mathbf{X}^\top \mathbf{X}\, \mathbf{w} = \mathbf{X}^\top \mathbf{y} \notag \\
&\Rightarrow \hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} \tag{21}
\end{align}
$$

This is the well-known **ordinary least squares (OLS)** solution. The key insight is that **minimizing MSE in linear regression is exactly MLE under a Gaussian noise assumption**.

## Connection to KL Divergence

Minimizing the NLL is equivalent to minimizing the KL divergence between $p_{data}$ and $p_\theta$:

$$
\begin{align}
\mathcal{D}_{KL}(p_{data} \| p_\theta) &= \mathbb{E}_{\mathbf{x} \sim p_{data}}\left[\log \frac{p_{data}(\mathbf{x})}{p_\theta(\mathbf{x})}\right] \notag \\
&= -\mathbb{E}_{\mathbf{x} \sim p_{data}}\left[\log p_\theta(\mathbf{x})\right] + \underbrace{\mathbb{E}_{\mathbf{x} \sim p_{data}}\left[\log p_{data}(\mathbf{x})\right]}_{\text{constant w.r.t. } \theta} \tag{22}
\end{align}
$$

Since the second term does not depend on $\theta$, minimizing $\mathcal{D}_{KL}$ reduces exactly to minimizing the NLL.


## Summary

This blog has covered MLE and its application to simple parametric distribution families, illustrated through two classical examples: Bernoulli coin flipping and Gaussian linear regression. Note that in practice, $p_\theta(\mathbf{x})$ rarely admits a closed-form solution like in the examples above — it is often a deep, expressive neural network, in which case the MLE objective must be optimized iteratively via [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).


