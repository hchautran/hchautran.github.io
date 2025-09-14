---
title: Covariance and Correlation
draft: true
tags:
  - Probabilities and Statistic 
  - notes
---

## Covariance

Covariance measures the **linear relationship** between two random variables. It tells us how much two variables change together:

$$
Cov(X,Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
$$

> [!note] Intuitive Interpretation
> - **Positive covariance**: When $X$ is above its mean, $Y$ tends to be above its mean too
> - **Negative covariance**: When $X$ is above its mean, $Y$ tends to be below its mean
> - **Zero covariance**: No linear relationship between the variables
Properties of Covariance:
1. **Symmetry**: $Cov(X,Y) = Cov(Y,X)$
2. **Linearity**: $Cov(aX + b, cY + d) = ac \cdot Cov(X,Y)$
3. **Variance connection**: $Cov(X,X) = Var(X)$
4. **Independence**: If $X$ and $Y$ are independent, then $Cov(X,Y) = 0$


## Correlation Coefficient

The correlation coefficient normalizes covariance to a scale between -1 and 1:

$$
\rho_{XY} = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}} = \frac{Cov(X,Y)}{\sigma_X \sigma_Y}
$$

Correlation Interpretation
 - $|\rho_{XY}| = 1$: Perfect linear relationship
 - $0.7 < |\rho_{XY}| < 1$: Strong linear relationship  
 - $0.3 < |\rho_{XY}| < 0.7$: Moderate linear relationship
 - $0 < |\rho_{XY}| < 0.3$: Weak linear relationship
 - $\rho_{XY} = 0$: No linear relationship


## Covariance Matrix

When dealing with multiple random variables, we organize all pairwise covariances into a **covariance matrix**:

For $n$ random variables $X_1, X_2, \ldots, X_n$, the covariance matrix is:

$$
\mathbf{\Sigma} = \begin{pmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Var}(X_n)
\end{pmatrix}
$$

Properties of Covariance Matrix
 - **Symmetric**: $\mathbf{\Sigma} = \mathbf{\Sigma}^T$ (since $\text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i)$)
 - **Positive semi-definite**: All eigenvalues are non-negative
 - **Diagonal elements**: $\Sigma_{ii} = \text{Var}(X_i)$
 - **Off-diagonal elements**: $\Sigma_{ij} = \text{Cov}(X_i, X_j)$ for $i \neq j$

## Independence vs. Uncorrelation

> [!warning] Important Distinction
> When $\rho_{XY} = 0$, $X$ and $Y$ are uncorrelated but not necessarily independent. Independence implies uncorrelation, but not vice versa.

> [!example]-  Uncorrelated but Not Independent
> Consider $X \sim \text{Uniform}(-1, 1)$ and $Y = X^2$. 
> 
> **Check correlation:**
> - $E[X] = 0$ (symmetric around 0)
> - $E[Y] = E[X^2] = \int_{-1}^{1} x^2 \cdot \frac{1}{2} dx = \frac{1}{3}$
> - $E[XY] = E[X \cdot X^2] = E[X^3] = 0$ (odd function, symmetric around 0)
> - $Cov(X,Y) = E[XY] - E[X]E[Y] = 0 - 0 \cdot \frac{1}{3} = 0$
> 
> So $X$ and $Y$ are **uncorrelated** ($\rho_{XY} = 0$).
> 
> **Check independence:**
> - If $X = 0.5$, then $Y = 0.25$ with probability 1
> - If $X = -0.5$, then $Y = 0.25$ with probability 1
> - But $P(Y = 0.25 | X = 0.5) = 1 \neq P(Y = 0.25) = \frac{1}{2}$
> 
> So $X$ and $Y$ are **not independent** - knowing $X$ completely determines $Y$!


## Key takeaways
 ### Mathematical Relationships
 - **Covariance**: Measures linear relationships: $Cov(X,Y) = E[XY] - E[X]E[Y]$
 - **Correlation**: Normalized covariance: $\rho_{XY} = \frac{Cov(X,Y)}{\sigma_X \sigma_Y}$
 
 ### Important Distinctions
 - **Independence ≠ Uncorrelation**: Independence implies uncorrelation, but not vice versa
 - **Correlation only measures linear relationships**: Non-linear relationships can have zero correlation
