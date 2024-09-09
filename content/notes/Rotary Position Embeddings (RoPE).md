---
title: Rotary Position Embeddings (RoPE)
draft: false
tags:
  - RPE
  - read_later
---
![[Rotary Position Embeddings (RoPE).png]]
RoPE is a popular [[Relative Positional Encoding]] implementation which is used in [[Large Language Model]] like [[LLaMA]].


In order to generalize our results in 2D to any $x_i \in \mathbb{R}^d$ where $d$ is even, we divide the $d$-dimension space into $d/2$ sub-spaces and combine them in the merit of the linearity of the inner product, turning $f_{\{q,k\}}$ into:

$$
    f_{\{q,k\}}(x_m,m) = R_{\Theta,m}^d W_{\{q,k\}} x_m
$$

where

$$
    R_{\Theta,m}^d =
    \begin{pmatrix}
        \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
        \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
        0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
        0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
        0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
        0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} \\
    \end{pmatrix}
$$

is the rotary matrix with pre-defined parameters $\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, \dots, d/2]\}$. A graphic illustration of RoPE is shown in Figure (1). Applying our RoPE to self-attention in Equation (2), we obtain:

$$
    q_m^T k_n = \left(R_{\Theta,m}^d W_q x_m \right)^T \left(R_{\Theta,n}^d W_k x_n \right) = x^T W_q R_{\Theta,m}^d R_{\Theta,n-m}^d W_k x_n
$$

where $R_{\Theta,n-m}^d = \left(R_{\Theta,m}^d\right)^T R_{\Theta,n}^d$. Note that $R_{\Theta}^d$ is an orthogonal matrix, which ensures stability during the process of encoding position information.



---
## Reference
