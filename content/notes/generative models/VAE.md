---
title: Variational AutoEncoder
draft: false
tags:
  - generative models
  - notes
---



In a lot of generative models courses, the starting point of training a neural network to generate new realistic data is the Variational Autoencoder (VAE). This model has its origins in the AutoEncoder (AE), which serves a different purpose: to reconstruct data:

Formally, AE consists of two parts: an encoder $f_\phi$ that compresses the input $\mathbf{x}$ into a compact latent representation $\mathbf{z} = f_\phi(\mathbf{x})$, and a decoder $g_\theta$ that reconstructs the input from that representation $\hat{\mathbf{x}} = g_\theta(\mathbf{z})$. The network is trained end-to-end by minimizing a reconstruction loss, typically mean squared error:

$$
\mathcal{L} = \| \mathbf{x} - g_\theta(f_\phi(\mathbf{x})) \|^2
$$

The bottleneck forces the encoder to learn a compressed, meaningful representation of the data. Once trained, the latent space can be used for tasks like dimensionality reduction or feature extraction. However, autoencoders have a critical limitation to serve as a generative model: the latent space has no guaranteed structure. Points in latent space are not organized in any principled way, so randomly sampling an arbitrary $\mathbf{z}$ and decoding it often yields garbage. There is no way to smoothly interpolate between examples or generate novel, realistic samples.

The Variational Autoencoder (VAE), introduced by [Kingma & Welling (2013)](https://arxiv.org/abs/1312.6114), addresses this by imposing a probabilistic structure on the latent space. Instead of mapping $\mathbf{x}$ to a fixed point $\mathbf{z}$, the encoder $q_\phi(\mathbf{z}\mid \mathbf{x})$ outputs the parameters of a distribution (usually a Gaussian). A latent vector $\mathbf{z}$ is then sampled from this distribution rather than deterministically computed. The decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$ learns to reconstruct $\mathbf{x}$ from these sampled latents. A prior $p(\mathbf{z})$ is placed over the latent space, and the encoder is regularized to stay close to this prior via KL divergence. This shift — from a deterministic bottleneck to a learned posterior — gives the latent space two important properties:

- **Continuity** -- Nearby points in latent space decode to similar outputs. Because the encoder maps each input $\mathbf{x}$ to a distribution over $\mathbf{z}$ rather than a single point, inputs that are similar naturally produce overlapping distributions — and thus neighboring regions in latent space correspond to similar decoded outputs.

- **Completeness** -- Any point sampled from the prior produces a meaningful output. By regularizing the encoder's posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$ to stay close to the prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$, the model ensures that the high-probability regions of the latent space are densely covered with meaningful structure, so random samples from the prior reliably decode into coherent outputs.

---
## 1. Constructions of VAE
Suppose we have a dataset of samples drawn i.i.d. from an unknown [distribution](join_prob.md) $p_{data}(\mathbf{x})$. Since the true form of $p_{data}$ is unknown, we cannot sample from it directly. The goal of a generative model is to learn a tractable approximation $p_\theta(\mathbf{x})$ from this finite dataset by minimizing a divergence $\mathcal{D}_f$ between the two distributions. In the case of VAEs, $\mathcal{D}_f$ is the KL divergence $\mathcal{D}_{KL}$:

$$
\mathcal{D}_{KL}(p_{data}(\mathbf{x}) \| p_{\theta}(\mathbf{x}))
$$

> [!note]- KL divergence intuition
>$$
>\begin{align}
>\mathcal{D}_{KL}(p_{data}(\mathbf{x}) \| p_\theta(\mathbf{x})) &= \int p_{data}(\mathbf{x}) \log \frac{p_{data}(\mathbf{x})}{p_{\theta}(\mathbf{x})} \, d\mathbf{x} \\
>&= \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}\left[\log \frac{p_{data}(\mathbf{x})}{p_{\theta}(\mathbf{x})}\right] \\
>&= \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}\left[\log p_{data}(\mathbf{x}) - \log p_{\theta}(\mathbf{x})\right] \\
>\end{align}
>$$
> As we can see, the KL divergence measures the expected log-likelihood difference between $p_{data}(\mathbf{x})$ and $p_\theta(\mathbf{x})$. Therefore, minimizing $\mathcal{D}_{KL}(p_{data}(\mathbf{x}) \| p_\theta(\mathbf{x}))$ pushes $p_\theta(\mathbf{x})$ to assign high likelihood to real data $\mathbf{x}$ sampled from $p_{data}(\mathbf{x})$.
> 

 Once the optimal parameters $\theta$ are found, $p_\theta(\mathbf{x})$  can be used to serve as a proxy for $p_{data}(\mathbf{x})$, enabling two key capabilities:

- **Generation**: Draw new, realistic samples from $p_{data}(\mathbf{x})$ via sampling methods such as [Monte Carlo Sampling](https://en.wikipedia.org/wiki/Monte_Carlo_method) via $p_\theta(\mathbf{x})$.
- **Evaluation**: Assess how likely a given sample $\mathbf{x}'$ is under the learned distribution $p_\theta(\mathbf{x})$ — for instance, judging whether an image $\mathbf{x}'$ looks realistic by computing the likelihood $p_\theta(\mathbf{x}')$.

Now having the target to optimize. We can rewrite the KL divergence as follow:

$$
\begin{align}
\mathcal{D}_{KL}(p_{data}(\mathbf{x}) \| p_\theta(\mathbf{x})) &= \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}\left[\log p_{data}(\mathbf{x}) - \log p_{\theta}(\mathbf{x})\right] \notag \\
&= -\mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}\left[\log p_{\theta}(\mathbf{x})\right] + \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}\left[\log p_{data}(\mathbf{x})\right] \notag \\
&= -\mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}\left[\log p_{\theta}(\mathbf{x})\right] + \mathcal{C} \notag
\end{align}
$$

The constant $\mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log p_{data}(\mathbf{x})]$ is simply the entropy of $p_{data}(\mathbf{x})$ and is independent of $\theta$. This is very convenient as $p_{data}(\mathbf{x})$ is unknown and minimizing $\mathcal{D}_{KL}$ is equivalent to maximizing the expected log-likelihood of the data $\mathbf{x}$ under $p_\theta(\mathbf{x})$:

$$
\boxed{\underset{\theta}{\arg\max} \; \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log p_\theta(\mathbf{x})]}
$$

Which is precisely the [maximum likelihood estimation (MLE)](MLE.md) objective. In practice we replace this population expectation $\mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}$ by its [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimate, yielding the empirical MLE objective now becomes:

$$
\hat{\mathcal{L}}_{\text{MLE}}(\theta) := -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(\mathbf{x}^{(i)})
$$

Where $N$ is the number of samples in the dataset. This objective is then optimized  via SGD  over minibatches.

### 1.1 Decoder (Generator)
Returning to the autoencoder setting, the goal is to generate a new sample $\mathbf{x}$ from a latent variable $\mathbf{z}$ via a neural network decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$. We can express the target distribution $p_\theta(\mathbf{x})$ as the [marginal distribution](join_prob.md):

$$
\boxed{p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}}
$$

Unfortunately, directly optimizing this objective via MLE is intractable: it requires integrating over the entire high-dimensional latent space, and since $p_\theta(\mathbf{x} \mid \mathbf{z})$ is a deep, expressive neural network with no closed-form solution, evaluating this integral exactly is computationally infeasible.  To make this optimization tractable, we need a way to focus only on latent states $\mathbf{z}$ that are likely to have generated the current input $\mathbf{x}$, rather than integrating over the entire latent space.


### 1.2 Encoder (Inference Model)
We can reframe the problem: instead of integrating over all possible $\mathbf{z}$, can we identify which latent states $\mathbf{z}$ are most likely to have produced the observed sample $\mathbf{x}$? This leads us to consider the [posterior distribution](join_prob.md) $p_\theta(\mathbf{z} \mid \mathbf{x})$, which by Bayes' rule is:

$$
p_\theta(\mathbf{z} \mid \mathbf{x}) = \frac{p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})}{p_\theta(\mathbf{x})}
$$

However, computing this posterior directly is equally intractable, as the denominator $p_\theta(\mathbf{x})$ is the same marginal likelihood we started with. This motivates approximating the true posterior with a learned inference model: 

$$
q_\phi(\mathbf{z}\mid\mathbf{x}) \approx p_\theta(\mathbf{z}\mid \mathbf{x})
$$  

And yes, this is exactly the encoder of the VAE! Which can be trained to concentrates probability mass on the $\mathbf{z}$ state that is most relevant to $\mathbf{x}$.


<!-- > [!note]- What make a tractable integral?
>
>
>
>   -->

<!-- > [!summary] TL;DR — Constructions of VAE
> The VAE consists of a **decoder** $p_\theta(\mathbf{x}\mid\mathbf{z})$ that generates data from latents, and an **encoder** $q_\phi(\mathbf{z}\mid\mathbf{x})$ that approximates the intractable posterior. -->

---
## 2. ELBO (Evidence Lower Bound)


Now that we have a controllable encoder model to generate $\mathbf{z}\sim q_\phi(\mathbf{z}\mid\mathbf{x})$. We can redefine the MLE optimization goal using $q_\phi(\mathbf{z} \mid \mathbf{x})$.

$$
\begin{align}
p_\theta(\mathbf{x}) &= \int p_\theta(\mathbf{z}, \mathbf{x}) d\mathbf{z} \notag \\
\log p_\theta(\mathbf{x}) &= \log \int p_\theta(\mathbf{z}, \mathbf{x}) d\mathbf{z} \notag \\
&= \log \int q_\phi(\mathbf{z} \mid \mathbf{x}) \frac{p_\theta(\mathbf{z}, \mathbf{x})}{q_\phi(\mathbf{z} \mid \mathbf{x})} d\mathbf{z} \notag \\
&= \log \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \frac{p_\theta(\mathbf{z}, \mathbf{x})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \right] \notag
\end{align}
$$
The learning objective is now tractable. Now according to [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality). We have the the evidence lower bound $\mathcal{L}_{ELBO}$ where:
$$
\begin{align}
\log \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \frac{p_\theta(\mathbf{z}, \mathbf{x})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \right] \geq \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{z}, \mathbf{x})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \right] = \mathcal{L}_{ELBO}
\end{align}
$$
Deriving further, we can see that $\mathcal{L}_{ELBO}$ consists of 2 terms:
$$
\begin{align}
  \mathcal{L}_{ELBO}  &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{z}, \mathbf{x})}{ q_\phi(\mathbf{z} \mid \mathbf{x})} \right] \notag \\
  &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}\mid \mathbf{z})p(\mathbf{z})}{ q_\phi(\mathbf{z} \mid \mathbf{x})} \right] \notag \\
  &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x} \mid \mathbf{z}) \right] - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p(\mathbf{z})} \right] \notag \\
  &= \boxed{\underbrace{\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x} \mid \mathbf{z}) \right]}_{\text{reconstruction error}} - \underbrace{\mathcal{D}_{KL}(q_{\phi}(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}))}_{\text{regularizing term}}}
\end{align}
$$

- **Reconstruction term** — $\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x} \mid \mathbf{z}) \right]$: This is the reconstruction objective from the standard AE, but now evaluated only over $\mathbf{z}$ sampled from the encoder, making it tractable.

- **Regularization term** — $\mathcal{D}_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$: This penalizes the encoder's posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$ for deviating from the prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, enforcing the latent space structure needed for generation.



> [!note] Why is the learning objective tractable now?
>
> The original MLE objective $\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}$ is intractable because it requires integrating $p_\theta(\mathbf{x} \mid \mathbf{z})$ — a neural network — over the entire latent space. The ELBO resolves this in two key ways:
>
> **1. Replacing the integral with a tractable expectation**:
> Instead of integrating over all $\mathbf{z}$, the reconstruction term
> $$\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$$
> only requires sampling $\mathbf{z}$ from the encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$, which concentrates mass on the latent regions most relevant to $\mathbf{x}$. 
>
> **2. A closed-form KL term**:
>  $q_\phi(\mathbf{z} \mid \mathbf{x})$ is usually modeled as a simple distribution, usually a gaussian and the KL divergence between two Gaussians has a closed-form solution — no integration is needed at all. And it can also be easily trainable via the [reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick)


Together, the two terms create a natural tension: maximizing $\mathcal{L}_{ELBO}$ encourages the decoder to recover the original input $\mathbf{x}$ as accurately as possible from latent samples $\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})$ (reconstruction term), while the regularization term pulls the encoder's posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$ back toward the prior $p(\mathbf{z})$. The VAE learns by striking a balance between these two competing objectives.


> [!note] ELBO as a Divergence Bound
>
> So what is the relationship between ELBO and the true MLE goal $p_\theta(\mathbf{x})$?
> Recall that maximum likelihood training amounts to minimizing the KL divergence between $p_{data}(\mathbf{x})$ and the learned distribution $p_\theta(\mathbf{x})$:
> $$
> \mathcal{D}_{KL}(p_{data}(\mathbf{x}) \| p_{\theta}(\mathbf{x}))
> $$
>  Since this term is intractable in general, the variational framework of VAE introduces a joint comparison $\mathbf{z}$. Specifically, consider two joint distributions 
>  - Generative Join -- Decoder: $p_\theta(\mathbf{z}, \mathbf{x})$ 
>  - Inference Join -- Encoder:  $q_\phi(\mathbf{z}, \mathbf{x})$ 
>
> The total error bound is to match these join together is:
> $$
> \begin{align}
> \mathcal{D}_{KL}(q_\phi(\mathbf{x}, \mathbf{z}) \| p_\theta(\mathbf{x}, \mathbf{z})) &= \iint q_\phi(\mathbf{x}, \mathbf{z}) \log \frac{q_\phi(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{x}, \mathbf{z})}  d\mathbf{x}  d\mathbf{z}  \notag \\
> &= \iint p_{data}(\mathbf{x}) q_{\phi}(\mathbf{z}\mid \mathbf{x}) \log(\frac{p_{data}(\mathbf{x})q_\phi(\mathbf{z}\mid \mathbf{x})}{p_\theta(\mathbf{x}) p_\theta(\mathbf{z}\mid \mathbf{x})}) d\mathbf{z}  d\mathbf{x}  \notag \\
> &=  \int p_{data}(\mathbf{x}) \log(\frac{p_{data}(\mathbf{x})}{p_\theta(\mathbf{x})}) d\mathbf{x}  \notag \\
> &+ \iint p_{data}(\mathbf{x}) q_{\phi}(\mathbf{z}\mid \mathbf{x}) \log(\frac{q_\phi(\mathbf{z}\mid \mathbf{x})}{ p_\theta(\mathbf{z}\mid \mathbf{x})}) d\mathbf{z}  d\mathbf{x}  \notag  \\
> &= \underbrace{\mathcal{D}_{KL}(p_{data}(\mathbf{x}) \| p_{\theta}(\mathbf{x}))}_{\text{True Modeling Error}}  \notag \\
> &+ \underbrace{\mathbb{E}_{\mathbf{x}\sim p_{data}(\mathbf{x})}\left[\mathcal{D}_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x}))  \right]}_{\text{Inference Error}} \notag\\
> \end{align} 
> $$
> Thus we have 
>$$
>\mathcal{D}_{KL}(q_\phi(\mathbf{x}, \mathbf{z}) \| p_\theta(\mathbf{x}, \mathbf{z})) \geq \mathcal{D}_{KL}(p_{data}(\mathbf{x}) \| p_{\theta}(\mathbf{x}))
>$$
> Where equality happens when inference error is zeros, which also means the encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ perfectly model the unknow posterior distribution $p_\theta(\mathbf{z} \mid \mathbf{x})$.  
>
> Note  that $\mathcal{L}_{ELBO}$ can also be rewritten as :  
> $$
>\begin{align}
> \mathcal{L}_{ELBO} = \log p_\theta(\mathbf{x}) - \mathcal{D}_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x}))  \notag \\ 
> \rightarrow \log p_\theta(\mathbf{x}) - \mathcal{L}_{ELBO} =  \mathcal{D}_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x}))  \notag 
>\end{align}
> $$
> We can see that the gap between the true log-likelihood $\log p_\theta(\mathbf{x})$ and the ELBO is precisely the inference error of the current sample $\mathbf{x}$. Maximizing the ELBO therefore directly reduces this gap. Specifically, optimizing the encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$ tightens the bound by bringing the approximate posterior closer to the true one $p_\theta(\mathbf{z}\mid\mathbf{x})$, while optimizing the decoder $p_\theta(\mathbf{x}\mid\mathbf{z})$ pushes the $p_\theta(\mathbf{x})$ itself upward — lifting the entire lower bound and improving the overall log-likelihood.
>

<!-- > [!summary] TL;DR — ELBO
> By Jensen's inequality, $\log p_\theta(\mathbf{x}) \geq \mathcal{L}_\text{ELBO}$. The ELBO decomposes into a **reconstruction term** (maximize decoder fidelity) minus a **KL term** (keep encoder close to prior), both of which are tractable to optimize. -->

---
## 3. Gaussian VAEs

The most common instantiation of the VAE framework is the **Gaussian VAE**, where the encoder, decoder and prior are modeled as Gaussians.

### 3.1 The encoder part
For each input $\mathbf{x}$, the encoder produces a Gaussian distribution centered at $\boldsymbol{\mu}_\phi(\mathbf{x})$ with variance $\boldsymbol{\sigma}^2_\phi(\mathbf{x})$, so that similar inputs yield overlapping distributions in the latent space:
$$
\begin{align}
\mathbf{z} &= \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \notag \\
&\Rightarrow q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z};\, \boldsymbol{\mu}_\phi(\mathbf{x}),\, \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))
\end{align}
$$

This is the reparameterization trick: by expressing $\mathbf{z}$ as a deterministic function of $\phi$ and a fixed noise variable $\boldsymbol{\varepsilon}$, the stochasticity is separated from the parameters, making the sampling step differentiable and allowing gradients to flow back through $\mathbf{z}$ to the encoder.

Since the prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ is also Gaussian, the KL divergence between two Gaussians admits a closed-form solution — no numerical integration required:

$$
\begin{align}
\mathcal{L}_{KL} = \mathcal{D}_{KL}( \mathcal{N}(\boldsymbol{\mu}_\phi, \text{diag}(\boldsymbol{\sigma}_\phi^2)) \| \mathcal{N}(\mathbf{0}, \mathbf{I})) \\  
\boxed{ \mathcal{L}_{KL} =  -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)}
\end{align}
$$
With $d$ is the number of dimension of the latent space. 

> [!note]- Derivation of closed-form KL loss
>Since both $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi, \text{diag}(\boldsymbol{\sigma}_\phi^2))$ and $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ are diagonal, the KL factorizes over dimensions. It suffices to derive for a single scalar dimension $z \sim \mathcal{N}(\mu, \sigma^2)$ vs $z \sim \mathcal{N}(0, 1)$:
>
>$$
>\begin{align}
>\mathcal{D}_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))
>&= \int \mathcal{N}(z;\mu,\sigma^2) \log \frac{\mathcal{N}(z;\mu,\sigma^2)}{\mathcal{N}(z;0,1)} \, dz \notag \\
>&= \mathbb{E}_q \left[ \log \mathcal{N}(z;\mu,\sigma^2) - \log \mathcal{N}(z;0,1) \right] \notag \\
>&= \mathbb{E}_q \left[ \left(-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}\right) - \left(-\frac{1}{2}\log(2\pi) - \frac{z^2}{2}\right) \right] \notag \\
>&= \mathbb{E}_q \left[ -\frac{1}{2}\log\sigma^2 - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2} \right] \notag \\
>&= -\frac{1}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\underbrace{\mathbb{E}_q[(z-\mu)^2]}_{=\,\sigma^2} + \frac{1}{2}\underbrace{\mathbb{E}_q[z^2]}_{=\,\sigma^2 + \mu^2} \notag \\
>&= -\frac{1}{2}\log\sigma^2 - \frac{1}{2} + \frac{\sigma^2 + \mu^2}{2} \notag \\
>&= -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right) \notag
>\end{align}
>$$
>
>Summing over all $d$ independent dimensions:
>
>$$
>\boxed{\mathcal{D}_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z})) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)}
>$$
>

Taking the gradient of the KL term with respect to $\mu_j$ and $\sigma_j^2$ we have: 

$$
\begin{align}
\frac{\partial}{\partial \mu_j} \mathcal{D}_{KL} &= \frac{\partial}{\partial \mu_j} \left[ -\frac{1}{2}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2) \right] = \mu_j \notag \\
\frac{\partial}{\partial \sigma_j^2} \mathcal{D}_{KL} &= -\frac{1}{2}\left(\frac{1}{\sigma_j^2} - 1\right) \notag
\end{align}
$$

Setting these to zero $\Rightarrow \hat{\mu}_j = 0 ,\quad \hat{\sigma}_j^2 = 1$. Therefore minimizing the KL term alone pushes the encoder toward:

$$
 q_\phi(\mathbf{z} \mid \mathbf{x}) \longrightarrow \mathcal{N}(\mathbf{0}, \mathbf{I}) = p(\mathbf{z})
$$

This is why the reconstruction term is essential: it pulls $\mu_j$ away from zero and $\sigma_j^2$ toward smaller values to make $\mathbf{z}$ informative about $\mathbf{x}$.


### 3.2 The Decoder part
To counteract collapse from the regularization term, the reconstruction term enforces that $\mathbf{z}$ remains informative about $\mathbf{x}$. Specifically, the decoder is trained to output a sample $\mathbf{x}'$ that resembles the original input as closely as possible, given a latent vector $\mathbf{z}$ drawn from the encoder's posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$. Note that $\mathbf{x}'$ need not be identical to $\mathbf{x}$: 
$$
\begin{align}
\mathbf{x} =  \boldsymbol{\mu}_\theta(\mathbf{z}) + \sigma \odot \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \notag  \notag\\ 
 \Rightarrow p_\theta(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\mathbf{x};\, \boldsymbol{\mu}_\theta(\mathbf{z}),\, \mathbf{I}\sigma)
\end{align}
$$
Here $\boldsymbol{\mu}_\theta(\mathbf{z})$ is the output of a neural network decoder, and $\sigma$ is a fixed hyperparameter controlling the spread of the output distribution — large $\sigma$ allows more deviation from the input, while small $\sigma$ forces the reconstruction to stay close to the input $\mathbf{x}$. The reconstruction loss can now be rewritten as:

$$
\begin{align}
\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x} \mid \mathbf{z}) \right] &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[-\frac{1}{2\sigma^2}\|\mathbf{x} -\boldsymbol{\mu}_\theta(\mathbf{z})\|^2 \right] + \log\!\left(\frac{1}{\sqrt{2\pi \sigma^2}}\right) \notag \\
&\propto -\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2 \right] \notag \\
\end{align}
$$
$$
\boxed{\mathcal{L}_{recon} = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2 \right]}
$$

This is equivalent to minimizing the expected MSE between the input $\mathbf{x}$ and the decoder output $\boldsymbol{\mu}_\theta(\mathbf{z})$ — which is similar to the original AE loss.   



### 3.3 Overall Training Procedure
With both the encoder and decoder defined, the full training procedure follows directly from maximizing the ELBO. Each training step processes a minibatch of inputs:

$$
\begin{array}{l}
\textbf{Algorithm: Training Gaussian VAE with ELBO} \\
\hline
\textbf{Input: } \text{Dataset } \mathcal{X} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\}, \text{ batch size } B, \text{ learning rate } \eta \\
\textbf{Output: } \text{Encoder parameters } \phi, \text{ Decoder parameters } \theta \\
\hline
1. \quad \text{Initialize } \phi, \theta \text{ randomly} \\
2. \quad \textbf{repeat} \\
3. \quad\quad \text{Sample minibatch } \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(B)}\} \sim \mathcal{X} \\
4. \quad\quad \textbf{// Encoder forward pass} \\
5. \quad\quad (\boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{(i)}) \leftarrow \text{Encoder}_\phi(\mathbf{x}^{(i)}) \quad \forall i \\
6. \quad\quad \textbf{// Reparameterization trick} \\
7. \quad\quad \boldsymbol{\varepsilon}^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
8. \quad\quad \mathbf{z}^{(i)} \leftarrow \boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\varepsilon}^{(i)} \\
9. \quad\quad \textbf{// Decoder forward pass} \\
10. \quad\quad \hat{\mathbf{x}}^{(i)} \leftarrow \text{Decoder}_\theta(\mathbf{z}^{(i)}) \quad \forall i \\
11. \quad\quad \textbf{// Compute ELBO} \\
12. \quad\quad \mathcal{L}_{\text{recon}} \leftarrow \frac{1}{B} \sum_i \log p_\theta(\mathbf{x}^{(i)} \mid \mathbf{z}^{(i)}) \\
13. \quad\quad \mathcal{L}_{\text{KL}} \leftarrow \frac{1}{B} \sum_i \mathcal{D}_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}^{(i)}) \| p(\mathbf{z})) \\
14. \quad\quad \mathcal{L}_{\text{ELBO}} \leftarrow \mathcal{L}_{\text{recon}} - \mathcal{L}_{\text{KL}} \\
15. \quad\quad \textbf{// Update parameters} \\
16. \quad\quad (\phi, \theta) \leftarrow (\phi, \theta) + \eta \cdot \nabla_{\phi, \theta}\, \mathcal{L}_{\text{ELBO}} \\
17. \quad \textbf{until } \text{convergence}\\
\hline
\end{array}
$$


---
## 4. Drawbacks of Gaussian VAEs

Despite its elegance, the Gaussian VAE has several well-known limitations:

### 4.1 Blurry reconstructions
Modeling $p_\theta(\mathbf{x} \mid \mathbf{z})$ as a Gaussian with a fixed variance corresponds to minimizing MSE, which tends to average over multiple plausible reconstructions. 

> [!note] Proof
> Recall the per-sample reconstruction loss:
> $$
> \mathcal{L}_{recon}=\mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\mid\mathbf{x})}\left[\|\mathbf{x}-\boldsymbol\mu_\theta(\mathbf{z})\|^2\right]
> $$
> When training over the full dataset, we optimize its expectation over all $\mathbf{x} \sim p_{data}$:
> $$
> \begin{align}
> &\mathbb{E}_{\mathbf{x}\sim p_{data},\, \mathbf{z}\sim q_\phi(\mathbf{z}\mid\mathbf{x})}\left[\|\mathbf{x}-\boldsymbol\mu_\theta(\mathbf{z})\|^2\right] \notag \\
> &= \iint p_{data}(\mathbf{x})\,q_\phi(\mathbf{z}|\mathbf{x})\,\|\mathbf{x}-\boldsymbol\mu_\theta(\mathbf{z})\|^2 \, d\mathbf{x}\, d\mathbf{z} \notag \\
> &= \iint q_\phi(\mathbf{z})\,q_\phi(\mathbf{x}|\mathbf{z})\,\|\mathbf{x}-\boldsymbol\mu_\theta(\mathbf{z})\|^2 \, d\mathbf{x}\, d\mathbf{z} \notag \\
> &= \int q_\phi(\mathbf{z}) \left[\int q_\phi(\mathbf{x}|\mathbf{z})\,\|\mathbf{x}-\boldsymbol\mu_\theta(\mathbf{z})\|^2 \, d\mathbf{x}\right] d\mathbf{z} \notag \\
> &= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z})}\!\left[\mathbb{E}_{\mathbf{x}\sim q_\phi(\mathbf{x} \mid \mathbf{z})}\!\left[\|\mathbf{x}-\boldsymbol\mu_\theta(\mathbf{z})\|^2\right]\right] \notag
> \end{align}
> $$
> Since $\theta$ only appears in the inner expectation and has no effect on the aggregate posterior $q_\phi(\mathbf{z})$, the outer expectation acts as a constant weight. It suffices to minimize the inner term with respect to $\boldsymbol\mu_\theta(\mathbf{z})$ for each fixed $\mathbf{z}$. Taking the gradient and setting it to zero:
> $$
> \begin{align}
> \frac{\partial}{\partial \boldsymbol\mu_\theta}\,\mathbb{E}_{\mathbf{x}\sim q_\phi(\mathbf{x}\mid\mathbf{z})}\!\left[\|\mathbf{x}-\boldsymbol\mu_\theta(\mathbf{z})\|^2\right]
> &= \mathbb{E}_{\mathbf{x}\sim q_\phi(\mathbf{x}\mid\mathbf{z})}\!\left[-2\left(\mathbf{x} - \boldsymbol\mu_\theta(\mathbf{z})\right)\right] = \mathbf{0} \notag \\
> &\Rightarrow\quad \boxed{\boldsymbol\mu_\theta^*(\mathbf{z}) = \mathbb{E}_{\mathbf{x}\sim q_\phi(\mathbf{x}\mid\mathbf{z})}\!\left[\mathbf{x}\right]}
> \end{align}
> $$
> The optimal decoder output is the **conditional mean** of $\mathbf{x}$ given $\mathbf{z}$ under the encoder's inverse distribution $q_\phi(\mathbf{x}\mid \mathbf{z})$. When multiple distinct images $\mathbf{x}$ map to similar latent codes $\mathbf{z}$, the MSE loss forces the decoder to output their average — producing blurry reconstructions.

For image data, this produces blurry outputs rather than sharp, realistic samples.



### 4.2 Limited posterior expressiveness
The diagonal Gaussian assumption for $q_\phi(\mathbf{z} \mid \mathbf{x})$ restricts the approximate posterior to an axis-aligned ellipsoid (i.e., zero off-diagonal [covariance](covariance.md)). If the true posterior $p_\theta(\mathbf{z} \mid \mathbf{x})$ has complex, multimodal, or highly correlated structure, a single Gaussian cannot capture it — leading to a persistently loose ELBO bound regardless of encoder capacity.

### 4.3 Mismatch between aggregate posterior and prior
Even if each individual posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$ is close to the prior, the **aggregate posterior** $q_\phi(\mathbf{z}) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[q_\phi(\mathbf{z} \mid \mathbf{x})]$ may not match $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$. This mismatch creates "holes" in the latent space — regions with high prior probability but low posterior density — causing poor sample quality at generation time.

These limitations motivate more expressive extensions, such as **Hierarchical VAEs**, which stack multiple layers of latent variables to capture richer structure.


---
## 5. Conclusion

The Variational Autoencoder is a foundational generative model that elegantly combines probabilistic inference with deep learning. By replacing the deterministic bottleneck of a standard autoencoder with a learned posterior distribution, VAEs endow the latent space with a structured, continuous geometry that supports both generation and interpolation. The ELBO provides a tractable training objective that simultaneously encourages faithful reconstruction and regularizes the latent space toward a simple prior — a tension that lies at the heart of all latent-variable generative models.

That said, the Gaussian VAE is far from perfect. The three drawbacks discussed above — blurry reconstructions from the MSE objective, limited posterior expressiveness from the diagonal Gaussian assumption, and aggregate posterior mismatch — are not merely implementation details; they are fundamental limitations that arise from the design choices made to keep the ELBO tractable.

**What comes next?** Two important lines of work build directly on these observations:

- [Hierarchical VAEs (HVAEs)](HVAE.md) address the expressiveness problem by stacking multiple layers of stochastic latent variables. Rather than compressing $\mathbf{x}$ into a single $\mathbf{z}$, HVAEs learn a hierarchy $\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_L$ where each layer captures structure at a different level of abstraction. This allows the model to represent far richer posteriors, and the ELBO generalizes naturally to the hierarchical setting.

- [Denoising Diffusion Probabilistic Models (DDPMs)](DDPM.md) take a different philosophical path. Instead of learning a compact latent code, diffusion models define a fixed forward process that gradually corrupts data with Gaussian noise over $T$ steps, then learn to reverse this process step by step. Remarkably, this can be seen as a special case of a hierarchical latent-variable model where the encoder is fixed (the forward noising process) and only the decoder (the denoising network) is learned. This design sidesteps the blurry-reconstruction and posterior-collapse problems entirely — the fixed encoder cannot collapse, and the step-by-step denoising objective enforces sharp, high-frequency detail at each scale. 



---
## References

[1]-- Kingma, D. P., & Welling, M. (2013). *Auto-encoding variational Bayes*. 

[2]--The Principles of Diffusion Models. (n.d.). <https://the-principles-of-diffusion-models.github.io/>

[3]--Wikipedia contributors. (n.d.). *Jensen's inequality*. Wikipedia. <https://en.wikipedia.org/wiki/Jensen%27s_inequality>

[4]--Wikipedia contributors. (n.d.). *Monte Carlo method*. Wikipedia. <https://en.wikipedia.org/wiki/Monte_Carlo_method>

[5]--Wikipedia contributors. (n.d.). *Reparameterization trick*. Wikipedia. <https://en.wikipedia.org/wiki/Reparameterization_trick>



> [!note]- Notations
> See the [notation reference](notation.md) for a summary of symbols used across all notes.
