---
title: Denoising Diffusion Probabilistic Models (DDPMs)
description: A ground-up derivation of the DDPM.
draft: false
tags:
  - generative models
  - notes
---
...Writing....

![VAE](HVAE.svg) *Figure 1: Hierachical Variational AutoEncoder*

<!-- The Variational Autoencoder is an elegant framework, but its Gaussian design comes with fundamental limitations: blurry reconstructions from the MSE objective, an approximate posterior restricted to a diagonal Gaussian, and an aggregate posterior that can drift away from the prior. All three problems share a common root — a single latent variable $\mathbf{z}$ is too weak a representation to capture the full complexity of real-world data.

The Hierarchical VAE (HVAE) addresses this by stacking multiple layers of stochastic latent variables:

$$\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_L$$

Rather than compressing $\mathbf{x}$ into one flat code, each layer $\mathbf{z}_l$ is responsible for capturing structure at a different level of abstraction — low-level texture and local features at the bottom, high-level semantic content at the top. This hierarchical factorization allows the model to represent far richer posteriors than any single Gaussian can, and the ELBO from the standard VAE extends naturally to the hierarchical setting. -->