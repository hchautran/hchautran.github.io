---
title: 2. Denoising Diffusion Probabilistic Models (DDPMs)
description: A ground-up derivation of the DDPM.
date: 2026-04-01
lastmod: 2026-04-02
draft: false
tags:
  - generative models
  - notes
---

## 0 Hierachical HVAE
![VAE](HVAE.svg) _Figure 1: Hierachical Variational AutoEncoder_


## 1. Denoising Diffusion Probabilistic Models 
![DDMP](DDPM.svg) _Figure 2: Denoising Diffusion Probabilistic Models: DDPMS consist of 2 component. where $q(.)$ is a fixed kernel adding gaussian noise to the data, and $p_\theta$ is a learned reverse process that try to generate new same from noise.




