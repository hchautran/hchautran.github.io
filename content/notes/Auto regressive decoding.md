---
title: Auto regressive decoding
draft: false
tags:
  - llm
  - decoding_algorithms
---
 **Auto-regressive decoding** is fundamental in LLM inference for generating coherent and con- textually appropriate text. It ensures that each token generated is conditioned on a comprehensive understanding of all previously generated content, allowing LLMs to produce highly relevant and fluent text sequences.
![[Auto regressive decoding illustrate.png]]
![[AutoRegessive.png]]

Here, $P(y|X_{t-1})$ represents the probability of the next token 𝑦 given the current sequence $X_{t-1}$, and $\oplus$ denotes the concatenation operation. The $argmax$ function is used to select the most probable next token at each step.

