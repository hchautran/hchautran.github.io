---
title: Layer Norm
draft: false
tags:
  - normalization
  - regularlization
---
![[normalization methods.png]]

## Motivation
**LayerNorm** is often preferred in models where sequences are involved, sequence length is variable within the batch size. [[Batch Norm]], on the other hand, works well in many standard deep learning models with large, fixed-size sequence length (like image ) where its ability to stabilize training through batch-wide statistics is beneficial. LayerNorm is widely used in many architecture that often process sequence data like  [[RNN]], [[Transformer]].
## Layer Norm


Layer normalization is a technique used to normalize the inputs across the features in a layer. Given an input vector $x = (x_1, x_2, \dots, x_n)$ of length $n$, layer normalization is performed as follows:  
1. **Compute the Mean:** $$ \mu = \frac{1}{n} \sum_{i=1}^{n} x_i $$ 
2. **Compute the Variance:** $$ \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 $$ 
3. **Normalize:** Each feature in the input is normalized by subtracting the mean and dividing by the standard deviation: $$ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$ Here, $\epsilon$ is a small constant added for numerical stability. 
4. **Apply Scaling and Shifting:** Finally, the normalized values are scaled and shifted using learned parameters $\gamma$ and $\beta$: $$ y_i = \gamma \hat{x}_i + \beta $$ where $\gamma$ and $\beta$ are trainable parameters that allow the network to maintain the representational power.

