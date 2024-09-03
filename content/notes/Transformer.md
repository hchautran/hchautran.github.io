---
title: Transformer
draft: false
tags:
  - transformers
---
![[Transformer.png]]

Transformers are a type of deep learning model that revolutionized natural language processing (NLP) and other fields like computer vision. Introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017, Transformers leverage self-attention mechanisms to process and generate data more efficiently than previous models like RNNs and LSTMs. They excel at handling sequential data, enabling breakthroughs in tasks such as machine translation, text generation, and even image recognition. The architecture's scalability and versatility have made it a foundational tool in modern AI research and applications.

In this post, I'll focus mainly on two main component of the Transformer architecture
- The  Multi-head Self-Attention mechanism 
- Position-wise Feed-Forward Network

## Multi-head Self Attention

Multi-head attention is a core component of the Transformer architecture. The idea behind multi-head attention is to allow the model to focus on different parts of the input sequence from multiple perspectives simultaneously. This is achieved by using multiple attention heads, each of which computes scaled dot-product attention independently. Given an input sequence represented by a matrix $X \in \mathbb{R}^{n \times d}$, where $n$ is the sequence length and $d$ is the dimensionality of each input vector, multi-head attention is computed as follows: 
1. **Linear Projections:** The input $X$ is linearly projected into queries, keys, and values using learned weight matrices $W_Q$, $W_K$, and $W_V$ respectively: 
$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$
2. **Scaled Dot-Product Attention:** Each attention head computes the scaled dot-product attention: $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$where $d_k$ is the dimensionality of the keys (or queries), and the scaling factor $\frac{1}{\sqrt{d_k}}$ helps to stabilize the gradients. 

3. **Multi-Head Attention:** The outputs from each attention head are concatenated and linearly transformed using a weight matrix $W_O$: $$
 \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W_O
$$where $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$, and $h$ is the number of attention heads. By using multiple heads, the model can capture different aspects of the input data, leading to richer representations and better performance on various tasks.

## Position-wise Feed-Forward Network


In the Transformer architecture, the Position-wise Feed-Forward Network (FFN) is applied *independently* to each position in the sequence. It provides additional non-linearity and depth to the model. The FFN is composed of two linear transformations with a ReLU activation in between. Given an input $x \in \mathbb{R}^d$, where $d$ is the dimensionality of the input, the FFN can be defined as: $$ \text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2 $$ Here: 
- $W_1 \in \mathbb{R}^{d \times d_{ff}}$ and $W_2 \in \mathbb{R}^{d_{ff} \times d}$ are learned weight matrices, 
- $b_1 \in \mathbb{R}^{d_{ff}}$ and $b_2 \in \mathbb{R}^{d}$ are bias terms, 
- $d_{ff}$ is the dimensionality of the hidden layer, typically larger than $d$, 
- $\text{max}(0, x)$ represents the ReLU activation function. 

The Position-wise FFN is applied to each position in the sequence separately and identically. Despite its simplicity, the FFN contributes significantly to the Transformer's ability to model complex patterns in the data.


## Other components

Apart from the two main components mentioned above, the Transformer's strength also inherits several powerful features from previous architectures. These include the ability to handle sequential data like [[RNN]]-based models, while overcoming their limitations, such as vanishing gradients and slow training times
##### **Embedding Layer**

The input tokens (words or subwords) are first converted into dense vectors (embeddings) that capture their meanings in a continuous space. These embeddings are learned during training or set fixed depending on the choice of the author.

#####  **Positional Encoding**

Since the Transformer does not have a built-in sense of sequence order (unlike RNNs or LSTMs), positional encoding is added to the input embeddings to inject information about the position of each token in the sequence. This encoding is often implemented using sine and cosine functions of different frequencies.

##### **Layer Normalization**

[[Layer Norm]] is applied after the attention mechanism and the feed-forward network to stabilize and speed up training. It normalizes the outputs of these layers, ensuring that the model can learn effectively. 

##### **Residual Connections**

To help the model retain information from earlier layers, [[Residual Connection]] are added around the self-attention and feed-forward layers. These connections allow the model to learn the difference between the input and output of each layer, making training easier and improving performance.