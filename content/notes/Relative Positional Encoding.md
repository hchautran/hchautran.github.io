---
title: Relative Positional Encoding
draft: false
tags:
  - transformers
  - deep_learning
  - llm
---
## Introduction
In practice, there are scenarios where a model is trained on sequences with a limited window size but is later required to process much longer documents during inference. The motivation behind **Relative Position Encodings (RPE)** is to provide position information relative to other tokens, avoiding the use of absolute positional embeddings like those in traditional [[Transformer]].
> [!note] Note
> RPE helps mitigate performance degradation when dealing with sequences of unseen lengths.

Unlike absolute embeddings, which inject positional information directly into the input embeddings—affecting all **queries, keys**, and **values** throughout all layers. Previous works on **RPE** often incorporate relative position information by influencing the attention scores. This can be done either by: 
- Adding a bias to the attention scores as seen in [[Attention with Linear Biases (ALiBi)]].
- Modifying the query and key vectors as seen in [[Rotary Position Embeddings (RoPE)]]. 


---
## Reference
[[@pressTrainShortTest2021]]
[[@suRoFormerEnhancedTransformer2021]]