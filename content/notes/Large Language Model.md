---
title: Large Language Model (llm)
draft: false
tags:
  - llm
  - transformers
  - decoder
  - deep_learning
---
![[Large Language Model.png]] 


Large Language Models (LLMs) have marked a significant shift in the field of natural language processing, introducing a new paradigm for understanding and generating human language. Central to this innovation is the [[Transformer]] architecture.


During inference LLM often employs an [[Auto regressive decoding]] approach. This method is central to how these models generate text, ensuring that each new word or token produced takes into account the entire sequence generated so far. Auto-regressive decoding operates under the principle of sequentially predicting the next token in a sequence, given all the previous ones,
