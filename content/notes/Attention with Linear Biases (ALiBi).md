---
title: Attention with Linear Biases (ALiBi)
draft: false
tags:
  - RPE
---
**ALiBi** is a [[Relative Positional Encoding]]  method, which is proposed to **add an offset to the** **attention matrix**  (defined as relative distance). The idea here is to change the position to be the relative distance between each token instead of absolute index with the hope of "*train short, test long*" 

![[Attention with Linear Biases (ALiBi).png]]

As you can see in the figure above, another offset matrix (right) is added to the attention score matrix. The values in the offset matrix are defined as the difference between the index of two token ($-1 * |i-j|$) where is $i$ is row and $j$ is column indexes. Here, $m$ is a scaler that is manually set to scale the offset matrix  

---
## Reference

[[@pressTrainShortTest2021]]
