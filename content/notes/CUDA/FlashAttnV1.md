---
title: Flash Attention Work Log 
description: My Work Log of implementing Flash Attention using CuTe DSL.
draft: false
tags:
  - CUDA 
  - notes 
  - work log 
---

My Work Log implementing Flash Attention using [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

---
## 1. The Math


---
## 2. The Algorithms

$$
\begin{array}{l}
\textbf{Algorithm 1: FlashAttention-2 Forward Pass} \\
\hline
\textbf{Require: } \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d} \text{ in HBM, block sizes } B_c, B_r \\
\hline
1. \quad T_r = \lceil N / B_r \rceil,\; T_c = \lceil N / B_c \rceil \\
\quad\quad \text{Divide } \mathbf{Q} \text{ into blocks } \mathbf{Q}_1, \dots, \mathbf{Q}_{T_r} \text{ of size } B_r \times d \\
\quad\quad \text{Divide } \mathbf{K}, \mathbf{V} \text{ into blocks } \mathbf{K}_1, \dots, \mathbf{K}_{T_c},\; \mathbf{V}_1, \dots, \mathbf{V}_{T_c} \text{ of size } B_c \times d \\
2. \quad \text{Divide } \mathbf{O} \in \mathbb{R}^{N \times d} \text{ into } T_r \text{ blocks of size } B_r \times d \\
\quad\quad \text{Divide logsumexp } L \text{ into } T_r \text{ blocks of size } B_r \\
3. \quad \textbf{for } 1 \leq i \leq T_r \textbf{ do} \\
4. \quad\quad \text{Load } \mathbf{Q}_i \text{ from HBM to on-chip SRAM} \\
5. \quad\quad \text{Initialize } \mathbf{O}_i^{(0)} = \mathbf{0}_{B_r \times d},\quad \ell_i^{(0)} = \mathbf{0}_{B_r},\quad m_i^{(0)} = (-\infty)_{B_r} \\
6. \quad\quad \textbf{for } 1 \leq j \leq T_c \textbf{ do} \\
7. \quad\quad\quad \text{Load } \mathbf{K}_j, \mathbf{V}_j \text{ from HBM to on-chip SRAM} \\
8. \quad\quad\quad \mathbf{S}_i^{(j)} = \mathbf{Q}_i \mathbf{K}_j^\top \in \mathbb{R}^{B_r \times B_c} \\
9. \quad\quad\quad m_i^{(j)} = \max\!\left(m_i^{(j-1)},\, \text{rowmax}(\mathbf{S}_i^{(j)})\right) \\
\quad\quad\quad\quad \tilde{\mathbf{P}}_i^{(j)} = \exp\!\left(\mathbf{S}_i^{(j)} - m_i^{(j)}\right) \quad \text{(pointwise)} \\
\quad\quad\quad\quad \ell_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}}\, \ell_i^{(j-1)} + \text{rowsum}\!\left(\tilde{\mathbf{P}}_i^{(j)}\right) \\
10. \quad\quad\quad \mathbf{O}_i^{(j)} = \text{diag}\!\left(e^{m_i^{(j-1)} - m_i^{(j)}}\right)^{-1} \mathbf{O}_i^{(j-1)} + \tilde{\mathbf{P}}_i^{(j)} \mathbf{V}_j \\
11. \quad\quad \textbf{end for} \\
12. \quad\quad \mathbf{O}_i = \text{diag}\!\left(\ell_i^{(T_c)}\right)^{-1} \mathbf{O}_i^{(T_c)} \\
13. \quad\quad L_i = m_i^{(T_c)} + \log\!\left(\ell_i^{(T_c)}\right) \\
14. \quad\quad \text{Write } \mathbf{O}_i \text{ to HBM as the } i\text{-th block of } \mathbf{O} \\
15. \quad\quad \text{Write } L_i \text{ to HBM as the } i\text{-th block of } L \\
16. \quad \textbf{end for} \\
17. \quad \textbf{return } \mathbf{O},\; L \\
\hline
\end{array}
$$

---
## 3. Matmul V1

### 3.0 Initialize Input

### 3.1 Make A, B Load Layout

### 3.2 Make C Store Layout 

### 3.3 Construct Copy Atom

### 3.4 Construct MMA Atom

### 3.5 Kernel Implementation

### 3.6 Comparison To Torch's Matmul

--- 
## 4. Matmul V2

---
## 4. Online Softmax


--- 
## 5. Flash Attention V1


---
## 6. Flash Attention V2

---
## 7. Flash Attention V3

