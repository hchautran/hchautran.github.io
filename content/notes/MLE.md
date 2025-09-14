---
title: Maximum Likelihood Estimation (MLE)
draft: true
tags:
  - Machine Learning
  - notes
---


Let's examine a realistic joint distribution based on survey data:

| Gender ($G$) | Hours Worked ($H$) | Wealth ($W$) | Probability ($P(G,H,W)$) |
|--------|--------------|--------|-------------|
| female | < 40.5 | poor  | 0.2531      |
| female | < 40.5 | rich  | 0.0246      |
| female | ≥ 40.5 | poor  | 0.0422      |
| female | ≥ 40.5 | rich  | 0.0116      |
| male   | < 40.5 | poor  | 0.3313      |
| male   | < 40.5 | rich  | 0.0972      |
| male   | ≥ 40.5 | poor  | 0.1341      |
| male   | ≥ 40.5 | rich  | 0.1059      |


## Example: Gender, Work Hours, and Wealth


### Analysis Questions

> [!example] Analysis Questions
> From this joint distribution, we can answer questions like:
> 
> > [!example]- 1. What's the probability of being male? 
> > 
> > $p_G(\text{male}) = 0.6685$
> 
> > [!example]- 2. Given someone is female, what's the probability they work $\geq$ 40.5 hours?
> > 
> > $p_{H|G}(\text{≥40.5} | \text{female}) ≈ 0.162$
> 
> > [!example]- 3. Are gender and wealth independent?
> > 
> > $p_{G,W}(\text{female, rich}) = 0.0362$
> >
> > $p_{G}(\text{female}) \cdot p_{W}(\text{rich}) ≈ 0.0793$
> >
> > Since $0.0362 ≠ 0.0793$, gender and wealth are **not independent**.
>
> > [!example]- 3. Compare with marginal probabilities
> > 
> > - $P(\text{rich}) = 0.2393$ (23.93% of all people are rich)
> > - $P(\text{≥40.5 hours}) = 0.2938$ (29.38% of all people work long hours)
> > 