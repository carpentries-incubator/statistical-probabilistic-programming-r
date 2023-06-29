---
title: 'Model checking'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- What is model checking?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Prior/Posterior predictive check
- Learn how assess model fit
  - Bayesian residuals?
  
- Model comparison with 
  - AIC, BIC, WAIC


::::::::::::::::::::::::::::::::::::::::::::::::

## Posterior predictive check

Gelman: "If the model fits, then replicated data generated under the model should look similar to observed data."

Idea: simulate data from the posterior predictive distribution and compare it to the observed data. Discrepancies between the simulated and observed data imply shortcomings in the model. 

Posterior predictive distribution presented in Episode Working with samples. 

## Information criteria


::::::::::::::::::::::::::::::::::::: keypoints 

- point 1

::::::::::::::::::::::::::::::::::::::::::::::::



## Reading

- Statistical Rethinking: Ch. 7
- BDA3: p.143: 6.3 Posterior predictive checking
