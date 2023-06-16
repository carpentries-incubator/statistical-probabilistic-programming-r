---
title: 'Hierarchical Models'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are Bayesian hierarchical models?
- What are they good for?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the idea of hierarchical models
- Learn how to build and with hierarchical models with Stan

::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::: callout
TODO: 
shrinkage
:::::::::::::::::::::::::


## Hierachical models

Bayesian hierarchical models are a class of models suited for modeling scenarios where the study population consists of separate but related groups. Hierarchical structure refers to this organization of data into multiple levels or groups, where each level can have its own set of parameters. These parameters are connected trough a common prior that is also learned when fitting the model. Some or all of the hyperparameters of the priors are unknown model parameters and they are given hyperpriors. 

One key advantage of Bayesian hierarchical models is their ability to borrow strength across groups. By pooling information from multiple groups, these models can provide more stable estimates, especially when individual groups have limited data. This pooling of information is particularly beneficial when there are sparse observations or when data from different groups exhibit similar patterns.


Examples of scenarios where hierarchical model could be a natural choice: 



## Example: Hierarchical binomial model

Let's take a look at a hierarchical binomial model. Let $X = \{X_1, X_2, \ldots, X_N\}$ be a set of observations representing the number of successes in $n$ Bernoulli trials in $N$ different scenarios. We assume that these scenarios are not identical, so there are $N$ unknown probability parameters, $p_1, p_2, \ldots, p_N$. This model can be specified as follows.

\begin{align}
X_i &\sim Binom(n, p_i) \\
p_i &\sim Beta(\alpha, \beta) \\
\alpha, \beta &\sim Gamma(2, 1).
\end{align}

The difference to the binomial model as used in the previous episodes is that the parameters $p_i$ have a prior with unknown hyperparameters $\alpha$ and $\beta.$ These hyperparameters are given a $Gamma$ prior and learned in the inference. 

:::::::::::::::::::::::::::::::::::::: challenge

Hierarchical models are also called partially pooled models in contrast to unpooled and completely pooled models. The former mean that the model parameters are assumed to be completely independent, while in the latter type the (parallel) parameters are equal. Write the unpooled and completely pooled variants of the hierarchical binomial model. 

:::::::::::::::::::: solution

Unpooled: 

\begin{align}
X_i &\sim Binom(n, p_i) \\
p_i &\sim Beta(2, 2) \\
\end{align}

Completely pooled: 

\begin{align}
X_i &\sim Binom(n, p) \\
p &\sim Beta(2, 2) \\
\end{align}

:::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::




Let's then implement the hierarchical, along with the unpooled and completely pooled binomial model in Stan. 

Hierarchical binomial model 


```stan

```




```r
sampling(hierarchical_binomial_model)
```

```{.error}
Error in sampling(hierarchical_binomial_model): could not find function "sampling"
```





::::::::::::::::::::::::::::::::::::: keypoints 

- point 1

::::::::::::::::::::::::::::::::::::::::::::::::

