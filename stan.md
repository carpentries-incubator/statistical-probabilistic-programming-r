---
title: 'Stan'
teaching: 10
exercises: 2
---






:::::::::::::::::::::::::::::::::::::: questions 

- What is Stan?
- How to efficiently generate samples from a posterior?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Install Stan and get it running
- Learn how to specify models in Stan
- How to generate samples with Stan
- How to process samples generated with Stan 

::::::::::::::::::::::::::::::::::::::::::::::::

Stan is a high-level programming language that can be used for sampling from the posterior distribution of a statistical model. It provides a flexible and efficient means the for inference of Bayesian models. Stan generates samples from the posterior of a defined statistical model utilizing Markov Chain Monte Carlo (MCMC) sampling, and more specifically its the Hamiltonian Monte Carlo variant which is a highly effective variant of MCMC. 


Follow the instruction at https://mc-stan.org/users/interfaces/ to install Stan on your local computer. 


## Basic program structure

- Write the program in a separate text file

- Call Stan from R, command line, or several other languages. This creates a collection of posterior samples. 

- Analyse the posterior samples with tools presented in the previous episode. 


Next, let's write the models we have previously worked with in Stan. 

## Example 1: Binomial model

A Stan program is structured into several blocks that define the statistical model and specify the algorithm for inference. A Stan program typically (but not necessarily) includes the following blocks:

1. Data block: This block is used to declare the input data fed into the model. It specifies the types and dimensions of the data variables used in the model.

2. Parameters block: In this block, the model's parameters, that are inferred, are declared. 

3. Model block: The model block contains the specification of the statistical model, consisting of the likelihood and the prior distributions. 



```stan
  data{
    int<lower=1> N; 
    int<lower=0> x; 
  }
  
  parameters{
    real<lower=0, upper=1> p;
  }
  
  model{
    
    // Likelihood
    x ~ binomial(N, p);
    
    // Uniform prior for p
    
  }
```

Notice that the types of each data variable and parameter needs to be specified. Moreover, it is helpful to specify the ranges the variables can get. For example, a variable that can only get positive values can be specified with a `<lower=0>` after the type specification. 

The sampling statement `x ~ binomial(N, p);` in the model block defines that the likelihood for the data is the binomial distribution. No prior distribution is specified for the parameter $p$ which implicitly imposes a uniform distribution. 
 
Next, let's define the data as a list and then call the Stan program. 


```r
binom_data <- list(N = 50, x = 7)

binom_samples <- sampling(binomial_model, binom_data)
```

```{.output}

SAMPLING FOR MODEL 'b30a4b20165ba5a239ed8d361ba485d2' NOW (CHAIN 1).
Chain 1: 
Chain 1: Gradient evaluation took 8e-06 seconds
Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.08 seconds.
Chain 1: Adjust your expectations accordingly!
Chain 1: 
Chain 1: 
Chain 1: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 1: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 1: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 1: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 1: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 1: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 1: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 1: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 1: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 1: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 1: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 1: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 1: 
Chain 1:  Elapsed Time: 0.007641 seconds (Warm-up)
Chain 1:                0.006366 seconds (Sampling)
Chain 1:                0.014007 seconds (Total)
Chain 1: 

SAMPLING FOR MODEL 'b30a4b20165ba5a239ed8d361ba485d2' NOW (CHAIN 2).
Chain 2: 
Chain 2: Gradient evaluation took 3e-06 seconds
Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.03 seconds.
Chain 2: Adjust your expectations accordingly!
Chain 2: 
Chain 2: 
Chain 2: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 2: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 2: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 2: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 2: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 2: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 2: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 2: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 2: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 2: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 2: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 2: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 2: 
Chain 2:  Elapsed Time: 0.007607 seconds (Warm-up)
Chain 2:                0.006282 seconds (Sampling)
Chain 2:                0.013889 seconds (Total)
Chain 2: 

SAMPLING FOR MODEL 'b30a4b20165ba5a239ed8d361ba485d2' NOW (CHAIN 3).
Chain 3: 
Chain 3: Gradient evaluation took 3e-06 seconds
Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.03 seconds.
Chain 3: Adjust your expectations accordingly!
Chain 3: 
Chain 3: 
Chain 3: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 3: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 3: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 3: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 3: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 3: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 3: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 3: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 3: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 3: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 3: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 3: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 3: 
Chain 3:  Elapsed Time: 0.007683 seconds (Warm-up)
Chain 3:                0.006846 seconds (Sampling)
Chain 3:                0.014529 seconds (Total)
Chain 3: 

SAMPLING FOR MODEL 'b30a4b20165ba5a239ed8d361ba485d2' NOW (CHAIN 4).
Chain 4: 
Chain 4: Gradient evaluation took 4e-06 seconds
Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.04 seconds.
Chain 4: Adjust your expectations accordingly!
Chain 4: 
Chain 4: 
Chain 4: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 4: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 4: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 4: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 4: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 4: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 4: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 4: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 4: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 4: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 4: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 4: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 4: 
Chain 4:  Elapsed Time: 0.011887 seconds (Warm-up)
Chain 4:                0.008681 seconds (Sampling)
Chain 4:                0.020568 seconds (Total)
Chain 4: 
```


With the default setting Stan runs 4 MCMC chains with 2000 iterations (more about this in Episode 5 on MCMC). Running `binom_samples` prints a posterior summary for the model parameter $p$ that allows you to quickly review the results. 


```r
binom_samples
```

```{.output}
Inference for Stan model: b30a4b20165ba5a239ed8d361ba485d2.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

       mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
p      0.15    0.00 0.05   0.07   0.12   0.15   0.18   0.26  1553    1
lp__ -22.82    0.02 0.71 -24.94 -22.98 -22.54 -22.38 -22.33  1615    1

Samples were drawn using NUTS(diag_e) at Fri Jun 16 07:40:02 2023.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

This summary can also be accessed as a matrix with `summary(binom_samples)$summary`.

Often, however, it is necessary to work with the individual samples. These can be extracted as follows:


```r
p_samples <- extract(binom_samples, "p")[["p"]]
```

Now we can use the methods presented in the previous Episode to compute posterior summaries, credible intervals and to generate figures. 


:::::::::::::::::::::::::::::::::::: challenge

(Revision)

Compute the 95% credible intervals for the samples drawn with Stan. What is the probability that $p \in [0.05, 0.15]$? Plot a histogram of the posterior samples. 


::::::::::::::::::::: solution


```r
CI95 <- quantile(p_samples, probs = c(0.025, 0.975))
p_between_0.05_0.15 <- mean(p_samples>0.05 & p_samples<0.15)


p <- ggplot(data = data.frame(p = p_samples)) +
  geom_histogram(aes(x = p), bins = 30) +
  coord_cartesian(xlim = c(0, 1))


print(p)
```

<img src="fig/stan-rendered-unnamed-chunk-6-1.png" style="display: block; margin: auto;" />


::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::::::::::



:::::::::::::::::::::::::::::::::::: challenge
It recommended that the Stan program is specified in a separate text file that is then called from R. Copy the binomial model from above into a text file and save it as `binomial.stan`. Then call it from R with `binomial_model <- stan_model("binomial.stan")` and generate samples with `rstan::sampling(binomial_model, stan_data)`.

Try setting a $Beta$ prior for p. 

Can you modify the Stan program further so that you can set the hyperparameters $\alpha, \beta$ as part of the data? What is the benefit of using this approach?


::::::::::::::::::::: solution

If the data block is modified so that it declares the hyperparameters as data (e.g. `real<lower=0> alpha;`), it enables setting the hyperparameter values as part of data. This way several hyperparameters can be tried out quickly, without modifying the Stan file. 


::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::



Let's use this Stan program to fit the handedness data encountered in the previous episodes.

## Example 2: normal model

Next, let's implement the normal model in Stan. First generate some data with unknown mean and standard deviation parameters $\mu$ and $\sigma$


```r
# Sample size
N <- 99

# Generate data with unknown parameters
unknown_sigma <- runif(1, 0, 10)
unknown_mu <- runif(1, -5, 5)

X <- rnorm(n = N,
           mean = unknown_mu,
           sd = unknown_sigma) 
```


Then the stan program, which the following piece of code stores as `normal_model`.


```stan

data {
  int<lower=0> N;
  vector[N] X;
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  
  // likelihood is vectorized!
  X ~ normal(mu, sigma);
  
  // Priors
  mu ~ normal(0, 1);
  sigma ~ gamma(2, 1);
  
}

```


Notice that the likelihood in the model block is vectorized. Alternatively, one could write a for loop that samples each data point individually from the likelihood: `X[i] ~normal(\mu, \sigma)`. This would be an inefficient way of implementing the likelihood function and vectorization is recommended. However, when writing complex models it may be useful to initially write the model in an unvectorized format so debugging is easier.  

Let's again fit the model on the data 


```r
# Call Stan
normal_samples <- rstan::sampling(normal_model, 
                                  list(N = N, X = X))
```

```{.output}

SAMPLING FOR MODEL 'ce5e697fa501eb08497df31f1fdce3a0' NOW (CHAIN 1).
Chain 1: 
Chain 1: Gradient evaluation took 9e-06 seconds
Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.09 seconds.
Chain 1: Adjust your expectations accordingly!
Chain 1: 
Chain 1: 
Chain 1: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 1: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 1: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 1: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 1: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 1: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 1: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 1: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 1: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 1: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 1: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 1: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 1: 
Chain 1:  Elapsed Time: 0.01404 seconds (Warm-up)
Chain 1:                0.011656 seconds (Sampling)
Chain 1:                0.025696 seconds (Total)
Chain 1: 

SAMPLING FOR MODEL 'ce5e697fa501eb08497df31f1fdce3a0' NOW (CHAIN 2).
Chain 2: 
Chain 2: Gradient evaluation took 5e-06 seconds
Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.05 seconds.
Chain 2: Adjust your expectations accordingly!
Chain 2: 
Chain 2: 
Chain 2: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 2: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 2: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 2: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 2: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 2: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 2: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 2: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 2: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 2: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 2: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 2: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 2: 
Chain 2:  Elapsed Time: 0.01263 seconds (Warm-up)
Chain 2:                0.0117 seconds (Sampling)
Chain 2:                0.02433 seconds (Total)
Chain 2: 

SAMPLING FOR MODEL 'ce5e697fa501eb08497df31f1fdce3a0' NOW (CHAIN 3).
Chain 3: 
Chain 3: Gradient evaluation took 5e-06 seconds
Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.05 seconds.
Chain 3: Adjust your expectations accordingly!
Chain 3: 
Chain 3: 
Chain 3: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 3: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 3: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 3: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 3: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 3: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 3: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 3: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 3: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 3: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 3: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 3: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 3: 
Chain 3:  Elapsed Time: 0.016276 seconds (Warm-up)
Chain 3:                0.012066 seconds (Sampling)
Chain 3:                0.028342 seconds (Total)
Chain 3: 

SAMPLING FOR MODEL 'ce5e697fa501eb08497df31f1fdce3a0' NOW (CHAIN 4).
Chain 4: 
Chain 4: Gradient evaluation took 4e-06 seconds
Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.04 seconds.
Chain 4: Adjust your expectations accordingly!
Chain 4: 
Chain 4: 
Chain 4: Iteration:    1 / 2000 [  0%]  (Warmup)
Chain 4: Iteration:  200 / 2000 [ 10%]  (Warmup)
Chain 4: Iteration:  400 / 2000 [ 20%]  (Warmup)
Chain 4: Iteration:  600 / 2000 [ 30%]  (Warmup)
Chain 4: Iteration:  800 / 2000 [ 40%]  (Warmup)
Chain 4: Iteration: 1000 / 2000 [ 50%]  (Warmup)
Chain 4: Iteration: 1001 / 2000 [ 50%]  (Sampling)
Chain 4: Iteration: 1200 / 2000 [ 60%]  (Sampling)
Chain 4: Iteration: 1400 / 2000 [ 70%]  (Sampling)
Chain 4: Iteration: 1600 / 2000 [ 80%]  (Sampling)
Chain 4: Iteration: 1800 / 2000 [ 90%]  (Sampling)
Chain 4: Iteration: 2000 / 2000 [100%]  (Sampling)
Chain 4: 
Chain 4:  Elapsed Time: 0.015885 seconds (Warm-up)
Chain 4:                0.011912 seconds (Sampling)
Chain 4:                0.027797 seconds (Total)
Chain 4: 
```


 and plot the posterior

```r
# Extract parameter samples
par_samples <- extract(normal_samples, c("mu", "sigma")) %>% 
  do.call(cbind, .) %>% 
  data.frame


# Full posterior
p_posterior <- ggplot(data = par_samples) + 
  geom_point(aes(x = mu, y = sigma)) +
  geom_point(aes(x = unknown_mu, y = unknown_sigma),
             color = "red", size = 5)

# Marginal posteriors
p_marginals <- ggplot(data = par_samples %>% gather) + 
  geom_histogram(aes(x = value), bins = 40) + 
  geom_vline(data = data.frame(key = c("mu", "sigma"), 
                               value = c(unknown_mu, unknown_sigma)), 
             aes(xintercept = value), color = "red", linewidth = 1) +
  facet_wrap(~key, scales = "free")


p <- cowplot::plot_grid(p_posterior, p_marginals,
                  ncol = 1)

print(p)
```

<img src="fig/stan-rendered-unnamed-chunk-10-1.png" style="display: block; margin: auto;" />


## Example 3: Linear regression



:::::::::::::::::::::::::::::::::::: challenge

Write a Stan program for linear regression with one dependent variable. 

::::::::::::::::::::: solution


```stan
data {
  int<lower=0> N; // Sample size
  vector[N] x; // x-values
  vector[N] y; // y-values
}
parameters {
  real alpha; // intercept
  real beta;  // slope
  real<lower=0> sigma; // noise
}

model {
  
  // Likelihood
  y ~ normal(alpha + beta * x, sigma);
  
  // Priors
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ gamma(2, 1);
}

```


::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::: challenge

Write a Stan program for linear regression with $M$ dependent variables. 

::::::::::::::::::::: solution


```stan

data {
  int<lower=0> N; // Sample size
  int<lower=0> M; // Number of features
  matrix[N, M] x; // x-values
  vector[N] y; // y-values
}
parameters {
  real alpha; // intercept
  vector[M] beta;  // slopes
  real<lower=0> sigma; // noise
}

model {
  
  // Likelihood
  y ~ normal(alpha + x * beta, sigma);
  
  // Priors
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ gamma(2, 1);
}

```


::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::















::::::::::::::::::::::::::::::::::: callout

- With Stan, you can fit model that have continuous parameters. Models with discrete parameters such as most classification models are typically impossible to fit, although some workarounds have been implemented. 

- Bayesplot, rstanarm, brms

:::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints 

- Stan is...

::::::::::::::::::::::::::::::::::::::::::::::::





## Reading

- Installation guide: https://mc-stan.org/users/interfaces/
- User's manual: https://mc-stan.org/docs/stan-users-guide/index.html
- Bayes Rules!: Chp. 6.2 https://www.bayesrulesbook.com/chapter-6.html#markov-chains-via-rstan
