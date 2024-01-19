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

Stan is a programming language that can be used to generate samples from the posterior distribution. Stan does this by utilizing a Markov Chain Monte Carlo (MCMC) algorithm, and more specifically a variant of it called Hamiltonian Monte Carlo.

The next episode is devoted to MCMC but in this one we will learn how to run it using Stan.

Follow the instruction at https://mc-stan.org/users/interfaces/ to install Stan on your local computer. 

While Stan makes model specifying a statistical model relatively easy and automates the sampling, the bottle-neck of fitting is not fully removed. In theory, MCMC methods will always converge to the target distribution (posterior for us). However, in practice, convergence issues can arise. Luckily‚ Stan produces warnings automatically which can be used in assessing model convergence. 


## Basic program structure

A Stan program is structured into several blocks that define the statistical model. A Stan program typically (but not necessarily) will include the following blocks:

1. Data block: This block is used to declare the input data fed into the model. It specifies the types and dimensions of the data variables used in the model.

2. Parameters block: In this block, the model's parameters are declared. 

3. Model block: The likelihood and prior distributions are included here in the form of sampling statements. 

It is recommended that you specify the Stan programs in a separate text files and call it from R (or other supported interface). The extension of the file must be `.stan`. 

## Example 1: Beta-binomial model
  
The following Stan program specifies the Beta-binomial model. There program consists of data, parameters, and model blocks. 

The data variables are the total sample size $N$ and the outcome of a binary variable (coin flip, handedness etc.). The declared data type in `int` for integer and the variables have a lower bound 1 and 0 for $N$ and $x$, respectively.  Notice that each line ends with a semicolon.

In the parameters block we declare $\theta$, the probability for a success. Since this parameter is a probability, it is restricted between 0 and 1.

In the model block, the likelihood is specified with the sampling statement `x ~ binomial(N, theta)`. This line includes the binomial distribution $Bin(x | N, theta)$ in the target distribution. The prior could be set similarly but omitting it implies a uniform prior. 
  


```stan
  data{
    int<lower=1> N; 
    int<lower=0> x; 
  }
  
  parameters{
    real<lower=0, upper=1> theta;
  }
  
  model{
    
    // Likelihood
    x ~ binomial(N, theta);
    
    // Uniform prior for theta
  }
```

Once the Stan program has been saved we need to compile it. In R, this is done by running


```r
binomial_model <- stan_model("binomial_model.stan")
```

Once the program has been compiled, it can be used to generate the posterior samples with the function `sampling()`. The data needs to be defined as a list. While running, Stan prints progress.


```r
binom_data <- list(N = 50, x = 7)

binom_samples <- sampling(binomial_model, binom_data)
```

```{.output}

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
Chain 1: 
Chain 1: Gradient evaluation took 5e-06 seconds
Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.05 seconds.
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
Chain 1:  Elapsed Time: 0.004 seconds (Warm-up)
Chain 1:                0.004 seconds (Sampling)
Chain 1:                0.008 seconds (Total)
Chain 1: 

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
Chain 2: 
Chain 2: Gradient evaluation took 1e-06 seconds
Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.01 seconds.
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
Chain 2:  Elapsed Time: 0.004 seconds (Warm-up)
Chain 2:                0.004 seconds (Sampling)
Chain 2:                0.008 seconds (Total)
Chain 2: 

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 3).
Chain 3: 
Chain 3: Gradient evaluation took 1e-06 seconds
Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.01 seconds.
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
Chain 3:  Elapsed Time: 0.004 seconds (Warm-up)
Chain 3:                0.003 seconds (Sampling)
Chain 3:                0.007 seconds (Total)
Chain 3: 

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 4).
Chain 4: 
Chain 4: Gradient evaluation took 2e-06 seconds
Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.02 seconds.
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
Chain 4:  Elapsed Time: 0.004 seconds (Warm-up)
Chain 4:                0.003 seconds (Sampling)
Chain 4:                0.007 seconds (Total)
Chain 4: 
```


With the default setting Stan runs 4 MCMC chains with 2000 iterations (more about this in Episode 5 on MCMC). Running `binom_samples` prints a summary for the model parameter $p$  allowing quick reviewing of result. 


```r
binom_samples
```

```{.output}
Inference for Stan model: anon_model.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

        mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
theta   0.15    0.00 0.05   0.07   0.12   0.15   0.19   0.26  1349    1
lp__  -22.83    0.02 0.70 -24.88 -22.98 -22.56 -22.38 -22.33  1569    1

Samples were drawn using NUTS(diag_e) at Fri Jan 19 13:09:38 2024.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```


This summary can also be accessed as a matrix with `summary(binom_samples)$summary`.

Often, however, it is necessary to work with the individual samples. These can be extracted as follows:


```r
p_samples <- extract(binom_samples, "theta")[["theta"]]
```

Now we can use the methods presented in the previous Episode to compute posterior summaries, credible intervals and to generate figures. 


:::::::::::::::::::::::::::::::::::: challenge

Compute the 95% credible intervals for the samples drawn with Stan. What is the probability that $theta \in (0.05, 0.15)$? Plot a histogram of the posterior samples. 


::::::::::::::::::::: solution


```r
CI95 <- quantile(theta_samples, probs = c(0.025, 0.975))
```

```{.error}
Error in eval(expr, envir, enclos): object 'theta_samples' not found
```

```r
theta_between_0.05_0.15 <- mean(theta_samples>0.05 & theta_samples<0.15)
```

```{.error}
Error in eval(expr, envir, enclos): object 'theta_samples' not found
```

```r
p <- ggplot(data = data.frame(theta = theta_samples)) +
  geom_histogram(aes(x = theta), bins = 30) +
  coord_cartesian(xlim = c(0, 1))
```

```{.error}
Error in eval(expr, envir, enclos): object 'theta_samples' not found
```

```r
print(p)
```

```{.error}
Error in eval(expr, envir, enclos): object 'p' not found
```


::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::::::::::



:::::::::::::::::::::::::::::::::::: challenge

Try modifying the Stan program so that you add a $Beta(\alpha, \beta)$ prior for $\theta$.

Can you modify the Stan program further so that you can set the hyperparameters $\alpha, \beta$ as part of the data? What is the benefit of using this approach?


::::::::::::::::::::: solution

If the data block is modified so that it declares the hyperparameters as data (e.g. `real<lower=0> alpha;`), it enables setting the hyperparameter values as part of data. This way hyperparameters can be changed without modifying the Stan file. 


::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::

## Additional Stan blocks

Functions
For user-defined functions
Must be the first block!

Transformed data
Transformations of the data variables

Transformed parameters
Transformations of the parameters
Remember Jacobian in the model block!

Generated quantities
Define quantities based on data and model parameters
Produces a posterior


## Stan shortcuts

- Modeling: rstanarm, brms; Plotting: bayesplot; Summaries and convergence: ShinyStan
- Syntax similar to standard R 
  - Linear model: fit <- stan_glm(y ~ x, data = df, family = gaussian)


We will not use these!
We will learn to	
Write Stan programs from scratch → flexibility
Access and manipulate the output
Generate visualization


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

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).
Chain 1: 
Chain 1: Gradient evaluation took 5e-06 seconds
Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.05 seconds.
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
Chain 1:  Elapsed Time: 0.008 seconds (Warm-up)
Chain 1:                0.007 seconds (Sampling)
Chain 1:                0.015 seconds (Total)
Chain 1: 

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 2).
Chain 2: 
Chain 2: Gradient evaluation took 2e-06 seconds
Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.02 seconds.
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
Chain 2:  Elapsed Time: 0.008 seconds (Warm-up)
Chain 2:                0.007 seconds (Sampling)
Chain 2:                0.015 seconds (Total)
Chain 2: 

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 3).
Chain 3: 
Chain 3: Gradient evaluation took 1e-06 seconds
Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.01 seconds.
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
Chain 3:  Elapsed Time: 0.008 seconds (Warm-up)
Chain 3:                0.006 seconds (Sampling)
Chain 3:                0.014 seconds (Total)
Chain 3: 

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 4).
Chain 4: 
Chain 4: Gradient evaluation took 2e-06 seconds
Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.02 seconds.
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
Chain 4:  Elapsed Time: 0.008 seconds (Warm-up)
Chain 4:                0.007 seconds (Sampling)
Chain 4:                0.015 seconds (Total)
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

<img src="fig/stan-rendered-unnamed-chunk-11-1.png" style="display: block; margin: auto;" />


## Example 3: Linear regression
## Example 4: Random walk



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





## Resources

- Official release paper https://www.jstatsoft.org/article/view/v076i01 
- User’s guide https://mc-stan.org/docs/2_18/stan-users-guide/
- Function’s reference https://mc-stan.org/docs/functions-reference/
- Reference manual https://mc-stan.org/docs/reference-manual/
- Stan forum https://discourse.mc-stan.org 
- Case studies https://mc-stan.org/users/documentation/case-studies

## Reading

- BDA3: Ch. 12.6, Appendix C
- Bayes Rules!: Ch. 6.2 
