---
title: 'Stan'
teaching: 10
exercises: 2
---






:::::::::::::::::::::::::::::::::::::: questions 

- What is Stan?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

Learn how to:
- implement statistical models in Stan
- generate posterior samples with Stan
- extract and process samples generated with Stan 

::::::::::::::::::::::::::::::::::::::::::::::::

Stan, a programming language, is as a tool for generating samples from the posterior distribution. It achieves this by applying a Markov Chain Monte Carlo (MCMC) algorithm, specifically a variant known as Hamiltonian Monte Carlo. In the next episode, we will delve into MCMC, but for now, our focus is on understanding how to execute it using Stan.

To get started, follow the instructions provided at https://mc-stan.org/users/interfaces/ to install Stan on your local computer.


::::::::::::::::::::::::::::::::::: callout

With Stan, you can fit model that have continuous parameters. Models with discrete parameters such as most classification models are typically impossible to fit, although some workarounds have been implemented. 

:::::::::::::::::::::::::::::::::::::::::::




## Basic program structure

A Stan program is organized into several blocks that collectively define the model. Typically, a Stan program includes at least the following blocks:

1. Data: This block is used to declare the input data provided to the model. It specifies the types and dimensions of the data variables incorporated into the model.

2. Parameters: In this block, the model parameters are declared. 

3. Model: The likelihood and prior distributions are included here through sampling statements. 

For best practices, it is recommended to specify Stan programs in separate text files with a .stan extension, which can then be called from R or other supported interfaces.


### Example 1: Beta-binomial model
  
The following Stan program specifies the Beta-binomial model, and consists of data, parameters, and model blocks. 

The data variables are the total sample size $N$ and the outcome of a binary variable (coin flip, handedness etc.). The declared data type is `int` for integer, and the variables have a lower bound 1 and 0 for $N$ and $x$, respectively.  Notice that each line ends with a semicolon.

In the parameters block we declare $\theta$, the probability for a success. Since this parameter is a probability, it is a real number restricted between 0 and 1.

In the model block, the likelihood is specified with the sampling statement `x ~ binomial(N, theta)`. This line includes the binomial distribution $Bin(x | N, theta)$ in the target distribution. The prior is set similarly, and omitting the prior implies a uniform prior. 
  


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
    
    // Prior is uniform
  }
```

Once the Stan program has been saved we need to compile it. In R, this is done by running the following line, where `"binomial_model.stan"` is the path of the program. 


```r
binomial_model <- stan_model("binomial_model.stan")
```

Once the program has been compiled, it can be used to generate the posterior samples by calling the function `sampling()`. The data needs to be defined as a list. 


```r
binom_data <- list(N = 50, x = 7)

binom_samples <- sampling(object = binomial_model,
                          data = binom_data)
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
Chain 1:                0.003 seconds (Sampling)
Chain 1:                0.007 seconds (Total)
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
Chain 4: Gradient evaluation took 1e-06 seconds
Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.01 seconds.
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
Chain 4:                0.004 seconds (Sampling)
Chain 4:                0.008 seconds (Total)
Chain 4: 
```


With the default settings, Stan executes 4 MCMC chains, each with 2000 iterations (more about this in the next episode on MCMC). During the run, Stan provides progress information, aiding in estimating the running time, particularly for complex models or extensive datasets. In this case the sampling took only a fraction of a second. 

When running `binom_samples`, a summary for the model parameter $p$ is printed, facilitating a quick review of the results.


```r
binom_samples
```

```{.output}
Inference for Stan model: anon_model.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

        mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
theta   0.15    0.00 0.05   0.07   0.12   0.15   0.19   0.27  1367    1
lp__  -22.87    0.02 0.75 -25.05 -23.04 -22.56 -22.38 -22.33  1464    1

Samples were drawn using NUTS(diag_e) at Fri Mar  8 10:26:06 2024.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```


This summary can also be accessed as a matrix with `summary(binom_samples)$summary`.

Often, however, it is necessary to work with the individual samples. These can be extracted as follows:


```r
theta_samples <- extract(binom_samples, "theta")[["theta"]]
```

Now we can use the methods presented in the previous Episode to compute posterior summaries, credible intervals and to generate figures. 


:::::::::::::::::::::::::::::::::::: challenge

Compute the 95% credible intervals for the samples drawn with Stan. What is the probability that $\theta \in (0.05, 0.15)$? Plot a histogram of the posterior samples. 


::::::::::::::::::::: solution


```r
CI95 <- quantile(theta_samples, probs = c(0.025, 0.975))
theta_between_0.05_0.15 <- mean(theta_samples>0.05 & theta_samples<0.15)


p <- ggplot(data = data.frame(theta = theta_samples)) +
  geom_histogram(aes(x = theta), bins = 30) +
  coord_cartesian(xlim = c(0, 1))


print(p)
```

<img src="fig/stan-rendered-unnamed-chunk-7-1.png" style="display: block; margin: auto;" />


::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::::::::::



:::::::::::::::::::::::::::::::::::: challenge

Try modifying the Stan program so that you add a $Beta(\alpha, \beta)$ prior for $\theta$.

Can you modify the Stan program further so that you can set the hyperparameters $\alpha, \beta$ as part of the data? What is the benefit of using this approach?


::::::::::::::::::::: solution

If the data block is modified so that it declares the hyperparameters as data (e.g. `real<lower=0> alpha;`), it enables setting the hyperparameter values as part of data. This makes it possible to change the hyperparameters without modifying the Stan file. 


::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::

## Additional Stan blocks

In addition the data, parameters, and model blocks there are additional blocks that can be included in the program. 

1. Functions: For user-defined functions. This block must be the first in the Stan program. It allows users to define custom functions.

2. Transformed data: This block is used for transformations of the data variables. It is often employed to preprocess or modify the input data before it is used in the main model. Common tasks include standardization, scaling, or other data adjustments.

3. Transformed parameters: In this block, transformations of the parameters are defined. If transformed parameters are used on the left-hand side of sampling statements in the model block, the Jacobian adjustment for the posterior density needs to be included in the model block as well. 

4. Generated quantities: This block is used to define quantities based on both data and model parameters. These quantities are not part of the model but are useful for post-processing. 

Examples of usage will be included in subsequent illustrations. 


::::::::::::::::::::::::::::::::::: callout

There are tools like Bayesplot, rstanarm, and brms built on top of Stan that make it easier to use many common statistical models and tools for analyzing and visualizing results. However, we won't use these much in this course. The idea is that learning Stan from the basics helps you understand Bayesian modeling better. It lets you have more control over your models and helps you learn how to build, fix, and improve them. So, by working directly with Stan and its details, you'll get better at making your own models and doing more customized Bayesian analyses later on. Furthermore, top-down tools can later be adopted with more confidence as you will understand what is happening under the proverbial hood. 
:::::::::::::::::::::::::::::::::::::::::::

## Example 2: normal model

Next, let's implement the normal model in Stan. First generate some data $X$ with unknown mean and standard deviation parameters $\mu$ and $\sigma$


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


The Stan program for the normal model is specified in the next code chunk. It introduces a new data type (vector) and leverages vectorization in the likelihood statement. Toward the end of the program, a generated quantities block is included, generating new data (X_tilde) to estimate what unseen data points might look like. This resulting distribution is referred to as the posterior predictive distribution. The way this works is by generating a random realization from the normal distribution for each posterior sample of $\mu$ and $\sigma$.




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
  // Likelihood is vectorized!
  X ~ normal(mu, sigma);
  
  // Priors
  mu ~ normal(0, 1);
  sigma ~ inv_gamma(1, 1);
}
generated quantities {
  real X_tilde;
  X_tilde = normal_rng(mu, sigma);
}
```


Instead of vectorizing the likelihood, one could write a for loop with sampling statements for each individual data point, such as `X[i] ~ normal(\mu, \sigma)`. However, this approach would result in an inefficient implementation, and vectorization is generally recommended for improved performance. Nevertheless, when dealing with complex models, it may be useful to initially write the model in an unvectorized format to facilitate easier debugging.

Let's again fit the model to the data 


```r
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
Chain 1:  Elapsed Time: 0.007 seconds (Warm-up)
Chain 1:                0.007 seconds (Sampling)
Chain 1:                0.014 seconds (Total)
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
Chain 2:  Elapsed Time: 0.007 seconds (Warm-up)
Chain 2:                0.007 seconds (Sampling)
Chain 2:                0.014 seconds (Total)
Chain 2: 

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 3).
Chain 3: 
Chain 3: Gradient evaluation took 2e-06 seconds
Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.02 seconds.
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


Next, we'll extract posterior samples and generate a plot for the joint, and marginal posteriors. The true unknown parameter values are included in the plots in red. 

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


Let's also plot the posterior predictive distribution: 


```r
PPD <- extract(normal_samples, c("X_tilde"))[[1]] %>% 
  data.frame(X_tilde = . )

p_PPD <- PPD %>% 
  ggplot() + 
  geom_histogram(aes(x = X_tilde), 
                 bins = 40, fill = posterior_color)

print(p_PPD)
```

<img src="fig/stan-rendered-unnamed-chunk-12-1.png" style="display: block; margin: auto;" />



## Example 3: Linear regression

:::::::::::::::::::::::::::::::::::: challenge

Write a Stan program for linear regression with one dependent variable. 

Generate data from the linear model and use the Stan program to estimate the intercept $\alpha$, slope $\beta$ and noise term $\sigma$.

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
  sigma ~ inv_gamma(1, 1);
}

```


::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::: challenge

Modify the program for linear regression so it facilitates $M$ dependent variables. 

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
  sigma ~ inv_gamma(1, 1);
}

```


::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::












::::::::::::::::::::::::::::::::::::: keypoints 

- Stan is a powerful tool for generating posterior distribution samples. 
- A Stan program is specified in a separate text file that consists of code blocks, with the data, parameters, and model blocks being the most crucial ones.

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
