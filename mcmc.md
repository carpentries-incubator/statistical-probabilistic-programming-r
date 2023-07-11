---
title: 'MCMC'
teaching: 10
exercises: 2
---





:::::::::::::::::::::::::::::::::::::: questions 

- What is MCMC?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Idea behind MCMC

- Learn how to
  - assess convergence
  - implement MCMC


::::::::::::::::::::::::::::::::::::::::::::::::


Computing the posterior analytically poses an insurmountable challenge in the general case. Even if we were aware of the analytical form, marginalizing it to recover posteriors for the individual model parameters would still be difficult. In Episode 3, we saw that drawing conclusions about the inference could be achieved relatively easily by working on samples from the posterior distribution. However, obtaining such samples from the posterior distribution is a non-trivial task, unless the analytical posterior is known. In this episode, we will delve into Markov chain Monte Carlo methods (MCMC) that are the most extensively employed solution for generating these samples.

## Metropolis-Hastings algorithm

MCMC methods draw samples from the posterior distribution by constructing sequences (chains) of values in the parameter space that ultimately converge to the posterior. While there are other variants of MCMC, on this course we will mainly focus on the Metropolis-Hasting algorithm outlined below 

A chain starts at some initial value $\theta^{0}$, which can be random or based on some more informed criterion. The only precondition is that $p(\theta^{0} | X) > 0$. Then a transition distribution $T_i$ is used to generate a proposal for the subsequent value. An often-used solution is the normal distribution centered at the current value, $\theta^* \sim N(\theta^{i}, \sigma^2)$. This is where the term "Markov chain" comes from, each element is generated based on only the previous one. 

Next, the generated proposal $\theta^*$ is either accepted or rejected. If each proposal was accepted, the sequence would simply be a random walk in the parameter space and would not approximate the posterior to any degree. The rule that determines the acceptance should reflect this; proposals towards higher posterior densities should be favored over proposals toward low density areas. The solution is to compute the ratio

$$r = \frac{p(\theta^* | X) / T_i(\theta^* | \theta^{i})}{p(\theta^i | X) / T_i(\theta^{i} | \theta^{*})}.$$
In situations where the transition density is symmetric, such as with the normal distribution, $r$ reduces simply to the ratio of the posterior values.

The next element in the chain is then chosen: with probability $r$ the chain moves to the proposal, $\theta^{i+1} = \theta^*$ and with probability $1-r$ stays at the current value, $\theta^{i+1} = \theta^{i}.$ 

As the algorithm is ran long enough, convergence is guaranteed and eventually the samples will start approximating the posterior distribution. 







## Example

Let's look at the normal model and implement the Metropolis-Hastings algorithm to sample the posterior. First we'll simulate some data 


```r
N <- 100
mu_true <- -1.25
sigma_true <- 0.6
X <- rnorm(N, mu_true, sigma_true)
p <- data.frame(X) %>%
  ggplot() +
  geom_histogram(aes(x = X),
                 bins = 20)

print(p)
```

<img src="fig/mcmc-rendered-unnamed-chunk-1-1.png" style="display: block; margin: auto;" />

Then, we'll write some functions. Below, `pars` is of the form `c(mu, sigma)`.

First the point-wise log likelihood. Remember that the likelihood is the product of likelihoods for individual data points, which after a log transformation turns into a sum. 


```r
log_likelihood <- function(X, pars) {
  
  log_lh <- dnorm(X,
                  mean = pars[1],
                  sd = pars[2],
                  log = TRUE) %>%
    sum
  
  return(log_lh)
  
}
```


Next, well define normal and Gamma priors as the priors. The log posterior is the sum of log likelihood and log priors. 


```r
# Priors: mu ~ normal, sigma ~ gamma  
log_prior <- function(pars) {
  log_mu_prior <- dnorm(x = pars[1],
                        mean = 0, sd = 1,
                        log = TRUE)
  
  log_sigma_prior <- dgamma(x = pars[2],
                            shape = 2, rate = 1,
                            log = TRUE)
  
  return(log_mu_prior + log_sigma_prior)
  
}


log_posterior <- function(X, pars) {
  
  log_likelihood(X, pars) + log_prior(pars)
  
}
```


The next function implements the transition density. We'll use the normal distribution for both parameters. However, as $\sigma$ cannot be negative, we'll take the absolute value to ensure positivity. 


```r
generate_proposal <- function(pars_old, sd) {
  
  # proposal 
  pars_new <- rnorm(2, mean = pars_old, sd = sd)
  
  # make sure sigma proposal > 0
  pars_new[2] <- abs(pars_new[2])
  
  return(pars_new)
}
```


The next function computes the acceptance ratio. Since the normal distribution is a symmetric proposal, it suffices to compute the ratio of the posteriors. 


```r
# Acceptance probability
get_ratio <- function(X, pars_old, pars_new) {
  
  # Ratio of posteriors
  # No need to include ratio of proposal densities
  # because Gaussian density is symmetric
  r <- exp(log_posterior(X, pars_new) - log_posterior(X, pars_old))
  
  return(r)
}
```


Finally, here is the Metropolis-Hastings sampler for the normal model. 


```r
metropolis_sampler <- function(X, inits, n_samples = 1000, jump_sd) {
  
  pars <- matrix(nrow = n_samples, ncol = length(inits))
  pars[1, ] <- inits
  
  for(i in 2:n_samples) {
    
    # Current parameters
    pars_old <- pars[i-1, ]
    
    # Proposal
    pars_new <- generate_proposal(pars_old, jump_sd)
    
    # Ratio
    r <- get_ratio(X, pars_old, pars_new)
    
    # Does the sampler move?
    move <- runif(n = 1, min = 0, max = 1) <= r
    # OR: 
    # move <- sample(x = c(TRUE, FALSE), size = 1, prob = c(r, 1-r))
    
    if(move) {
      pars[i, ] <- pars_new
    } else {
      pars[i, ] <- pars_old
    }
    
  }
  
  pars <- data.frame(mu = pars[, 1], sigma = pars[, 2])
  
  return(pars) 
  
}
```



Then, we run the sampler. We'll use 4 chains with 5000 samples each. The transition density standard deviation is set to 0.05. 


```r
n_samples <- 5000
n_chains <- 4
jump_sd <- 0.05

# Warmup 50%
warmup <- 0.5


# Random initials for each chain
inits <- apply(X = matrix(c(1:n_chains)),
               MARGIN = 1,
               FUN = function(x) rnorm(2, 0, 1)) %>% 
  t 

# Make sure sigma initial >0
inits[, 2] <- abs(inits[, 2])

# Run the chains
samples <- lapply(1:nrow(inits), function(i) {
  
  my_df <- metropolis_sampler(X = X,
                              inits = c(inits[i, 1], inits[i, 2]),
                              jump_sd = jump_sd,
                              n_samples = n_samples) %>% 
      data.frame()
  
  
  # Add column for chain and 50% warmup
  my_df <- my_df %>% 
    mutate(chain = as.factor(i), 
           warmup = c(rep(TRUE, nrow(my_df)*warmup),
                      rep(FALSE, nrow(my_df)*(1-warmup))))
  
  }) %>%
  do.call(rbind, .)
```


Plot results. Uncomment the line `fill = warmup` below so see the effect of removing the initial 50% of the samples. 


```r
samples_w <- samples %>%
  gather(key = "par", value = "value", -c(chain, warmup))

# Posterior histogram

p_posterior <- ggplot() + 
  geom_histogram(data = samples_w,
                 aes(x = value,
                     # fill = warmup
                     ),
                 bins = 50, alpha = 0.75, position = "identity") +
  geom_vline(data = data.frame(par = c("mu", "sigma"),
                               value = c(mu_true, sigma_true)), 
             aes(xintercept = value)) + 
  facet_wrap(~par, scales = "free")

# p_posterior_chains <- ggplot(samples_l) +
#   geom_histogram(aes(x = value, fill = chain),
#                  bins = 50, alpha = 0.75, 
#                  position = "identity") +
#   geom_vline(data = data.frame(par = c("mu", "sigma"),
#                                value = c(mu_true, sigma_true)), 
#              aes(xintercept = value)) + 
#   scale_fill_grafify()


print(p_posterior)
```

<img src="fig/mcmc-rendered-unnamed-chunk-8-1.png" style="display: block; margin: auto;" />



## Assessing convergence

Although converge is guaranteed in theory, it is not so in practice. 

Like we saw above, unless the chains' initial values are in high posterior density areas, the early chain samples will bias the target distribution estimate. For this reason, it is customary to discard a proportion of the chain as "warmup." Often 50% is used and this is also the default in Stan. 


- Sample autocorrelation, effective sample size
  - ideally, the samples would be independent
- Mixing
- $\hat{R}$



Let's plot the generated trajectories and the true parameter value. Here, the initial 50% of the chains are colored dim to illustrate the fact that convergence to the posterior distribution requires long enough chains.


```r
p_traj <- ggplot() +
  geom_path(data = samples,
            aes(x = mu, y = sigma, color = chain), alpha = 0.25) +
  # Post warmup samples
  geom_path(data = samples %>% filter(warmup == FALSE),
            aes(x = mu, y = sigma, color = chain)) +
  geom_point(aes(x = mu_true, y = sigma_true),
             size = 4) + 
  scale_color_grafify() +
  labs(title = "Red point = true value")

print(p_traj)
```

<img src="fig/mcmc-rendered-unnamed-chunk-9-1.png" style="display: block; margin: auto;" />




Next, we'll make the trace plots, which are comprised of the chains of individual parameters. The warmup iterations are again colored dimmer. Trace plots can give quick visual information about the chains. When there are no converge issues, the trace plots should look like "hairy caterpillars," with sample of each chain located around the same value. In this case everything look ok. A standard convergence metric is the $\hat{R}$ which compares the variances between chains and to the variance within each chain. We'll omit the definition here but, generally, value $\hat{R} \geq 1.1$  are seen as a sign of convergence issues. However, with modern samplers such as Stan, thresholds closer to 1 are recommended (https://mc-stan.org/rstan/reference/Rhat.html).

Clearly convergence is reached fairly quickly after initialization, in some dozen iterations. Moreover, the chain autocorrelation low. This is desirable because it implies that the drawn samples are independent. Another popular convergence metric is the effective sample size. It quantifies the number of independent samples produced by the sampler.   


```r
# Long --> wide format
samples_w <- samples %>% 
  mutate(sample = rep(1:n_samples, 4)) %>%
  gather(key = "parameter", value = "value", -c(chain, warmup, sample))

# Trace plot
p_trace <- ggplot() + 
  geom_line(data = samples_w,
            aes(x = sample, y = value, color = chain), alpha = 0.25) + 
  geom_line(data = samples_w %>% filter(warmup == FALSE),
            aes(x = sample, y = value, color = chain)) + 
  facet_wrap(chain ~ parameter,
             scales = "free",
             ncol = 2) + 
  scale_color_grafify()

print(p_trace)
```

<img src="fig/mcmc-rendered-unnamed-chunk-10-1.png" style="display: block; margin: auto;" />



Let's compute the number of rejected proposals to get some idea about the efficiency of the algorithm. The less samples are rejected, the more efficient the sampler performs.  

```r
# Proportion of post-warmup samples where the proposal was rejected
sum(table(samples %>%
            filter(warmup == FALSE) %>%
            pull(mu))-1)/(n_chains*n_samples*(1-warmup))
```

```{.output}
[1] 0.4556
```



::::::::::::::::::::::::::::::::::::: challenge

Try different proposal distributions (e.g. 0.005, 0.5) standard deviations in the MCMC example above. How does this affect the inference and convergence? Why?

:::::::::::::::::::::::::::::::::::::::::::::::



## Hamiltonian Monte Carlo

Hamiltonian Monte Carlo (HMC) is a variant of the Metropolis-Hastings algorithm implemented in Stan. The defining feature is the elaborate scheme it uses to generate proposals. Briefly, the idea is to simulate the dynamics of a particle moving in a potential landscape defined by the posterior. At each iteration, the particle is given a random momentum vector and then its dynamics are simulated forward for some time. The end of the trajectory is then taken as the proposal value. 

Compared to the random walk Metropolis-Hastings we implemented in this episode, HMC is very efficient. The main advantages of HMC is its ability to explore high-dimensional spaces more effectively, making it especially useful in complex models with many parameters

A type of convergence criterion exclusive to HMC are divergent transitions. In region of the parameter space where the posterior has high curvature, the simulated particle dynamics can produce spurious transitions which do not represent the posterior accurately. Such transitions are called divergent and signal that the particular area of parameter space is not explored accurately. Stan provides information about divergent transitions. 


::::::::::::::::::::::::::::::::::::: keypoints 

- MCMC is ....

- Convergence should be monitored
  - mixing: $\hat{R}$
  - divergent transitions

::::::::::::::::::::::::::::::::::::::::::::::::



## Reading

- See interactive visualization of different MCMC algorithms: https://chi-feng.github.io/mcmc-demo/app.html

- Statistical Rethinking
- BDA3
- Bayes Rules!
