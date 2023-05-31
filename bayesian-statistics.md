---
title: 'Bayesian Statistics'
teaching: 10
exercises: 2
---



:::::::::::::::::::::::::::::::::::::: questions 

- What is Bayesian statistics?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the basic idea of Bayesian statistical thinking.
- Bayesian formula: prior, likelihood, posterior
- Implement grid approximation for a Bayesian model

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

The fundamental ingredient of Bayesian statistics and probabilistic thinking is the Bayes' theorem, stated as

$$
  p(\theta | X) = \frac{p(\theta)  p(X | \theta)}{p(X)} \\
$$

Given a statistical model, the theorem can be used to infer probabilities of the values of the model parameters $\theta$ conditional on the available data $X$. These probabilities are quantified by the *posterior distribution* $p(\theta | X)$. The *prior distribution* $p(\theta)$ is used to impose beliefs about $\theta$ without taking the data into account. The *likelihood function* $p(X | \theta)$ gives the probability of the data conditional $\theta$ and specifies the effect of data on the posterior. The denominator on the right-hand side $p(X)$ is called the marginal probability, which is often practically impossible to compute, and usually the proportional version of the Bayes' formula is used:


$$
p(\theta | X) \propto p(\theta)  p(X | \theta).
$$

The proportional Bayes' formula produces an unnormalized posterior distribution which can then be normalized to access the normalized posterior. 

## Example: handedness

Let us illustrate the use of the Bayes' theorem with an example. 

Assume we are interested in estimating the prevalence of left-handedness based on a sample of 50 children. In this sample 7 children reported left-handedness and 43 were right-handed. Since the outcome is binary and the children independent (assumption) we can model left-handedness with the binomial distribution:

$$
\text{Number of left-handed} \sim Bin(n, p)
$$

The parameters $n$ and $p$ refer to the total number of children and proportion of the left-handed in the population, respectively. The likelihood for the data is $p(X|\theta) = Bin(7 | 50, p).$

Next, we should think what sort of prior information we'd like to use. For instance, the following distributions might be considered: $Unif(0, 1); \, N(0, 1)$ and $Beta(1, 10)$. 

::::::::::::::::::::::::::::::::::::: discussion

What could be the rationale for choosing each of these prior distributions?

::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: instructor

For example: 

- uniform = absolutely no idea about $p$ a-priori
- normal = a parsimonious distribution, often a good choice
- Beta = conjugate prior, hyperparameters can be interpreted as prior data, 1 out of 10+1 people is a leftie. 

Extra-reading: BDA p.34: interpreting hyperparameters as prior data, conjugate prior

::::::::::::::::::::::::::::::::::::::::::::::::


### Grid approximation

In many cases, analytical computations of the posterior are not feasible and approximation methods need to be used. Although the analytical posterior for the binomial model (with Beta prior) is available, we'll next illustrate fitting this model with a grid approximation. 

The grid approximation is a method for approximating the posterior distribution. The idea is to discretize the parameter space into a grid and then calculate the likelihood and prior at each grid point. The product of these values forms the approximate unnormalized posterior. To obtain a proper probability distribution, the unnormalized posterior is then normalized by dividing each value by the sum of all values times the discretization interval (in 1D, area in 2D etc.). This results in an approximation of the posterior distribution. 

In R, the model can be implemented as follows. First we define the data variables: 


```r
# Sample size
N <- 50

# 7/50 are left-handed
x <- 7

# Define a grid of points in the interval [0, 1], with 0.01 interval
delta <- 0.01
p_grid <- seq(from = 0, to = 1, by = delta)
```

Next, we'll define the likelihood, uniform prior and posterior functions. 


```r
likelihood <- dbinom(x = x, size = N, prob = p_grid)
prior <- rep(1, length(p_grid))
posterior <- likelihood*prior

# normalize posterior
posterior <- posterior/(sum(posterior)*delta)

# Make data frame
df <- data.frame(p = p_grid, likelihood, prior, posterior)
```

Finally, we can plot these functions


```r
# wide to long format
df_l <- df %>%
  gather(key = "func", value = "value", -p)

# Plot
p1 <- ggplot(df_l, 
       aes(x = p, y = value, color = func)) + 
  geom_point() +
  geom_line() +
  scale_color_grafify()

p1
```

<img src="fig/bayesian-statistics-rendered-unnamed-chunk-4-1.png" style="display: block; margin: auto;" />


::::::::::::::::::::::::::::::::::::: instructor

Take a moment to analyze the figure. 

::::::::::::::::::::::::::::::::::::::::::::::::


Notice that the likelihood function is not a distribution in terms of the parameter $p$, so it doesn't sum to one. Below, we normalize it for better illustration. 


Now that we have a posterior distribution (approximation) available, we can try to quantify it by computing features of it. The mode (maximum a posteriori, or MAP), average and variance and commonly employed point estimates: 


```r
data.frame(Estimate = c("Mode", "Mean", "Variance"), 
           Value = c(df[which.max(df$posterior), "p"],
                     sum(df$p*df$posterior*delta), 
                     sum(df$p^2*df$posterior*delta) - sum(df$p*df$posterior*delta)^2))
```

```{.output}
  Estimate      Value
1     Mode 0.14000000
2     Mean 0.15384615
3 Variance 0.00245618
```


::::::::::::::::::::::::::::::::::::: discussion

These are point estimates. What information do they give? What is absent? Think of other ways in which you could quantify the posterior. How could you quantify the uncertainty. 

What is the conclusion of the analysis in terms of handedness? 

::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: instructor

Actual value from a study from 1975 with 7,688 children in US grades 1-6 was 9.6%

Hardyck, C. et al. (1976), Left-handedness and cognitive deficit
https://en.wikipedia.org/wiki/Handedness

::::::::::::::::::::::::::::::::::::::::::::::::



### Effect of the prior

Next, let's compare the effect of the prior, using the distributions mentioned above. 


```r
uniform_prior <- dunif(x = p_grid, min = 0, max = 1)
normal_prior <- dnorm(x = p_grid, mean = 0, sd = 0.1)
beta_prior <- dbeta(x = p_grid, shape1 = 2, shape2 = 10)

posterior1 <- likelihood*uniform_prior/(sum(likelihood*uniform_prior)*delta)
posterior2 <- likelihood*normal_prior/(sum(likelihood*normal_prior)*delta)
posterior3 <- likelihood*beta_prior/(sum(likelihood*beta_prior)*delta)

# Normalized likelihood
likelihood <- likelihood/(sum(likelihood)*delta)


df2 <- data.frame(p = rep(p_grid, 3), 
                  likelihood = rep(likelihood, 3),
                  prior = c(uniform_prior,  normal_prior, beta_prior), 
                  posterior = c(posterior1, posterior2, posterior3), 
                  prior_type = rep(c("uniform", "normal", "beta"),
                                   each = length(p_grid)))

df2_w <- df2 %>% gather(key = "func", value = "value", -c(p, prior_type))

p2 <-ggplot(df2_w,
         aes(x = p, y = value, color = func)) + 
    geom_point() + 
    geom_line() +
    facet_wrap(~prior_type,
               scales = "free", 
               ncol = 1) +
  scale_color_grafify()
    
p2    
```

<img src="fig/bayesian-statistics-rendered-unnamed-chunk-6-1.png" style="display: block; margin: auto;" />


::::::::::::::::::::::::::::::::::::::::::::::::::::: challenge

Play around with the parameters of the prior distributions and see how it affects the posterior. 

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

The main limitation of the grid approximation method is that it becomes impractical for models with a large number of parameters. The reason is that the number of computations grows as $O \{ n^p \}$ where $n$ is the number of grid points per model parameter and $p$ the number of parameters. This quickly becomes prohibitive, and the grid approximation is seldom used in practice. The standard approach to posterior computations is to draw samples from it with Markov chain Monte Carlo (MCMC) methods, which we will go through later. In the following episodes, we will learn how to perform computations on samples drawn from a distribution and eventually implement our own MCMC algorithms. 

## Example: normal model 

Let us implement another standard statistical model, the normal model, with the grid approximation. We'll assume that the variance of the model in known, $\sigma^2=1,$ and we'd like to learn the mean parameter $\mu.$

Let us first generate some data from the model, 5 independent observations from a normal distribution with unknown mean parameter:


```r
# Sample size
N <- 5

# Generate data
sigma <- 1
unknown_mu <- runif(1, -5, 5)
x <- rnorm(n = N,
           mean = unknown_mu,
           sd = sigma) 


# Define a grid of points for mu
delta <- 0.01
mu_grid <- seq(from = -5, to = 5, by = delta)
```


Similarly as before, we'll then define the prior, likelihood, and compute posterior distribution. We'll use a normal prior with mean 0 and standard deviation 1 for $\mu$. As the observations are assumed to be independent, the likelihood function is product of the likelihoods of the individual points:

$$p(X | \theta) = \prod_{i = 1}^{N} p(X_i | \theta),$$

where $X_i$ are individual data points and $N$ the sample size. 

$$\log p(X | \theta) = \sum_{i = 1}^{N} \log p(X_i | \theta)$$


::::::::::::::::::::::::::::::::::::::::::: challenge

Implement the grid approximation the normal model with the generated data as described above.

Work with logarithms to avoid underflow. 

:::::::::::::::::::::: solution



```r
df <- data.frame(mu = mu_grid)

# Log likelihood
for(i in 1:nrow(df)) {
  # print(i)
  df[i, "log_likelihood"] <- sum(dnorm(x = x,
                                   mean = df[i, "mu"], sd = sigma,
                                   log = TRUE))
}

df <- df %>% 
  mutate(likelihood = exp(log_likelihood))

# Prior: mu ~ N(0, 1)
df <- df %>% 
  mutate(log_prior = dnorm(x = mu,
                           mean = 0 ,
                           sd = 1,
                           log = TRUE)) %>% 
  mutate(prior = exp(log_prior))

# Posterior
df <- df %>% 
  mutate(log_posterior = log_prior + log_likelihood) %>% 
  mutate(posterior = exp(log_posterior)) %>% 
  mutate(posterior = posterior/(sum(posterior)*delta)) # normalize

# Normalize likelihood (for better illustration)
df <- df %>% 
  mutate(likelihood = likelihood/(sum(likelihood)*delta))
```

::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::


Now, we can plot the prior, likelihood, and posterior, along with the unknown true value of $\mu$ (black vertical line).    



```r
# Wide --> long format
df_l <- df %>% 
  gather(key = "func", value = "value", -c("mu"))


# In log scale
p_log <- ggplot(df_l %>% 
         filter(grepl("log", func)), 
       aes(x = mu, y = value, color = func)) +
  geom_line() +
  geom_vline(xintercept = unknown_mu)

# In regular scale
p_reg <- ggplot(df_l %>% 
         filter(!grepl("log", func)), 
       aes(x = mu, y = value, color = func)) +
  geom_line() +
  geom_vline(xintercept = unknown_mu, 
             color = "blue") +
  scale_color_grafify()

p_reg
```

<img src="fig/bayesian-statistics-rendered-unnamed-chunk-9-1.png" style="display: block; margin: auto;" />


::::::::::::::::::::::::::::::::::::: discussion

Play around with different samples sizes $N$ and priors for $\mu$ and see how the results change. 

What happens if the true value of $\mu$ is not within the defined grid?

:::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: challenge

Implement the grid approximation with both unkown mean $\mu$ and standard deviation $\sigma:$

1. Generate data
2. Define grid for $mu$ and $sigma$
3. Compute prior, likelihood and posterior at grid points
  - Use normal and gamma priors for $\mu$ and $\sigma$, respectively
    - The prior is the product of priors of $\mu$ and $\sigma$
4. Plot prior, likelihood and posterior (in 2D)
5. Compute marginal prior, likelihood and posterior for $\mu$
  - Marginalize (integrate) over sigma
  - Normalize
  - Plot

:::::::::::::::::::::::: solution
Included in the instructor's notes. 
::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: instructor


```r
## Data *************************** ####

# Sample size
N <- 15


# Generate data
x <- rnorm(n = N, mean = 2.5, sd = 0.5) # mu = 2.5, sigma = 0.5


# Define a grid of points for mu
delta <- 0.01
mu_grid <- seq(from = -5, to = 5, by = delta)
sigma_grid <- seq(from = 0.01, to = 2, by = delta)



## Fit model ********************** ####

df <- expand.grid(mu = mu_grid, sigma = sigma_grid)

# Log likelihood
for(i in 1:nrow(df)) {
  # print(i)
  df[i, "log_likelihood"] <- sum(dnorm(x = x,
                                   mean = df[i, "mu"], sd = df[i, "sigma"],
                                   log = TRUE))
}

df <- df %>% 
  mutate(likelihood = exp(log_likelihood))

# Priors: mu ~ N(0, 5), sigma ~ Gamma(2, 1)
df <- df %>% 
  mutate(log_prior = dnorm(x = mu,
                           mean = 0 ,
                           sd = 1,
                           log = TRUE) + dgamma(x = sigma, 
                                                shape = 2,
                                                scale = 1,
                                                log = TRUE)) %>% 
  mutate(prior = exp(log_prior))

# Posterior
df <- df %>% 
  mutate(log_posterior = log_prior + log_likelihood) %>% 
  mutate(posterior = exp(log_posterior)) %>% 
  mutate(posterior = posterior/(sum(posterior)*delta^2)) # normalize




# Plot
p_posterior <- ggplot(df, 
             aes(x = mu, y = sigma, fill = posterior)) + 
  geom_tile() +
  scale_fill_gradientn(colours = rainbow(5))

p_likelihood <- ggplot(df, 
                      aes(x = mu, y = sigma, fill = likelihood)) + 
  geom_tile() +
  scale_fill_gradientn(colours = rainbow(5))

p_prior <- ggplot(df, 
                       aes(x = mu, y = sigma, fill = prior)) + 
  geom_tile() +
  scale_fill_gradientn(colours = rainbow(5))


p <- plot_grid(p_prior, p_likelihood, p_posterior)


## Marginalize ******************** ####

# long to wide format
df_w <- df %>%
  select(-c(log_likelihood, log_prior, log_posterior)) %>% 
  gather(key = "func", value = "value", -c(mu, sigma))


df_w_marginal <- df_w %>% 
  group_by(mu, func) %>% 
  summarise(marginal_distribution = sum(value)) %>% # integrate over sigma
  ungroup %>% 
  group_by(func) %>% # normalize marginal distributions
  mutate(marginal_distribution = marginal_distribution/(sum(marginal_distribution)*delta))

p_marginal <- df_w_marginal %>% 
  ggplot(data = ., aes(x = mu, y = marginal_distribution, color = func)) + 
  geom_line()
```
:::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: keypoints 

- Likelihood determines the probability of data conditional on the model parameters
- Prior distribution encodes prior beliefs about the model parameters
- Posterior distribution quantifies the probability of parameter values conditional on the data.
- Posterior is a compromise between the data and prior. The less data available, the bigger the effect of the prior. 
- The grid approximation is a means for accessing posterior distributions which may be difficult to compute analytically


::::::::::::::::::::::::::::::::::::::::::::::::

## Reading 

- Gelman *et al.*, Bayesian Data Analysis (3rd ed.): Ch. 1, 2
- McElreath, Statistical Rethinking (2nd ed.): Ch. 2



::::::::::::::::::::::::::::::::::::: instructor

Here are some exercises that can be used as additional challenges. 

1. 

:::::::::::::::::::::::::::::::::::::::::::::::



