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

- Model comparison with 
  - AIC, BIC, WAIC
  
- Bayesian cross-validation

::::::::::::::::::::::::::::::::::::::::::::::::

This episode focuses on model checking, a crucial step in Bayesian data analysis when dealing with competing models that require systematic comparison. We'll explore three different approaches for this purpose.

Firstly, we'll delve into posterior predictive checks, a method that involves comparing a fitted model's predictions with observed data.

Next, we'll examine information criteria, a tool that helps strike a balance between model complexity and goodness-of-fit.

Finally, we'll wrap up the episode with an exploration of Bayesian cross-validation.

Throughout the episode, we'll use the same simulated dataset for examples. 

## Data

For data, we're using $N=88$ univariate numerical data 



```{.output}
 [1]  -2.270   1.941   0.502  -0.378  -0.226  -0.786  -0.209  -0.637   0.814
[10]   0.566  -1.901  -2.047  -0.689  -3.509   0.133  -4.353   1.067   0.722
[19]   0.861   0.523   0.681   2.982   0.429  -0.539  -0.512  -1.090  -8.044
[28]  -0.387  -0.007 -11.126   1.036   1.734  -0.203   1.036   0.582  -2.922
[37]  -0.543  -6.120  -0.649   4.547  -0.867   1.942   7.148  -0.044  -0.681
[46]  -3.461  -0.142   0.678   0.644  -0.039   0.354   1.783   0.369   0.175
[55]   0.980  -0.097  -4.408   0.442   0.158   0.255   0.084   0.775   2.786
[64]   0.008  -0.664  43.481   1.943   0.334  -0.118   3.901   1.736  -0.665
[73]   2.695   0.002  -1.904  -2.194  -4.015   0.329   1.140  -3.816 -14.788
[82]   0.047   6.205   1.119  -0.003   3.618   1.666 -10.845
```

<img src="fig/model-critisism-rendered-unnamed-chunk-1-1.png" style="display: block; margin: auto;" />


## Posterior predictive check

Gelman: "If the model fits, then replicated data generated under the model should look similar to observed data."

The idea of posterior predictive checking is to use the posterior predictive distribution to simulate a replicate data set and compare it to the observed data. Discrepancies between the simulated and observed data can imply shortcomings in the model. Comparison between simulated and actual data can be done in different ways. Visual check is one option but a more approach is to compute the posterior predictive p-value (ppp). 

Difference to posterior predictive... 

1. Fit a Bayesian Model:
 Develop and fit a Bayesian model to observed data.
2. Generate Posterior Predictive Samples:
 Use the posterior distribution of model parameters to simulate new datasets.
3. Compare Simulated Data:
 Assess how well the simulated datasets align with the observed data using statistical measures or graphical tools.
4. Assess Model Fit:
 Evaluate the model's appropriateness based on the comparison results.
5. Iterate or Modify:
  Iterate or modify the model if needed to enhance its performance.
  
Posterior predictive checking ensures the model's ability to generate data resembling observed patterns, aiding in model refinement and validation.




### Normal model

Here we fit the normal model and generate posterior predictions $X_{rep}$. Notice, that this is slightly different from the posterior predictive distribution $\tilde{X}$. The former consists of replications of the original data, in this case 88 observations, the latter doesn't specify the sample size. 


```stan
data {
  int<lower=0> N;
  vector[N] X;
}
parameters {
  real<lower=0> sigma;
  real mu;
}
model {
  // Likelihood
  X ~ normal(mu, sigma);
  
  // Prior
  mu ~ normal(0, 1);
  sigma ~ gamma(2, 1);
}

generated quantities {
  vector[N] X_rep;

  // Posterior predictive density
  for(i in 1:N) {
    X_rep[i] = normal_rng(mu, sigma);
  }
}

```




```r
normal_fit <- sampling(normal_model,
                       list(N = N, X = df$X), 
                       refresh = 0)

# Extract samples
X_rep <- rstan::extract(normal_fit, "X_rep")[[1]] %>% 
  data.frame() %>%
  mutate(sample = 1:nrow(.))
```



Below is a comparison of 12 samples of $\tilde{X}$ against the data (the panel titles correspond to MCMC sample numbers). The large discrepancy between the data and posterior predictions indicates that something is wrong with our model. It seems like the normal model misestimates the tails of the data, and that, likely, the normal model is a poor choice for such data in any case.


```r
# Subset
X_rep_sub <- X_rep %>% filter(sample %in%
                                    sample(X_rep$sample,
                                       12,
                                       replace = FALSE))

# Wide --> long
X_rep_sub_l <- X_rep_sub %>% gather(key = "key", value = "value", -sample)

p_norm_hist <- ggplot() +
  geom_histogram(data = X_rep_sub_l,
                 aes(x = value
                     # y = after_stat(density)
                     ),
               bins = 50,
               fill = posterior_color, alpha = 0.8) +
  facet_wrap(~sample, scales = "free") +
  geom_histogram(data = df,
                 aes(x = X
                     # y = after_stat(density)
                     ),
                 bins = 50,
                 alpha = 0.8)


print(p_norm_hist)
```

<img src="fig/model-critisism-rendered-unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

It's visually apparent that there is a discrepancy between the posterior replications and data. 

Let's quantify this using the maximum of the data as a test statistic. The maximum of the original data is max($X$) = 43.481. The following histogram shows this value (vertical line) against the maximum compute for each replicate data set $\tilde{X}$. 



```r
## Compute X_rep max
rep_maxs <- X_rep %>%
  select(-sample) %>%
  apply(MARGIN = 1, FUN = max) %>%
  data.frame(max = ., sample = 1:length(.))

ggplot() +
  geom_histogram(data = rep_maxs,
                 aes(x = max),
                 bins = 50, fill = posterior_color) +
  geom_vline(xintercept = max(df$X)) +
  labs(title = "Max value of the replicate data sets")
```

<img src="fig/model-critisism-rendered-unnamed-chunk-5-1.png" style="display: block; margin: auto;" />

The proportion of replications $X_{rep}$ that produce at least as extreme values as the data is called the posterior predictive p-value ($ppp$). The $ppp$ quantifies the evidence for the suitability of the model for the data with higher $ppp$ implying a lesser conflict. In this case, the value is $ppp =$ 1 which means that the maximum was as at least as large as in the data in0% replications.


### Cauchy model

Let's do a similar analysis utilizing the Cauchy model, and compute the $ppp$ for this model. 

The code is essentially copy-pasted from above, with the distinction of the Stan program.


```stan
data {
  int<lower=0> N;
  vector[N] X;
}

parameters {
  // Scale
  real<lower=0> sigma;
  // location
  real mu;
}

model {
  
  // Likelihood (vectorized)
  // location = mu and scale = sigma
  X ~ cauchy(mu, sigma);
  
  // Prior
  mu ~ normal(0, 1);
  sigma ~ gamma(2, 1);
  
}

generated quantities {
  vector[N] X_rep;
  for(i in 1:N) {
    X_rep[i] = cauchy_rng(mu, sigma);
  }
}

```




```r
cauchy_fit <- sampling(cauchy_model, list(N = N, X = df$X), 
                       refresh = 0)

X_rep <- rstan::extract(cauchy_fit, "X_rep")[[1]] %>% data.frame() %>%
  mutate(sample = 1:nrow(.))

# Subset
X_rep_sub <- X_rep %>% filter(sample %in%
                                sample(X_rep$sample,
                                       12,
                                       replace = FALSE))

# Wide --> long
X_rep_sub_l <- X_rep_sub %>% gather(key = "key", value = "value", -sample)


p_cauchy_hist <- ggplot() +
  geom_histogram(data = X_rep_sub_l,
                 aes(x = value
                     # y = after_stat(density)
                 ),
                 bins = 50, fill = posterior_color) +
  facet_wrap(~sample, scales = "free") +
  geom_histogram(data = df,
                 aes(x = X
                     # y = after_stat(density)
                 ),
                 bins = 50)

print(p_cauchy_hist)
```

<img src="fig/model-critisism-rendered-unnamed-chunk-7-1.png" style="display: block; margin: auto;" />
The figure below contains again the distribution of maximum value for each replicate sets. Here the visually apparent differences present in the previous section are less apparent. 

The $ppp$ is large, indicating no issues with the suitability of the model on the data. 


```r
## Compute ppp
rep_maxs <- X_rep %>%
  select(-sample) %>%
  apply(MARGIN = 1, FUN = max) %>%
  data.frame(max = ., sample = 1:length(.))

ggplot() +
  geom_histogram(data = rep_maxs,
                 aes(x = max),
                 bins = 10000, fill = posterior_color) +
  geom_vline(xintercept = max(df$X)) +
  labs(title = paste0("ppp = ", mean(rep_maxs$max > max(df$X)))) +
  # set plot limits to aid with visualizations
  coord_cartesian(xlim = c(0, 1000)) 
```

<img src="fig/model-critisism-rendered-unnamed-chunk-8-1.png" style="display: block; margin: auto;" />


## Information criteria

Let's then compare the normal and Cauchy models with the WAIC. First we'll need to fit both models on the data. 


```r
stan_data <- list(N = N, X = df$X)

# Fit
normal_fit <- sampling(normal_model, stan_data,
                       refresh = 0)
cauchy_fit <- sampling(cauchy_model, stan_data, 
                       refresh = 0)


# Extract samples
normal_samples <- rstan::extract(normal_fit, c("mu", "sigma")) %>% data.frame
cauchy_samples <- rstan::extract(cauchy_fit, c("mu", "sigma")) %>% data.frame
```


Then we can write a function that compute the WAIC. 


```r
WAIC <- function(samples, data, model){
  
  
  # Loop over data points
  pp_dens <- lapply(1:length(data), function(i) {
    
    my_x <- data[i]
    
    # Loop over posterior samples  
    point_pp_dens <- lapply(1:nrow(samples), function(S) {
      
      my_mu <- samples[S, "mu"]
      my_sigma <- samples[S, "sigma"]
      
      if(model == "normal") {
        # Model: y ~ normal(mu, sigma)
        dnorm(x = my_x,
              mean = my_mu,
              sd = my_sigma)
      } else if(model == "cauchy") {
        # Model: y ~ cauchy(mu, sigma)
        dcauchy(x = my_x,
                location = my_mu,
                scale = my_sigma)
      }
      
      
      
    }) %>%
      unlist()
    
    return(point_pp_dens)
    
  }) %>%
    do.call(rbind, .)
  
  
  # See BDA3 p.169
  lppd <- apply(X = pp_dens,
                MARGIN = 1, 
                FUN = function(x) log(mean(x))) %>% 
    sum
  
  # See BDA3 p.173
  bias <- apply(X = pp_dens,
                MARGIN = 1, 
                FUN = function(x) var(log(x))) %>% 
    sum
  
  # WAIC
  waic = -2*(lppd - bias)
  
  return(waic)
}
```

Applying this function to the posterior samples, we'll recover a lower value for the Cauchy model, implying a better fit on the data. 


```r
WAIC(normal_samples, df$X, model = "normal")
```

```{.output}
[1] 582.2829
```

```r
WAIC(cauchy_samples, df$X, model = "cauchy")
```

```{.output}
[1] 413.9462
```


## Bayesian cross-validation

Idea in Bayesian cross-validation is.... 


Helper function that computes.. 


```r
# Get log predictive density for a point x,
# given data X and posterior samples
# See BDA3 p.175
get_lpd <- function(x, X, samples, model) {
  
  # Loop over posterior samples  
  pp_dens <- lapply(1:nrow(samples), function(S) {
    
    if(model == "normal") {
      # Normal(x | mu, sigma^2)
      dnorm(x = x,
            mean = samples[S, "mu"],
            sd = samples[S, "sigma"])
    } else if (model == "cauchy") {
      # Cauchy(x | location = mu, scale = sigma^2)
      dcauchy(x = x,
              location = samples[S, "mu"],
              scale = samples[S, "sigma"])
    }
    
    
  }) %>%
    unlist()
  
  lpd <- log(mean(pp_dens))
  
  return(lpd)
}
```


Now we can perform CV


```r
# Loop over data partitions
normal_loo_lpds <- lapply(1:N, function(i) {
  
  # Training set
  my_X <- X[-i]
  
  # Test set
  my_x <- X[i]
  
  # Fit model
  my_normal_fit <- sampling(normal_model,
                            list(N = length(my_X),
                                 X = my_X),
                            refresh = 0 # omits output
                            ) 
  
  # Get data
  my_samples <- rstan::extract(my_normal_fit, c("mu", "sigma")) %>% 
    do.call(cbind, .) %>% 
    set_colnames(c("mu", "sigma"))
  
  # Get lpd
  my_lpd <- get_lpd(my_x, my_X, my_samples, "normal")
  
  data.frame(i, lpd = my_lpd, model = "normal_loo")
  
}) %>%
  do.call(rbind, .)

# Predictive density for data points using full data in training
normal_full_lpd <- lapply(1, function(dummy) {
  
  # Fit model
  my_normal_fit <- sampling(normal_model,
                            list(N = length(X),
                                 X = X), 
                            refresh = 0)
  
  # Get data
  my_samples <- rstan::extract(my_normal_fit, c("mu", "sigma")) %>% 
    do.call(cbind, .) %>% 
    set_colnames(c("mu", "sigma"))
  
  # Compute lpds
  lpds <- lapply(1:N, function(i) {
    
    my_lpd <- get_lpd(X[i], X, my_samples, "normal")
    
    data.frame(i, lpd = my_lpd, model = "normal")
  }) %>% do.call(rbind, .)
  
  return(lpds)
}) %>%
  do.call(rbind, .)


# Same for Cauchy:
cauchy_loo_lpds <- lapply(1:N, function(i) {
  print(i)
  # Subset data
  my_X <- X[-i]
  my_x <- X[i]
  
  # Fit model
  my_normal_fit <- sampling(cauchy_model,
                            list(N = length(my_X),
                                 X = my_X), 
                            refresh = 0)
  
  # Get data
  my_samples <- rstan::extract(my_normal_fit, c("mu", "sigma")) %>% 
    do.call(cbind, .) %>% 
    set_colnames(c("mu", "sigma"))
  
  # Get lpd
  my_lpd <- get_lpd(my_x, my_X, my_samples, "cauchy")
  
  data.frame(i, lpd = my_lpd, model = "cauchy_loo")
  
}) %>%
  do.call(rbind, .)
```

```{.output}
[1] 1
[1] 2
[1] 3
[1] 4
[1] 5
[1] 6
[1] 7
[1] 8
[1] 9
[1] 10
[1] 11
[1] 12
[1] 13
[1] 14
[1] 15
[1] 16
[1] 17
[1] 18
[1] 19
[1] 20
[1] 21
[1] 22
[1] 23
[1] 24
[1] 25
[1] 26
[1] 27
[1] 28
[1] 29
[1] 30
[1] 31
[1] 32
[1] 33
[1] 34
[1] 35
[1] 36
[1] 37
[1] 38
[1] 39
[1] 40
[1] 41
[1] 42
[1] 43
[1] 44
[1] 45
[1] 46
[1] 47
[1] 48
[1] 49
[1] 50
[1] 51
[1] 52
[1] 53
[1] 54
[1] 55
[1] 56
[1] 57
[1] 58
[1] 59
[1] 60
[1] 61
[1] 62
[1] 63
[1] 64
[1] 65
[1] 66
[1] 67
[1] 68
[1] 69
[1] 70
[1] 71
[1] 72
[1] 73
[1] 74
[1] 75
[1] 76
[1] 77
[1] 78
[1] 79
[1] 80
[1] 81
[1] 82
[1] 83
[1] 84
[1] 85
[1] 86
[1] 87
[1] 88
```

```r
cauchy_full_lpd <- lapply(1, function(dummy) {
  
  # Fit model
  my_cachy_fit <- sampling(cauchy_model,
                           list(N = length(X),
                                X = X), 
                           refresh = 0)
  
  # Get data
  my_samples <- rstan::extract(my_cachy_fit, c("mu", "sigma")) %>% 
    do.call(cbind, .) %>% 
    set_colnames(c("mu", "sigma"))
  
  # Compute lpds
  lpds <- lapply(1:N, function(i) {
    
    my_lpd <- get_lpd(X[i], X, my_samples, "cauchy")
    
    data.frame(i, lpd = my_lpd, model = "cauchy")
  }) %>% do.call(rbind, .)
  
  return(lpds)
}) %>%
  do.call(rbind, .)
```



```r
# Combine
lpds <- rbind(normal_loo_lpds, 
              normal_full_lpd, 
              cauchy_loo_lpds,
              cauchy_full_lpd)

lpd_summary <- lpds %>% 
  group_by(model) %>% 
  summarize(lppd = sum(lpd))


# Effective number of parameters
p_loo_cv_normal <- lpd_summary[lpd_summary$model == "normal", "lppd"] - lpd_summary[lpd_summary$model == "normal_loo", "lppd"]
p_loo_cv_cauchy <- lpd_summary[lpd_summary$model == "cauchy", "lppd"] - lpd_summary[lpd_summary$model == "cauchy_loo", "lppd"]


paste0("Effective number of parameters, normal = ", p_loo_cv_normal)
```

```{.output}
[1] "Effective number of parameters, normal = 33.7046896782354"
```

```r
paste0("Effective number of parameters, cauchy = ", p_loo_cv_cauchy)
```

```{.output}
[1] "Effective number of parameters, cauchy = 1.903135942788"
```




::::::::::::::::::::::::::::::::::::: keypoints 

- point 1

::::::::::::::::::::::::::::::::::::::::::::::::



## Reading

- Statistical Rethinking: Ch. 7
- BDA3: p.143: 6.3 Posterior predictive checking
