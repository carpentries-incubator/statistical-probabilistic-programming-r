---
title: 'Working with samples'
teaching: 10
exercises: 2
---





:::::::::::::::::::::::::::::::::::::: questions 

- How do you work with posterior samples?
- How can the posterior information be handled?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Learn how to
  - work with posterior samples
  - compute posterior intervals


::::::::::::::::::::::::::::::::::::::::::::::::

In the previous episode, we were introduced to the Bayesian formula and learned how to fit binomial and normal models using the grid approximation. However, the poor scalability of the grid approximation makes it impractical to use on models of even moderate size. To overcome this challenge, the standard solution is to employ Markov chain Monte Carlo (MCMC) methods, which involve drawing random samples from the posterior distribution. While we will delve into MCMC methods in subsequent lessons, we will now learn working with samples.

## Example: binomial model

Let's revisit the binomial model considered in the previous episode. The binomial model with a beta distribution is an example of a model where the analytical shape of the posterior is known. 

$$p(\theta | X) = Beta(\alpha + x, \beta + N - x),$$
where $\alpha$ and $\beta$ are the hyperparameters and $x$ the number of successes out of $N$ trials. 

::::::::::::::::::::::::::::::::::::::::: challenge

Derive the analytical posterior distribution for the Beta-Binomial model. 


::::::::::::::::::::::::::::::: solution


\begin{align}
p(\theta | X) &\propto  p(X | \theta) p(\theta) \\
              &= ... 
\end{align}


::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::


Let's generate samples from the prior and posterior distributions, using the handedness data of the previous episode but first redefine the data. 


```r
# Sample size
N <- 50

# 7/50 are left-handed
x <- 7
```

In R, the standard statistical distributions can be sampled from using readily available functions. For instance, the binomial, normal and beta distributions can be sampled, respectively, with `rbinom, rnorm` and `rbeta` functions. Let's draw 5000 samples from the prior and posterior distributions. 


```r
# Number of samples
n_samples <- 5000

# Prior is Beta(1, 10)
alpha <- 1
beta <- 10

# Draw random values from the prior
prior_samples <- rbeta(n = n_samples,
                       shape1 = alpha,
                       shape2 = beta)

# Draw random values from the posterior
posterior_samples <- rbeta(n = n_samples,
                           shape1 = alpha + x, 
                           shape2 = beta + N - x)

samples_df <- data.frame(prior = prior_samples, 
                         posterior = posterior_samples)
```


Let's plot histograms for these samples along with the analytical densities, the normalized likelihood, and the "true" value (blue) based on a larger population sample. 


```r
# Wide --> long format
samples_df_w <- samples_df %>% gather(key = "Samples")

# Define analytical distributions
delta <- 0.001
analytical_df <- data.frame(p = seq(0, 1, by = delta)) %>% 
  mutate(
    prior = dbeta(x = p , alpha, beta), 
    posterior = dbeta(x = p , alpha + x, beta + N - x), 
    likelihood = dbinom(size = N, x = x, prob = p)
    ) %>% 
  mutate(likelihood = likelihood/(sum(likelihood)*delta)) %>%  # normalize likelihood for better presentation
  gather(key = "Analytical function", value = "value", -p) %>%  # wide --> long
  mutate(`Analytical function` = factor(`Analytical function`, levels = c("posterior", "prior", "likelihood")))


# Frequency in a large population sample (Hardyck, C. et al., 1976)
p_true <- 9.6/100


p <- ggplot(samples_df_w) + 
  geom_histogram(aes(x = value, y = after_stat(density),
                     fill = Samples),
                 bins = 50, 
                 position = "identity", alpha = 0.75) +
  geom_line(data = analytical_df, 
            aes(x = p, y = value, color = `Analytical function`), 
            linewidth = 1) +
  geom_vline(xintercept = p_true,
             color = "blue", 
             linewidth = 1) +
  scale_color_grafify() + 
  scale_fill_grafify() +
  labs(x = "p")

print(p)
```

<img src="fig/sampling-rendered-unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

## Posterior intervals

In Episode 1, we summarized the posterior with the posterior mode (MAP), mean and variance. While these points estimates are informative and often utilized, they and not informative enough for many scenarios. 

More specified information about the posterior can be communicated with *credible intervals* (CI). These intervals refer to areas of the space where a certain amount of posterior mass is located. Usually CIs are computed as quantiles of posterior, so for instance the 90\% CI would be located between the 5\% and 95\% percentiles. 

Let us now compute the percentile-based CIs for the handedness example, along with the posterior mode (MAP), and include them in the figure. The x-axis is zoomed so for clarity. 


```r
# MAP
posterior_density <- density(posterior_samples)
MAP <- posterior_density$x[which.max(posterior_density$y)]

# 90% credible interval
CIs <- quantile(posterior_samples, probs = c(0.05, 0.95))

p <- p +
  geom_vline(xintercept = CIs, linetype = "dashed") + 
  geom_vline(xintercept = MAP) +
  geom_vline(xintercept = p_true, color = "blue", size = 1) +
  coord_cartesian(xlim = c(0, 0.5)) +
  labs(title = "Black = MAP and 90% CIs") 
```

```{.warning}
Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
ℹ Please use `linewidth` instead.
This warning is displayed once every 8 hours.
Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
generated.
```

```r
print(p)
```

<img src="fig/sampling-rendered-unnamed-chunk-5-1.png" style="display: block; margin: auto;" />

The 90% interval in this example is 0.07, 0.21 which means that, according to the analysis, there is a 90% probability that the proportion of left-handed people is between these values. 


Another approach to reporting posterior information is to find the amount of the posterior mass in a given interval (or some more general set). This approach enables determining probabilities for hypotheses. For instance, we might be interested in knowing the probability that the target parameter is less than 0.2, between 0.05 and 0.10, or less than 0.1 or greater than 0.20. Such probabilities can be recovered based on samples simply by computing the proportion of samples in these sets. 


```r
p_less_than_0.15 <- mean(posterior_samples < 0.15)
p_between_0.05_0.1 <- mean(posterior_samples > 0.05 & posterior_samples < 0.1)
p_outside_0.1_0.2 <- mean(posterior_samples < 0.1 | posterior_samples > 0.2)
```

Let's visualize these probabilities as shaded areas of the analytical posterior:

<img src="fig/sampling-rendered-unnamed-chunk-7-1.png" style="display: block; margin: auto;" />




::::::::::::::::::::::::::::::::::::: discussion

How would you compute CIs based on an analytical posterior density?

Can you draw samples from the likelihood?

:::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: challenge

Another approach to posterior summarizing to compute the smallest interval that contains p% of the posterior. Such sets are called highest posterior density intervals (HPDI). More generally, this set can be a more general set than an interval. 

Write a function that returns the highest posterior density interval based on a set of posterior samples. Compute the 95% HPDI for the posterior samples generated (`samples_df`) and compare it to the 95% CIs.


:::::::::::::::::::::: hint

If you sort the posterior samples in order, each set of $n$ consecutive samples contains $100 \cdot n/N \%$ of the posterior, where $N$ is the total number of samples.  

:::::::::::::::::::::::::::


:::::::::::::::::::::::::::: solution

Let's write the function for computing the HPDI


```r
get_HPDI <- function(samples, prob) {
  
  # Total samples
  n_samples <- length(samples)
  
  # How many samples constitute prob of the total number?
  prob_samples <- round(prob*n_samples)
  
  # Sort samples
  samples_sort <- samples %>% sort
  
  
  # Each samples_sort[i:(i + prob_samples - 1)] contains prob of the total distribution mass
  # Find the shortest such interval 
  min_i <- lapply(1:(n_samples - prob_samples), function(i) {
    
    samples_sort[i + prob_samples - 1] - samples_sort[i]
    
  }) %>% unlist %>% which.min()
  
  # Get correspongind values
  hpdi <- samples_sort[c(min_i, min_i + prob_samples)]
  
  return(hpdi)
}
```

Then we can compute the 95% HPDI and compare it to the corresponding CIs


```r
data.frame(HPDI = get_HPDI(samples_df$posterior, 0.95), 
           CI = quantile(posterior_samples, probs = c(0.025, 0.975))) %>% 
  t %>% 
  data.frame() %>% 
  mutate(length = X97.5. - X2.5.)
```

```{.output}
          X2.5.    X97.5.    length
HPDI 0.05376876 0.2159918 0.1622231
CI   0.05974777 0.2240324 0.1642846
```

Both intervals contain the same mass but the HPDI is (slightly) shorter. 

:::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints 

- The posterior distribution can be summarized with point estimates or computing the posterior mass in some set. 

::::::::::::::::::::::::::::::::::::::::::::::::

