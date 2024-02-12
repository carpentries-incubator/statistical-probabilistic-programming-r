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

- Learn how to construct  hierarchical models and fit them with Stan

::::::::::::::::::::::::::::::::::::::::::::::::


Bayesian hierarchical models are a class of models suited for scenarios where the study population consists of separate but related groups. For instance, analyzing student performance in different schools, income levels within various regions, or animal behavior within distinct populations are scenarios where such models might be appropriate.
This structure can be incorporated into a model by including group-wise parameters, each with a prior that, in turn, contains unknown parameters that also receive a prior. In other words, the prior parameters, hyperparameters, are given a prior, a hyperprior, and the hyperparameters are learned in the fitting process. The hyperparameters and hyperpriors can be thought to exist on another level of hierarchy, hence the name.
As an example, consider the beta-binomial model presented in Episode 2. It was used to estimate the prevalence of left-handedness based on a sample of 50 students. If we had additional information, such as study majors, we could include this information in the model, like so:
$$X_g \sim \text{Bin}(N_g, \theta_g) \\
\theta_g \sim \text{Beta}(\alpha, \beta) \\
\alpha, \beta \sim \Gamma(2, 0.1).$$


Here, the subscript $g$ indexes the study major groups. The group-specific prevalences for left-handedness $\theta_g$ are given a beta prior with hyperparameters $\alpha, \beta$ that are random variables. The final line denotes the hyperprior $\Gamma(2, 0.1)$ that controls the prior beliefs about the hyperparameters.

In this hierarchical beta-binomial model, students are modeled as identical within the groups but no longer on the population level. On the other hand, there is an assumed similarity between the groups since they share a common prior. It is said that the groups are partially pooled, as they are not equal but not entirely independent either.

One key advantage of Bayesian hierarchical models is their ability to borrow strength across groups. By pooling information from multiple groups, these models can provide more stable estimates, especially when only limited data is available.


Another difference from non-hierarchical models is that the prior, or the **population distribution** of the parameters, is learned in the process. The population distribution can provide insights into parameter variation in a larger context, for groups where we have no data. For instance, if we had gathered data on the handedness of students majoring in natural sciences, the population distribution could offer insights into students in humanities and social sciences as well.

In the following example, we'll conduct a hierarchical analysis of human heights in different countries.





## Example: human height 

Let's analyze human adult height in different countries. We'll use the normal model with unknown mean $\mu$ and standard deviation $\sigma$ as the generative model, and give these parameters the hierarchical treatment. 

We'll analyze simulated data based on actual measurements in different countries. The simulations are based on data in: Height: Height and body-mass index trajectories of school-aged children and adolescents from 1985 to 2019 in 200 countries and territories: a pooled analysis of 2181 population-based studies with 65 million participants. Lancet 2020, 396:1511-1524

First, let's read the data and check its structure. 


```r
height <- read.csv("data/height_data.csv")

head(height)
```

```{.output}
      Country  Sex Year Age.group Mean.height
1 Afghanistan Boys 1985         5    103.3152
2 Afghanistan Boys 1985         6    109.2357
3 Afghanistan Boys 1985         7    114.7595
4 Afghanistan Boys 1985         8    120.0023
5 Afghanistan Boys 1985         9    125.0773
6 Afghanistan Boys 1985        10    130.0964
  Mean.height.lower.95..uncertainty.interval
1                                   92.91241
2                                   99.91444
3                                  106.31005
4                                  112.20252
5                                  117.88036
6                                  123.38137
  Mean.height.upper.95..uncertainty.interval Mean.height.standard.error
1                                   113.7128                   5.295555
2                                   118.2826                   4.718901
3                                   123.0034                   4.270250
4                                   127.5500                   3.924385
5                                   132.1538                   3.662401
6                                   136.7418                   3.465345
```

Let's subset this data to simplify the analysis and focus on the height of adult women measured in 2019.


```r
height_women <- height %>% 
  filter(
    # 2019 measurements
    Year == 2019,        
    # use only a single age group
    Age.group == 19, 
    # Consider girls only
    Sex == "Girls"
    ) %>% 
  # Select variables of interest.
  select(Country, Sex, Mean.height, Mean.height.standard.error)
```

Let's select 10 countries randomly


```r
# Select countries
N_countries <- 10
Countries <- sample(unique(height_women$Country),
                    size = N_countries,
                    replace = FALSE) %>% sort

height_women10 <- height_women %>% filter(Country %in% Countries)

height_women10
```

```{.output}
       Country   Sex Mean.height Mean.height.standard.error
1    Argentina Girls    161.2204                  0.6207879
2       Bhutan Girls    155.1526                  1.1308066
3     Botswana Girls    162.0620                  1.0115264
4       Gambia Girls    161.7282                  0.9603018
5   Montenegro Girls    169.9609                  0.8887514
6  New Zealand Girls    164.6600                  0.4656146
7   Seychelles Girls    162.4656                  0.8525184
8    Swaziland Girls    158.9214                  0.8919865
9      Tunisia Girls    161.6855                  0.9464602
10     Uruguay Girls    161.5585                  0.7967392
```


### Simulate data

Now, we can treat the values in the table above as ground truth and simulate some data based on them. Let's generate $N=25$ samples for each country from the normal model with $\mu = \text{Mean.height}$ and $\sigma = \text{Mean.height.standard.error}$.


```r
# Sample size per group 
N <- 25

# For each country, generate some random girl's heights
height_sim <- lapply(1:nrow(height_women10), function(i) {
  
  my_df <- height_women10[i, ]
  
  data.frame(Country = my_df$Country, 
             # Random normal values based on measured mu and sd
             Height = rnorm(N, my_df$Mean.height, my_df$Mean.height.standard.error))

}) %>% 
  do.call(rbind, .)


# Plot
height_sim %>% 
  ggplot() +
  geom_point(aes(x = Country, y = Height)) + 
  coord_flip() + 
  labs(title = "Simulated data")
```

<img src="fig/hierarchical-models-rendered-unnamed-chunk-5-1.png" style="display: block; margin: auto;" />

### Modeling

Let's build a normal model that uses partial pooling for the country means and standard deviations. The model can be written as follows. We'll use $g$ to index the country, and $i$ for individuals.

\begin{align}
X_{gi} &\sim \text{N}(\mu_g, \sigma_g) \\
\mu_g &\sim \text{N}(\mu_\mu, \sigma_\mu) \\
\sigma_g &\sim \Gamma(\alpha_\sigma, \beta_\sigma) \\
\mu_\mu &\sim \text{N}(0, 100)\\
\sigma_\mu &\sim \Gamma(2, 0.1) \\
\alpha_\sigma, \beta_\sigma  &\sim \Gamma(2, 0.01)
\end{align}


Here is the Stan program for the hierarchical normal model. The data points are input as a concatenated vector `X` as this would allow using data with uneven sample sizes. The country-specific start and end indices are computed in the transformed data block. The parameters block contains the declarations of vectors for the means and standard deviations, along with the hyperparameters. The hyperparameter subscripts denote the parameter they are assigned to so, for instance, $\sigma_{\mu}$ is the standard deviation of the mean parameter $\mu$. The generated quantities block generates samples from the population distributions of $\mu$ and $\sigma$ and a posterior predictive distribution. 



```stan
data {
  int<lower=0> G; // number of groups
  int<lower=0> N[G]; // sample size within each group
  vector[sum(N)] X; // concatenated observations
}

transformed data {
  // get first and last index for each group in X
  int start_i[G];
  int end_i[G];
  
  for(g in 1:G) {
    
    if(g == 1) {
      start_i[1] = 1;
    } else {
      start_i[g] = start_i[g-1] + N[g-1];
    }
    
    end_i[g] = start_i[g] + N[g]-1;
  }
}

parameters {
  
  // parameters
  vector[G] mu;
  vector<lower=0>[G] sigma;
  
  // hyperparameters
  real mu_mu;
  real<lower=0> sigma_mu;
  real<lower=0> alpha_sigma;
  real<lower=0> beta_sigma;
}

model {
  
  // Likelihood for each group
  for(i in 1:G) {
    X[start_i[i]:end_i[i]] ~ normal(mu[i], sigma[i]);
  }
  
  // Priors
  mu ~ normal(mu_mu, sigma_mu);
  sigma ~ gamma(alpha_sigma, beta_sigma);
  
  // Hyperpriors
  mu_mu ~ normal(0, 100);
  sigma_mu ~ inv_gamma(2, 0.1);
  alpha_sigma ~ gamma(2, 0.01);
  beta_sigma ~ gamma(2, 0.01);
}

generated quantities {
  
  real mu_tilda;
  real<lower=0> sigma_tilda;
  real X_tilda; 
  
  // Population distributions
  mu_tilda = normal_rng(mu_mu, sigma_mu);
  sigma_tilda = gamma_rng(alpha_sigma, beta_sigma);
  
  // Posterior predictive distribution
  X_tilda = normal_rng(mu_tilda, sigma_tilda);
  
} 

```


Now we can call Stan and fit the model. Hierarchical models can often encounter convergence issues and for this reason, we'll use 10000 iterations and set `adapt_delta = 0.99`. Moreover, we'll speed up the inference by running 2 chains in parallel by setting `cores = 2`. 


```r
stan_data <- list(G = length(unique(height_sim$Country)), 
                  N = rep(N, length(Countries)), 
                  X = height_sim$Height)

normal_hier_fit <- rstan::sampling(normal_hier_model,
                              stan_data, 
                              iter = 10000,
                              chains = 2,
                                    # Use to get rid of divergent transitions:
                                    control = list(adapt_delta = 0.99), 
                              cores = 2,
                            # Track progress every 5000 iterations
                            refresh = 5000
                            )
```



<!-- Non-pooled analysis -->



<!-- Fit -->





## Results

### Country-specific estimates

Let's first compare the marginal posteriors for the country-specific estimates: 


```r
par_summary <- rstan::summary(normal_hier_fit, c("mu", "sigma"))$summary %>% 
  data.frame() %>%
  rownames_to_column(var = "par") %>%
  separate(par, into = c("par", "country"), sep = "\\[") %>%
  mutate(country = gsub("\\]", "", country)) %>%
  mutate(country = Countries[as.integer(country)])
```



```r
# Plot

  ggplot() + 
  geom_point(data = par_summary, aes(x = country, y = mean),
             color = posterior_color) +
  geom_errorbar(data = par_summary, aes(x = country, ymin = X2.5., ymax = X97.5.),
                color = posterior_color) + 
  geom_point(data = height_women10 %>% 
               rename_with(~ c('mu', 'sigma'), 3:4) %>% 
               gather(key = "par",
                      value = "value",
                      -c(Country, Sex)), 
             aes(x = Country, y = value)) + 
  facet_wrap(~ par, scales = "free", ncol = 1) +
  coord_flip() +
  labs(title = "Blue = posterior; black = true value")
```

<img src="fig/hierarchical-models-rendered-unnamed-chunk-11-1.png" style="display: block; margin: auto;" />



:::::::::::::::::::::::::::::: challenge

Experiment with the data and fit. Explore the effect of sample size, unequal sample sizes between countries, and the amount of countries, for example. 

::::::::::::::::::::::::::::::::::::::::


## Hyperparameters

Let's then plot the population distribution's parameters, that is, the hyperparameters. The sample-based values are included in the plots of $\mu_\mu$ and $\sigma_\mu$ (why not for the other two hyperparameters?). 


```r
## Population distributions:
population_samples_l <- rstan::extract(normal_hier_fit,
                                       c("mu_mu", "sigma_mu", "alpha_sigma", "beta_sigma")) %>% 
  do.call(cbind, .) %>% 
  set_colnames(c("mu_mu", "sigma_mu", "alpha_sigma", "beta_sigma")) %>% 
  data.frame() %>% 
  mutate(sample = 1:nrow(.)) %>% 
  gather(key = "hyperpar", value = "value", -sample)


ggplot() +
  geom_histogram(data = population_samples_l, 
                 aes(x = value),
                 fill = posterior_color,
                 bins = 100) + 
  geom_vline(data = height_women %>% 
               rename_with(~ c('mu', 'sigma'), 3:4) %>%
               filter(Sex == "Girls") %>% 
               summarise(mu_mu = mean(mu), sigma_mu = sd(mu)) %>% 
               gather(key = "hyperpar", value = "value"),
             aes(xintercept = value)
             )+
  facet_wrap(~hyperpar, scales = "free")
```

<img src="fig/hierarchical-models-rendered-unnamed-chunk-12-1.png" style="display: block; margin: auto;" />

## Population distributions 

Let's then plot the population distributions and compare to the sample $\mu$'s and $\sigma$'s


```r
population_l <- rstan::extract(normal_hier_fit, c("mu_tilda", "sigma_tilda")) %>% 
  do.call(cbind, .) %>% 
  data.frame() %>% 
  set_colnames( c("mu", "sigma")) %>% 
  mutate(sample = 1:nrow(.)) %>% 
  gather(key = "par", value = "value", -sample)


 
ggplot() + 
  geom_histogram(data = population_l,
                 aes(x = value, y = ..density..),
                 bins = 100, fill = posterior_color) +
    geom_histogram(data = height_women %>%
                     rename_with(~ c('mu', 'sigma'), 3:4) %>%
                     gather(key = "par", value = "value", -c(Country, Sex)) %>% 
                     filter(Sex == "Girls"), 
                   aes(x = value, y = after_stat(density)), 
                   alpha = 0.75, bins = 30) +
  facet_wrap(~par, scales = "free") + 
  labs(title = "Blue = posterior; black = sample")
```

```{.warning}
Warning: The dot-dot notation (`..density..`) was deprecated in ggplot2 3.4.0.
ℹ Please use `after_stat(density)` instead.
This warning is displayed once every 8 hours.
Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
generated.
```

<img src="fig/hierarchical-models-rendered-unnamed-chunk-13-1.png" style="display: block; margin: auto;" />


## Posterior predictive distribution

Finally, let's plot the posterior predictive distribution. Let's overlay it with the simulated data based on all countries.


```r
# Sample size per group 
N <- 25

# For each country, generate some random girl's heights
Height_all <- lapply(1:nrow(height_women), function(i) {
  
  my_df <- height_women[i, ] %>% 
    rename_with(~ c('mu', 'sigma'), 3:4)
  
  data.frame(Country = my_df$Country, 
             Sex = my_df$Sex, 
             # Random normal values based on sample mu and sd
             Height = rnorm(N, my_df$mu, my_df$sigma))

}) %>% 
  do.call(rbind, .)
```




```r
# Extract the posterior predictive distribution
PPD <- rstan::extract(normal_hier_fit, c("X_tilda")) %>% 
  data.frame() %>% 
  set_colnames( c("X_tilda")) %>% 
  mutate(sample = 1:nrow(.))


ggplot() + 
  geom_histogram(data = PPD, 
                 aes(x = X_tilda, y = after_stat(density)),
                 bins = 100,
                 fill = posterior_color) +
  geom_histogram(data = Height_all, 
                 aes(x = Height, y = after_stat(density)), 
                 alpha = 0.75, 
                 bins = 100)
```

<img src="fig/hierarchical-models-rendered-unnamed-chunk-15-1.png" style="display: block; margin: auto;" />








## Extensions

We analyzed women's heights in a few countries and modeled them hierarchically. You could make the structure richer in many ways, for instance by adding hierarchy between sexes, continents, developed/developing countries etc. 


::::::::::::::::::::::::::::::::::::: keypoints 

- Hierarchical models are appropriate for scenarios where the study population naturally divides into subgroups. 
- Hierarchical model borrow statistical strength across the groups. 
- Population distributions hold information about the variation of the model parameters in the whole population. 

::::::::::::::::::::::::::::::::::::::::::::::::

## Reading
