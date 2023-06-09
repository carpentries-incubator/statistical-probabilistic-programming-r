---
title: 'exercises'
teaching: 10
exercises: 2
---



:::::::::::::::::::::::::::::::::::::: questions 

- How can I get routine in probabilistic programming?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

The purpose of this episode is to provide material for practicing probabilistic programming. The exercises are listed approximately based on the episode they refer to. 

::::::::::::::::::::::::::::::::::::::::::::::::



## Bayesian statistics

### (easy) Explanation
In your own words, explain the following concepts 

1. Posterior distribution
2. Informative prior distribution
3. Grid approximation
4. Effect of prior when different amount of data is available
5. Marginal posterior distribution




### (easy) Formula explanations

For the models described by the following formulas, come up with realistic scenarios where the models could be used. 

1. 
\begin{align}
y_i & \sim Normal(\mu, \sigma) \\
\mu & \sim Normal(0, 1) \\
\sigma & \sim Exponential(1)
\end{align}

2. 
\begin{align}
y_i & \sim Normal(\alpha + \beta x_i, \sigma) \\
\alpha & \sim Normal(0, 10) \\
\beta & \sim Normal(0, 1) \\
\sigma & \sim Exponential(1)
\end{align}

Formulate statistical models for the following scenarios. Give rationalizations for the likelihood and priors. 

1. A batch of 10000 avocados is shipped from South America to North Europe. You'd like to estimate the proportion of the batch that doesn't get bad in transit. 

2. A dart-throwing robot is poorly calibrated. Its throws produce a dense, normally distributed cluster with an approximately identity covariance matrix. However, the throws seems to veer down and left from bull's eye. You want the calibrate the robot based on a set of previous throws.  

Provide examples not found in the course material. 


### (easy) Formula explanations 2

What prior distributions would you use for the following parameters? Justify your choices. 

1. Life expectancy in the developed world
2. The probability of a successful 3 point basketball throw (for yourself)
3. The regression coefficient for milage per liter of gas and car weight
4. The waiting time for a government bureau helpline
5. The effect of national energy drink consumption on yearly rainfall


### (medium) Analyze grid approximation

In Cox regression, the risk for an event (such as the onset of a disease) can be estimated based on follow-up data and background variables at the beginning of the study. Assume the effect of a cancer medicine is studied in control and treatment groups. The data frame below gives the posterior distribution for the effect size $\beta$ of the medicine on survival compared to the control group. What assumptions were made in the analysis that most likely will produce incorrect conclusions about the efficacy of the drug?


```r
df <- data.frame(
  beta = c(-0.25, -0.219, -0.188, -0.156, -0.125, -0.094, -0.062, -0.031, 0, 0.031, 0.062, 0.094, 0.125, 0.156, 0.188, 0.219, 0.25, 0.281, 0.312, 0.344, 0.375, 0.406, 0.438, 0.469, 0.5),
  posterior = c(1.052, 1.478, 2.025, 2.707, 3.533, 4.498, 5.59, 6.779, 8.023, 9.266, 10.444, 11.487, 12.33, 12.915, 13.202, 13.17, 12.821, 12.18, 11.293, 10.217, 9.021, 7.773, 6.536, 5.363, 4.295)
)
```


### (medium) Grid approximation for the linear model




On the planet Neptunus, a ball is dropped from $h=8$ meters $N=25$ times and the falling time $t$ is measured for each drop. From previous experiments, it is known that the measurement error is normally distributed with standard deviation of 0.2. The measurement times are as follows. 


```r
Time <- c(1.28, 1.18, 1.37, 1.17, 0.98, 1.02,
          1.16, 1.19, 1.55, 1.13, 1.57, 1.26,
          1.39, 1.1, 1.4, 0.92, 1.15, 1.19, 1.25,
          1.01, 0.82, 1.46, 1.06, 1.38, 1.24)
```


According to Newton's theory of gravity, $h$ and $t$ are connected via the relation of $h = \frac{1}{2}gt^2$, where $g$ is the gravitational constant. 

Implement the grid approximation for standard linear regression and estimate $g$. Give the posterior mode and 90% CIs. Use some reasonable prior distribution. 


#### Hint
Solve formula $h = \frac{1}{2}gt^2$ for $t$ so the model error measures time measurement error. Further, do a parameter transformation so the formula doesn't contain squares of variables.  

## Working with samples

### (easy) Sampling the posterior

Assume we model the following observations $X$ with the exponential likelihood:


```r
X <- c(0.166, 1.08, 1.875, 0.413, 1.369, 0.463, 0.735,
       0.24, 0.774, 1.09, 0.463, 0.916, 0.225, 0.889,
       0.051, 0.688, 0.119, 0.078, 1.624, 0.553, 0.523,
       0.644, 0.284, 1.744, 1.468)
```

If we add a $\Gamma(2, 1)$ prior, the posterior distribution can be shown to be 

$$p(\lambda | X) = \Gamma(2 + n, 1 + \sum_{i}^{n}X_i),$$
where $n$ is the number of observations. 

Produce 1000 samples from the posterior and compute
1. posterior mean and mode
2. 50%, 90% and 95% posterior intervals
3. the probabilities that $\lambda > 1$, $\lambda < 1.5$, and $1<\lambda<1.5$ 


### (easy) Marginal posterior

Suppose the following piece of code produces samples from a (bivariate) posterior for $\theta = (\theta_1, \theta_2).$


```r
posterior_samples <- mvtnorm::rmvnorm(1000,
                                      mean = c(1, 0),
                                      sigma = matrix(c(1, 0.65, 0.65, 2), 2, 2))

# Make into a data frame
posterior_samples <- data.frame(posterior_samples) %>%
  set_colnames(c("theta1", "theta2"))
```

Produce marginal posterior samples for $\theta_1$ and $\theta_2$. Plot the full and marginal posteriors. What information is lost if you only look at the marginal posteriors?


### (medium) Posterior of difference

Consider the posterior distribution of the previous exercise. What is the probability that $\theta_1$ is larger than $\theta_2$?

### (medium) Sample size estimation

Suppose we aim to estimate p, the proportion of left-handed people very precisely (assume true p = 0.094). Specifically, we want the 99% percentile interval of the posterior distribution of p to be only 0.05 wide. Approximately how large of a study sample do we need? Simulate, no need to do analytical calculations. Use a prior of your liking.


### (medium) Less data means bigger prior effect

Explore the effec of sample size on the location and variance of the posterior. 

Instructions: 
  - Set p ~ (0, 1) and generate data from the binomial model. Simulate a sequence of 200 throws. 
  - Draw samples form the the analytical posterior using the first 5, 10, 15,..., 200 throws
  - Compute the posterior mode and 90% CIs for each fit
  - Visualize the results by generating a figure with: 
    - the mode and CIs as the function of data size
    - true parameter value 
    - prior mode and CIs.
      - Check formulas e.g. from Wikipedia
    - true analytical posterior mode and CIs using all 200 throws
      
  
### (hard) Highest posterior density set

Assume the following piece of code generates 5000 samples from a posterior distribution.


```r
posterior_samples <- c(rnorm(2500, -1, 0.5), rnorm(2500, 1, 0.75))
```

Compute the 50% highest posterior density set (not necessarily an interval!) of this distribution. Plot the set over a histogram of the samples. 

#### Hint

- Generate the posterior density from the samples with the `density` function. 
- Starting from the highest density, scan the density values in a decreasing order. Determine the mass of the posterior corresponding to the posterior above the particular density values. What does this set represent?


## Stan

### (easy) Analyze Stan programs 

Write the statistical models implemented in the following Stan programs and give some explanation on what they model. 

1. 

```stan

data{
  int<lower=0> N;
  vector[N] X;
}

parameters {
  real mu;
  real<lower=0> tau;
}

model {
  X ~ normal(mu, 1/tau);
  
  mu ~ normal(0, 10);
  tau ~ Gamma(2, 1);
}

```


2. 

```stan

data{
  int<lower=0> N;
  vector[N] X;
}

parameters {
  vector[3] phi;
  real<lower=0> sigma;
}

model {

  for(i in 3:N) {
    X[i] ~ normal(phi[1] + phi[2]*X[i-1] + phi[3]*X[i-2], sigma);
    }

  phi ~ normal(0, 1);
}

```

3. 

```stan

data{
  int<lower=0> n_groups;
  vector[n_groups] X;
  int<lower=0> N;
}

parameters {
  vector[n_groups] theta;
}

model {

  for(g in n_groups) {
    X[g] ~ binomial(N, theta[g]);
  }
  
  theta ~ beta(alpha, beta);
  
  alpha ~ gamma(2, 0.1); 
  beta ~ gamma(2, 0.1);

}

```


### (medium) Estimate dice fairness

Estimate the fairness of a 6 sided dice, that is, the probabilities of getting each face on a random roll. The results from 99 rolls is stored in the vector `rolls`.


```r
rolls <- c(3, 2, 6, 3, 6, 2, 5, 6, 5, 6, 4, 1, 4,
           2, 5, 4, 6, 6, 5, 4, 1, 3, 3, 4, 2, 3,
           4, 4, 4, 1, 1, 3, 4, 4, 1, 6, 4, 6, 5,
           5, 2, 6, 1, 1, 4, 4, 1, 6, 6, 1, 6, 4,
           5, 5, 3, 4, 2, 6, 6, 5, 2, 6, 1, 1, 4,
           4, 4, 6, 3, 5, 3, 6, 5, 3, 3, 2, 3, 3,
           5, 3, 3, 4, 6, 4, 3, 6, 6, 4, 4, 6, 5,
           1, 3, 5, 1, 2, 4, 4, 1)
```

Write a Stan program that implements the following statistical model: 

\begin{align}
y & \sim categorial(\theta) \\
\theta & \sim Dir(\textbf{1}),
\end{align}

Where $Dir$ is the Dirichlet distribution with parameter $\alpha = 1, 1, \ldots, 1.$ Plot the marginal posteriors for each $\theta_i$. Is the dice fair? Quantify this: give some probability for the hypothesis "the dice is fair" (for example, compute the probability of for $\theta_i > \theta_j$ for some $i$ and $j$). 
### (hard) Write Stan program: OUP

The Ornstein-Uhlenbeck process (OUP) can be used (among other things) as a part in algorithmic trading strategies. The model is a stochastic differential equation defined as

$$dX_t = \phi(\mu - X_t) + \sigma dW,$$
where $X_t$ is the state variable at time $t$, $\mu$ is the mean level, $\phi$ the mean-reversion rate, and $\sigma$ determines the level of stochastic variations of the process. $dW$ is a so-called Wiener process, of which you can read more about elsewhere. The parameter $\phi$ determines how quickly the process returns towards $\mu$ if the system is driven away from $\mu.$ 

The nice thing about the OUP is that it is analytically tractable, meaning that the transition density between different time points is known. In other words, $p(X_{t + dt} | X_t) = N(\mu - (\mu - X_t)e^{-\lambda t}, \frac{\sigma^2}{2\phi}(1-e^{-2\phi t})).$ This allows writing the likelihood for the data!

Implement the OUP in Stan and fit it on the data provided below. The data is the daily stock price of Exxon Mobil over the past 2 years (minus the 20 day exponential moving average, provided by Yahoo). Plot the posteriors of the model parameters. How could you use the analysis to make trading decisions (this is not an investment recommendation)?


```r
exxon <- c(-6.71, -8.31, -5.79, -7.72, -8.37, -5.16, -4.55, -4.51, -5.11, -4.32, -5.45, -3.66, -1.9, 0.24, 1.1, -0.35, -0.93, 1.79, 1.3, 2.93, 3.62, 7.16, 4.26, 3.48, 0.39, -3.11, -1.68, -1.06, 0.52, 1.25, 2.71, 3.18, 1.36, 0.45, 1.08, 2.95, 2.39, 2.1, 5.51, 5.51, 5.45, 3.83, 5.5, 1.53, 0.73, -0.9, 0.75, 0.1, -0.65, 0.11, 1.54, 2.4, 0.11, 2.21, -0.57, -1.98, -1.8, -2.32, -3.44, -3.46, -7.49, -8.38, -5.99, -2.6, -2.51, -3.51, 0.99, 3.93, 7.04, 9.03, 7.24, 4.57, 3.37, 3.44, 6.22, 3.21, 4.2, 3.96, 6.29, 5.81, 7.01, 7.01, 5.69, 6.29, 6.06, 8.33, 7.64, 7.91, 5.07, 5.94, 6.47, 7.05, 6.71, 1.46, 2.76, 5.62, 4.56, 4.82, 3.25, 3.08, 1.9, 0.71, 3.55, 2.7, 2.08, -1.2, -0.42, 0.34, -0.18, -1.01, -3.64, -5.98, -5.62, -4.39, -4.77, -2, -0.76, -1.41, -2.19, -2.66, -1.98, -0.41, 0.87, -1.19, 1.46, 2.68, 0.79, 1.46, 2.31, -1.34, -0.93, 1.32, 2.39, 0.3, 1.74, 2.73, 4.14, 3.69, 3.14, 0.74, 1.31, 3.02, 2.2, 2.94, 2.12, 6.03, 3.51, 1.32, 3.41, 1.94, -1.49, -0.65, -0.76, 2.2, 1.08, 1.37, 5.6, 3.83, 2.21, 1.69, 1.22, -2.92, -2.75, -3.79, -2.51, -2.26, -2.23, -2.6, -1.46, -0.86, 0.5, 1.35, -0.77, -2.17, -2.73, -3.69, -4.46, -3.68, -8.14, -7.9, -8.22, -5.08, -0.44, -2.62, -3.43, -3, -0.66, 0.6, 2.21, 2.48, 2.39, 8.02, 6.25, 7.44, 4.97, 4.04, 4.39, 3.86, 3.98, 3.86, 2.27, 4.08, 3.35, 2.19, 2.32, 4.08, 2.17, 1, 2.15, 3.31, -0.32, -4.43, -5.97, -7.11, -4.05, -3.27, -2.93, -3.92, -5.31, -4.81, -5, -6.83, -4.1, -2.85, -2.13, -3.1, -1.51, -0.29, -2.01, -2.44, -3.05)
```


### (hard) Implement logistic growth

In ecology, evolution of population size is often modeled using logistic growth which is defined by the differential equation 

$$\frac{dP}{dt} = rP(1-\frac{P}{K}),$$
where $P$ is the population size, $t$ time, $r$ the growth rate. The carrying capacity $K$ determines the maximum size of the population, and without it the population would grow exponentially. The solution to this differential equation is 

$$P(t) = \frac{K}{1 + (\frac{K}{P_0} - 1)e^{-rt}}.$$ 
In other words, the solution gives the population size at time $t$, given the initial size $P_0$ and the model parameters. 

The following data contains (noisy and scaled) measurements of an **E. coli** bacteria culture measured for 1 day (Sprouffske, Wagner, Growthcurver (2016): an R package for obtaining interpretable metrics from microbial growth curves (2016)). 


```r
E.coli <- data.frame(
  time = c(0, 0.167, 0.333, 0.5, 0.667, 0.833, 1, 1.167, 1.333, 1.5, 1.667, 1.833, 2, 2.167, 2.333, 2.5, 2.667, 2.833, 3, 3.167, 3.333, 3.5, 3.667, 3.833, 4, 4.167, 4.333, 4.5, 4.667, 4.833, 5, 5.167, 5.333, 5.5, 5.667, 5.833, 6, 6.167, 6.333, 6.5, 6.667, 6.833, 7, 7.167, 7.333, 7.5, 7.667, 7.833, 8, 8.167, 8.333, 8.5, 8.667, 8.833, 9, 9.167, 9.333, 9.5, 9.667, 9.833, 10, 10.167, 10.333, 10.5, 10.667, 10.833, 11, 11.167, 11.333, 11.5, 11.667, 11.833, 12, 12.167, 12.333, 12.5, 12.667, 12.833, 13, 13.167, 13.333, 13.5, 13.667, 13.833, 14, 14.167, 14.333, 14.5, 14.667, 14.833, 15, 15.167, 15.333, 15.5, 15.667, 15.833, 16, 16.167, 16.333, 16.5, 16.667, 16.833, 17, 17.167, 17.333, 17.5, 17.667, 17.833, 18, 18.167, 18.333, 18.5, 18.667, 18.833, 19, 19.167, 19.333, 19.5, 19.667, 19.833, 20, 20.167, 20.333, 20.5, 20.667, 20.833, 21, 21.167, 21.333, 21.5, 21.667, 21.833, 22, 22.167, 22.333, 22.5, 22.667, 22.833, 23, 23.167, 23.333, 23.5, 23.667, 23.833, 24), 
  abundance = c(0.01, 0.005, 0.013, 0.008, 0.002, 0.01, 0.006, 0.006, 0.003, 0.009, 0.01, 0.006, 0.011, 0, 0.001, 0.006, 0.007, 0.004, 0.006, 0.005, 0.003, 0.008, 0.006, 0.011, 0.008, 0.015, 0.008, 0.006, 0.013, 0.01, 0.013, 0.016, 0.018, 0.013, 0.015, 0.013, 0.019, 0.021, 0.024, 0.03, 0.03, 0.036, 0.04, 0.045, 0.059, 0.064, 0.072, 0.083, 0.093, 0.111, 0.123, 0.137, 0.154, 0.176, 0.183, 0.201, 0.218, 0.231, 0.248, 0.261, 0.267, 0.287, 0.288, 0.293, 0.299, 0.306, 0.309, 0.314, 0.321, 0.324, 0.326, 0.331, 0.328, 0.33, 0.325, 0.337, 0.331, 0.337, 0.333, 0.335, 0.338, 0.335, 0.332, 0.332, 0.333, 0.332, 0.332, 0.337, 0.336, 0.335, 0.33, 0.34, 0.333, 0.332, 0.337, 0.337, 0.336, 0.336, 0.336, 0.335, 0.334, 0.339, 0.335, 0.336, 0.337, 0.335, 0.332, 0.333, 0.334, 0.337, 0.334, 0.33, 0.337, 0.333, 0.339, 0.335, 0.329, 0.332, 0.338, 0.333, 0.329, 0.334, 0.338, 0.344, 0.335, 0.334, 0.337, 0.337, 0.334, 0.334, 0.331, 0.34, 0.336, 0.34, 0.34, 0.336, 0.337, 0.34, 0.337, 0.333, 0.328, 0.339, 0.334, 0.332, 0.336))
```

Write a Stan program for the logistic growth model and estimate the growth rate and carrying capacity for this **E. coli** strain. Plot the marginal posteriors for $P_0$, $r$ and $K.$ Generate a posterior of growth trajectories and plot it against the data. 





## MCMC 
## Hierarchical models 

###    (easy) Analyze models

Are the following statistical models hierarchical? If not, make modifications that turn them into hierarchical. 

1. 

\begin{align}
  y_{group, i} & \sim Normal(a_{group} + b x_{group, i}, \sigma) \\
  a_{group} & \sim Normal(0, 1) \\
  b & \sim Normal(0, 1) \\
  \sigma & \sim Exponential(1)
\end{align}

2. 

\begin{align}
  y_i &\sim Binomial(1, p_i) \\
  logit(p_i) &= a_{group, i} + b x_i \\
  a_{group} &\sim N(\alpha, 1) \\
  \alpha &\sim N(0, 100) \\
  b & \sim Normal(0, 1)
\end{align}

3. 

\begin{align}
  X_i & \sim \Gamma(\alpha, \beta) \\
  \alpha &\sim \Gamma(1, 1) \\
  \beta &\sim \Gamma(1, 1) \\
\end{align}


## Model checking 
## Other topics



::::::::::::::::::::::::::::::::::::: keypoints 

- Point

::::::::::::::::::::::::::::::::::::::::::::::::

