---
title: 'Other topics'
teaching: 10
exercises: 2
---





:::::::::::::::::::::::::::::::::::::: questions

- Which packages take advantage of Stan and how to use them?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives


- Learn to use Stan with additional R packages


::::::::::::::::::::::::::::::::::::::::::::::::


In this chapter we will introduce packages that take advantage of Stan and `rstan`. The packages covered are `loo`, `bayesplot` and `brms`. From these `loo` enables easy cross-validation of Bayesian models and `bayesplot` provides tools for plotting the models. `brms` allows user fit Bayesian models using Stan without having to write the Stan code.


## `loo` R package

The first package that we will introduce is the `loo` R package. It allows user to compute efficient approximate leave-one-out cross-validation for fitted Bayesian models. It can also compute model weights that can be used to average predictive distributions. `loo` uses Pareto smoothed importance sampling (PSIS) to compute LOO-CV. It can also return approximate standard errors for estimated predictive errors. The package can also be used for calculating WAIC.

### Example 1

We will demonstrate the use of the `loo` package on the two models introduced in chapter 5. We will evaluate the fit of the models and compare them with the tools provided by `loo`.

To begging we need to add log-likelihood calculation to the Stan code. This is done by adding `log_lik` to the generated quantities block of the code. Below we demonstrate how this is done with the two models we will be comparing.


``` stan
// Normal model
data {
  int<lower=0> N;
  vector[N] X;
}
parameters {
  real<lower=0> sigma;
  real mu;
}
model {
  X ~ normal(mu, sigma);
  
  mu ~ normal(0, 1);
  sigma ~ gamma(2, 1);
}

generated quantities {
  vector[N] X_rep;
  
  for(i in 1:N) {
    X_rep[i] = normal_rng(mu, sigma);
  }
  
  // Calculating log-likelihood for loo
  vector[N] log_lik;
  
  for (i in 1:N) {
  log_lik[i] = normal_lpdf(X[i] | mu, sigma);
  }
}
```



``` stan
// Cauchy model
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
  // location = mu and scale = sigma
  X ~ cauchy(mu, sigma);
  
  mu ~ normal(0, 1);
  sigma ~ gamma(2, 1);
}
generated quantities {
  vector[N] X_rep;
  for(i in 1:N) {
    X_rep[i] = cauchy_rng(mu, sigma);
  }
  
  // Calculating log-likelihood for loo
  vector[N] log_lik;
  
  for (i in 1:N) {
  log_lik[i] = cauchy_lpdf(X[i] | mu, sigma);
  }
}

```

After adding the log-likelihood calculation into the code the model can be fit in the usual way.




``` r
# Fitting normal model
normal_fit <- rstan::sampling(normal_model_loo,
                       list(N = N, X = df5$X), 
                       refresh = 0, seed = 2024)
# Fitting cauchy model
cauchy_fit <- rstan::sampling(cauchy_model_loo,
                       list(N = N, X = df5$X), 
                       refresh = 0, seed = 2024)
```

We can now compute PSIS-LOO for both of the models with `loo::loo` function. After the computations we can get more information about the fit by printing the `loo` objects.


``` r
# PSIS-LOO computation for normal model
normal_loo <- loo::loo(normal_fit)
```

``` warning
Warning: Some Pareto k diagnostic values are too high. See help('pareto-k-diagnostic') for details.
```

``` r
print(normal_loo)
```

``` output

Computed from 4000 by 88 log-likelihood matrix.

         Estimate   SE
elpd_loo   -288.8 41.6
p_loo        18.5 17.6
looic       577.7 83.3
------
MCSE of elpd_loo is NA.
MCSE and ESS estimates assume MCMC draws (r_eff in [0.8, 0.9]).

Pareto k diagnostic values:
                         Count Pct.    Min. ESS
(-Inf, 0.7]   (good)     87    98.9%   2167    
   (0.7, 1]   (bad)       0     0.0%   <NA>    
   (1, Inf)   (very bad)  1     1.1%   <NA>    
See help('pareto-k-diagnostic') for details.
```

``` r
# PSIS-LOO computation for cauchy model
cauchy_loo <- loo::loo(cauchy_fit)
print(cauchy_loo)
```

``` output

Computed from 4000 by 88 log-likelihood matrix.

         Estimate   SE
elpd_loo   -206.9 14.7
p_loo         2.0  0.0
looic       413.8 29.3
------
MCSE of elpd_loo is 0.0.
MCSE and ESS estimates assume MCMC draws (r_eff in [0.7, 0.8]).

All Pareto k estimates are good (k < 0.7).
See help('pareto-k-diagnostic') for details.
```

Using print returns $\widehat{\text{elpd}}_{\text{loo}}$ (expected log pointwise predictive density), $\hat{p}_{loo}$ (estimated number of parameters) and $\text{looic}$ (LOO information criterion) values and their standard errors. It also return a table with the Pareto $k$ diagnostic values. These values are used to asses the reliability of the estimates. Values over 1 means that the PSIS estimate and the corresponding Monte Carlo standard error are not well defined.

If we want to compare the models, this can be done by using `loo::loo_compare` function on the `loo` objects. It will compare the models based on their elpd values. 


``` r
# Comparing models based on loo
loo::loo_compare(normal_loo, cauchy_loo)
```

``` output
       elpd_diff se_diff
model2   0.0       0.0  
model1 -81.9      36.2  
```

From the comparison we see that the elpd difference is larger than the standard error which indicates that the cauchy model is expected to have better predictive performance than the normal model. This is in line with what we saw in chapter 5.


::::::::::::::::::::::::::::::::::::::::: challenge

`loo` can also be used to compute WAIC for Bayesian models. Calculate WAIC for the two models and then compare them.

::::::::::::::::::::::::::::::: solution

First we need to extract the log-likelihood values from the fitted model object.


``` r
# Extracting loglik
normal_loglik <- loo::extract_log_lik(normal_fit)
cauchy_loglik <- loo::extract_log_lik(cauchy_fit)

# Computing WAIC for the models
normal_waic <- loo::waic(normal_loglik)
```

``` warning
Warning: 
1 (1.1%) p_waic estimates greater than 0.4. We recommend trying loo instead.
```

``` r
print(normal_waic)
```

``` output

Computed from 4000 by 88 log-likelihood matrix.

          Estimate   SE
elpd_waic   -290.0 42.8
p_waic        19.7 18.8
waic         580.1 85.6

1 (1.1%) p_waic estimates greater than 0.4. We recommend trying loo instead. 
```

``` r
cauchy_waic <- loo::waic(cauchy_loglik)
print(cauchy_waic)
```

``` output

Computed from 4000 by 88 log-likelihood matrix.

          Estimate   SE
elpd_waic   -206.9 14.7
p_waic         2.0  0.0
waic         413.8 29.3
```

Computing WAIC for the model return values for $\widehat{\text{eldp}}_{\text{WAIC}}$, $\hat{p}_{\text{WAIC}}$ and $\widehat{\text{WAIC}}$. Models can be compared based on WAIC with the same function as with PSIS-LOO.


``` r
# Comparing models based on WAIC
loo::loo_compare(normal_waic, cauchy_waic)
```

``` output
       elpd_diff se_diff
model2   0.0       0.0  
model1 -83.1      37.4  
```

::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::


## `bayesplot` R package

The next package we will cover is the `bayesplot` R package. The package provides a large library of different kinds of plotting functions to use after fitting a Bayesian model. The plots created by the package are `ggplot` objects, which means that the plots can be customized with the functions from `ggplot2` package. The package provides functions for plotting posterior draws, visual MCMC diagnostics and graphical posterior or prior predictive checking. The functions in the package also works with models fit by other packages like `brms` and `rstanarm`.

### Continuing with example 1

We will demonstrate using `bayesplot` with the Cauchy model used in the first exercise. Before we can start using the plotting functions we must first extract the draws. After this we will plot uncertainty intervals for mu and sigma.


``` r
# Extracting draws
cauchy_draws <- as.array(cauchy_fit)

# Plotting uncertainty intervals
bayesplot::mcmc_intervals(cauchy_draws, pars = c("mu", "sigma"))
```

<img src="fig/other-topics-rendered-unnamed-chunk-10-1.png" style="display: block; margin: auto;" />

If we instead wanted to plot estimated posterior density curves with uncertainty intervals as shaded areas or histograms of marginal posterior distributions, we can use the following functions:


``` r
# Plotting estimated density curves
bayesplot::mcmc_areas(cauchy_draws, pars = c("mu", "sigma"),
                      prob = 0.95,
                      point_est = "mean")
```

<img src="fig/other-topics-rendered-unnamed-chunk-11-1.png" style="display: block; margin: auto;" />

``` r
# Plotting histogram
bayesplot::mcmc_hist(cauchy_draws, pars = c("mu", "sigma"))
```

<img src="fig/other-topics-rendered-unnamed-chunk-11-2.png" style="display: block; margin: auto;" />

`bayesplot` also provides a lot of functions meant for assessing the convergence of the MCMC chains and visualizing other fit diagnostics. For example we can easily plot the trace plots for the chains.


``` r
# Plotting trace plot
bayesplot::mcmc_trace(cauchy_draws, pars = c("mu", "sigma"),
                      facet_args = list(ncol = 1))
```

<img src="fig/other-topics-rendered-unnamed-chunk-12-1.png" style="display: block; margin: auto;" />

In addition to the demonstrated functions, `bayesplot` has many more plotting functions. 


::::::::::::::::::::::::::::::::::::::::: challenge

`bayesplot` provides tools for doing graphical posterior predictive checks. Plot density estimates of $X_{rep}$ overlaid with density of $X$ for the Cauchy model using `bayesplot`. You can also plot the histograms for $X$ and $X_{rep}$.

::::::::::::::::::::::::::::::: solution



``` r
# Extracting replicates and getting a subset
set.seed(2024)
X_rep <- rstan::extract(cauchy_fit, "X_rep")[[1]] %>%
  data.frame() %>%
    mutate(sample = 1:nrow(.))

N_rep <- 9

X_rep_sub <- X_rep %>% filter(sample %in%
                                sample(X_rep$sample,
                                       N_rep,
                                       replace = FALSE))
X_rep_sub <- X_rep_sub[, -89] %>%
  as.matrix()
```


``` r
# Plotting density
bayesplot::ppc_dens_overlay(y = df5$X, yrep = X_rep_sub) + xlim(-25, 50)
```

<img src="fig/other-topics-rendered-unnamed-chunk-14-1.png" style="display: block; margin: auto;" />


``` r
# Plotting histograms
bayesplot::ppc_hist(y = df5$X, yrep = X_rep_sub) + xlim(-25,50)
```

<img src="fig/other-topics-rendered-unnamed-chunk-15-1.png" style="display: block; margin: auto;" />

In both cases it is recommended to use `xlim` to make the results more clear.

::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::


## `brms` R package

We will now introduce the `brms` R package. The purpose of the package is fitting Bayesian generalized (non-)linear multivariate multilevel models using Stan. It allows user to use familiar syntax from other linear fit functions, while writing and compiling the Stan model on the backend. `brms` supports large range of distributions and link functions. User can also easily specify a lot of different kinds of priors for the models.

The package also provides a lot of tools for evaluating the fit and convergence of the MCMC chains. It does this by using `loo` and `bayesplot` packages, so it uses a lot of the same tools we have already covered in this chapter. You can for example easily plot the trace plots and conditional effects. You can also use different functions that take advantage of PSIS-LOO to assess the fit or compare different models. User can also easily compute posterior draws of the posterior predictive distribution.

We will next demonstrate the use of the package with two different examples.

### Example 2: Cox proportional hazard model

As mentioned `brms` can be used to fit large array of different models. We will now fit a Cox proportional hazard model on the `lung` dataset from the `survival` R package. Before getting into fitting the model we will shortly introduce survival modeling and the Cox model.

In survival modeling the outcome of interest is time to event, meaning the time it takes to move from one state to another. This can be the time it takes to move from alive to dead or from sick to healthy. The risk of transitioning to a state at time $t$, for example the risk of dying at time $t$, is described by hazard function $\lambda(t)=\text{lim}_{h \to 0+} \frac{1}{h}P(t \le T<t+h|T\ge t)$. Hazard function basically describes the possibility of the event happening in the time interval $h$ conditioned on it not having happened by time $t$.

In the Cox model we will be fitting, the hazard function is of form $\lambda(t_i,Z_i,\theta)=\lambda_0(t_i)\text{exp}(\beta^\prime Z_i)$. Now there is a baseline hazard $\lambda_0(t_i)$ that is same for every subject $i$. The hazard function also has subject specific covariate part $\text{exp}(\beta^\prime Z_i)$. The covariates $Z_i$ can be things like age of the patient or an indicator for if the subject is getting a placebo or a drug. It is good to note that $\beta$ is log hazard ratio. Hazard ratio measures the hazard in one group against hazard in a another group.

When fitting Cox proportional hazard model, `brms` will use M-splines for the baseline hazard. Splines are functions defined piecewise by polynomials. This means that the the baseline hazard is a combination of several different polynomial functions. M-splines are non-negative spline functions.

Before actually fitting the model, let's take a look at the lung dataset we will be using. The dataset consists of survival times of patients with advanced lung cancer. 


``` r
# Getting the dataset
lung <- survival::lung

# First few rows of data
head(lung)
```

``` output
  inst time status age sex ph.ecog ph.karno pat.karno meal.cal wt.loss
1    3  306      2  74   1       1       90       100     1175      NA
2    3  455      2  68   1       0       90        90     1225      15
3    3 1010      1  56   1       0       90        90       NA      15
4    5  210      2  57   1       1       90        60     1150      11
5    1  883      2  60   1       0      100        90       NA       0
6   12 1022      1  74   1       1       50        80      513       0
```

Status variable describes if the event of interest (death) was observed or if the observation was censored. Censoring is common in time-to-event datasets as it is often infeasible to follow all the subjects until the event of interest. Censoring means that the event time is larger than the observed time. Censoring must be taken into account during modelling or it will bias the results.

We will fit a Cox proportional hazard model with three covariates: age, sex and ph.karno. Variable ph.karno describes how well patient can perform daily activities rated by a physician. We will split the variable into two categories high and mid. Cox model can be fit with `brms::brm()` function by specifying `family = brmsfamily("cox")`. We also need to specify when observations are censored, which is done with `cens(1 - status)` argument. We will use $\text{Normal}(0, 10)$ as prior for the population-level effects. By default `brms` will use an improper flat prior over the reals for population-level effects. Prior is defined with the `prior(normal(0,10), class = b)` argument, where `class = b` means that the prior is set for all population-level effects.
 

``` r
# Let's change status coding from 2/1 to 1/0
lung$status <- lung$status - 1

# Remove observations with NA ph.karno
lung <- lung[!is.na(lung$ph.karno),]

# Creating new variable for ph.karno status
lung$ph.karno_status <- cut(lung$ph.karno,
                            breaks = c(0, 70, 100),
                            labels = c("mid", "high"))

# Fitting the model
fit_cox <- brms::brm(time | cens(1 - status) ~ sex + age + ph.karno_status,
             data = lung, family = brmsfamily("cox"), seed = 2024,
             silent = 2, refresh = 0, cores = 4,
             prior = prior(normal(0,10), class = b))
```

``` error
Error: Please install the 'splines2' package.
```

``` r
# Summary of the fit
summary(fit_cox)
```

``` error
Error in eval(expr, envir, enclos): object 'fit_cox' not found
```

The summary output of the `brms` fit returns estimates for coefficients and their credible intervals. It also returns Rhat, Bulk_ESS and Tail_ESS values, which can be used to assess the convergence of the model. 

It is important to notice that the coefficients are the log hazard ratios, which means that we still need to take an exponential of the values. We will also plot the credible intervals. The `bayesplot::mcmc_intervals()` function allows transforming the parameters before plotting with `transform = "exp"` argument.


``` r
# Getting hazard values to normal scale
sum_cox <- summary(fit_cox)
```

``` error
Error in eval(expr, envir, enclos): object 'fit_cox' not found
```

``` r
exp(sum_cox$fixed[,1:4])
```

``` error
Error in eval(expr, envir, enclos): object 'sum_cox' not found
```

``` r
# Credible intervals
bayesplot::mcmc_intervals(fit_cox, pars = c("b_sex", "b_age", "b_ph.karno_statushigh"),
                          transform = "exp")
```

``` error
Error in eval(expr, envir, enclos): object 'fit_cox' not found
```

Based on the estimate and credible interval it seems that age has a very little effect on the hazard. Females and patients with high ph.karno value have smaller hazard, meaning that the risk of death is smaller.

After fitting the model you can print info on the priors used with the function `brms::get_prior`.


``` r
# Getting priors for the cox model
brms::get_prior(fit_cox)
```

``` error
Error in eval(expr, envir, enclos): object 'fit_cox' not found
```

As we can see the population-level effects has the normal prior we specified. Intercept has the default prior used by `brms` for intercept, which is Student's t-distribution with three degrees of freedom. If we want to print the whole Stan code used in fitting the model, we can do this with the `brms::stancode` function.


``` r
# Printing the Stan code
brms::stancode(fit_cox)
```

``` error
Error in eval(expr, envir, enclos): object 'fit_cox' not found
```


### Example 3: Hierarchical binomial model

We will next show how to fit hierarchical models using `brms`. Fitting hierarchical models can be considered one of the main focuses of the `brms` package. If you have ever used the `lme4` package, you are already quite familiar with the syntax used in `brm()` to fit hierarchical models, as both packages use similar formula syntax. 

The data we will be using is the `VerbAgg` dataset from `lme4` package. The dataset consist of item responses to a questionnaire on verbal aggression. 


``` r
# Getting the data
VerbAgg <- lme4::VerbAgg

# First few rows of the data
head(VerbAgg)
```

``` output
  Anger Gender        item    resp id btype  situ mode r2
1    20      M S1WantCurse      no  1 curse other want  N
2    11      M S1WantCurse      no  2 curse other want  N
3    17      F S1WantCurse perhaps  3 curse other want  Y
4    21      F S1WantCurse perhaps  4 curse other want  Y
5    17      F S1WantCurse perhaps  5 curse other want  Y
6    21      F S1WantCurse     yes  6 curse other want  Y
```

We will use Anger, Gender, btype and situ as population-level effects for the model. The model will also have group-level intercept for id. The variable of interest is the binary r2, which describes response to questionnaire question. For priors we will use $\text{Normal}(0, 10)$ as prior for all the population-level effects. The group-level effect's standard deviation parameter will have $\text{Cauchy}(0, 5)$ prior (this is Half-Cauchy prior as the values are restricted to be positive). By default `brms` uses Half-Student's t-distribution with three degrees of freedom for standard deviation parameters. The group-level intercept for variable id is specified with the argument `(1|id)`. Let's now fit the model.


``` r
# Changing coding for r2
VerbAgg <- VerbAgg %>%
  mutate(r2 = ifelse(r2 == "N", 0, 1))
# Fitting the model
fit_hier <- brms::brm(r2 ~ Anger + Gender + btype + situ + (1|id), family = bernoulli, 
                data = VerbAgg, seed = 2024, cores = 4, silent = 2, refresh = 0,
                prior = prior(normal(0, 10), class = b) + 
                  prior(cauchy(0,5), class = sd))
# Summary of the fit
summary(fit_hier)
```

``` output
 Family: bernoulli 
  Links: mu = logit 
Formula: r2 ~ Anger + Gender + btype + situ + (1 | id) 
   Data: VerbAgg (Number of observations: 7584) 
  Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
         total post-warmup draws = 4000

Multilevel Hyperparameters:
~id (Number of levels: 316) 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
sd(Intercept)     1.30      0.07     1.17     1.43 1.00     1086     1687

Regression Coefficients:
           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept      0.21      0.34    -0.49     0.87 1.01      637     1035
Anger          0.05      0.02     0.02     0.09 1.01      664     1026
GenderM        0.31      0.19    -0.05     0.68 1.00      580     1257
btypescold    -1.03      0.07    -1.17    -0.90 1.00     4963     3254
btypeshout    -2.00      0.07    -2.14    -1.85 1.00     4867     3286
situself      -1.01      0.06    -1.12    -0.89 1.00     5877     3022

Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1).
```

With `brms` we can easily plot conditional effects of predictors by using the function `brms::conditional_effects`.


``` r
# Conditional effects
plots <- plot(conditional_effects(fit_hier), plot = FALSE)
cowplot::plot_grid(plots[[1]], plots[[2]], plots[[3]], plots[[4]])
```

<img src="fig/other-topics-rendered-unnamed-chunk-23-1.png" style="display: block; margin: auto;" />

We can also plot interactions using the function. Let's plot the conditional effect for interaction of Anger and btype.


``` r
# Plotting conditional effect for interaction of Anger and btype
plot(conditional_effects(fit_hier, effects = "Anger:btype"))
```

<img src="fig/other-topics-rendered-unnamed-chunk-24-1.png" style="display: block; margin: auto;" />

Let's fit another model with the same population-level effects, but we will add another group-level intercept for the item variable. The priors are same as for the first model. We don't have to rewrite the whole formula when updating the model. We can use the `update` function for updating the formula.


``` r
# Updating the model
fit_hier2 <- update(fit_hier, formula. = ~ . + (1|item), newdata = VerbAgg, seed = 2024,
                    cores = 4, silent = 2, refresh = 0)
# Summary of the updated fit
summary(fit_hier2)
```

``` output
 Family: bernoulli 
  Links: mu = logit 
Formula: r2 ~ Anger + Gender + btype + situ + (1 | id) + (1 | item) 
   Data: VerbAgg (Number of observations: 7584) 
  Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
         total post-warmup draws = 4000

Multilevel Hyperparameters:
~id (Number of levels: 316) 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
sd(Intercept)     1.36      0.07     1.23     1.51 1.00     1347     1922

~item (Number of levels: 24) 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
sd(Intercept)     0.59      0.11     0.42     0.84 1.00     1404     2252

Regression Coefficients:
           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept      0.18      0.44    -0.65     1.05 1.01      831     1522
Anger          0.06      0.02     0.02     0.09 1.00      801     1568
GenderM        0.32      0.20    -0.07     0.72 1.00      847     1333
btypescold    -1.06      0.30    -1.63    -0.43 1.00     1191     1561
btypeshout    -2.11      0.30    -2.71    -1.52 1.00     1284     1889
situself      -1.05      0.25    -1.54    -0.55 1.00     1156     1828

Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1).
```

Another useful thing about `update` is that it allows for resampling from the model with, for example, different number of iterations without having to recompile the model. If you update the formula or priors, the model has to be recompiled.

To end let's compare the two models by using `brms::loo()`. This works in the same way as the `loo::loo_compare`.


``` r
# Comparing the models
brms::loo(fit_hier, fit_hier2)
```

``` output
Output of model 'fit_hier':

Computed from 4000 by 7584 log-likelihood matrix.

         Estimate   SE
elpd_loo  -4004.9 42.9
p_loo       268.0  3.7
looic      8009.7 85.8
------
MCSE of elpd_loo is 0.2.
MCSE and ESS estimates assume MCMC draws (r_eff in [1.1, 2.2]).

All Pareto k estimates are good (k < 0.7).
See help('pareto-k-diagnostic') for details.

Output of model 'fit_hier2':

Computed from 4000 by 7584 log-likelihood matrix.

         Estimate   SE
elpd_loo  -3866.9 43.9
p_loo       287.0  4.1
looic      7733.8 87.8
------
MCSE of elpd_loo is 0.2.
MCSE and ESS estimates assume MCMC draws (r_eff in [1.1, 2.6]).

All Pareto k estimates are good (k < 0.7).
See help('pareto-k-diagnostic') for details.

Model comparisons:
          elpd_diff se_diff
fit_hier2    0.0       0.0 
fit_hier  -138.0      16.2 
```

Based on the output the second model is clearly better fit when compared to the first model.


::::::::::::::::::::::::::: challenge

Experiment with different priors for the model. How much does the choice of prior effect the results? Is there a big difference between a flat and the used weakly informative prior?

:::::::::::::::::::::::::::::::::::::


## Other packages built on Stan

In addition to the packages covered here, there are several other packages that take advantage of Stan. Here we will shortly introduce some of them.  [CmdStanR](https://mc-stan.org/cmdstanr/index.html) is a lightweight interface for Stan and provides and alternative for rstan interface. [rstanarm](https://mc-stan.org/rstanarm/) emulates R model-fitting functions using Stan. The package can do lot of the same things as `brms`, but they do have differences, for example `rstanarm` models come pre-compiled while `brms` compiles the models when fitted. 

[shinystan](https://mc-stan.org/shinystan/) uses Shiny and provides user with informative, customizable visual and numerical summaries of model parameters and convergence diagnostics. [projpred](https://mc-stan.org/projpred/) performs projection predictive variable selection for various models. The package works with models from `brms` and `rstanarm`. [posterior](https://mc-stan.org/posterior/) provides tools for converting between different formats of draws, consistent methods for operations commonly performed on draws (like subsetting and binding), summaries of draws and posterior diagnostics. 

You can learn a lot more about the packages above and the packages covered in this chapter by visiting the associated websites and reading the provided examples, vignettes and articles.

::::::::::::::::::::::::::::::::::::: keypoints 

- There are several R packages that make use Stan enabling more user-friendly ways of fitting Bayesian models and allowing for more advanced inference.
- `brms` package can be used to fit a fast array of different Bayesian models.
- Extensive set of different plots for Bayesian models can be plotted with `bayesplot` package.
- Leave-one-out cross-validation can be performed with `loo` package. 

::::::::::::::::::::::::::::::::::::::::::::::::



## Reading

- [brms website](https://paul-buerkner.github.io/brms/index.html)

- [bayesplot website](http://mc-stan.org/bayesplot/)

- [loo website](https://mc-stan.org/loo/)

