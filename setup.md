---
title: Setup
---

Welcome to this introductory course on probabilistic programming and Bayesian data analysis. The course takes an application-oriented approach, introducing the theoretical background only where needed. Links to textbooks and other resources are provided for learners who wish to explore the topics in greater depth.

The central aim of these lessons is to introduce Stan, a probabilistic programming language for specifying and fitting a wide range of statistical models. Stan can be used through interfaces for Python, Julia, R, and several other languages. Here, we focus on its integration with R through the CmdStanR package.

Instructions for installing CmdStanR and Stan are available [here](https://mc-stan.org/cmdstanr/).

The lessons require relatively little prior knowledge. However, familiarity with basic concepts in probability and statistics (summary statistics, probability densities etc.). Learners should also have a solid working knowledge of R, including data wrangling and visualization.

The primary sources in preparing the material were A. Gelman et al., "[Bayesian Data Analysis](https://users.aalto.fi/~ave/BDA3.pdf)" (3rd ed.), and R. McElreath's ”[Statistical Rethinking](https://xcelab.net/rm/)” (2nd edition). The [Stan User's Guide](https://mc-stan.org/docs/stan-users-guide/index.html), Michael Betancourt's [writings](https://betanalpha.github.io/writing/), and the [website](https://avehtari.github.io/BDA_course_Aalto/index.html) of Aki Vehtari's Bayesian Data Analysis course were also consulted.

The lessons also require some additional R packages. Run the following code to install and load the packages and set the graphical theme.
```R
package_list <- c("tidyverse", "cowplot", "grafify", "cmdstanr",
                  "mvtnorm", "loo", "bayesplot", "brms")

for (p in package_list){
     if(!p %in% installed.packages()){
         install.packages(p)
     }
     require(p, character.only = TRUE)
}

theme_set(theme_bw(15))
prior_color <- "#009E73"
likelihood_color <- "#E69F00"
posterior_color <- "#56B4E9"
```

## Data sets

The data sets used in the lesson can be accessed  [here](https://github.com/carpentries-incubator/statistical-probabilistic-programming-r/tree/main/episodes/data) where you can find them packaged in  `lesson-data.zip`.


<!--
## Contributors
Ville Laitinen, Eetu Tammi, Aleksi Lahtinen, Leo Lahti
-->

<!--
FIXME: place any data you want learners to use in `episodes/data` and then use
       a relative link ( [data zip file](data/lesson-data.zip) ) to provide a
       link to it, replacing the example.com link.
-->



