---
title: Setup
---

Welcome to an introductory course on probabilistic programming! The aim of this course is to learn the basics of the topic with an application-oriented approach. Theoretical details are provided to a minimal extent. However, links to textbooks are provided and it's recommended that the student follow these texts along with the lesson material. 

In order to make most of this material, the student should have a good grasp of basic concepts in probability and statistics (distribution, probability density, Bayes' rule, basic summary statistics)  programming with R, including data manipulation and visualization. 

The primary sources in preparing the material were A. Gelman et al., "[Bayesian Data Analysis](https://users.aalto.fi/~ave/BDA3.pdf)" (3rd ed.), and R. McElreath's "Statistical Rethinking" (2nd edition). The [Stan User's Guide](https://mc-stan.org/docs/2_18/stan-users-guide/index.html) and the [website](https://avehtari.github.io/BDA_course_Aalto/index.html) of Aki Vehtari's course Bayesian Data Analysis were also utilized.

The lesson makes use of two programming languages: R and Stan. While Stan is compatible with Python, Julia, and several other interfaces, our focus here will be on its integration with R. Make sure you are using the latest version of R. It's also recommended to use RStudio, an integrated development environment, that makes using R easier. 

Instructions for installing Stan can be found [here](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started). 

For the lessons, you'll need to have several R packages installed and loaded. Run the following code to load the libraries and to set the graphical theme. 

```R
package_list <- c("magrittr", "tidyverse", "cowplot", "grafify", "rstan", "mvtnorm")

for (p in package_list){
     if(! p %in% package_list()){
         package_list(i, dependencies = TRUE)
     }
     require(i)
}

theme_set(theme_bw(15))
prior_color <- "#009E73"
likelihood_color <- "#E69F00"
posterior_color <- "#56B4E9"
```

## Data Sets

The data sets used in the lesson can be accessed  [here](https://github.com/carpentries-incubator/statistical-probabilistic-programming-r/tree/main/episodes/data) where you can find them packaged in  `lesson-data.zip`.

<!--
FIXME: place any data you want learners to use in `episodes/data` and then use
       a relative link ( [data zip file](data/lesson-data.zip) ) to provide a
       link to it, replacing the example.com link.
-->



