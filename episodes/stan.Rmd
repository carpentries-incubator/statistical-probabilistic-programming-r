---
title: 'Stan'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- What is Stan?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Learn how to specify models in Stan
- How to generate samples with Stan
- How to process samples generated with Stan 

::::::::::::::::::::::::::::::::::::::::::::::::

Stan is a high-level programming language used for statistical modeling and computation. It provides a flexible and efficient means the for inference of Bayesian models. The syntax of Stan if user-friendly which simplifies the implementation of complex statistical models and makes the language relatively approachable. 


Stan generates samples from the posterior of a defined statistical model utilizing Markov Chain Monte Carlo (MCMC) sampling, and more specifically its the Hamiltonian Monte Carlo variant which is a highly effective variant of MCMC. 


You can fit model that have continuous parameters. Models with discrete parameters such as classification models are typically impossible to fit, although some workarounds have been implemented. 


## Stan installation

Follow the instruction at https://mc-stan.org/users/interfaces/ to install Stan. 

## Basic program structure

- Write the program in a separate text file

- Call Stan from R, command line, or several other languages. This creates a collection of posterior samples. 

- Analyse the posterior samples with tools presented in the previous episode. 

## Example 1: Binomial model
## Example 2: normal model
## Example 3: Linear regression

::::::::::::::::::::::::::::::::::::: keypoints 

- Stan is...

::::::::::::::::::::::::::::::::::::::::::::::::

