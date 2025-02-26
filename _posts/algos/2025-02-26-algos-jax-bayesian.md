---
title: JAX - Bayesian Inference with Hamiltonian Monte Carlo for Financial Time Series
date: 2025-02-26
tags:
  - ml
  - finance
  - jax
---

**Bayesian Inference with Hamiltonian Monte Carlo for Financial Time Series**

### Introduction
Financial time series often exhibit complex dynamics, making them difficult to model using traditional statistical methods. Bayesian inference provides a probabilistic framework for estimating model parameters while incorporating prior knowledge. However, many Bayesian models rely on sampling techniques such as Markov Chain Monte Carlo (MCMC), which can be computationally expensive. Hamiltonian Monte Carlo (HMC) is an advanced MCMC method that leverages Hamiltonian dynamics to improve sampling efficiency, making it well-suited for Bayesian modeling of financial data. In this post, we explore how to apply HMC using JAX to estimate parameters in a financial time series model.

### The Traditional Approach to Bayesian Inference
The standard approach to Bayesian inference involves defining a likelihood function for the observed data, specifying prior distributions for model parameters, and using MCMC to generate samples from the posterior distribution. The posterior is given by Bayes’ theorem:

\\[
P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)}
\\]

where:
- \\( P\left( \theta \mid D \right) \\) is the posterior distribution of parameters \\( \theta \\) given data \\( D \\),
- \\( P\left( D \mid \theta \right) \\) is the likelihood of the data given parameters,
- \\( P\left( \theta \right) \\) is the prior distribution of parameters,
- \\( P\left( D \right) \\) is the evidence (a normalizing constant).

The primary challenge is efficiently sampling from \\( P(\theta \mid D) \\).
Traditional MCMC methods, such as Metropolis-Hastings, struggle with high-dimensional parameter spaces and correlated parameters. HMC addresses these challenges by incorporating gradient information to guide the sampling process.

### How Hamiltonian Monte Carlo Works
HMC improves sampling by treating parameter estimation as a physical system evolving under Hamiltonian dynamics. It introduces auxiliary momentum variables \\( p \\) and defines a Hamiltonian function:

\\[
H(\theta, p) = U(\theta) + K(p)
\\]

where:
- \\( U\left( \theta \right) = -\log P\left( D \mid \theta \right) - \log P\left( \theta \right) \\) is the potential energy (negative log-posterior),
- \\( K\left( p \right) = \frac{1}{2} p^T M^{-1} p \\) is the kinetic energy (typically Gaussian distributed),
- \\( M \\) is a mass matrix that determines the momentum scaling.

HMC samples from the posterior by simulating Hamiltonian dynamics using numerical integrators such as the leapfrog method, making long-distance moves in the parameter space while maintaining high acceptance rates.

### Implementing HMC from Scratch in JAX
Instead of using NumPyro’s built-in HMC implementation, we implement HMC from scratch to understand its mechanics fully.

#### **Step 1: Define the Leapfrog Integrator**
```python
import jax.numpy as jnp
from jax import grad

def leapfrog(theta, p, step_size, num_steps, potential_fn):
    p -= 0.5 * step_size * grad(potential_fn)(theta)
    for _ in range(num_steps - 1):
        theta += step_size * p
        p -= step_size * grad(potential_fn)(theta)
    theta += step_size * p
    p -= 0.5 * step_size * grad(potential_fn)(theta)
    return theta, -p
```

The leapfrog integration, in numerical analysis, is a method for numerically integrating differential equations of the form
\\[
\displaystyle {\ddot {x}}={\frac {d^{2}x}{dt^{2}}}=A(x)
\\]

#### **Step 2: Define the Hamiltonian and Acceptance Step**
```python
def hamiltonian(theta, p, potential_fn):
    kinetic = 0.5 * jnp.sum(p ** 2)
    potential = potential_fn(theta)
    return kinetic + potential

def metropolis_hastings(theta_new, p_new, theta, p, potential_fn, key):
    h_new = hamiltonian(theta_new, p_new, potential_fn)
    h_old = hamiltonian(theta, p, potential_fn)
    accept_prob = jnp.exp(h_old - h_new)
    return jnp.where(jax.random.uniform(key) < accept_prob, theta_new, theta)
```

#### **Step 3: Define the HMC Sampler**
```python
def hmc_sampler(potential_fn, theta_init, num_samples, step_size, num_steps, key):
    samples = []
    theta = theta_init
    for i in range(num_samples):
        key, subkey = jax.random.split(key)
        p = jax.random.normal(subkey, shape=theta.shape)
        theta_new, p_new = leapfrog(theta, p, step_size, num_steps, potential_fn)
        theta = metropolis_hastings(theta_new, p_new, theta, p, potential_fn, subkey)
        samples.append(theta)
    return jnp.array(samples)
```

### Bayesian Time Series Model
We now define a stochastic volatility model and apply our HMC implementation.

#### **Step 4: Define the Model and Potential Function**
```python
def potential_fn(theta):
    mu, phi, tau = theta
    prior = -dist.Normal(0, 1).log_prob(mu) - dist.Beta(5, 1).log_prob(phi) - dist.HalfNormal(0.5).log_prob(tau)
    return prior
```

```python
import yfinance as yf

data = yf.download("SPY", start="2020-01-01", end="2023-01-01")
returns = jnp.log(data["Adj Close"]).diff().dropna().values

key = jax.random.PRNGKey(0)
theta_init = jnp.array([0.0, 0.9, 0.1])
samples = hmc_sampler(potential_fn, theta_init, num_samples=1000, step_size=0.01, num_steps=10, key)
```

