---
title: "Algos - Stochastic Programming"
date: 2025-02-25
tags: algos
---

# Guaranteeing Minimum Wealth via Two-Stage Stochastic Programming in Finance

In many real-world financial settings—such as asset–liability management for pension funds or risk management for hedge funds—
a portfolio manager must ensure that the portfolio’s value never falls below a critical threshold, which also helps with reducing liquidity risk.
If it does, extra capital must be injected at a cost.
This two-stage decision problem, where the first stage involves choosing a portfolio allocation and
the second stage deals with corrective actions under uncertainty, is a perfect example to explain stochastic programming in
a more practical context (which is the aim of the Algos blog series).

Below, we build a model using JAX that minimizes the expected cost of capital injections.
We also explain the mathematical underpinnings of the code,
including why the reparameterization works and a sketch of the convergence proof under standard assumptions.

## Step 1: Setting Up the Environment

We begin by importing the necessary libraries and setting a random seed.
This is just for reproducibility—ensuring that our stochastic simulations are consistent every time we run the code.

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax  # Optimizers for JAX
import numpy as np

# Set a random seed for reproducibility.
key = jax.random.PRNGKey(0)
```

Computationally, we load JAX and its NumPy wrapper, along with tools for differentiation and optimization.
Setting the PRNG key guarantees that our stochastic sampling (used later to simulate asset returns) produces the same results across runs.

## Step 2: Generating Asset Return Scenarios

Imagine a small portfolio consisting of four assets.
We model the uncertainty in their returns by assuming that they follow a multivariate normal distribution.
We specify the expected returns and standard deviations for each asset and then sample 1,000 possible scenarios.

The equations for the expected returns and standard deviations are:

$$
\mu = \begin{bmatrix} 0.001 & 0.002 & 0.0015 & 0.0005 \end{bmatrix}, \quad
\Sigma = \begin{bmatrix} 0.01^2 & 0 & 0 & 0 \\ 0 & 0.015^2 & 0 & 0 \\ 0 & 0 & 0.012^2 & 0 \\ 0 & 0 & 0 & 0.008^2 \end{bmatrix}.
$$

```python
n_assets = 4
n_scenarios = 1000  # Number of possible future scenarios

mu = jnp.array([0.001, 0.002, 0.0015, 0.0005])
sigma = jnp.array([0.01, 0.015, 0.012, 0.008])

# Assume asset returns are independent, so the covariance matrix is diagonal.
Sigma = jnp.diag(sigma ** 2)

def sample_returns(key, mu, Sigma, n):
    return jax.random.multivariate_normal(key, mu, Sigma, (n,))  # Sample returns from a multivariate normal distribution.

key, subkey = jax.random.split(key)
r_scenarios = sample_returns(subkey, mu, Sigma, n_scenarios)
```

We assume that asset returns \\( r
\\) are distributed as
\\[
r \sim \mathcal{N}(\mu, \Sigma),
\\]
where \\(\mu \in \mathbb{R}^{4}
\\) is the vector of expected returns and \\(\Sigma
\\) is the diagonal covariance matrix.
This step is fundamental because the 1,000 sampled scenarios represent the uncertainty that the portfolio manager faces.

If this assumption is unclear, I suggest reading about the multivariate normal distribution and stylized facts of asset returns in finance,
depending on which part you find challenging.


## Step 3: Defining the Recourse Cost Function

Once the asset returns are realized, the portfolio’s wealth will be calculated.
If the portfolio wealth is below our target, the manager must inject extra capital, which incurs a cost.
The following code defines the recourse cost for each scenario.

```python
target = 1.02  # Our target is to achieve at least a 2% gain.
c = 1.05       # Cost factor: each unit of capital injection costs 5% extra.

# For a given portfolio x and asset return vector r, compute the recourse cost.
def recourse_cost(x, r):
    wealth = 1.0 + jnp.dot(r, x)  # Portfolio wealth: initial capital (1) plus returns.
    injection = jnp.maximum(0.0, target - wealth)  # Shortfall if wealth < target.
    return c * injection  # Total cost incurred.

v_recourse_cost = jax.vmap(lambda r, x: recourse_cost(x, r), in_axes=(0, None))
```

Computationally, the function `recourse_cost` calculates the shortfall when the portfolio wealth \\( W = 1 + r^\top x
\\) is less than the target wealth.


The function is then vectorized using `jax.vmap` to efficiently compute the cost across all scenarios simultaneously.
Mathematically, this corresponds to evaluating

\\[
y(r) = \max\{0, W_{\text{target}} - (1 + r^\top x)\},
\\]
and the cost is given by \\( c \cdot y(r)
\\).
The convexity of the \\(\max \\) function ensures that our objective will be convex under certain conditions.

To read more about vectorization and the `vmap` function,
I recommend the JAX documentation or the book "Neural Networks and Deep Learning" by Charu Aggarwal.

If you prefer a simpler explanation, see our post on "Vectorizing high-dimensional functions with JAX."


## Step 4: Building the Objective Function

Our aim is to choose portfolio weights that minimize the expected cost of capital injection. However, portfolio weights must lie in the probability simplex (nonnegative and summing to one). We address this constraint by reparameterizing using the softmax function.

```python
# Define the softmax function to map an unconstrained vector to the probability simplex.
def softmax(z):
    exp_z = jnp.exp(z - jnp.max(z))
    return exp_z / jnp.sum(exp_z)

# Define the overall objective: the expected recourse cost.
def objective(z, r_scenarios):
    x = softmax(z)  # Convert unconstrained parameters z into portfolio weights x.
    costs = v_recourse_cost(r_scenarios, x)
    return jnp.mean(costs)
```

By using the softmax transformation, we ensure that the portfolio weights \\( x \\) satisfy \\( x_i \ge 0 \\) and \\( \sum_i x_i = 1 \\).
The objective function is then the expected recourse cost, mathematically expressed as:

$$
f(x) = \mathbb{E}_{r}\left[ c \cdot \max\{0, W_{\text{target}} - (1 + r^\top x)\} \right].
$$

This function is convex (since the max function and expectation preserve convexity)
and smooth with respect to the unconstrained parameter \\( z \\) after applying the softmax mapping.



## Step 5: Optimizing the Objective with Optax

Now that our objective function is defined, we minimize it by optimizing over \\( z \\) using a gradient-based method.
We use the Adam optimizer from Optax, which adapts the learning rate during training.

If you would like to learn more about the Adam optimizer, I recommend the original paper by Kingma and Ba, "Adam: A Method for Stochastic Optimization."
Or, if you feel like diving into the math behind the optimizer, check out the blog post "Understanding the Mathematics of Optimization using JAX."

```python
z_init = jnp.zeros(n_assets)  # unconstrained parameter vector z

optimizer = optax.adam(learning_rate=0.1)

def init_state(params):
    opt_state = optimizer.init(params)
    return params, opt_state

params, opt_state = init_state(z_init)

# Single optimization step with just-in-time compilation for efficiency.
@jit
def step(params, opt_state, r_scenarios):
    loss, grads = value_and_grad(objective)(params, r_scenarios)  # Compute loss and gradients.
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```


Here we initialize \\( z \\) as a zero vector, so the initial portfolio is uniform.
The function `step` computes both the loss and the gradient of our objective with respect to \\( z \\) using JAX’s automatic differentiation.
The optimizer (Adam) updates \\( z \\) based on these gradients.
JIT compilation speeds up this repetitive process.
The convergence of gradient descent methods in convex optimization is well understood:
under Lipschitz continuity and convexity assumptions,
gradient descent converges at a rate of \\( \mathcal{O}(1/k)\\).
Although we use Adam here, recent theoretical work supports its convergence in convex settings with proper hyperparameter choices.


## Step 6: Running the Optimization Loop

Finally, we run the optimization loop for a set number of iterations.
In each iteration, the parameters are updated, and we monitor the loss periodically.
At the end of the optimization, the final portfolio weights are obtained by applying the softmax function to the optimized \\( z \\).

```python
n_steps = 500
loss_history = []

params_current = params
for i in range(n_steps):
    params_current, opt_state, loss_val = step(params_current, opt_state, r_scenarios)
    loss_history.append(loss_val)
    if i % 50 == 0:
        print(f"Step {i}, Loss: {loss_val:.6f}")

x_opt = softmax(params_current)
print("\nOptimal portfolio weights:", x_opt)
```


This loop represents the iterative process of minimizing the expected recourse cost.
At every step, our optimizer adjusts \\( z \\) to decrease the loss:

$$
F(z) = f(\operatorname{softmax}(z)) = \mathbb{E}_{r}\left[ c \cdot \max\{0, W_{\text{target}} - (1 + r^\top \operatorname{softmax}(z))\} \right].
$$

The convergence of our gradient-based method is assured under convexity and smoothness assumptions (with further improvements provided by Adam).
Once the loop completes, the softmax transformation provides the optimal portfolio weights \\( x^*\\).

## Rationale and Short Convergence Proof

### Why Does This Work?

Our two-stage stochastic programming problem is formulated as:

$$
\min_{x \in \Delta} f(x) = \mathbb{E}_{r}\left[ c \cdot \max\{0, W_{\text{target}} - (1 + r^\top x)\} \right],
$$

where \\(\Delta
\\) is the probability simplex and \\( f(x) \\) is the expected recourse cost.

To handle the constraint \\( x \in \Delta \\) smoothly, we reparameterize using:

\\[
x = \operatorname{softmax}(z).
\\]

This gives us a new optimization problem:

\\[
\min_{z \in \mathbb{R}^n} F(z) = f(\operatorname{softmax}(z)).
\\]

Even though the softmax is not a convex mapping, if \\( f(x)
\\) is convex over \\(\Delta \\), then \\( F(z) \\) is smooth and can be minimized using gradient descent.

Under standard assumptions—if the gradient \\( \nabla F(z) \\) is \\( L \\)-Lipschitz continuous—the gradient descent update

\\[
z_{k+1} = z_k - \alpha \nabla F(z_k)
\\]

with a step size \\( \alpha \leq 1/L
\\) guarantees convergence at a rate:

\\[
F(z_k) - F(z^*) \leq \frac{L \|z_0 - z^*\|^2}{2k},
\\]

where \\( z^* \\) is an optimal solution. In our code, we use Adam, which is known to converge under similar conditions when the objective is convex.

If these assumptions are unclear, I recommend reading about convex optimization and the convergence of gradient-based methods.
For a more detailed explanation, see the book "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe.

A simple way to understand this is that we are optimizing a convex function over a convex set, and the softmax transformation ensures that the constraints are satisfied.
We are, after all transformations, still just minimizing a convex function over a convex set using gradient-based methods.

### Convergence Proof Sketch

1. **Smoothness and Convexity:**
We assume that \\( f(x) \\) is convex over the simplex and that the composite function \\( F(z) = f(\operatorname{softmax}(z)) \\) is smooth.
This assumption is justified because the max function is convex, the dot product is linear, and the expectation preserves convexity.

2. **Gradient Descent Convergence:**
   For a function with an \\( L \\)-Lipschitz continuous gradient, standard theory shows that gradient descent converges as

   $$
   F(z_k) - F(z^*) \leq \frac{L \|z_0 - z^*\|^2}{2k}.
   $$

3. **Adaptive Methods:**
   Although Adam employs adaptive step sizes, under proper conditions and with appropriate hyperparameter tuning,
   its convergence in the convex setting is also guaranteed.

Thus, our iterative updates in the code converge to a (global) minimizer, and by mapping the optimal \\( z^* \\) back via the softmax,
we obtain the optimal portfolio weights \\( x^* \\).



## Conclusion

In this post, we implemented a two-stage stochastic programming model using JAX to ensure that a portfolio achieves a minimum target wealth.
We broke the code into segments, explaining each step both computationally and mathematically.
Our approach begins by generating asset return scenarios and defining a recourse cost function that calculates
the cost incurred if the portfolio falls short of the target.
By reparameterizing the portfolio weights using softmax, we maintain the required constraints while optimizing an unconstrained variable.
We then minimize the expected recourse cost using Adam, and, under standard assumptions, the convergence of this method is guaranteed.

Expected dates for the continuation of this series are as follows:

- **Algos - Stochastic Programming**: February 25, 2025
- **Algos - Stochastic Programming (Part 2)**: March 1, 2025
- **Algos - Stochastic Programming (Part 3)**: March 4, 2025

Expected dates for the other two posts on JAX referenced in this series are as follows:
- **Vectorizing high-dimensional functions with JAX**: February 27, 2025
- **Understanding the Mathematics of Optimization using JAX**: February 28, 2025


Happy coding!
