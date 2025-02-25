---
title: "Quantitative Training - Month 1"
date: 2025-02-25
tags: monthly-problemset
---

## Part 1: Fundamental Exercises {#part-1}

### Exercise 1: Differentiation in Option Pricing {#exercise-1}

The Black-Scholes price for a European call option is given by:

\\[
C = S_0 N(d_1) - K e^{-rT} N(d_2),
\\]

where \\( d_1 \\) and \\( d_2 \\) are:

\\[
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \\
d_2 = d_1 - \sigma\sqrt{T}.
\\]

Compute \\( \frac{\partial C}{\partial S_0} \\) and interpret its meaning in the context of financial Greeks.

### Exercise 2: Integration in Continuous Compounding {#exercise-2}

Calculate the present value of a continuously compounded bond that pays \$1000 in 5 years, given a risk-free rate of \\( r = 5\% \\) per annum.

### Exercise 3: Eigenvalues in Portfolio Optimization {#exercise-3}

Consider the covariance matrix of asset returns:

\\[
\Sigma = \begin{bmatrix}
0.04 & 0.02 \\
0.02 & 0.03
\end{bmatrix}.
\\]

Find the eigenvalues and eigenvectors of \\( \Sigma \\) and discuss their significance in risk factor analysis.

### Exercise 4: Constrained Optimization in Portfolio Theory {#exercise-4}

Using Lagrange multipliers, determine the optimal portfolio weights in a two-asset portfolio under the constraints:

\\[
w_1 + w_2 = 1, \quad E[R_p] = 10\%, \quad \sigma_p = \min.
\\]

### Exercise 5: Probability in Asset Returns {#exercise-5}

Assume daily stock returns follow a normal distribution with a mean \\( \mu = 0.001 \\) and standard deviation \\( \sigma = 0.02 \\). Calculate the probability that a stock will experience a return below \\(-2\%\\) on a given day.

## Part 2: Advanced Exercises {#part-2}

### Exercise 6: Stochastic Processes in Stock Price Modeling {#exercise-6}

Given that stock prices follow a geometric Brownian motion:

\\[
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t,
\\]

derive the expected value of \\( S_T \\) given \\( S_0 \\) at \\( t=0 \\).

### Exercise 7: Hypothesis Testing in Market Efficiency {#exercise-7}

A hedge fund claims to have developed a trading strategy that consistently outperforms the market. Design a statistical hypothesis test to evaluate whether their results are statistically significant.

### Exercise 8: GARCH Modeling in Volatility Forecasting {#exercise-8}

Consider the GARCH(1,1) model:

\\[
\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2.
\\]

Explain the procedure to estimate \\( \alpha_0 \\), \\( \alpha_1 \\), and \\( \beta_1 \\) using maximum likelihood estimation.

### Exercise 9: Monte Carlo Simulation in Option Pricing {#exercise-9}

Simulate the price of a European call option using Monte Carlo methods with the following parameters: \\( S_0 = 100 \\), \\( K = 100 \\), \\( r = 5\% \\), \\( \sigma = 20\% \\), and \\( T = 1 \\) year. Perform 10,000 simulations.

### Exercise 10: Markov Chains in Credit Ratings {#exercise-10}

A company's credit rating follows a Markov process with the transition matrix:

$$
P = \begin{bmatrix}
0.9 & 0.1 & 0 \\\\
0.05 & 0.9 & 0.05 \\\\
0 & 0.1 & 0.9
\end{bmatrix}.
$$

If the company starts with a rating of 2, determine the probability of being in rating 1 after two transitions.


See the solutions
at [(Solutions)]({{ site.baseurl }}/2025/02/25/solutions-month-01).
