---
title: "Quantitative Training (Solutions) - Month 1"
date: 2025-02-25
tags: monthly-solutions
---

<!--more-->

See the original post
at [Quantitative Training - Month 1]({{ site.baseurl }}/2025/02/25/problemset-month-01/).

## Part 1: Fundamental Exercises

### Exercise 1: Differentiation in Option Pricing

<details>
  <summary>Solution</summary>
  The Black-Scholes price for a European call option is given by:

\[
C = S_0 N(d_1) - K e^{-rT} N(d_2),
\]

where \( d_1 \) and \( d_2 \) are:

\[
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T}.
\]

To compute \( \frac{\partial C}{\partial S_0} \), we apply the chain rule to
differentiate the equation for \( C \):

\[
\frac{\partial C}{\partial S_0} = \frac{\partial}{\partial S_0} \left( S_0 N(
d_1) - K e^{-rT} N(d_2) \right)
\]

Using the derivative of \( N(d_1) \) with respect to \( S_0 \):

\[
\frac{\partial C}{\partial S_0} = N(d_1) + S_0 \frac{\partial N(d_1)}{\partial
S_0} - 0
\]

where \( \frac{\partial N(d_1)}{\partial S_0} = \frac{1}{\sigma \sqrt{T}}
e^{-d_1^2/2} \), the partial derivative involves both the normal cumulative
distribution and the derivative of the standard normal probability.

**Interpretation:** The partial derivative \( \frac{\partial C}{\partial S_0} \)
represents the sensitivity of the option price to changes in the underlying
asset price, which is the **delta** of the option. A higher delta means the
option's price is more sensitive to movements in the underlying asset.
</details>

### Exercise 2: Integration in Continuous Compounding

<details>
  <summary>Solution</summary>
  To calculate the present value \( PV \) of a bond with continuous compounding
  that pays \$1000 in 5 years, we use the formula:

\[
PV = 1000 e^{-rT}
\]

Given \( r = 5\% = 0.05 \) and \( T = 5 \) years, we get:

\[
PV = 1000 e^{-0.05 \times 5} = 1000 e^{-0.25} \approx 1000 \times 0.7788 = 778.8
\]

So, the present value of the bond is approximately \$778.80.
</details>

### Exercise 3: Eigenvalues in Portfolio Optimization

<details>
  <summary>Solution</summary>
  Consider the covariance matrix \( \Sigma \):

\[
\Sigma = \begin{bmatrix}
0.04 & 0.02 \\
0.02 & 0.03
\end{bmatrix}.
\]

To find the eigenvalues \( \lambda \), solve the characteristic equation \(
\text{det}(\Sigma - \lambda I) = 0 \):

\[
\text{det}\left( \begin{bmatrix} 0.04 & 0.02 \\ 0.02 & 0.03 \end{bmatrix} -
\lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) = 0
\]

This simplifies to:

\[
\text{det}\left( \begin{bmatrix} 0.04 - \lambda & 0.02 \\ 0.02 & 0.03 - \lambda
\end{bmatrix} \right) = 0
\]

Expanding the determinant:

\[
(0.04 - \lambda)(0.03 - \lambda) - 0.02^2 = 0
\]

Solving this quadratic equation gives the eigenvalues \( \lambda_1 = 0.05 \)
and \( \lambda_2 = 0.02 \).

**Significance:** The eigenvalues represent the variance along the principal
axes of the asset returns. The eigenvectors give the directions of maximum
variance, which are critical for risk factor analysis in portfolio optimization.
</details>

### Exercise 4: Constrained Optimization in Portfolio Theory

<details>
  <summary>Solution</summary>
  Using Lagrange multipliers, we need to maximize the utility function subject to
  constraints:

\[
L(w_1, w_2, \lambda) = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2w_1w_2 \sigma_
{12} + \lambda(w_1 + w_2 - 1)
\]

The optimal portfolio weights are obtained by differentiating the Lagrange
function with respect to \( w_1, w_2, \) and \( \lambda \), and setting the
derivatives equal to zero. Solving these will yield the weights that minimize
risk under the given constraints.
</details>

### Exercise 5: Probability in Asset Returns

<details>
  <summary>Solution</summary>
  Given that daily stock returns follow a normal distribution with \( \mu =
  0.001 \) and \( \sigma = 0.02 \), we calculate the probability that the return
  is below \( -2\% \).

The Z-score for \( -0.02 \) is:

\[
Z = \frac{-0.02 - 0.001}{0.02} = \frac{-0.021}{0.02} = -1.05
\]

Using the standard normal table, the cumulative probability for \( Z = -1.05 \)
is approximately 0.146.

Thus, the probability of a return below \( -2\% \) is approximately **14.6%**.
</details>

<!-- ADVANCED EXERCISES -->

## Part 2: Advanced Exercises

### Exercise 6: Stochastic Processes in Stock Price Modeling

<details>
  <summary>Solution</summary>
  <p>
  <b>Problem Recap:</b>
  Given that stock prices follow a geometric Brownian motion:
  \[
  dS_t = \mu S_t \, dt + \sigma S_t \, dW_t,
  \]
  derive the expected value of \( S_T \) given \( S_0 \) at \( t=0 \).
  </p>
<p>
<b>Solution Outline:</b>
The solution uses the fact that the stochastic differential equation (SDE) for
geometric Brownian motion has the explicit solution
\[
S_T = S_0 \exp\left\{\left(\mu - \frac{1}{2}\sigma^2\right)T + \sigma
W_T\right\}.
\]
Taking the expectation, and noting that \( W_T \) is normally distributed with
mean 0 and variance \( T \), we use the moment generating function of the normal
distribution:
\[
E\left[e^{\sigma W_T}\right] = e^{\frac{1}{2}\sigma^2 T}.
\]
Thus,
\[
E[S_T] = S_0 e^{\left(\mu - \frac{1}{2}\sigma^2\right)T} e^{\frac{1}{2}\sigma^2
T} = S_0 e^{\mu T}.
\]
</p>
</details>

### Exercise 7: Hypothesis Testing in Market Efficiency

<details>
  <summary>Solution</summary>
  <p>
  <b>Problem Recap:</b>
  A hedge fund claims to have developed a trading strategy that consistently outperforms the market. Design a statistical hypothesis test to evaluate whether their results are statistically significant.
  </p>

<p>
<b>Solution Outline:</b>
<ol>
<li>
<p><strong>Define Hypotheses:</strong></p>
<ul>
<li><p>( H_0 ): The hedge fund&#39;s strategy does not outperform the market (any
observed outperformance is due to chance).</p></li>
<li><p>( H_1 ): The strategy does outperform the market.</p></li>
</ul>
</li>
<li><p><strong>Choose a Test:</strong>
A common choice is a t-test (either one-sample or two-sample, depending on
available data) comparing the mean excess return to zero or to the market
return.</p></li>
<li>
<p><strong>Test Procedure:</strong></p>
<ul>
<li><p>Gather a sample of returns from the hedge fund strategy.</p></li>
<li>Compute the sample mean and standard deviation.</li>
<li>Calculate the test statistic and corresponding p-value.</li>
<li>Compare the p-value against a chosen significance level (e.g., 0.05).</li>
</ul>
</li>
</ol>
</p>
</details>

### Exercise 8: GARCH Modeling in Volatility

<details>
  <summary>Solution</summary>
  <p><strong>Problem Recap:</strong></p>
  <p>Consider the GARCH(1,1) model:</p>
  <p>
    \[
    \sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2.
    \]
  </p>
  <p>Explain the procedure to estimate \( \alpha_0 \), \( \alpha_1 \), and \( \beta_1 \) using maximum likelihood estimation.</p>

  <p><strong>Solution Outline:</strong></p>
  <ol>
    <li><strong>Specify the Model:</strong> Assume that the returns (or errors \( \epsilon_t \)) are conditionally normally distributed given past information.</li>
    <li><strong>Write the Likelihood Function:</strong> The likelihood for the observed series is the product over time of the normal density:
      <p>
        \[
        L(\alpha_0, \alpha_1, \beta_1) = \prod_{t=1}^{T} \frac{1}{\sqrt{2 \pi \sigma_t^2}} \exp\left(-\frac{\epsilon_t^2}{2\sigma_t^2}\right).
        \]
      </p>
    </li>
    <li><strong>Log-Likelihood:</strong> Take the logarithm to simplify the product into a sum:
      <p>
        \[
        \ln L = -\frac{1}{2} \sum_{t=1}^{T} \left( \ln(2\pi \sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2} \right).
        \]
      </p>
    </li>
    <li><strong>Numerical Optimization:</strong> Use numerical methods (such as the BFGS algorithm) to maximize the log-likelihood function with respect to the parameters, subject to the constraints (e.g., \( \alpha_0 > 0 \), \( \alpha_1, \beta_1 \ge 0 \), and \( \alpha_1 + \beta_1 < 1 \)).</li>
  </ol>
  Or, compute the maximum likelihood estimates for the GARCH(1,1) model using an analytical approach.
  This involves taking the first-order conditions of the log-likelihood function and solving the resulting system of equations.

  The analytical solution for the GARCH(1,1) model is:
  \[
  \begin{aligned}
  \alpha_0 &= \frac{\hat{\sigma}^2}{1 - \hat{\alpha_1} - \hat{\beta_1}}, \\
  \alpha_1 &= \frac{\sum_{t=1}^{T} \epsilon_{t-1}^2}{\sum_{t=1}^{T} \hat{\sigma}_t^2}, \\
  \beta_1 &= \frac{\sum_{t=1}^{T} \hat{\sigma}_{t-1}^2}{\sum_{t=1}^{T} \hat{\sigma}_t^2},
  \end{aligned}
  \]
  where \( \hat{\sigma}_t^2 \) is the estimated conditional variance at time \( t \).
</details>

### Exercise 9: Monte Carlo Simulation in Option Pricing

<details>
  <summary>Solution</summary>
  <p><strong>Problem Recap:</strong></p>
  <p>Simulate the price of a European call option using Monte Carlo methods with the parameters: \( S_0 = 100 \), \( K = 100 \), \( r = 5\% \), \( \sigma = 20\% \), and \( T = 1 \) year, performing 10,000 simulations.</p>

  <p><strong>Solution Outline:</strong></p>
  <ol>
    <li><strong>Generate Simulated Stock Prices:</strong> For each simulation, use the formula for geometric Brownian motion:
      <p>
        \[
        S_T = S_0 \exp\left\{\left(r - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z\right\},
        \]
        where \( Z \sim N(0,1) \).
      </p>
    </li>
    <li><strong>Compute Payoffs:</strong> For each simulated \( S_T \), calculate the option payoff:
      <p>
        \[
        \text{Payoff} = \max(S_T - K, 0).
        \]
      </p>
    </li>
    <li><strong>Discount Payoffs:</strong> Average the payoffs across all simulations and discount the average back to the present:
      <p>
        \[
        C = e^{-rT} \times \text{Average Payoff}.
        \]
      </p>
    </li>
  </ol>
</details>

### Exercise 10: Markov Chains in Credit Ratings
<details>
  <summary>Solution</summary>
  <p><strong>Problem Recap:</strong></p>
  <p>A company's credit rating follows a Markov process with the transition matrix:</p>
  <p>
    \[
    P = \begin{bmatrix}
    0.9 & 0.1 & 0 \\
    0.05 & 0.9 & 0.05 \\
    0 & 0.1 & 0.9
    \end{bmatrix}.
    \]
  </p>
  <p>If the company starts with a rating of 2, determine the probability of being in rating 1 after two transitions.</p>

  <p><strong>Solution Outline:</strong></p>
  <ol>
    <li><strong>Initial State Vector:</strong> Represent the starting condition as a vector. For a starting rating of 2 (assuming states 1, 2, and 3 correspond to positions in the vector), use:</li>
  </ol>
</details>
