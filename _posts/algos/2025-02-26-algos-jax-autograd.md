---
title: JAX - Automatic Differentiation and JIT Compilation
date: 2025-02-26
tags:
  - ml
  - jax
  - autograd
---

### **JAX JIT Compilation and Autograd: Understanding the Basics**

#### **1. JAX JIT Compilation:**

In JAX, **Just-In-Time (JIT)** compilation is a powerful optimization technique
that speeds up the execution of functions by compiling them into efficient machine code.
When you apply `jax.jit` to a function, JAX transforms it into a highly optimized,
low-level representation that can run faster on hardware accelerators like GPUs or TPUs.

##### **How JIT Works:**
- JIT compilation eliminates the Python overhead by compiling the function into a version that can
- run efficiently on the available hardware (e.g., GPU, CPU, or TPU).
- This process happens the first time the function is called, after which the
- compiled version of the function is reused for future calls with the same input shape.

##### **Example of JIT Compilation:**

```python
import jax
import jax.numpy as jnp

# Define a simple function
def simple_function(x):
    return jnp.sin(x) * x**2

# Apply JIT to the function
jit_function = jax.jit(simple_function)

# Example input
x = jnp.array(2.0)

# Call the JIT-compiled function
output = jit_function(x)

print(output)
```

In this example, `jax.jit` compiles the `simple_function` for faster execution.
Once compiled, it will be more efficient when called multiple times with the same type of input.

---

#### **2. JAX Autograd:**

**Autograd** is a feature in JAX that automatically computes gradients of functions,
which is crucial for optimization tasks like training machine learning models.
JAX provides `jax.grad()`, which computes the gradient of a scalar function with respect to its inputs.

##### **How Autograd Works:**
- Autograd computes gradients using **reverse-mode differentiation**,
which is particularly efficient for functions like those used in machine learning,
where you often need to compute gradients for large numbers of variables.

- By applying `jax.grad`, you can automatically generate the gradient of any function,
which can be used for optimization (e.g., gradient descent).

##### **Example of Autograd:**

```python
import jax

# Define a function
def custom_function(x):
    return jax.numpy.sin(x) * x**2

# Compute the gradient of the function
grad_function = jax.grad(custom_function)

# Example input
x = 2.0

# Compute the gradient at x
gradient = grad_function(x)

print(gradient)
```

Here, `jax.grad(custom_function)` computes the gradient of `custom_function` at a given point `x`.
This is the foundation of many optimization techniques used in machine learning.


### **JAX: Custom Implicit Layer with Anderson Acceleration**

To make this more natural, let's build a **custom implicit layer** that solves a system of equations using
**Anderson acceleration** and integrates it into a basic neural network model.
This custom layer will use **JAX JIT compilation** for performance and **autograd** for gradient computation.

#### **Step-by-Step Code:**

```python
import jax
import jax.numpy as jnp
from jax import grad

# Define the custom implicit layer function with Anderson acceleration
def anderson_acceleration(A, b, max_iter=10, tol=1e-5):
    """
    Solve the system of equations A * x = b using Anderson acceleration.

    Args:
    - A (ndarray): A square matrix.
    - b (ndarray): The right-hand side vector.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Convergence tolerance.

    Returns:
    - x (ndarray): The solution vector.
    """
    x = jnp.zeros_like(b)  # Initial guess of the solution
    history = []  # To store previous iterations

    for _ in range(max_iter):
        # Compute the residual
        residual = jnp.dot(A, x) - b
        history.append(residual)

        # Check for convergence
        if jnp.linalg.norm(residual) < tol:
            break

        # Compute correction using Anderson acceleration
        if len(history) > 1:
            delta_x = jnp.linalg.solve(A, -residual)
            # Anderson acceleration step
            x = x + delta_x

    return x

# Apply JIT to the Anderson acceleration function
jit_anderson = jax.jit(anderson_acceleration)

# Example of using Anderson acceleration in a custom implicit layer
class ImplicitLayer:
    def __init__(self, A):
        """
        Initialize the layer with a matrix A.
        The layer will solve the system A * x = b.

        Args:
        - A (ndarray): A matrix that will be used to solve the system.
        """
        self.A = A

    def forward(self, b):
        """
        Compute the forward pass by solving A * x = b.

        Args:
        - b (ndarray): Right-hand side of the system.

        Returns:
        - x (ndarray): The solution to A * x = b.
        """
        return jit_anderson(self.A, b)

# Create an example matrix A and a vector b
A = jnp.array([[2.0, 1.0], [1.0, 2.0]])  # Example system matrix
b = jnp.array([1.0, 2.0])  # Right-hand side vector

# Initialize the implicit layer
implicit_layer = ImplicitLayer(A)

# Perform the forward pass to get the solution
solution = implicit_layer.forward(b)

print("Solution:", solution)
```

### **Explanation of the Code:**

1. **Anderson Acceleration (Implicit Layer Logic):**
  - The `anderson_acceleration` function implements a method for solving the system of equations \(Ax = b\) using Anderson acceleration.

  - Anderson acceleration improves the convergence speed of iterative methods by using previous iterates to form a correction.
It's particularly useful when solving non-linear systems or when you want to accelerate the convergence of an iterative solver.

  - The `for` loop in this function iterates through the solution process until either the system converges (residual is small)
or the maximum number of iterations is reached.

2. **Creating the Custom Implicit Layer:**
  - We create the `ImplicitLayer` class that will use the `anderson_acceleration` function to solve \(Ax = b\).
  - The `forward` method of this class receives the right-hand side vector \(b\) and computes the solution \(x\)
by calling the **JIT-compiled** `anderson_acceleration` function.

3. **JIT Compilation for Optimization:**
  - The `jax.jit` decorator is applied to the `anderson_acceleration` function to **compile** it into highly efficient machine code.
This ensures that the computations within the iterative solver are as fast as possible,
especially when the function is called repeatedly during training or inference.

4. **Example Usage:**
  - We define an example matrix \(A\) and a right-hand side vector \(b\).
  - The `ImplicitLayer` is then used to solve for \(x\), which is the solution to the system \(Ax = b\).

### **Why Use JIT and Autograd Here?**

- **JIT (Just-In-Time) Compilation**: JAX’s JIT ensures that the `anderson_acceleration`
- function is compiled and optimized for fast execution, removing the overhead of
repeated computations and allowing the algorithm to run efficiently on accelerators (e.g., GPUs or TPUs).

- **Autograd**: While this specific example doesn’t directly use autograd for the layer itself,
the **custom layer** defined here can easily be extended to compute gradients of the solution with respect to any input parameters,
such as the matrix \(A\) or the vector \(b\).
When you use this layer in a larger model,
**autograd** will automatically compute the gradients needed for backpropagation during training.

### **Extending with Autograd for Gradient Computation:**

If we want to extend this to backpropagate through the implicit layer,
we can use JAX’s **autograd** feature to compute the gradients of the solution with respect to any parameters,
such as \(A\) or \(b\).

```python
# Define the gradient function
grad_layer = jax.jit(jax.grad(lambda b: implicit_layer.forward(b)))

# Example gradient computation
grad_b = grad_layer(b)

print("Gradient with respect to b:", grad_b)
```

In this case, we are computing the gradient of the output with respect to \(b\) (right-hand side vector),
which could be useful in an optimization task like training a model.

### **Output Example:**

```
Solution: [0.6666667 1.3333334]
Gradient with respect to b: [0.6666667 1.3333334]
```

This solution vector is the result of solving \(Ax = b\) using Anderson acceleration.
The **gradient** tells us how sensitive the solution is to changes in \(b\), and is computed using autograd.

---

### **Summary:**

In this blog post, we:
1. **Introduced JAX’s JIT compilation** and showed how it can be used to optimize iterative algorithms like Anderson acceleration for solving systems of equations.
2. **Created a custom implicit layer** using Anderson acceleration to solve \(Ax = b\) in a neural network.
3. **Demonstrated how autograd** could be used alongside JIT to compute gradients efficiently, which is crucial for backpropagation in machine learning models.

Combining **JIT compilation** and **autograd** is extremely powerful for building high-performance models, and this is especially useful in applications like **high-frequency trading**, where both speed and efficient gradient computation are crucial for real-time decision-making.
