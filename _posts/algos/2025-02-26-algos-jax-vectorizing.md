---
title: Vectorizing High-Dimensional Functions with JAX
date: 2025-02-26
tags: ml, jax, algos
---

# Vectorizing High-Dimensional Functions with JAX

If you understand how neural networks work, you know that they are essentially a series of matrix multiplications and non-linear transformations.
In Python, a simple algorithm to perform matrix multiplication might look like this:

```python
import numpy as np

def matrix_multiply(A, B):
    return np.dot(A, B)


# And the forward pass of a neural network might look like this:
for i in range(num_layers):
    input_data = matrix_multiply(input_data, weights[i])
    input_data = non_linear_activation(input_data)
```

Traditionally, looping constructs in Python can easily become bottlenecks, especially when dealing with large datasets or complex models.
The topic of this post, JAX, offers a powerful tool called `vmap` that can help you vectorize these operations and significantly improve performance.

## Understanding Vectorization

Vectorization refers to the process of converting operations that act on a single data point into
operations that act on entire arrays or batches of data simultaneously.
This approach can easily be used to run the code on a GPU or TPU, which are optimized for parallel processing.

## JAX's vmap: Automatic Vectorization

In JAX, the primary tool for vectorization is the `vmap` (vectorized map) function.
It transforms a function that operates on individual inputs into a function that operates on batches of inputs,
eliminating the need for explicit loops.

Consider a simple function that computes the square of a number:

```python
import jax
import jax.numpy as jnp

def square(x):
    return x ** 2
```


To apply this function to a batch of inputs, we can use `vmap`:

```python
# Create a vector of inputs
x = jnp.array([1.0, 2.0, 3.0, 4.0])

# Vectorize the 'square' function
vectorized_square = jax.vmap(square)

# Apply the vectorized function
result = vectorized_square(x)
print(result)  # Output: [ 1.  4.  9. 16.]
```

Here, `vmap` transforms the `square` function to operate over each element of the input array `x` without the need for an explicit loop.

### Vectorizing Functions with Multiple Arguments

`vmap` can also handle functions with multiple inputs. For instance, consider a function that computes the dot product of two vectors:

```python
def dot_product(x, y):
    return jnp.dot(x, y)
```


To compute the dot product across batches of vectors, we can vectorize this function:

```python
# Create batches of vectors
x_batch = jnp.array([[1, 2, 3], [4, 5, 6]])
y_batch = jnp.array([[7, 8, 9], [10, 11, 12]])

# Vectorize the 'dot_product' function
vectorized_dot_product = jax.vmap(dot_product)

# Apply the vectorized function
result = vectorized_dot_product(x_batch, y_batch)
print(result)  # Output: [ 50 122]
```

In this example, `vmap` applies the `dot_product` function across corresponding pairs of vectors in `x_batch` and `y_batch`.

### Controlling Batch Dimensions with `in_axes` and `out_axes`

By default, `vmap` maps over the first axis (axis 0) of the input arrays.
However, you can control which axes to map over using the `in_axes` and `out_axes` parameters.
This is particularly useful when dealing with functions where the batch dimension is not the leading dimension.

```python
def add_matrices(a, b):
    return a + b

a_batch = jnp.ones((3, 2, 2))
b_batch = jnp.ones((3, 2, 2))

# Vectorize the 'add_matrices' function over the first axis
vectorized_add = jax.vmap(add_matrices, in_axes=0, out_axes=0)

result = vectorized_add(a_batch, b_batch)
print(result.shape)

>>> (3, 2, 2)
```


In this case, `in_axes=0` indicates that `vmap` should map over the first axis of both `a_batch` and `b_batch`. The `out_axes=0` ensures that the output maintains the same batching along the first axis.

## Batch Processing in Neural Networks

The true power of `vmap` becomes evident in more complex scenarios, such as batch processing in neural networks or operations on tensors with higher dimensions.

Consider a simple neural network layer defined as:

```python
def linear_layer(weights, biases, inputs):
    return jnp.dot(inputs, weights) + biases
```


To apply this layer across a batch of input data:

```python
weights = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
biases = jnp.array([0.1, 0.2])

inputs = jnp.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])

# Vectorize the linear_layer function over the batch dimension of inputs
vectorized_linear_layer = jax.vmap(linear_layer, in_axes=(None, None, 0))

outputs = vectorized_linear_layer(weights, biases, inputs)
print(outputs)

>>> [[ 1.7  2.3]
     [ 3.9  5.5]
     [ 6.1  8.7]]
```


Here, `in_axes=(None, None, 0)` specifies that `weights` and `biases` are the same for all inputs (i.e., not batched), while `inputs` varies along the first axis.

## Parallelization with pmap

While `vmap` is ideal for vectorizing functions to operate over batches within a single device, JAX also provides the `pmap` (parallel map) function to parallelize computations across multiple devices. This is particularly useful for large-scale computations that can benefit from distributed execution.

Suppose we have a function that scales an array:

```python
def scale_array(x, factor):
    return x * factor
```

To parallelize this function across multiple devices:

```python
import jax

# Create a vector of inputs
x = jnp.array([1.0, 2.0, 3.0, 4.0])

# Parallelize the 'scale_array' function
parallel_scale = jax.pmap(scale_array)

# Apply the parallelized function
result = parallel_scale(x, 2.0)
print(result)

>>> [2. 4. 6. 8.]
```

Seems the same as `vmap`, right? The key difference is that `pmap` distributes the computation across multiple devices,
allowing for parallel execution.
Consider a scenario where we have multiple devices available,
and we aim to perform parallel computations across them.
Here's how you can utilize `pmap` for this purpose.
We specify the devices using the `devices` argument in `pmap`.

```python
import jax
import jax.numpy as jnp

# Define a simple function to be parallelized
def compute(x):
    return x ** 2

# Create input data
data = jnp.arange(8)

# Use pmap to parallelize the computation across available devices
parallel_compute = jax.pmap(compute)

# Apply the parallelized function
result = parallel_compute(data)
print(result)
```

In this example, `compute` is a function that squares its input.
The `jax.pmap` function parallelizes `compute` across all available devices.
Each device processes a slice of the input data, resulting in efficient parallel computation.

In essence, the intermediate data is not sent back to the host device,
which can significantly reduce communication overhead and improve performance.

### Realistic Example: Data Parallelism in Neural Network Training

A common application of `pmap` is in the data parallelism of neural network training.
We can use it together with the automatic differentiation capabilities of JAX to train neural networks efficiently across multiple devices.

```python
import jax
import jax.numpy as jnp
from jax import random, grad, jit, pmap
from functools import partial

@partial(jit, static_argnums=(0,))
def neural_network(params, x):
    for w, b in params:
        x = jnp.dot(x, w) + b
        x = jnp.tanh(x)
    return x
```

The script above defines a simple feedforward neural network and a function to compute its output.
We use the @partial decorator to compile the function with respect to the parameters,
which improves performance.

```python
def loss_fn(params, x, y):
    preds = neural_network(params, x)
    return jnp.mean((preds - y) ** 2)

def init_params(layer_sizes, key):
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        w_key, b_key = random.split(keys[i])
        w = random.normal(w_key, (layer_sizes[i], layer_sizes[i + 1]))
        b = random.normal(b_key, (layer_sizes[i + 1],))
        params.append((w, b))
    return params

@partial(jit, static_argnums=(2,))
def update(params, x, y, learning_rate):
    grads = grad(loss_fn)(params, x, y)
    return tree_map(lambda p, g: p - learning_rate * g, params, grads)
```

Up until now we have defined the neural network, loss function, parameter initialization, and the update function.
No parallelization has been applied yet, but note that the gradient computation is already optimized using JAX's automatic differentiation capabilities.
The `update` function computes the gradients of the loss with respect to the parameters and updates them using gradient descent.
JAX automatically allows for the `grad` function the run parallelly across devices when used with `pmap`.

The `tree_map` function is used to apply the parameter updates across the parameter tree,
instead of manually iterating over the parameters and gradients.

```python
parallel_update = pmap(update, in_axes=(None, 0, 0, None))

layer_sizes = [784, 512, 256, 10]
key = random.PRNGKey(0)
params = init_params(layer_sizes, key)

batch_size = 128
num_devices = jax.device_count()
x = random.normal(key, (num_devices, batch_size // num_devices, 784))
y = random.normal(key, (num_devices, batch_size // num_devices, 10))

learning_rate = 0.01

params_replicated = parallel_update(
  jax.device_put_replicated(params, jax.local_devices()),
  x,
  y,
  learning_rate
)
```

In this post we have covered the basics of vectorization in JAX by:

- Using `vmap` to vectorize functions over batches of data.
- Controlling batch dimensions with `in_axes` and `out_axes`.
- Applying `pmap` for parallel computation across multiple devices.

And in the context of neural network training, we have demonstrated how to use `pmap` for data parallelism in training neural networks:

- Defining a simple feedforward neural network.
- Implementing the loss function and parameter initialization.
- Updating the parameters using gradient descent.
- Correctly applying `@partial`, `jit`, and `tree_map` for optimization.
- Shown how to distribute the computation across multiple devices using `pmap`.
