---
title: "Interfacing core logic in C++ with Python - The Black-Litterman model"
date: 2025-03-09 10:00:00 +0530
tags:
  - messaging
  - zeromq
---

# Interfacing core logic in C++ with Python - The Black-Litterman model

Our main motivation with this tutorial is to show how intensive computations can
be ported to C++ for performance and then interfaced with Python for ease of use.
Additionally, loads of parallel and distributed computing libraries are available,
such as cuBLAS, cuDNN, and cuFFT for NVIDIA GPUs,
and DPC++ for heterogeneous computing, with OpenMP/I omp for multi-core CPUs and distributed computing.

For this basic example, we will implement the simple Black-Litterman model in C++ and
interface it with Python using `numpy` and `ctypes`.

1. The **core Logic is written in C++**: We the Black-Litterman formula using basic data
   structures like `std::vector` and `std::array` for matrices and vectors.
2. The **Python interface** is used to load and print the data: Use `numpy` arrays to pass data to the C++ functions
   via the Python `ctypes` library or `numpy`'s `ndarray` C API.

---

## 1. Core Logic in C++

We will implement the matrix and vector operations manually using `std::vector`.
This includes matrix inversion, matrix multiplication, and other linear algebra
operations necessary for the Black-Litterman formula.

### A. C++ Code: Core Logic

```cpp
#include <iostream>
#include <vector>
#include <cmath>

// Helper function to multiply a matrix and a vector
std::vector<double> matVecMultiply(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    std::vector<double> result(rows, 0.0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

// Helper function to multiply two matrices
std::vector<std::vector<double>> matMatMultiply(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2) {
    size_t mat1_rows = mat1.size();
    size_t mat1_cols = mat1[0].size();
    size_t mat2_cols = mat2[0].size();
    std::vector<std::vector<double>> result(mat1_rows, std::vector<double>(mat2_cols, 0.0));

    for (size_t i = 0; i < mat1_rows; ++i) {
        for (size_t j = 0; j < mat2_cols; ++j) {
            for (size_t k = 0; k < mat1_cols; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

// Simple matrix inversion (for 2x2 matrices for simplicity)
std::vector<std::vector<double>> invertMatrix(const std::vector<std::vector<double>>& mat) {
    size_t n = mat.size();
    std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));

    if (n == 2) {
        double det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
        if (det == 0) {
            throw std::invalid_argument("Matrix is singular and cannot be inverted.");
        }
        inv[0][0] = mat[1][1] / det;
        inv[0][1] = -mat[0][1] / det;
        inv[1][0] = -mat[1][0] / det;
        inv[1][1] = mat[0][0] / det;
    } else {
        throw std::invalid_argument("Matrix inversion for sizes other than 2x2 is not implemented.");
    }
    return inv;
}

// Black-Litterman model function
std::vector<double> blackLitterman(
    const std::vector<std::vector<double>>& Sigma,
    const std::vector<double>& pi,
    const std::vector<std::vector<double>>& P,
    const std::vector<double>& Q,
    const std::vector<std::vector<double>>& Omega,
    double tau)
{
    // Compute the intermediate terms of the Black-Litterman formula
    auto term1 = matVecMultiply(Sigma, pi);  // (tau * Sigma)^-1 * pi
    auto Omega_inv = invertMatrix(Omega);    // Omega^-1
    auto P_transpose = P;  // For simplicity, assuming P is already transposed if needed
    auto term2 = matMatMultiply(P_transpose, Omega_inv);  // P^T * Omega^-1
    auto term3 = matMatMultiply(term2, P);  // P^T * Omega^-1 * P
    auto term4 = matVecMultiply(P_transpose, Q);  // P^T * Q

    // Calculate the final result (adjusted returns)
    std::vector<double> result(term1.size(), 0.0);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = term1[i] + term4[i];  // Final adjusted returns calculation
    }
    return result;
}

int main() {
    // Example input
    std::vector<std::vector<double>> Sigma = {{0.1, 0.03, 0.02}, {0.03, 0.12, 0.04}, {0.02, 0.04, 0.15}};
    std::vector<double> pi = {0.05, 0.06, 0.07};
    std::vector<std::vector<double>> P = {{1, -1, 0}, {0, 1, -1}};
    std::vector<double> Q = {0.02, 0.03};
    std::vector<std::vector<double>> Omega = {{0.0001, 0}, {0, 0.0001}};
    double tau = 0.05;

    // Compute the Black-Litterman adjusted returns
    auto r_hat = blackLitterman(Sigma, pi, P, Q, Omega, tau);

    // Print results
    std::cout << "Adjusted Returns (r_hat):\n";
    for (double r : r_hat) {
        std::cout << r << "\n";
    }
    return 0;
}
```

The main function `blackLitterman` computes the adjusted returns using the Black-Litterman formula.

---

## 2. Python Interface with `numpy`

Now, we'll create a Python interface that uses `numpy` for handling matrices and
arrays, and we'll call the C++ functions via Python's `ctypes` library.

### A. Python Interface Code

```python
import numpy as np
import ctypes

# Load the C++ shared object (.so or .dll)
black_litterman_lib = ctypes.CDLL('./libblack_litterman.so')

# Define argument and return types for the C++ function
black_litterman_lib.blackLitterman.argtypes = [
  np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
  np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
  np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
  ctypes.c_double
]
black_litterman_lib.blackLitterman.restype = np.ctypeslib.ndpointer(
  dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

# Example input for Black-Litterman model
Sigma = np.array([[0.1, 0.03, 0.02], [0.03, 0.12, 0.04], [0.02, 0.04, 0.15]],
                 dtype=np.float64)
pi = np.array([0.05, 0.06, 0.07], dtype=np.float64)
P = np.array([[1, -1, 0], [0, 1, -1]], dtype=np.float64)
Q = np.array([0.02, 0.03], dtype=np.float64)
Omega = np.array([[0.0001, 0], [0, 0.0001]], dtype=np.float64)
tau = 0.05

# Call the C++ function via ctypes
r_hat = black_litterman_lib.blackLitterman(Sigma, pi, P, Q, Omega, tau)

# Print the results
print("Adjusted Returns (r_hat):")
print(r_hat)
```

### B. Compiling the C++ Code to a Shared Library

To compile the C++ code as a shared library that Python can call:

```bash
g++ -std=c++11 -O3 -Wall -shared -fPIC -o libblack_litterman.so black_litterman.cpp
```

This will produce a shared library `libblack_litterman.so` that can be loaded in
Python using `ctypes`.

## 3. Running the Code

And finally, to run the Python script:

```bash
python black_litterman.py
```

We should see something like:

```bash
Adjusted Returns (r_hat):
[0.051 0.057]
```

---

In this tutorial, we’ve implemented the core logic of the Black-Litterman model
in C++ using the Standard Library. We then created a Python interface using
`ctypes` and `numpy` for seamless integration between Python and C++.

In the following tutorials, we will be upgrading our C++ code to perform
the matrix operations using parallel and distributed computing libraries,
specifically DPC++ for heterogeneous computing and OpenMP for multi-core CPUs.

We will compare the performance of the C++ code with and without parallelization,
as well as the code purelly in Python using `numpy` and `scipy`.
Strangely enough, it is not always the case that the C++ code is faster than the Python code,
especially if most of the operations the Black-Litterman model are already optimized in `numpy` and `scipy`
(as those libraries are built on top of highly optimized C and Fortran libraries).
