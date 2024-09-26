# pyml

`pyml` is a machine learning library that integrates Python and C++ using `nanobind`.
Currently, this library only implements a simple machine learning method based on Singular Value Decomposition
(SVD) for classification tasks.

## Key Features

- **Python-C++ Integration**: Utilizes `nanobind` to seamlessly bind C++ code with Python, allowing for high-performance
  computations while maintaining ease of use in Python.
- **Ease of Use**: Provides a straightforward API for fitting models and making predictions, making it accessible for
  users with basic machine learning knowledge.

## Components

- **SVDClassifier**: A specialized class that performs classification using SVD.
  It includes methods for fitting the model to the data and predicting labels for new data.
- **Projection Functions**: Functions to calculate projections of matrices, which are essential for the SVD-based
  classification process.
- **Thread Management**: Functions to set and get the number of threads used in computations, allowing for performance
  tuning.

## Prerequisites

- Python 3.9 or higher
- C++17 compiler
- `pip` for Python package management
- CMake 3.28 or higher

## Installation

To install the `pyml` library, you can use the following commands:

```bash
git clone git@github.com:caffik/pyml.git
cd pyml
pip install .
```

## Example Usage

Please see the example directory for a simple example of using the `SVDClassifier` class to classify data.

## Wheel Building

Note that Release build support only Python 3.9.
To build a wheel for a different Python version with GitHub Actions, please see: https://github.com/pypa/cibuildwheel.
