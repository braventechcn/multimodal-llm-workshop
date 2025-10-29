#!/usr/bin/env python3

import numpy as np
from typing import Iterable, Optional, Union

# Define a type alias for array-like inputs
# Eg: np.ndarray, list of floats/ints
ArrayLike = Union[np.ndarray, Iterable[float], Iterable[int]]


def softmax(x: ArrayLike, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Compute the softmax of input x along a given axis in a numerically stable way.
    Params:
    x : array-like
        the input array or sequence of arbitrary shape; integer types will be promoted to float.
    axis : int, default -1
        the axis along which to compute the softmax (default is the last axis).
    temperature : float, default 1.0
        the temperature parameter; >1.0 makes the distribution smoother, <1.0 makes it sharper.

    Returns:
    np.ndarray
        the probability distribution with the same shape as x, where each element is non-negative and sums to 1 along the specified axis.
    """
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.floating):
        # use float64 for better numerical stability, default to float64
        arr = arr.astype(np.float64)

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Apply temperature scaling
    # - All logits are divided by T, which shrinks or expands the differences between them; 
    #   the larger the difference, the more the softmax favors the maximum item.
    # - T > 1.0 makes the distribution smoother (more uniform);
    #   T < 1.0 makes it sharper (more peaked).
    #   T = 1.0 is the standard softmax.
    arr = arr / float(temperature)

    # ----- Standard Softmax operation -----
    # Subtract the maximum value of each sample (axis direction) to prevent exp overflow
    max_ = np.max(arr, axis=axis, keepdims=True)
    e_k = np.exp(arr - max_)
    sum_e_i = np.sum(e_k, axis=axis, keepdims=True)
    y = e_k / sum_e_i
    # --------------------------------------
    
    return y


if __name__ == "__main__":
    x = [0.3, 2.9, 4.0]
    for T in [0.5, 1.0, 2.0]:
        y = softmax(x, temperature=T)
        print(f"T={T}: {np.round(y, 5)} sum={y.sum():.4f}")
    x1 = np.array([[1., 2., 3.], [3., 2., 1.]])
    y1 = softmax(x1)
    print(y1)
    print(np.round(y1, 2)) # Print rounded for better readability