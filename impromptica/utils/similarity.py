"""Utilities for comparing the similarity of musical or other features."""
import math

import numpy as np


def l1(a, b):
    """Returns the L1-norm similarity between vectors a and b.

    The L1-norm is the sum of the absolute values of the differences of the
    components of a and b.

    `a` and `b` must be equal-length one-dimensional numpy arrays.
    """
    return np.sum(np.absolute(a - b))


def l2(a, b):
    """Returns the L2-norm similarity between vectors a and b.

    The L2-norm is the square root of the squares of the differences of the
    components of a and b.

    `a` and `b` must be equal-length one-dimensional numpy arrays.
    """
    return math.sqrt(np.sum(np.square(a - b)))
