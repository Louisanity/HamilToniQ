from typing import List

import numpy as np


def return_hardness(return_vec: List(float)) -> float:
    """
    A function calculating the QAOA hardness according of a return vector
    Args:
        return_vec (List(float)): the return vector
    return:
        hardness (float): the hardness in the range between -1 and 1.
    """
    hardness = np.var(return_vec)
    return hardness


def covariance_hardness(covariance: List(List(float))) -> float:
    """
    A function calculating the QAOA hardness according of a covariance matrix
    Args:
        covariance (List(List(float)): the covariance matrix
    return:
        hardness (float): the hardness in the range between -1 and 1.
    """
    normalized_covariance = [
        covariance[i, j] / np.sqrt(covariance[i, i] * covariance[j, j])
        for j in range(i + 1)
        for i in range(len(covariance))
    ]
    hardness = np.var(normalized_covariance)
    return hardness
