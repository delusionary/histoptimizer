"""
## Recursive Solver for the Minimum Variance Linear Partition problem.

<b>Problem</b>: Integer Partition to minimize σ<sup>2</sup> without Rearrangement

<b>Input</b>: An arrangement S of nonnegative numbers {s<sub>1</sub>, . . . , s<sub>n</sub>} and an integer k.

<b>Output</b>: Partition S into k ranges to minimize the variance (σ<sup>2</sup>), without reordering any of the
numbers.

Given a list of n items, we wish to return a list of divider locations (dividers go *after* the given index)
that creates partitions, or buckets, such that the variance/standard deviation between the size of the buckets
minimized.
"""
import numpy as np
import histoptimizer

name = 'recursive_numba'


@jit(void(float32[:], int32, int32, float32)
def min_cost_partition(items, k, last_item, mean):
    """

    """
    n = len(items)
    j = k - 1
    if mean is None:
        mean = sum(items) / k
    if last_item is None:
        last_item = n - 1
    first_possible_position = j
    best_cost = np.inf

    # The base case is that we are being called to find the optimum location of the first divider for a given
    # location of the second divider
    if j == 0:
        return (sum(items[0:last_item + 1]) - mean)**2, []

    for current_divider_location in range(first_possible_position, last_item + 1):
        for previous_divider_location in range(j - 1, current_divider_location):
            (lh_cost, previous_dividers) = min_cost_partition(items, k - 1, last_item=current_divider_location - 1,
                                                              mean=mean)
            rh_cost = (sum(items[current_divider_location:last_item + 1]) - mean) ** 2
            cost = lh_cost + rh_cost
            if cost < best_cost:
                best_cost = cost
                dividers = previous_dividers + [current_divider_location]
    return best_cost, dividers


def partition(items, k, debug_info=None):
    variance, dividers = min_cost_partition(items, k)
    return dividers, variance / k

