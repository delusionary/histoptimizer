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
import sys

import numpy as np

from histoptimizer import Histoptimizer


class RecursiveCacheOptimizer(Histoptimizer):
    name = 'recursive_cache'

    @classmethod
    def min_cost_partition(cls, items: list, k: int, last_item=None, mean=None,
                           cache=None):
        """

        """
        # Return cache hit if there is one.
        if cache is None:
            cache = {}
        if cache.get(k, None) is None:
            cache[k] = {}
        cached_value = cache[k].get(last_item, None)
        if cached_value is not None:
            return cache[k][last_item]

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
            return (sum(items[0:last_item + 1]) - mean) ** 2, []

        for current_divider_location in range(first_possible_position,
                                              last_item + 1):
            (lh_cost, previous_dividers) = cls.min_cost_partition(items,
                                                                  k - 1,
                                                                  last_item=current_divider_location - 1,
                                                                  mean=mean,
                                                                  cache=cache)
            rh_cost = (sum(items[
                           current_divider_location:last_item + 1]) - mean) ** 2
            cost = lh_cost + rh_cost
            if cost < best_cost:
                best_cost = cost
                dividers = previous_dividers + [current_divider_location]

        cache[k][last_item] = (best_cost, dividers)
        return best_cost, dividers

    @classmethod
    def partition(cls, items, k, debug_info=None):
        cache = {}
        variance, dividers = cls.min_cost_partition(items, k, cache=cache)
        if debug_info is not None:
            debug_info['cache'] = sys.getsizeof(cache)
        return dividers, variance / k
