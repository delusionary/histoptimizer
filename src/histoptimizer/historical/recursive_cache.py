"""
## Recursive Solver for the Minimum Variance Linear Partition problem.

This is a simple recursive solution with a cache added.
"""
import sys

import numpy as np

from histoptimizer import Histoptimizer


class RecursiveCacheOptimizer(Histoptimizer):
    """Recursive Histoptimizer implementation with a cache.

    This is intended to be the same as ``RecursiveOptimizer``, but with an
    added cache so that the function is never called with the same parameters
    twice.
    """
    name = 'recursive_cache'

    @classmethod
    def recurse(cls, k: int, last_item: int, mean: float,
                prefix_sums: list, cache: dict):
        n = len(prefix_sums)
        first_possible_position = k - 1
        best_cost = np.inf

        # The base case remains broken out. These could be integrated into the main loop
        # and the first / only call intercepted, but that does not suit my purposes.
        if k == 1:
            if 1 not in cache:
                cache[1] = {}
            if last_item not in cache[1]:
                (previous_dividers, lh_cost) = [], \
                    (prefix_sums[last_item + 1] - mean) ** 2
                cache[1][last_item] = (previous_dividers, lh_cost)
                return previous_dividers, lh_cost

        for cur_div_loc in range(first_possible_position, last_item + 1):
            if k - 1 in cache and (cur_div_loc - 1) in cache[k - 1]:
                (previous_dividers, lh_cost) = cache[k - 1][cur_div_loc - 1]
            else:
                (previous_dividers, lh_cost) = cls.recurse(
                    k - 1,
                    cur_div_loc - 1,
                    mean,
                    prefix_sums,
                    cache)
            rh_cost = ((prefix_sums[last_item + 1] - prefix_sums[cur_div_loc])
                       - mean) ** 2
            cost = lh_cost + rh_cost
            if cost < best_cost:
                best_cost = cost
                dividers = previous_dividers + [cur_div_loc]

        if k not in cache:
            cache[k] = {}
        cache[k][last_item] = (dividers, best_cost)
        return dividers, best_cost

    @classmethod
    def partition(cls, items, k, debug_info=None):
        cache = {}
        prefix_sums = cls._get_prefix_sums(items)
        dividers, msd = cls.recurse(k, len(items) - 1,
                                    sum(items) / k,
                                    prefix_sums, cache)
        return dividers, msd / k
