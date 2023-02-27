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

from histoptimizer import Histoptimizer


class RecursiveOptimizer(Histoptimizer):
    name = 'recursive'

    @classmethod
    def recurse(cls, k: int, last_item: int, mean: float,
                prefix_sums: list):
        """Recursively find a solution to the minimum variance linear partition problem.

        Takes:
            items: A list of item sizes
            k: Number of buckets to partition items into.
            last_item:
        """
        first_possible_position = k - 1
        best_cost = np.inf

        # The base case is that we are being called to find the optimum location of the first divider for a given
        # location of the second divider
        if k == 1:
            return [], (prefix_sums[last_item + 1] - mean) ** 2

        for cur_div_loc in range(first_possible_position,
                                 last_item + 1):
            (previous_dividers, lh_cost) = cls.recurse(k - 1,
                                                       cur_div_loc - 1,
                                                       mean,
                                                       prefix_sums)
            rh_cost = ((prefix_sums[last_item + 1] - prefix_sums[cur_div_loc])
                       - mean) ** 2
            cost = lh_cost + rh_cost
            if cost < best_cost:
                best_cost = cost
                dividers = previous_dividers + [cur_div_loc]
        try:
            return dividers, best_cost
        except Exception as e:
            pass

    @classmethod
    def partition(cls, items, k, debug_info=None):
        prefix_sums = cls._get_prefix_sums(items)
        dividers, msd = cls.recurse(k, len(items) - 1,
                                    sum(items) / k,
                                    prefix_sums)
        return dividers, msd / k
