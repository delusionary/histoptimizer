"""
dynamic_numpy implements the Histoptimizer API using NumPy `vectorize`
"""
import numpy as np
from numpy import vectorize
from histoptimizer import Histoptimizer


def get_min_cost(prefix_sum, previous_row, mean):
    current_row_cost = np.zeros(previous_row.shape)
    current_row_dividers = np.zeros(previous_row.shape)
    current_row_cost[0] = previous_row[0]
    current_row_cost[1] = previous_row[1]
    for item in range(2, len(prefix_sum)):
        min_cost = np.inf
        divider_location = 0
        for previous_item in range(2, item + 1):
            cost = previous_row[previous_item] + ((prefix_sum[item] -
                                                   prefix_sum[
                                                       previous_item]) - mean) ** 2
            if cost < min_cost:
                min_cost = cost
                divider_location = previous_item
        current_row_cost[item] = min_cost
        current_row_dividers[item] = divider_location
    return current_row_cost, current_row_dividers


get_min_cost_vector = vectorize(get_min_cost, signature='(m),(n),()->(n),(n)')


class NumpyOptimizer(Histoptimizer):
    name = 'dynamic_numpy'

    @classmethod
    def build_matrices(cls, min_cost, divider_location, prefix_sum):
        mean = prefix_sum[-1] / (min_cost.shape[1] - 1)
        for bucket in range(2, min_cost.shape[1]):
            min_cost[:, bucket], divider_location[:, bucket] =\
                get_min_cost_vector(prefix_sum, min_cost[:, bucket - 1], mean)

        return min_cost, divider_location

    @classmethod
    def partition(cls, items, buckets: int, debug_info: dict = None) -> list:
        """
        Implements a histoptimizer.partitioner-compliant partitioner.

        Args:
            items (iterable): An iterable of float- or float-compatible values
                representing a sorted series of item sizes.
            buckets (int): Number of buckets to partition items into.
            debug_info: A dictionary to be populated with debugging information.

        Returns:
            dividers (list): A list of divider locations that partitions items into `buckets` partitions such that
                the variance of the partition item sums is minimized.
            variance: The resulting variance.
        """

        prefix_sum = cls.get_prefix_sums(items)

        min_cost, divider_location = cls.init_matrices(buckets, prefix_sum)
        cls.build_matrices(min_cost, divider_location, prefix_sum)

        if debug_info is not None:
            debug_info['items'] = items
            debug_info['prefix_sum'] = prefix_sum
            debug_info['min_cost'] = min_cost
            debug_info['divider_location'] = divider_location

        partition = cls.reconstruct_partition(divider_location, len(items),
                                              buckets)

        return partition, min_cost[len(items) - 1, buckets]
