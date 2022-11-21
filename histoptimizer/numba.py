import numpy as np
import os
from numba import guvectorize, float32, int64
from histoptimizer import Histoptimizer, get_prefix_sums, partitioner

# os.environ['NUMBA_DISABLE_JIT'] = '1'

@guvectorize(
    ['i4, f4[:], f4[:], f4, f4[:], f4[:]'],
    '(),(m),(n),()->(n),(n)',
    nopython=True,
    target='cpu'
)
def _get_min_cost(bucket, prefix_sum, previous_row, mean, current_row_cost,
                 current_row_dividers):  # pragma: no cover
    current_row_cost[0] = previous_row[0]
    current_row_cost[1] = previous_row[1]
    current_row_dividers[0] = 0
    current_row_dividers[1] = 0
    for item in range(2, len(prefix_sum)):
        min_cost = np.inf
        divider_location = 0
        for previous_item in range(bucket - 1, item):
            cost = previous_row[previous_item] + ((prefix_sum[item] -
                                                   prefix_sum[
                                                       previous_item]) - mean) ** 2
            if cost < min_cost:
                min_cost = cost
                divider_location = previous_item
        current_row_cost[item] = min_cost
        current_row_dividers[item] = divider_location

class NumbaOptimizer(Histoptimizer):
    name = 'numba'
    @classmethod
    def precompile(cls):
        cls.partition([1, 4, 6, 9], 3)

    @classmethod
    def get_min_cost(cls, bucket, prefix_sum, previous_row, mean):
        return _get_min_cost(bucket, prefix_sum, previous_row, mean)

    @classmethod
    def build_matrices(cls, min_cost, divider_location,
                       num_buckets, prefix_sum):
        mean = prefix_sum[-1] / num_buckets
        for bucket in range(2, min_cost.shape[1]):
            min_cost[:, bucket], divider_location[:, bucket] =\
                cls.get_min_cost(bucket, prefix_sum, min_cost[:, bucket-1], mean)

        return min_cost, divider_location

    # noinspection DuplicatedCode
    # @classmethod
    # def partition(cls, item_sizes, num_buckets: int, debug_info: dict = None
    #               ) -> list:
    #     """
    #     Implements a histoptimizer.partitioner-compliant partitioner.
    #
    #     Args:
    #         item_sizes (iterable): An iterable of float- or float-compatible values representing a sorted series of item sizes.
    #         num_buckets (int): Number of buckets to partition items into.
    #         debug_info: A dictionary to be populated with debugging information.
    #
    #     Returns:
    #         dividers (list): A list of divider locations that partitions items into `buckets` partitions such that
    #             the variance of the partition item sums is minimized.
    #         variance: The resulting variance.
    #     """
    #     prefix_sum = cls.get_prefix_sums(item_sizes)
    #
    #     min_cost, divider_location = cls.init_matrices(num_buckets, prefix_sum)
    #     cls.build_matrices(min_cost, divider_location, prefix_sum)
    #
    #     if debug_info is not None:
    #         debug_info['items'] = item_sizes
    #         debug_info['prefix_sum'] = prefix_sum
    #         debug_info['min_cost'] = min_cost
    #         debug_info['divider_location'] = divider_location
    #
    #     partition = cls.reconstruct_partition(divider_location, len(item_sizes), num_buckets)
    #     return partition, min_cost[len(item_sizes), num_buckets] / num_buckets
