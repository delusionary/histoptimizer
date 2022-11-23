import numpy as np
from numba import guvectorize, float32, int64, prange
from histoptimizer import Histoptimizer


# os.environ['NUMBA_DISABLE_JIT'] = '1'
# noinspection DuplicatedCode
@guvectorize(
    ['i4[:], i4, f4[:], f4[:,:], i4[:,:]'],
    '(k),(),(n)->(n,k),(n,k)',
    nopython=True,
    target='cpu'
)
def build_matrices(bucket_list, buckets, prefix_sum, min_cost, divider_location):

    mean = prefix_sum[-1] / buckets
    for item in range(1, len(prefix_sum)):
        # min_cost[item, 1] = prefix_sum[item]
        min_cost[item, 1] = (prefix_sum[item] - mean)**2

    mean = prefix_sum[-1] / (min_cost.shape[1] - 1)

    for bucket in range(2, min_cost.shape[1]):
        min_cost[0, bucket] = min_cost[0, bucket - 1]
        min_cost[1, bucket] = min_cost[0, bucket - 1]
        for item in prange(2, len(prefix_sum)):
            min_cost_tmp = np.inf
            divider_location_tmp = 0
            for previous_item in prange(bucket - 1, item):
                cost = min_cost[previous_item, bucket - 1] + ((prefix_sum[item] - prefix_sum[previous_item]) - mean) ** 2
                if cost < min_cost_tmp:
                    min_cost_tmp = cost
                    divider_location_tmp = previous_item
            min_cost[item, bucket] = min_cost_tmp
            divider_location[item, bucket] = divider_location_tmp


class NumbaOptimizerDraft2(Histoptimizer):
    name = 'dynamic_numba_2'

    @classmethod
    def precompile(cls):
        cls.partition([1, 4, 6, 9], 3)

    @classmethod
    def partition(cls, items, buckets: int, debug_info: dict = None) -> list:
        """
        Implements a histoptimizer.partitioner-compliant partitioner.

        Args:
            items (iterable): An iterable of float- or float-compatible values representing a sorted series of item sizes.
            buckets (int): Number of buckets to partition items into.
            debug_info: A dictionary to be populated with debugging information.

        Returns:
            dividers (list): A list of divider locations that partitions items into `buckets` partitions such that
                the variance of the partition item sums is minimized.
            variance: The resulting variance.
        """
        prefix_sum = cls.get_prefix_sums(items)

        #min_cost, divider_location = init_matrices(buckets, prefix_sum)
        bucket_list = np.zeros((buckets + 1), dtype=int)
        min_cost, divider_location = build_matrices(bucket_list, buckets, prefix_sum)

        if debug_info is not None:
            debug_info['items'] = items
            debug_info['prefix_sum'] = prefix_sum
            debug_info['min_cost'] = min_cost
            debug_info['divider_location'] = divider_location

        partition = cls.reconstruct_partition(divider_location, len(items), buckets)
        return partition, min_cost[len(items), buckets] / buckets
