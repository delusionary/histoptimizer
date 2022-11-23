import math

from numba import cuda
import numpy as np

from histoptimizer.cuda import CUDAOptimizer

# import os; os.environ['NUMBA_ENABLE_CUDASIM'] = '1'


# Instead of doing one thread per bucket, Do one item at a time and divvy up the work for the items in each
# thread.

@cuda.jit
def init_items_kernel(min_cost, prefix_sum):
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    item = thread_idx + (block_idx * block_size)
    if item < prefix_sum.size:  # Check array boundaries
        min_cost[item, 1] = prefix_sum[item]


@cuda.jit
def init_buckets_kernel(min_cost, item):
    # item is a single-element array
    bucket = cuda.grid(1) + 1
    min_cost[1, bucket] = item[1]


@cuda.jit
def cuda_partition_kernel(min_cost, divider_location, prefix_sum, num_items, bucket, mean):
    """
    There is one thread for each pair of items.
    """
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    first_item = thread_idx + (block_idx * block_size)
    if first_item > (num_items[0] // 2) + 1:
        return

    if first_item > 1:
        divider = 0
        tmp = np.inf
        if first_item >= bucket[0]:
            for previous_item in range(bucket[0] - 1, first_item):
                rh_cost = ((prefix_sum[first_item] - prefix_sum[previous_item]) - mean[0]) ** 2
                lh_cost = min_cost[previous_item, bucket[0] - 1]
                cost = lh_cost + rh_cost
                if tmp > cost:
                    tmp = cost
                    divider = previous_item

        min_cost[first_item, bucket[0]] = tmp
        divider_location[first_item, bucket[0]] = divider

    second_item = num_items[0] - first_item

    if second_item == first_item:
        return

    divider = 0
    tmp = np.inf
    for previous_item in range(bucket[0] - 1, second_item):
        cost = min_cost[previous_item, bucket[0] - 1] + (
                    (prefix_sum[second_item] - prefix_sum[previous_item]) - mean[0]) ** 2
        if tmp > cost:
            tmp = cost
            divider = previous_item

    min_cost[second_item, bucket[0]] = tmp
    divider_location[second_item, bucket[0]] = divider
    return


class CUDAOptimizerItemPairs(CUDAOptimizer):
    name = 'cuda_2'

    @classmethod
    def precompile(cls):
        cls.partition([1, 4, 6, 9], 3)

