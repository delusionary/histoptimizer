import math

# import os; os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

from numba import cuda
import numpy as np

from histoptimizer.cuda import CUDAOptimizer

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
def cuda_partition_kernel(min_cost, divider_location, prefix_sum, mean):
    """
    There is one thread for each bucket.
    """
    bucket = cuda.grid(1) + 2
    divider = 0
    # Fill in the size of the first element at the top of each column
    # min_cost[1, bucket] = prefix_sum[1]
    cuda.syncthreads()
    for item in range(2, min_cost.shape[0] + 1):
        # tmp = prefix_sum[prefix_sum.shape[0]-1] + 1
        tmp = np.inf
        for previous_item in range(bucket - 1, item):
            cost = min_cost[previous_item, bucket - 1] + ((prefix_sum[item] - prefix_sum[previous_item]) - mean[0]) ** 2
            if tmp > cost:
                tmp = cost
                divider = previous_item
        min_cost[item, bucket] = tmp
        divider_location[item, bucket] = divider
        # All threads must finish the current item row before we continue.
        # This is probably not true; the previous thread just needs to be done?
        cuda.syncthreads()

class CUDAOptimizerBuckets(CUDAOptimizer):
    name = 'cuda_1'

    @classmethod
    def precompile(cls):
        cls.partition([1, 4, 6, 9], 3)

    @classmethod
    def partition(cls, items, num_buckets, debug_info=None):
        """

        """
        padded_items = [0]
        padded_items.extend(items)
        items = padded_items
        prefix_sum = np.zeros((len(items)), dtype=np.float32)
        item_cost = np.zeros((len(items)), dtype=np.float32)
        mean_bucket_sum = sum(items) / num_buckets

        # Pre-calculate prefix sums for items in the array.
        for item in range(1, len(items)):
            prefix_sum[item] = prefix_sum[item - 1] + items[item]
            item_cost[item] = (prefix_sum[item] - mean_bucket_sum)**2

        prefix_sum_gpu = cuda.to_device(prefix_sum)
        mean_value_gpu = cuda.to_device(np.array([mean_bucket_sum], dtype=np.float32))
        item_cost_gpu = cuda.to_device(item_cost)
        min_cost_gpu = cuda.device_array((len(items), num_buckets+1))
        divider_location_gpu = cuda.device_array((len(items), num_buckets+1), dtype=np.int)

        threads_per_block = 256
        num_blocks = math.ceil(len(items) / threads_per_block)
        init_items_kernel[num_blocks, threads_per_block](min_cost_gpu, item_cost_gpu)
        init_buckets_kernel[1, num_buckets](min_cost_gpu, item_cost_gpu)

        cuda_partition_kernel[1, num_buckets-1](min_cost_gpu, divider_location_gpu, prefix_sum_gpu, mean_value_gpu)

        min_variance, partition = cls.cuda_reconstruct_partition(items, num_buckets, min_cost_gpu, divider_location_gpu)

        cls.add_debug_info(debug_info, divider_location_gpu, items, min_cost_gpu, prefix_sum)

        return partition, min_variance


