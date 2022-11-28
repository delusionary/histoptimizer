"""Implementation of the Histoptimizer API on CUDA, parallelized by buckets.

This module is not production code, and is included as example
material for a tutorial. Use histoptimizer.CUDAOptimizer instead.

CUDAOptimizerBuckets was my first attempt at parallelizing Skiena's
algorithm for a GPU. I knew it was not correct, but writing it helped
me to figure out how it was wrong.

This implementation has one thread per *bucket* (column), and all the
threads must synchronize after every row. This dramatically limits the
parallelism of the solution.

Copyright (C) 2020 by Kelly Joyner (de@lusion.org)

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
"""
import math

import numpy as np
from numba import cuda
from numba.core import config

from histoptimizer.cuda import CUDAOptimizer


@cuda.jit
def init_items_kernel(min_cost, prefix_sum):
    """Initialize the first column of min_cost.

    The effect of running this kernel is that the first column of the min_cost
    array, which is zero when initially allocated, is set to the prefix sum
    of the corresponding item.

    Args:
        min_cost: Minimum cost array. A GPU Array.
        prefix_sum: Prefix sum of item sizes array. A GPU Array.
    """
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    item = thread_idx + (block_idx * block_size)
    if item < prefix_sum.size:  # Check array boundaries
        min_cost[item, 1] = prefix_sum[item]


@cuda.jit
def init_buckets_kernel(min_cost, item):
    """Initializes the first row of the min_cost array.

    This kernel initializes the first row of the min_cost array to the given
    item size.

    Args:
        min_cost: Minimum cost array. A GPU Array.
        item: A one-element array. A GPU Array.
    """
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
            cost = min_cost[previous_item, bucket - 1] + (
                        (prefix_sum[item] - prefix_sum[previous_item]) - mean[
                    0]) ** 2
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
        """Divide the given item sizes evenly into the given number of buckets.

        This module is not production code, and is included as example
        material for a tutorial. Use histoptimizers.CUDAOptimizer instead.

        CUDAOptimizerBuckets was my first attempt at parallelizing Skiena's
        algorithm for a GPU. I knew it was not correct, but writing it helped
        me to figure out how it was wrong.

        This implementation has one thread per *bucket* (column), and all the
        threads must synchronize after every row. This dramatically limits the
        parallelism of the solution.
        """

        # Record the state of then disable the Cuda low occupancy warning.
        warnings_enabled = config.CUDA_LOW_OCCUPANCY_WARNINGS
        config.CUDA_LOW_OCCUPANCY_WARNINGS = False

        padded_items = [0]
        padded_items.extend(items)
        items = padded_items
        prefix_sum = np.zeros((len(items)), dtype=np.float32)
        item_cost = np.zeros((len(items)), dtype=np.float32)
        mean_bucket_sum = sum(items) / num_buckets

        # Pre-calculate prefix sums for items in the array.
        for item in range(1, len(items)):
            prefix_sum[item] = prefix_sum[item - 1] + items[item]
            item_cost[item] = (prefix_sum[item] - mean_bucket_sum) ** 2

        prefix_sum_gpu = cuda.to_device(prefix_sum)
        mean_value_gpu = cuda.to_device(
            np.array([mean_bucket_sum], dtype=np.float32))
        item_cost_gpu = cuda.to_device(item_cost)
        min_cost_gpu = cuda.device_array((len(items), num_buckets + 1))
        divider_location_gpu = cuda.device_array((len(items), num_buckets + 1),
                                                 dtype=np.int32)

        threads_per_block = 256
        num_blocks = math.ceil(len(items) / threads_per_block)
        init_items_kernel[num_blocks, threads_per_block](min_cost_gpu,
                                                         item_cost_gpu)
        init_buckets_kernel[1, num_buckets](min_cost_gpu, item_cost_gpu)

        cuda_partition_kernel[1, num_buckets - 1](min_cost_gpu,
                                                  divider_location_gpu,
                                                  prefix_sum_gpu,
                                                  mean_value_gpu)

        min_variance, partition = cls.cuda_reconstruct_partition(items,
                                                                 num_buckets,
                                                                 min_cost_gpu,
                                                                 divider_location_gpu)

        cls.add_debug_info(debug_info, divider_location_gpu, items,
                           min_cost_gpu, prefix_sum)

        # Restore occupancy warning config setting.
        config.CUDA_LOW_OCCUPANCY_WARNINGS = warnings_enabled

        return partition, min_variance
