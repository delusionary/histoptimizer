"""Implementation of the Histoptimizer API on CUDA, parallelized by item pairs.

This module is not production code, and is included as example
material for a tutorial. Use histoptimizer.CUDAOptimizer instead.

CUDAOptimizerItemPairs was my second attempt at parallelizing Skiena's
algorithm for a GPU. It creates one thread for every pair of items in the
list (first paired with last, second with next to last, etc.). The kernel
runs once for every bucket, with the end of the job being the sync point.

This solution is much more performant than the first, and is very close in
performance to the production solution, which introduces the concept of multiple
threads per item pair.

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
def cuda_partition_kernel(min_cost, divider_location, prefix_sum, num_items,
                          bucket, mean):
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
                rh_cost = ((prefix_sum[first_item] - prefix_sum[
                    previous_item]) - mean[0]) ** 2
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
                (prefix_sum[second_item] - prefix_sum[previous_item]) - mean[
            0]) ** 2
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

    @classmethod
    def partition(cls, items, num_buckets, debug_info=None):
        """
        Highly parallel GPU-based implementation of Skiena's dynamic programming algorithm for the linear partition problem.

        Arguments:
            items: An ordered list of items sizes.
            num_buckets: The number of buckets to partition the items into.
            debug_info: A dictionary that can accept debug information.

        Returns:
            partition_locations: Index of dividers within items. Dividers come after the item in 0-based indexing and
            before the item in 1-based indexing.
            min_variance: The variance of the solution defined by partition_locations
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
        # Determine the min cost of placing the first divider at each item.
        # get_cost = np.vectorize(lambda x: (x-mean_bucket_sum)**2)
        # item_cost = get_cost(items)

        prefix_sum_gpu = cuda.to_device(prefix_sum)
        mean_value_gpu = cuda.to_device(
            np.array([mean_bucket_sum], dtype=np.float32))
        num_items_gpu = cuda.to_device(np.array([len(items) - 1]))
        item_cost_gpu = cuda.to_device(item_cost)
        min_cost_gpu = cuda.device_array((len(items), num_buckets + 1))
        divider_location_gpu = cuda.device_array(
            (len(items), num_buckets + 1), dtype=np.int)

        threads_per_block = 1024
        num_blocks = math.ceil((len(items) / 2) / threads_per_block)
        init_items_kernel[num_blocks, threads_per_block](min_cost_gpu,
                                                         item_cost_gpu)  # prefix_sum_gpu)
        # We don't really need this, could be a special case in kernel.
        init_buckets_kernel[1, num_buckets](min_cost_gpu,
                                            item_cost_gpu)  # prefix_sum_gpu)

        for bucket in range(2, num_buckets + 1):
            bucket_gpu = cuda.to_device(np.array([bucket]))
            cuda_partition_kernel[num_blocks, threads_per_block](
                min_cost_gpu, divider_location_gpu, prefix_sum_gpu,
                num_items_gpu, bucket_gpu, mean_value_gpu)

        min_variance, dividers = cls.cuda_reconstruct_partition(
            items, num_buckets,
            min_cost_gpu,
            divider_location_gpu
        )

        cls.add_debug_info(debug_info, divider_location_gpu, items,
                           min_cost_gpu, prefix_sum)

        # Restore occupancy warning config setting.
        config.CUDA_LOW_OCCUPANCY_WARNINGS = warnings_enabled

        return dividers, min_variance
