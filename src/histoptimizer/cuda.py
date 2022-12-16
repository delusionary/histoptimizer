"""CUDA-based parallel implementation of the histoptimizer algorithm.

The cuda module implements a parallelized version of Skiena's dynamic
programming algorithm for solving the linear partition problem. It relies on
the CUDA facilities of NVidia GPUs.

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

from histoptimizer import Histoptimizer

threads_per_pair = 256
item_pairs_per_block = 4
threads_per_block = threads_per_pair * item_pairs_per_block


@cuda.jit
def init_items_kernel(min_cost, divider_location,
                      prefix_sum):  # pragma: no cover
    """Initialize column 1 of the min_cost matrix.
    """
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    item = thread_idx + (block_idx * block_size)
    if item < prefix_sum.size:  # Check array boundaries
        min_cost[item, 1] = prefix_sum[item]
        min_cost[item, 0] = 0
        divider_location[item, 1] = 0
        divider_location[item, 0] = 0


@cuda.jit
def init_buckets_kernel(min_cost, divider_location, item):  # pragma: no cover
    """Initialize row 1 of the min_cost matrix.
    """
    # item is a single-element array
    bucket = cuda.grid(1) + 1
    min_cost[1, bucket] = item[1]
    min_cost[0, bucket] = 0
    divider_location[1, bucket] = 0
    divider_location[0, bucket] = 0


@cuda.jit
def cuda_reconstruct(divider_location, min_cost, num_items, num_buckets,
                     partitions, min_variance):  # pragma: no cover
    divider = num_buckets[0]
    next_location = num_items[0]
    min_variance[0] = min_cost[next_location, divider] / num_buckets[0]
    while divider > 2:
        partitions[divider - 2] = divider_location[next_location, divider]
        next_location = divider_location[next_location, divider]
        divider -= 1
    partitions[0] = divider_location[next_location, divider]


@cuda.jit
def _cuda_partition_kernel(min_cost, divider_location, prefix_sum, num_items,
                           bucket, mean, shared_mem_export):  # pragma: no cover
    """
    Main CUDA kernel.

    Together, all the instances of this kernel compute the item values for a
    single bucket. To implement Skiena's algorithm, this kernel must be invoked
    once for each bucket.

    Given a bucket, thread index, block index, and block size, derives:

      * A pair of item indices, symmetrical about the center of the item list.
      * A set of previous-row indices the thread will cover. The previous-row
        indices are divided by the number of threads per item pair, and
        distributed to threads in an interlaced manner to maximize memory
        bandwidth.

    For each item in the pair, each thread calculates the minimum cost for its
    set of buckets, and the associated divider location. After these have been
    calculated, the threads cooperate to reduce their results to the global
    minimum cost and divider location.

    These values are stored in the min_cost and divider_location matrices.

    Arguments:
        min_cost: Matrix from which to read minimum costs from the previous
            column, and to which to store minimum costs from the current column.
        divider_location: Matrix to write divider locations which give minimum
            cost for the current column.
        prefix_sum: List of the sum of the sizes of all items previous to _i_,
            for index _i_.
        num_items: The number of items in the problem.
        bucket: The index of the bucket/column for which minimum cost and
                divider location are to be computed. Single-item GPU array.
        mean: The mean value of the item sizes. Single-item GPU array.

    Returns:
        min_cost: Each item location in the given bucket now contains lowest
            cost attainable.
        divider_location: Each item location in the given bucket now contains
            the divider location that gives lowest cost.
    """

    # Offset of thread within the block.
    thread_idx = cuda.threadIdx.x
    # Offset of the block within the grid.
    block_idx = cuda.blockIdx.x
    # Number of threads in each block.
    block_size = cuda.blockDim.x
    # Threads are packed into blocks as follows: All the threads for the first
    # item pair, followed by all the threads for the second item pair.
    # Item pairs are allocated in order throughout the blocks of the grid.
    # So to get the index of the first item, get the absolute index of the
    # thread and then integer divide by the number of threads per item pair
    item1 = (thread_idx + (block_idx * block_size)) // threads_per_pair
    item2 = num_items[0] - item1

    # Offset within the item pair of this thread.
    thread_offset = thread_idx % threads_per_pair

    shared_cost = cuda.shared.array(
        shape=(2, 32),
        dtype=np.float32
    )
    shared_divider = cuda.shared.array(
        shape=(2, 32),
        dtype=np.int32
    )

    # The last block will have threads with values greater than half, those
    # threads are done.
    # if item1 > num_items[0] / 2:
    #    return

    # Find Item 1 thread local minimum.
    min_cost1 = np.inf
    divider1 = 0
    if item1 >= bucket[0] and (item1 <= (num_items[0] // 2)):
        for previous_item in range(bucket[0] - 1 + thread_offset,
                                   item1, threads_per_pair):
            cost = (
                    min_cost[previous_item, bucket[0] - 1] +
                    ((prefix_sum[item1] - prefix_sum[
                        previous_item]) -
                     mean[0]) ** 2
            )
            if min_cost1 > cost:
                min_cost1 = cost
                divider1 = previous_item

    shared_mem_export[item1, bucket[0], thread_offset] = min_cost1

    # Find Item 2 thread local minimum
    min_cost2 = np.inf
    divider2 = 0
    if (item1 < item2):
        for previous_item in range(bucket[0] - 1 + thread_offset,
                                   item2, threads_per_pair):
            cost = min_cost[previous_item, bucket[0] - 1] + \
                   ((prefix_sum[item2] - prefix_sum[previous_item])
                    - mean[0]) ** 2
            if min_cost2 > cost:
                min_cost2 = cost
                divider2 = previous_item

    # shared_mem_export[item2, bucket[0], thread_offset] = min_cost2

    cuda.syncthreads()

    delta = np.int32(cuda.warpsize // 2)
    while delta > 0:
        min_cost_next = cuda.shfl_down_sync(0xffffffff, min_cost1, delta)
        divider_next = cuda.shfl_down_sync(0xffffffff, divider1, delta)
        if min_cost1 > min_cost_next:
            min_cost1 = min_cost_next
            divider1 = divider_next
        delta //= 2

    if cuda.laneid == 0:
        shared_cost[0, thread_idx // cuda.warpsize] = min_cost1
        shared_divider[0, thread_idx // cuda.warpsize] = divider1

    cuda.syncthreads()

    delta = np.int32(cuda.warpsize // 2)
    while delta > 0:
        min_cost_next = cuda.shfl_down_sync(0xffffffff, min_cost2, delta)
        divider_next = cuda.shfl_down_sync(0xffffffff, divider2, delta)
        if min_cost2 > min_cost_next:
            min_cost2 = min_cost_next
            divider2 = divider_next
        delta //= 2

    if cuda.laneid == 0:
        shared_cost[1, thread_idx // cuda.warpsize] = min_cost2
        shared_divider[1, thread_idx // cuda.warpsize] = divider2

    cuda.syncthreads()

    # Linear search for the lowest cost. Should be a reduction like above.
    if thread_offset > 0:
        return

    base = thread_idx // cuda.warpsize
    min_cost1 = shared_cost[0, base]
    min_cost2 = shared_cost[1, base]
    divider1 = shared_divider[0, base]
    divider2 = shared_divider[1, base]
    for x in range(1, threads_per_pair // cuda.warpsize):
        if min_cost1 > shared_cost[0, base + x]:
            min_cost1 = shared_cost[0, base + x]
            divider1 = shared_divider[0, base + x]
        if min_cost2 > shared_cost[1, base + x]:
            min_cost2 = shared_cost[1, base + x]
            divider2 = shared_divider[1, base + x]

    if item1 > 1:
        min_cost[item1, bucket[0]] = min_cost1
        divider_location[item1, bucket[0]] = divider1
    if item2 > item1:
        min_cost[item2, bucket[0]] = min_cost2
        divider_location[item2, bucket[0]] = divider2


class CUDAOptimizer(Histoptimizer):
    name = 'cuda'

    @classmethod
    def add_debug_info(cls, debug_info, divider_location_gpu, items,
                       min_cost_gpu,
                       prefix_sum):
        if debug_info is not None:
            min_cost = min_cost_gpu.copy_to_host()
            divider_location = divider_location_gpu.copy_to_host()
            debug_info['prefix_sum'] = prefix_sum
            debug_info['items'] = items
            debug_info['min_cost'] = min_cost
            debug_info['divider_location'] = divider_location

    @classmethod
    def cuda_reconstruct_partition(cls, items, num_buckets, min_cost_gpu,
                                   divider_location_gpu):
        min_variance_gpu = cuda.device_array((1,), dtype=np.float32)
        num_items_gpu = \
            cuda.to_device(np.array([len(items) - 1], dtype=np.int32))
        num_buckets_gpu = \
            cuda.to_device(np.array([num_buckets], dtype=np.int32))
        partition_gpu = cuda.device_array(num_buckets - 1, dtype=np.int32)

        cuda_reconstruct[1, 1](divider_location_gpu, min_cost_gpu,
                               num_items_gpu, num_buckets_gpu, partition_gpu,
                               min_variance_gpu)
        partition = partition_gpu.copy_to_host()
        min_variance = min_variance_gpu.copy_to_host()[0]
        return min_variance, partition

    @classmethod
    def precompile(cls):
        """
        Invokes the paritioner on a simple problem, in order to invoke Numba CUDA JIT compiler.
        The bulk of the execution time of this function will be compilation overhead.
        """
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

        # Transfer input values to the GPU, and create on-GPU arrays for results.

        prefix_sum_gpu = cuda.to_device(prefix_sum)
        mean_value_gpu = cuda.to_device(
            np.array([mean_bucket_sum], dtype=np.float32))
        num_items_gpu = cuda.to_device(np.array([len(items) - 1]))
        item_cost_gpu = cuda.to_device(item_cost)
        min_cost_gpu = cuda.device_array((len(items), num_buckets + 1),
                                         dtype=np.float32)
        shared_mem = cuda.device_array(
            (len(items), num_buckets + 1, threads_per_pair), dtype=np.float32)
        divider_location_gpu = cuda.device_array((len(items), num_buckets + 1),
                                                 dtype=np.int32)

        # Initialize row 1 and column 1 of the min_cost matrix. These could be handled
        # Using logic in the main kernel, but it does not appear to improve performance.
        num_blocks = math.ceil(len(items) / threads_per_block)
        init_items_kernel[num_blocks, threads_per_block](min_cost_gpu,
                                                         divider_location_gpu,
                                                         item_cost_gpu)
        init_buckets_kernel[1, num_buckets](min_cost_gpu,
                                            divider_location_gpu,
                                            item_cost_gpu)

        # Invoke the main computation kernel once for each bucket.
        # Each pair of items (n/2) will have _threads_per_item_pair_ threads.
        num_blocks = math.ceil(
            (len(items) / 2) * threads_per_pair / threads_per_block)
        for bucket in range(2, num_buckets + 1):
            bucket_gpu = cuda.to_device(np.array([bucket]))
            _cuda_partition_kernel[num_blocks, threads_per_block](min_cost_gpu,
                                                                  divider_location_gpu,
                                                                  prefix_sum_gpu,
                                                                  num_items_gpu,
                                                                  bucket_gpu,
                                                                  mean_value_gpu,
                                                                  shared_mem)

        # Calculate the list of dividers from the min_cost and divider_location matrices
        min_variance, dividers = cls.cuda_reconstruct_partition(items,
                                                                num_buckets,
                                                                min_cost_gpu,
                                                                divider_location_gpu)
        cls.add_debug_info(debug_info, divider_location_gpu, items,
                           min_cost_gpu, prefix_sum)
        if debug_info:
            debug_info['shared_mem'] = shared_mem.copy_to_host()

        config.CUDA_LOW_OCCUPANCY_WARNINGS = warnings_enabled

        return dividers, min_variance
