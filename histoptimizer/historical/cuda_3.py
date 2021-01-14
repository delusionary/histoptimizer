import math

from numba import cuda
import numpy as np

from histoptimizer.cuda_common import add_debug_info, reconstruct_partition

# import os; os.environ['NUMBA_ENABLE_CUDASIM'] = '1'


# Instead of doing one thread per bucket, Do one thread at a time and divvy up the work for the items in each
# thread.

name = 'cuda_3'
threads_per_item_pair = 8
item_pairs_per_block = 8
threads_per_block = threads_per_item_pair * item_pairs_per_block


def reconstruct_partition(divider_location, num_items, num_buckets):
    if num_buckets < 2:
        return np.array(0)
    partitions = np.zeros((num_buckets - 1,), dtype=np.int)
    divider = num_buckets
    while divider > 2:
        partitions[divider - 2] = divider_location[num_items, divider]
        num_items = divider_location[num_items, divider]
        divider -= 1
    partitions[0] = divider_location[num_items, divider]
    return partitions


def precompile():
    partition([1, 4, 6, 9], 3)


@cuda.jit
def _init_items_kernel(min_cost, prefix_sum):  # pragma: no cover
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    item = thread_idx + (block_idx * block_size)
    if item < prefix_sum.size:  # Check array boundaries
        min_cost[item, 1] = prefix_sum[item]


@cuda.jit
def _init_buckets_kernel(min_cost, item):  # pragma: no cover
    # item is a single-element array
    bucket = cuda.grid(1) + 1
    min_cost[1, bucket] = item[1]

@cuda.jit
def _cuda_partition_kernel(min_cost, divider_location, prefix_sum, num_items, bucket, mean): # pragma: no cover
    """
    There is one thread for each pair of items.
    """

    shared_cost = cuda.shared.array(shape=(2, item_pairs_per_block, threads_per_item_pair), dtype=np.float32)
    shared_divider = cuda.shared.array(shape=(2, item_pairs_per_block, threads_per_item_pair), dtype=np.int32)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    first_item = (thread_idx + (block_idx * block_size)) // item_pairs_per_block
    item_pair_offset_within_block = thread_idx // threads_per_item_pair
    item_thread_id = (thread_idx + (block_idx * block_size)) % threads_per_item_pair

    if first_item > num_items[0] / 2:
        return

    if first_item == 1:
        min_cost[first_item, bucket[0]] = min_cost[first_item, bucket[0] - 1]

    if first_item > 1:
        divider = 0
        tmp = np.inf
        if first_item >= bucket[0]:
            for previous_item in range(bucket[0] - 1 + item_thread_id, first_item, threads_per_item_pair):
                cost = min_cost[previous_item, bucket[0] - 1] + ((prefix_sum[first_item] - prefix_sum[previous_item]) - mean[0]) ** 2
                if tmp > cost:
                    tmp = cost
                    divider = previous_item

        shared_cost[0, item_pair_offset_within_block, item_thread_id] = tmp
        shared_divider[0, item_pair_offset_within_block, item_thread_id] = divider

    second_item = num_items[0] - first_item

    if second_item != first_item:
        divider = 0
        tmp = np.inf
        if second_item >= bucket[0]:
            for previous_item in range(bucket[0] - 1 + item_thread_id, second_item, threads_per_item_pair):
                cost = min_cost[previous_item, bucket[0] - 1] + ((prefix_sum[second_item] - prefix_sum[previous_item]) - mean[0]) ** 2
                if tmp > cost:
                    tmp = cost
                    divider = previous_item

        shared_cost[1, item_pair_offset_within_block, item_thread_id] = tmp
        shared_divider[1, item_pair_offset_within_block, item_thread_id] = divider

    cuda.syncthreads()

    # Reduce the values from each thread in the shared memory segments to find the lowest overall value.
    s = 1
    while s < item_pairs_per_block:
        if item_thread_id % (2 * s) == 0:
            if shared_cost[0, item_pair_offset_within_block, item_thread_id] > shared_cost[0, item_pair_offset_within_block, item_thread_id + s]:
                shared_cost[0, item_pair_offset_within_block, item_thread_id] = shared_cost[0, item_pair_offset_within_block, item_thread_id + s]
                shared_divider[0, item_pair_offset_within_block, item_thread_id] = shared_divider[0, item_pair_offset_within_block, item_thread_id + s]
        cuda.syncthreads()
        s = s * 2

    if item_thread_id == 0 and first_item > 1:
        min_cost[first_item, bucket[0]] = shared_cost[0, item_pair_offset_within_block, item_thread_id]
        divider_location[first_item, bucket[0]] = shared_divider[0, item_pair_offset_within_block, item_thread_id]

    cuda.syncthreads()

    s = 1
    while s < item_pairs_per_block:
        if item_thread_id % (2 * s) == 0:
            if shared_cost[1, item_pair_offset_within_block, item_thread_id] > shared_cost[1, item_pair_offset_within_block, item_thread_id + s]:
                shared_cost[1, item_pair_offset_within_block, item_thread_id] = shared_cost[1, item_pair_offset_within_block, item_thread_id + s]
                shared_divider[1, item_pair_offset_within_block, item_thread_id] = shared_divider[1, item_pair_offset_within_block, item_thread_id + s]
        cuda.syncthreads()
        s = s * 2

    if item_thread_id == 0 and second_item != first_item:
        min_cost[second_item, bucket[0]] = shared_cost[1, item_pair_offset_within_block, item_thread_id]
        divider_location[second_item, bucket[0]] = shared_divider[1, item_pair_offset_within_block, item_thread_id]


def partition(items, num_buckets, debug_info=None):
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
    num_items_gpu = cuda.to_device(np.array([len(items) - 1]))
    item_cost_gpu = cuda.to_device(item_cost)
    min_cost_gpu = cuda.device_array((len(items), num_buckets+1))
    divider_location_gpu = cuda.device_array((len(items), num_buckets+1), dtype=np.int)

    num_blocks = math.ceil(len(items) / threads_per_block)
    _init_items_kernel[num_blocks, threads_per_block](min_cost_gpu, item_cost_gpu)
    # We don't really need this, could be a special case in kernel.
    # _init_buckets_kernel[1, num_buckets](min_cost_gpu, item_cost_gpu)

    num_blocks = math.ceil((len(items) / 2) * threads_per_item_pair / threads_per_block)
    for bucket in range(2, num_buckets + 1):
        bucket_gpu = cuda.to_device(np.array([bucket]))
        _cuda_partition_kernel[num_blocks, threads_per_block](min_cost_gpu, divider_location_gpu, prefix_sum_gpu, num_items_gpu, bucket_gpu, mean_value_gpu)

    min_cost = min_cost_gpu.copy_to_host()
    divider_location = divider_location_gpu.copy_to_host()
    partition = reconstruct_partition(divider_location, len(items) - 1, num_buckets)

    #min_variance, partition = reconstruct_partition(items, num_buckets, min_cost_gpu, divider_location_gpu)
    #add_debug_info(debug_info, divider_location_gpu, items, min_cost_gpu, prefix_sum)

    return partition, min_cost[len(items) - 1, num_buckets] / num_buckets
