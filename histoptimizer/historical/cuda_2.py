import math

from numba import cuda
import numpy as np

from histoptimizer.cuda_common import add_debug_info, reconstruct_partition

# import os; os.environ['NUMBA_ENABLE_CUDASIM'] = '1'


# Instead of doing one thread per bucket, Do one thread at a time and divvy up the work for the items in each
# thread.

name = 'cuda_2'


@cuda.jit
def cuda_reconstruct(divider_location, num_items, num_buckets, partitions):
    divider = num_buckets
    while divider > 2:
        partitions[divider - 2] = divider_location[num_items, divider]
        num_items = divider_location[num_items, divider]
        divider -= 1
    partitions[0] = divider_location[num_items, divider]


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


def partition(items, num_buckets, debug_info=None):
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
    # Determine the min cost of placing the first divider at each item.
    # get_cost = np.vectorize(lambda x: (x-mean_bucket_sum)**2)
    # item_cost = get_cost(items)

    prefix_sum_gpu = cuda.to_device(prefix_sum)
    mean_value_gpu = cuda.to_device(np.array([mean_bucket_sum], dtype=np.float32))
    num_items_gpu = cuda.to_device(np.array([len(items) - 1]))
    item_cost_gpu = cuda.to_device(item_cost)
    min_cost_gpu = cuda.device_array((len(items), num_buckets+1))
    divider_location_gpu = cuda.device_array((len(items), num_buckets+1), dtype=np.int)

    threads_per_block = 256
    num_blocks = math.ceil((len(items) / 2) / threads_per_block)
    init_items_kernel[num_blocks, threads_per_block](min_cost_gpu, item_cost_gpu) # prefix_sum_gpu)
    # We don't really need this, could be a special case in kernel.
    init_buckets_kernel[1, num_buckets](min_cost_gpu, item_cost_gpu) # prefix_sum_gpu)

    for bucket in range(2, num_buckets + 1):
        bucket_gpu = cuda.to_device(np.array([bucket]))
        cuda_partition_kernel[num_blocks, threads_per_block](min_cost_gpu, divider_location_gpu, prefix_sum_gpu, num_items_gpu, bucket_gpu, mean_value_gpu)

    min_variance, partition = reconstruct_partition(items, num_buckets, min_cost_gpu, divider_location_gpu)
    add_debug_info(debug_info, divider_location_gpu, items, min_cost_gpu, prefix_sum)

    return partition, min_variance
