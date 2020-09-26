from numba import cuda
import numpy as np
from timeit import default_timer as timer

# import os; os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

def python_reconstruct(divider_location, num_items, num_buckets):
    partition_lengths = np.zeros((num_buckets,), dtype=np.int)
    bucket = num_buckets
    while bucket > 1:
        partition_lengths[bucket - 1] = num_items - divider_location[num_items, bucket]
        num_items = divider_location[num_items, bucket]
        bucket -= 1
    partition_lengths[0] = num_items
    return partition_lengths

@cuda.jit
def cuda_reconstruct(divider_location, num_items, num_buckets, partitions):
    bucket = num_buckets
    while bucket > 0:
        partitions[bucket] = divider_location[num_items, bucket]
        num_items = partitions[bucket]
        bucket -= 1
    partitions[0] = num_items

@cuda.jit
def init_items_kernel(min_cost, prefix_sum):
    item = cuda.grid(1)
    min_cost[item, 1] = prefix_sum[item]

@cuda.jit
def init_buckets_kernel(min_cost, item):
    # item is a single-element array
    bucket = cuda.grid(1) + 1
    min_cost[1, bucket] = item[1]

@cuda.jit
def cuda_partition_kernel(min_cost, divider_location, prefix_sum):
    """
    There is one thread for each bucket.
    """
    bucket = cuda.grid(1) + 2
    divider = 0
    # Fill in the size of the first element at the top of each column
    min_cost[1, bucket] = prefix_sum[1]
    cuda.syncthreads()
    for item in range(2, min_cost.shape[0] + 1):
        tmp = prefix_sum[prefix_sum.shape[0]-1] + 1
        for previous_item in range(1, item):
            cost = max(min_cost[previous_item, bucket - 1], prefix_sum[item] - prefix_sum[previous_item])
            if tmp > cost:
                tmp = cost
                divider = previous_item
        min_cost[item, bucket] = tmp
        divider_location[item, bucket] = divider
        # All threads must finish the current item row before we continue.
        # This is probably not true; the previous thread just needs to be done?
        cuda.syncthreads()


def cuda_partition(items, num_buckets):
    items = [0] + items
    prefix_sum = np.zeros((len(items)))
    # Cache cumulative sums
    for item in range(1, len(items)):
        prefix_sum[item] = prefix_sum[item - 1] + items[item]

    prefix_sum_gpu = cuda.to_device(prefix_sum)
    min_cost_gpu = cuda.device_array((len(items), num_buckets+1))
    divider_location_gpu = cuda.device_array((len(items), num_buckets+1), dtype=np.int)

    init_items_kernel[1, len(items)](min_cost_gpu, prefix_sum_gpu)

    # min_cost = min_cost_gpu.copy_to_host()

    cuda_partition_kernel[1, num_buckets-1](min_cost_gpu, divider_location_gpu, prefix_sum)

    min_cost = min_cost_gpu.copy_to_host()
    divider_location = divider_location_gpu.copy_to_host()

    partitions_gpu = cuda.device_array(num_buckets, dtype=np.int)

    #cuda_reconstruct[1, 1](divider_location_gpu, len(items), num_buckets, partitions_gpu)
    partitions = python_reconstruct(divider_location, len(items) - 1, num_buckets)
    #partitions = partitions_gpu.copy_to_host()
    return partitions

# start = timer()
# partitions = numba_partition([1,2,3,4,5,6,7,8,9,10], 3)
# end = timer()