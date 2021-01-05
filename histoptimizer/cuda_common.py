import numpy as np
from numba import cuda


def add_debug_info(debug_info, divider_location_gpu, items, min_cost_gpu, prefix_sum):
    if debug_info is not None:
        min_cost = min_cost_gpu.copy_to_host()
        divider_location = divider_location_gpu.copy_to_host()
        debug_info['prefix_sum'] = prefix_sum
        debug_info['items'] = items
        debug_info['min_cost'] = min_cost
        debug_info['divider_location'] = divider_location


@cuda.jit
def cuda_reconstruct(divider_location, min_cost, num_items, num_buckets, partitions, min_variance):
    divider = num_buckets[0]
    next_location = num_items[0]
    min_variance[0] = min_cost[next_location, divider] / num_buckets[0]
    while divider > 2:
         partitions[divider - 2] = divider_location[next_location, divider]
         next_location = divider_location[next_location, divider]
         divider -= 1
    partitions[0] = divider_location[next_location, divider]


def reconstruct_partition(items, num_buckets, min_cost_gpu, divider_location_gpu):
    min_variance_gpu = cuda.device_array((1,), dtype=np.float32)
    num_items_gpu = cuda.to_device(np.array([len(items) - 1], dtype=np.int))
    num_buckets_gpu = cuda.to_device(np.array([num_buckets], dtype=np.int))
    partition_gpu = cuda.device_array(num_buckets - 1, dtype=np.int)
    cuda_reconstruct[1, 1](divider_location_gpu, min_cost_gpu, num_items_gpu, num_buckets_gpu, partition_gpu,
                           min_variance_gpu)
    partition = partition_gpu.copy_to_host()
    min_variance = min_variance_gpu.copy_to_host()[0]
    return min_variance, partition