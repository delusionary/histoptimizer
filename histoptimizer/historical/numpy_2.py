import numpy as np


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


# noinspection DuplicatedCode
def build_matrices(buckets, prefix_sum):
    n = len(prefix_sum)
    min_cost = np.zeros((n, buckets + 1), dtype=np.float32)
    divider_location = np.zeros((n, buckets + 1), dtype=np.int32)
    mean = prefix_sum[-1] / buckets
    for item in range(1, len(prefix_sum)):
        # min_cost[item, 1] = prefix_sum[item]
        min_cost[item, 1] = (prefix_sum[item] - mean)**2
    for bucket in range(1, buckets + 1):
        min_cost[1, bucket] = (prefix_sum[1] - mean)**2
    prev_divider_lowest_cost_location = np.full((buckets + 1,), 1, dtype=np.int)
    for item in range(2, len(prefix_sum)):
        # evaluate main recurrence
        for bucket in range(2, buckets + 1):
            min_cost_temp = np.finfo(dtype=np.float32).max
            divider_location_temp = 0
            if min_cost[item - 1, bucket - 1] < prev_divider_lowest_cost_location[bucket]:
                prev_divider_lowest_cost_location[bucket] = item - 1
            min_cost_temp = min_cost[prev_divider_lowest_cost_location[bucket], bucket - 1] + ((prefix_sum[item] - prefix_sum[prev_divider_lowest_cost_location[bucket]]) - mean)**2
            min_cost[item, bucket] = min_cost_temp
            divider_location_temp = prev_divider_lowest_cost_location[bucket]
            divider_location[item, bucket] = divider_location_temp
    return min_cost, divider_location


# noinspection DuplicatedCode
def numpy_partition(items, buckets, debug_info=None):
    num_items = len(items)
    padded_items = [0]
    padded_items.extend(items)
    items = padded_items

    prefix_sum = np.zeros((num_items + 1), dtype=np.float32)
    # Cache cumulative sums
    for item in range(1, num_items + 1):
        prefix_sum[item] = prefix_sum[item - 1] + items[item]

    (min_cost, divider_location) = build_matrices(buckets, prefix_sum)

    if debug_info is not None:
        debug_info['items'] = items
        debug_info['prefix_sum'] = prefix_sum
        debug_info['min_cost'] = min_cost
        debug_info['divider_location'] = divider_location

    partition = reconstruct_partition(divider_location, num_items, buckets)
    return partition
