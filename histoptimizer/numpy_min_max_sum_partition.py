import numpy
from histoptimizer.dynamic import reconstruct_partition


# noinspection DuplicatedCode
def build_matrices(buckets, prefix_sum):
    n = len(prefix_sum)
    min_cost = numpy.zeros((n, buckets + 1), dtype=numpy.float32)
    divider_location = numpy.zeros((n, buckets + 1), dtype=numpy.int32)

    for item in range(1, len(prefix_sum)):
        # min_cost[item, 1] = prefix_sum[item]
        min_cost[item, 1] = prefix_sum[item]
    for bucket in range(1, buckets + 1):
        min_cost[1, bucket] = prefix_sum[1]
    for item in range(2, len(prefix_sum)):
        # evaluate main recurrence
        for bucket in range(2, buckets + 1):
            min_cost_temp = numpy.finfo(dtype=numpy.float32).max
            divider_location_temp = 0
            for previous_item in range(1, item):
                cost = max(min_cost[previous_item, bucket - 1], prefix_sum[item] - prefix_sum[previous_item])
                if min_cost_temp > cost:
                    min_cost_temp = cost
                    divider_location_temp = previous_item
            min_cost[item, bucket] = min_cost_temp
            divider_location[item, bucket] = divider_location_temp

    return min_cost, divider_location


# noinspection DuplicatedCode
def numpy_min_max_sum_partition(items, buckets, debug_info=None):
    padded_items = [0]
    padded_items.extend(items)
    items = padded_items
    n = len(items) - 1
    prefix_sum = numpy.zeros((n + 1), dtype=numpy.float32)
    # Cache cumulative sums
    for item in range(1, n + 1):
        prefix_sum[item] = prefix_sum[item - 1] + items[item]

    (min_cost, divider_location) = build_matrices(buckets, prefix_sum)

    if debug_info is not None:
        debug_info['items'] = items
        debug_info['prefix_sum'] = prefix_sum
        debug_info['min_cost'] = min_cost
        debug_info['divider_location'] = divider_location

    partition = reconstruct_partition(divider_location, n, buckets)
    return partition
