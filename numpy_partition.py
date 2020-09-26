import numpy


def reconstruct_partition(divider_location, num_items, num_buckets):
    partitions = numpy.zeros((num_buckets - 1,), dtype=numpy.int)
    divider = num_buckets
    while divider > 2:
        partitions[divider - 2] = divider_location[num_items, divider]
        num_items = divider_location[num_items, divider]
        divider -= 1
    partitions[0] = divider_location[num_items, divider]
    return partitions


def numpy_partition(items, buckets):
    padded_items = [0]
    padded_items.extend(items)
    items = padded_items
    n = len(items) - 1
    min_cost = numpy.zeros((n + 1, buckets + 1), dtype=numpy.float32)       # pd.DataFrame(columns=range(0, buckets + 1))
    divider_location = numpy.zeros((n + 1, buckets + 1), dtype=numpy.int32)       # d = pd.DataFrame(columns=range(0, buckets + 1))
    prefix_sum = numpy.zeros((n + 1))                    # p = [None] * (n + 1)  # prefix sums array

    # Cache cumulative sums
    for item in range(1, n + 1):
        prefix_sum[item] = prefix_sum[item - 1] + items[item]
    for item in range(1, n + 1):
        min_cost[item, 1] = prefix_sum[item]
    for bucket in range(1, buckets + 1):
        min_cost[1, bucket] = items[1]
    for item in range(2, n + 1):
        # evaluate main recurrence
        for bucket in range(2, buckets + 1):
            min_cost[item, bucket] = numpy.finfo(dtype=numpy.float32).max
            for previous_item in range(1, item):
                cost = max(min_cost[previous_item, bucket - 1], prefix_sum[item] - prefix_sum[previous_item])
                if min_cost[item, bucket] > cost:
                    min_cost[item, bucket] = cost
                    divider_location[item, bucket] = previous_item
    print('hi')
    return reconstruct_partition(divider_location, n, buckets)
