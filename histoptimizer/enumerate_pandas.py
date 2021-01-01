import numpy as np
import pandas as pd
import histoptimizer

name = 'enumerate_pandas'


def get_partition_sums(dividers, items):
    """
    Given a list of divider locations and a list of items,
    return a list of partition sums.
    """
    #  fix this to take and use prefix sums.
    partitions = [.0]*(len(dividers)+1)
    for x in range(0, len(dividers)+1):
        if x == 0:
            left_index = 0
        else:
            left_index = dividers[x-1]
        if x == len(dividers):
            right_index = len(items)
        else:
            right_index = dividers[x]
        for y in range(left_index, right_index):
            partitions[x] += items[y]
    return partitions


def get_partitions(num_items, buckets, prefix):
    """
    Given a number of items and a number of buckets, return all possible combination of divider locations.

    Result is a list of lists, where each sub-list is set of divider locations. Each divider is placed after
    the 1-based index provided (or, alternately, *before* the zero-based index provided).
    """
    partitions = []
    num_dividers = len(prefix)
    if num_dividers:
        last_divider_loc = prefix[-1]
    else:
        last_divider_loc = 0
    remaining_dividers = (buckets - 1) - num_dividers
    for next_divider_loc in range(last_divider_loc + 1, num_items - (remaining_dividers - 1)):
        new_prefix = prefix.copy() + [next_divider_loc]
        if remaining_dividers == 1:
            partitions.append(new_prefix)
        else:
            partitions.extend(get_partitions(num_items, buckets, new_prefix))
    return partitions


def partition(items, num_buckets, debug_info=None):
    """
    Given an ordered list of items and number of buckets, return a list of divider locations
    that partitions the list in such a way as to minimize the standard deviation of the sum of the items
    in the bucket.

    This function operates by generating all possible partitionings and calculating the standard deviation
    for each. It will not complete in a reasonable amount of time for large numbers of items or buckets,
    and is intended to test the correctness of other algorithms.

    It is written for clarity rather than performance.
    """

    n = len(items)
    all_partitions = get_partitions(n, num_buckets, [])
    df = pd.DataFrame(pd.Series(all_partitions, name='dividers'))
    df['partition_sums'] = df['dividers'].apply(get_partition_sums, items=items)
    df['cost'] = df['partition_sums'].apply(np.var)
    dividers = df[df.cost == df.cost.min()].iloc[0]['dividers']
    if debug_info is not None:
        debug_info['df'] = df
    return dividers, df.cost.min()

