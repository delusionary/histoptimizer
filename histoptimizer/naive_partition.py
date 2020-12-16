import numpy as np
import pandas as pd


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


def slow_naive_partition(items, num_buckets, debug_info=None):
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
    df['cost'] = df['partition_sums'].apply(np.std)
    partition = np.array(df.sort_values('cost').reset_index().loc[0]['dividers'])
    if debug_info is not None:
        debug_info['df'] = df
    return partition


def partition_generator(num_items: int, num_buckets: int) -> list:
    """
    Given a number of items `num_items` and a number of buckets `num_buckets`, enumerate lists of all the possible
    combinations of divider locations that partition `num_items` into `num_buckets`.

    The strategy is to start at the enumeration that has each divider in its left-most possible location, and then
    iterate all possible locations of the last (right-most) divider before incrementing the next-to-last and again
    iterating all possible locations of the last divider.

    When there are no more valid locations for the next-to-last divider, then the previous divider is incremented and
    the process repeated, and so on until the first divider and all subsequent dividers are in their largest
    (right-most) possible locations.
    """
    num_dividers = num_buckets - 1
    last_divider = num_dividers - 1

    partition = [x for x in range(1, num_dividers+1)]  # Start with the first valid partition.
    last_partition = [x for x in range(num_items - num_dividers, num_items)] # We know what the last partition is.
    current_divider = last_divider

    # Deal with single-divider/two-bucket case
    if num_dividers == 1:
        for last_location in range(1, num_items):
            partition[0] = last_location
            yield partition
        return

    while True:
        if current_divider == last_divider:
            for last_location in range(partition[current_divider-1] + 1, num_items):
                partition[last_divider] = last_location
                yield partition
            if partition == last_partition:
                return
            # partition[last_divider] = 0
            current_divider -= 1
        else:
            if partition[current_divider] == 0:
                partition[current_divider] = partition[current_divider-1] + 1
                current_divider += 1
            elif partition[current_divider] < (num_items - (last_divider - current_divider)):
                partition[current_divider] += 1
                current_divider += 1
            else:
                for divider in range(current_divider, num_dividers):
                    partition[divider] = 0
                current_divider -= 1
        # if this is the last divider, then loop through all possible values yielding each
        #         then decrease the current divider location and set an increment flag
        # if not last divider:
        #   check the current location of the current divider
        #     if it is zero, set to the minimum valid value (previous divider location + 1)
        #     elif it is less than the max location value, increment it and move to the next divider location
        #     elif it is at the max location value, then set it and all subsequent location values to 0
        #       and move to previous divider.
    # End loop when all dividers are set at their last possible locations.


def naive_partition(items, num_buckets, debug_info=None, mean=None):
    min_variance = np.inf
    best_partition = None
    n = len(items)
    if mean is None:
        mean = sum(items) / num_buckets

    prefix_sums = [0]*len(items)
    prefix_sums[0] = items[0]
    for i in range(1, len(items)):
        prefix_sums[i] = prefix_sums[i-1] + items[i]

    previous_dividers = [0] * (num_buckets - 1)
    variances = [0.0] * num_buckets
    # partitition_sums = [0.0] * num_buckets
    for dividers in partition_generator(n, num_buckets):
        divider_index = 0
        variance = 0.0
        # Most of the time, only one divider location has changed.
        # Retain the previous prefix sums and variances to save time.
        # If there are only two buckets, the single divider location has always changed.
        while num_buckets > 2 and (dividers[divider_index] == previous_dividers[divider_index]):
            divider_index += 1
        for partition_index in range(0, num_buckets):
            if divider_index - 1 >= partition_index:
                pass  # variances[partition_index] already contains the correct value from the previous iteration.
            elif partition_index == 0:
                variances[0] = (prefix_sums[dividers[0] - 1] - mean)**2
            elif partition_index == (num_buckets - 1):
                variances[partition_index] = (prefix_sums[-1] - prefix_sums[dividers[-1] - 1] - mean) ** 2
            else:
                variances[partition_index] = (
                    (prefix_sums[dividers[partition_index] - 1] - prefix_sums[dividers[partition_index - 1] - 1] - mean) ** 2)
            variance += variances[partition_index]
        if variance < min_variance:
            min_variance = variance
            best_partition = dividers[:]
        previous_dividers[:] = dividers[:]
    if debug_info is not None:
        debug_info['variance'] = min_variance
    return np.array(best_partition)
