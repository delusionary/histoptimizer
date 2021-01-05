import pandas as pd
import numpy as np


def get_partitioner_dict(*modules):
    """
    Given a list of modules which have a `partition` function and
    optionally a `name` variable, return a dictionary that maps
    `name` -> `partition` for any modules that have a `name`.

    This allows for partitioner modules to specify a standardized
    name by which they can be referenced.
    """
    partitioners = {}
    for m in modules:
        if name := getattr(m, 'name', None):
            partitioners[name] = m.partition
    return partitioners


def cuda_supported():
    """
    In theory, returns True if Numba is installed and the system has a GPU.
    """
    try:
        from numba import cuda
        gpus = cuda.gpus
        return True
    except (cuda.CudaDriverError, cuda.CudaDriverError):
        return False


def get_partition_sums(dividers, items):
    """
    Given a list of divider locations and a list of items,
    return a list the sum of the items in each partition.
    """
    #  fix this to take and use prefix sums, but only after you
    #  have written a test.
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

def get_prefix_sums(items):
    """
    Given a list of item sizes, return a NumPy float32 array where the first item is 0.0 and subsequent items are the
    cumulative sum of the elements of the list.

    Args:
        items (iterable): A list of item sizes, integer or float.

    Returns:
        NumPy float32 array containing a [0]-prefixed cumulative sum.
    """
    prefix_sum = np.zeros((len(items) + 1), dtype=np.float32)
    prefix_sum[1:] = np.cumsum(items)
    return prefix_sum


def bucket_generator(dividers: np.array, num_items: int):
    """
    Iterate over a list of partitions, converting it into bucket numbers for each item in the
    partitioned series.

    Args:
        dividers (NumPY array): A list of divider locations. Dividers are considered as
        coming before the given list index with 0-based array indexing.
        num_items (int): The number of items in the list to be partitioned.

    Returns:
        Series: A series with an item for each item in the partitioned list, where
                the value of each item is the bucket number it belongs to, starting
                with bucket 1.

    Example:
        partitions = [12, 13, 18]  # Three dividers = 4 buckets
        num_items = 20

        Returns [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4]
    """
    for bucket in range(1, len(dividers) + 2):
        if bucket == 1:
            low_item = 0
        else:
            low_item = dividers[bucket - 2]
        if bucket == len(dividers) + 1:
            high_item = num_items
        else:
            high_item = dividers[bucket - 1]
        for item in range(low_item, high_item):
            yield bucket


def get_partition_series(data: pd.DataFrame, sizes: str, buckets: int, partitioner):
    """
    Takes a Pandas DataFrame and returns a Series that distributes rows sequentially into the given
    number of buckets with the minimum possible standard deviation.

    Args:
        data (DataFrame): The first parameter.
        sizes (str): Column to get size values from.
        buckets (int): Number of buckets to partition items into.
        partitioner (function): Partitioner function

    Returns:
        pandas.Series: Series thing.
    """
    items = data[[sizes]].astype('float32').to_numpy(dtype=np.float32)
    partitions, variance = partitioner(items, buckets)
    return pd.Series((b for b in bucket_generator(partitions, len(items))))


def histoptimize(data: pd.DataFrame, sizes: str, bucket_list: list, column_name: str,
                 partitioner_func, optimal_only=False):
    """
    Histoptimize takes a Pandas DataFrame and adds additional columns, one for each integer
    in bucket_list.

    The additional columns are named `column_name` + {bucket_list[i]} and contain for each
    row a bucket number such that the rows are distributed into the given number of buckets
    in such a manner as to minimize the variance/standard deviation over all buckets.

    Args:
        data (DataFrame): The first parameter.
        sizes (str): Column to get size values from.
        bucket_list (list): A list of integer bucket sizes.
        column_name (str): Prefix to be added to the number of buckets to get the column name.
        partitioner_func (function): Partitioner function
        optimal_only (bool): If true, add only one column, for the number of buckets with the
            lowest variance.

    Returns:
        DataFrame: Original DataFrame with one or more columns added.
    """
    partitions = pd.DataFrame(columns=('column_name', 'dividers', 'variance'))
    items = data[[sizes]].astype('float32').to_numpy(dtype=np.float32)
    for buckets in bucket_list:
        dividers, variance = partitioner_func(items, buckets)
        partitions.append({
          'column_name': f'{column_name}{buckets}',
          'dividers': dividers,
          'variance': variance})

    if optimal_only:
        partitions = partitions[partitions.variance == partitions.variance.min()].limit(1)
    for p in partitions.iterrows():
        data[p['column_name']] = pd.Series((b for b in bucket_generator(p['dividers'], len(items))))
    return data


def partitioner(partitioner_func):
    """
    Decorates partitioner functions and ensures that parameters are valid.

    Args:
        partitioner_func (function): function to decorate.

    Returns:
        A wrapped version of partitioner_function that validates input.
    """
    def checked_partitioner(items, buckets, debug_info=None):
        try:
            num_items = len(items)
        except TypeError:
            raise ValueError("items must be a container")
        if num_items < 3:
            raise ValueError("Must have at least 3 items to have a choice of partition location")
        if buckets < 2:
            raise ValueError("Must request at least two buckets")
        if buckets > num_items:
            raise ValueError("Cannot have more buckets than items")
        return partitioner_func(items, buckets, debug_info)

    return checked_partitioner
