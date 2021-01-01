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


def histoptimize(data: pd.DataFrame, sizes: str, buckets: int, column_name: str,
                 partitioner):
    """
    Histoptimize takes a Pandas DataFrame and returns a Series that distributes them
    sequentially into the given number of buckets with the minimum possible standard deviation.


    """
    items = data[sizes].astype('float32').to_numpy(dtype=np.float32)
    partitions, variance = partitioner(items, buckets)
    return pd.Series((b for b in bucket_generator(partitions, len(items))))
