import pandas as pd
import numpy as np


class Histoptimizer(object):
    """Base class for objects implementing the Histoptimizer API.

    Histoptimizer provides a basic, pure python solution for the linear
    partition problem with a minimum variance cost function.

    """
    name = 'dynamic'

    @classmethod
    def check_parameters(cls, items, buckets, debug_info):
        """Do basic validation on partition parameters.
        """
        try:
            num_items = len(items)
        except TypeError:
            raise ValueError("Items must be a container.")
        if num_items < 3:
            raise ValueError("Must have at least 3 items.")
        if buckets < 2:
            raise ValueError("Must request at least two buckets.")
        if buckets > num_items:
            raise ValueError("Cannot have more buckets than items.")
        if debug_info is not None and not isinstance(debug_info, dict):
            raise ValueError("debug_info should be None or a dictionary")


    @classmethod
    def reconstruct_partition(cls, divider_location, num_items, num_buckets):
        """Return a list of optimal divider locations given a location matrix.

        Arguments:
            divider_location: A matrix giving the location of dividers that
                minimize variance.
            num_items: The number of items to be partitioned.
            num_buckets: The number of buckets to partition the items into.
        """
        if num_buckets < 2:
            return np.array(0)
        partitions = np.zeros((num_buckets - 1,), dtype=int)
        divider = num_buckets
        while divider > 2:
            partitions[divider - 2] = divider_location[num_items, divider]
            num_items = divider_location[num_items, divider]
            divider -= 1
        partitions[0] = divider_location[num_items, divider]
        return partitions

    @classmethod
    def get_prefix_sums(cls, item_sizes):
        """
        Given a list of item sizes, return a NumPy float32 array where item 0 is
        0 and item *n* is the cumulative sum of item sizes 0..n-1.

        This transformation of item sizes makes it possible to determine the
        total size of the items between locations *m* and *n* by a single
        subtraction: prefix_sums[n] - prefix_sums[m]

        Args:
            item_sizes (iterable): A list of item sizes, integer or float.

        Returns:
            NumPy float32 array containing a [0]-prefixed cumulative sum.
        """
        prefix_sum = np.zeros((len(item_sizes) + 1), dtype=np.float32)
        prefix_sum[1:] = np.cumsum(item_sizes)
        return prefix_sum

    @classmethod
    def init_matrices(cls, num_buckets, prefix_sum):
        """Create and initialize min_cost and divider_location matrices.

        Creates the two matrices necessary to implement Skiena's algorithm, and
        initializes the cells where the base relation values are stored.

        Args:
            num_buckets: The number of
            prefix_sum: List of item sizes in prefix sum form.

        Returns:
            A tuple:

            min_cost:
            divider_location:
        """
        n = len(prefix_sum)
        min_cost = np.zeros((n, num_buckets + 1), dtype=np.float32)
        divider_location = np.zeros((n, num_buckets + 1), dtype=np.int32)
        mean = prefix_sum[-1] / num_buckets
        for item in range(1, len(prefix_sum)):
            min_cost[item, 1] = (prefix_sum[item] - mean) ** 2
        for bucket in range(1, num_buckets + 1):
            min_cost[1, bucket] = (prefix_sum[1] - mean) ** 2
        return min_cost, divider_location

    @classmethod
    def build_matrices(cls, min_cost, divider_location,
                       num_buckets, prefix_sum):
        """Compute min cost and divider location matrices.

        These matrices encode a full set of intermediate results that
        can be used to identify an optimal division of an ordered set of sized
        items in num_buckets partitions, such that the variance over the total
        size of each partition is minimized.

        Arguments:
            num_buckets: Number of buckets to distribute the items into.
            prefix_sum: List of sums such that prefix_sum[n] = sum(1..n)
                of item sizes. This representation is more efficient than
                storing the item sizes themselves, since only this sum is
                needed.

        Returns:
            min_cost: Matrix giving, For a given [item, divider] combination,
                the minimum achievable variance for placing [divider-1] dividers
                between elements 1..item.
            divider_location
                Last [divider-1] divider location that achieves the matching
                lowest cost in min_cost.

        """
        n = len(prefix_sum)
        mean = prefix_sum[-1] / num_buckets

        for bucket in range(2, num_buckets + 1):
            for item in range(2, len(prefix_sum)):
                # evaluate main recurrence relation.
                min_cost_temp = np.inf
                divider_location_temp = 0
                for previous_item in range(bucket - 1, item):
                    cost = min_cost[previous_item, bucket - 1] +\
                        (
                            (prefix_sum[item] - prefix_sum[previous_item]) - mean
                        ) ** 2

                    if cost < min_cost_temp:
                        min_cost_temp = cost
                        divider_location_temp = previous_item
                min_cost[item, bucket] = min_cost_temp
                divider_location[item, bucket] = divider_location_temp

        return min_cost, divider_location


    @classmethod
    def precompile(cls):
        """Precompile any accelerator code used by this class.

        Some implementations of the Histoptimizer API rely on the compilation
        of python code to GPU or SIMD machine code. For these implementations,
        `precompile` will run a trivial problem set in order to trigger JIT
        compilation.

        For the default implementation, this is a no-op.
        """
        pass

    # noinspection DuplicatedCode
    @classmethod
    def partition(cls, item_sizes, num_buckets: int, debug_info: dict = None
                  ) -> list:
        """Given a list of item sizes, partition the items into buckets evenly.

        This function returns a set of partition indexes, or divider locations,
        such that dividing the given ordered set of items into "buckets" at
        these indexes results in a set of buckets with the lowest possible
        variance over the sum of the items sizes in each bucket.

        The base implementation is a staightforward linear implementation
        of Skiena's dynamic programming algorithm for the linear partition
        problem, with a modified cost function.

        See: _The Algorithm Design Manual_, S. Skiena. Springer, London, 2008

        Arguments:
            item_sizes: An iterable of float- or float-compatible values
                        representing a sorted series of item sizes.
            num_buckets: The number of buckets to partition the items into.
            debug_info: A dictionary that can accept debug information.

        Returns:
            A tuple:

            partition_locations: Index of dividers within items. Dividers come
                after the item in 0-based indexing and before the item in
                1-based indexing.
            min_variance: The variance of the solution defined by
                `partition_locations`
        """
        num_items = len(item_sizes)

        prefix_sum = cls.get_prefix_sums(item_sizes)

        (min_cost, divider_locs) = cls.init_matrices(num_buckets, prefix_sum)
        (min_cost, divider_locs) = cls.build_matrices(min_cost,
                                                      divider_locs,
                                                      num_buckets,
                                                      prefix_sum)

        if debug_info is not None:
            debug_info['items'] = item_sizes
            debug_info['prefix_sum'] = prefix_sum
            debug_info['min_cost'] = min_cost
            debug_info['divider_locs'] = divider_locs

        partition = cls.reconstruct_partition(divider_locs, num_items,
                                               num_buckets)
        return partition, min_cost[num_items, num_buckets] / num_buckets


def cuda_supported():
    """Returns True if Numba is installed and the system has a GPU.
    """
    try:
        from numba import cuda
        gpus = cuda.gpus
        return True
    except (cuda.CudaDriverError, cuda.CudaDriverError):
        return False


def get_partition_sums(dividers, item_sizes):
    """Get a list of the total sizes of each partition.

    Given a list of divider locations and a list of items,
    Return a list the sum of the items in each partition.

    Args:
        dividers: A list of divider locations.
        item_sizes: A list of item sizes

    Returns:
        A list where item N is the sum of the item_sizes
        in partition N.
    """
    #  TODO: fix this to take and use prefix sums
    partitions = [.0]*(len(dividers)+1)
    for x in range(0, len(dividers)+1):
        if x == 0:
            left_index = 0
        else:
            left_index = dividers[x-1]
        if x == len(dividers):
            right_index = len(item_sizes)
        else:
            right_index = dividers[x]
        for y in range(left_index, right_index):
            partitions[x] += item_sizes[y]
    return partitions


def bucket_generator(dividers: np.array, num_items: int):
    """Convert divider locations into a series of bucket numbers for items.

    Given a list of divider locations and a total number of items, yield a list
    with a bucket number item for each item in the original list of items.

    This can be made into a series and concatenated to a data frame to provide
    a partitioning column.

    Args:
        dividers: A list of divider locations. Dividers are
            considered as coming before the given list index with 0-based array
            indexing.
        num_items: The number of items in the list to be partitioned.

    Yields:
        dividers[0] 1s, followed by divider[1]-divider[0] 2s,
        divider[2]-divider[1] 3s, ... , num_items-divider[-1] Ns.

    Example:
        partitions = [12, 13, 18]  # Three dividers = 4 buckets
        num_items = 20

        Yields [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4]
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


def get_partition_series(sizes: pd.Series, num_buckets: int, partitioner):
    """
    Takes a Pandas DataFrame and returns a Series that distributes rows sequentially into the given
    number of buckets with the minimum possible standard deviation.

    Args:
        data (DataFrame): The first parameter.
        sizes (str): Column to get size values from.
        num_buckets (int): Number of buckets to partition items into.
        partitioner (function): Partitioner function

    Returns:
        pandas.Series: Series thing.
    """
    item_sizes = sizes.astype('float32').to_numpy(dtype=np.float32)
    partitions, variance = partitioner.partition(item_sizes, num_buckets)
    return pd.Series((b for b in bucket_generator(partitions, len(item_sizes))))


def histoptimize(data: pd.DataFrame, sizes: str, bucket_list: list, column_name: str,
                 partitioner, optimal_only=False):
    """
    Histoptimize takes a Pandas DataFrame and adds additional columns, one for each integer
    in bucket_list.

    The additional columns are named `column_name` + {bucket_list[i]} and contain for each
    row a bucket number such that the rows are distributed into the given number of buckets
    in such a manner as to minimize the variance/standard deviation over all buckets.

    Args:
        data (DataFrame): The DataFrame to add columns to.
        sizes (str): Column to get size values from.
        bucket_list (list): A list of integer bucket sizes.
        column_name (str): Prefix to be added to the number of buckets to get the column name.
        partitioner (class): Class that implements the Histoptimizer API.
        optimal_only (bool): If true, add only one column, for the number of buckets with the
            lowest variance.

    Returns:
        DataFrame: Original DataFrame with one or more columns added.
        list(str): List of column names added to the original DataFrame
    """
    partitions = pd.DataFrame(columns=('column_name', 'dividers', 'variance'))
    items = data[[sizes]].astype('float32').to_numpy(dtype=np.float32)
    for buckets in bucket_list:
        dividers, variance = partitioner.partition(items, buckets)
        partitions = partitions.append({
            'column_name': f'{column_name}{buckets}',
            'dividers': dividers,
            'variance': variance},
            ignore_index=True)

    if optimal_only:
        partitions = partitions[partitions.variance == partitions.variance.min()].iloc[0:1]

    columns_added = []
    for p in partitions.itertuples():
        data[p.column_name] = pd.Series((b for b in bucket_generator(p.dividers, len(items))))
        columns_added.append(p.column_name)

    return data, columns_added
