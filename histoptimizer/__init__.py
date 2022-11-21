import pandas as pd
import numpy as np

class Histoptimizer(object):
    """Base class for objects implementing the Histoptimizer API.

    """

    @classmethod
    def reconstruct_partition(cls, divider_location, num_items, num_buckets):
        """
        Arguments:
            divider_location
                A matrix giving the
            num_items
                The number of items to be partitioned
            num_buckets
        """
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

    @classmethod
    def get_prefix_sums(cls, items):
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

    @classmethod
    def init_matrices(cls, num_buckets, prefix_sum):
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
            num_buckets
                Number of buckets to distribute the items into.
            prefix_sum
                List of sums such that prefix_sum[n] = sum(1..n) of item sizes.
                This representation is more efficient than storing the item
                sizes themselves, since only this sum is needed.
        Returns:
            min_cost
                Matrix giving, For a given [item, divider] combination, the
                minimum achievable variance for placing [divider-1] dividers
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
            partition_locations: Index of dividers within items. Dividers come
                after the item in 0-based indexing and before the item in
                1-based indexing.
            min_variance: The variance of the solution defined by
                `partition_locations`
        """
        num_items = len(item_sizes)
        #padded_items = [0]
        #padded_items.extend(item_sizes)
        #item_sizes = padded_items

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
            partitioners[name] = m
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


def bucket_generator(dividers: np.array, num_items: int):
    """
    Iterate over a list of partitions to create a series of bucket numbers for each item in the
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


def get_partition_series(sizes: pd.Series, buckets: int, partitioner):
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
    items = sizes.astype('float32').to_numpy(dtype=np.float32)
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
        data (DataFrame): The DataFrame to add columns to.
        sizes (str): Column to get size values from.
        bucket_list (list): A list of integer bucket sizes.
        column_name (str): Prefix to be added to the number of buckets to get the column name.
        partitioner_func (function): Partitioner function
        optimal_only (bool): If true, add only one column, for the number of buckets with the
            lowest variance.

    Returns:
        DataFrame: Original DataFrame with one or more columns added.
        list(str): List of column names added to the original DataFrame
    """
    partitions = pd.DataFrame(columns=('column_name', 'dividers', 'variance'))
    items = data[[sizes]].astype('float32').to_numpy(dtype=np.float32)
    for buckets in bucket_list:
        dividers, variance = partitioner_func(items, buckets)
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
            raise ValueError("Items must be a container.")
        if num_items < 3:
            raise ValueError("Must have at least 3 items.")
        if buckets < 2:
            raise ValueError("Must request at least two buckets.")
        if buckets > num_items:
            raise ValueError("Cannot have more buckets than items.")
        return partitioner_func(items, buckets, debug_info)

    return checked_partitioner


def get_prefix_sums(cls, items):
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


class OldHistoptimizer(object):
    """Base class for objects implementing the Histoptimizer API.

    """

    @classmethod
    def _reconstruct_partition(cls, divider_location, num_items, num_buckets):
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
    @classmethod
    def _build_matrices(cls, buckets, prefix_sum):
        n = len(prefix_sum)
        min_cost = np.zeros((n, buckets + 1), dtype=np.float32)
        divider_location = np.zeros((n, buckets + 1), dtype=np.int32)
        mean = prefix_sum[-1] / buckets
        for item in range(1, len(prefix_sum)):
            # min_cost[item, 1] = prefix_sum[item]
            min_cost[item, 1] = (prefix_sum[item] - mean) ** 2
        for bucket in range(1, buckets + 1):
            min_cost[1, bucket] = (prefix_sum[1] - mean) ** 2
        for bucket in range(2, buckets + 1):
            for item in range(2, len(prefix_sum)):
                # evaluate main recurrence
                min_cost_temp = np.inf
                divider_location_temp = 0
                for previous_item in range(bucket - 1, item):
                    cost = min_cost[previous_item, bucket - 1] + \
                           (
                                   (prefix_sum[item] - prefix_sum[
                                       previous_item]) - mean
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
    def partition(cls, item_sizes, buckets, debug_info=None):
        """Given a list of item sizes, partition the items into buckets evenly.

        The list of
         Skiena's dynamic programming algorithm for the linear partition problem.

        Arguments:
            item_sizes: An ordered list of item sizes.
            num_buckets: The number of buckets to partition the items into.
            debug_info: A dictionary that can accept debug information.

        Returns:
            partition_locations: Index of dividers within items. Dividers come after the item in 0-based indexing and
            before the item in 1-based indexing.
            min_variance: The variance of the solution defined by partition_locations
        """
        num_items = len(item_sizes)
        padded_items = [0]
        padded_items.extend(item_sizes)
        item_sizes = padded_items

        prefix_sum = np.zeros((num_items + 1), dtype=np.float32)
        # Cache cumulative sums
        for item in range(1, num_items + 1):
            prefix_sum[item] = prefix_sum[item - 1] + item_sizes[item]

        (min_cost, divider_location) = cls._build_matrices(buckets, prefix_sum)

        if debug_info is not None:
            debug_info['items'] = item_sizes
            debug_info['prefix_sum'] = prefix_sum
            debug_info['min_cost'] = min_cost
            debug_info['divider_location'] = divider_location

        partition = cls._reconstruct_partition(divider_location, num_items,
                                               buckets)
        return partition, min_cost[num_items, buckets] / buckets