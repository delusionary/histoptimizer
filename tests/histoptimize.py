import numpy as np
import pandas as pd
import pytest
import time

from histoptimizer.cli import clean_and_sort, partitioners
import histoptimizer
# TODO(kjoyner) Move get_partition_sums to __init__.py
from histoptimizer.enumerate_pandas import get_partition_sums
from histoptimizer.enumerate import partition_generator


# Test some static data sets for the naive_

@pytest.mark.skip(reason="not using this right now.")
def rando_debug():
    # data = pd.read_json('../county_narrow.json')
    # data = clean_and_sort(data, 'pct_trump16', True, 'total_population', 30, True)
    data = pd.read_json('../county_narrow.json')
    data = clean_and_sort(data, 'pct_trump16', True, 'total_population', 30, True)
    # result = histoptimize(data, 'pct_trump16', 'total_population', 6, True, False, 'bucket', True)

    cuda_debug_info = {}
    numpy_debug_info = {}
    # partition_numpy = numpy_partition(data['total_population'], 6, numpy_debug_info)
    assert True


def run_all_partitioners(items, num_buckets, exclude=[], include=None):
    """
    Compare different methods of distributing an ordered set of differently-sized items
    into a given number of buckets as evenly as possible.
    """

    results = {'items': items}
    if not include:
        include = partitioners.keys()
    for partitioner_type in set(include) - set(exclude):
        debug_info = {}
        start = time.time()
        dividers, variance = partitioners[partitioner_type](items, num_buckets, debug_info=debug_info)
        partitions = get_partition_sums(dividers, items)
        end = time.time()
        results.update({
            f'{partitioner_type}_dividers': dividers,
            f'{partitioner_type}_debug_info': debug_info,
            f'{partitioner_type}_std_dev': np.std(partitions),
            f'{partitioner_type}_max_sum': np.max(partitions),
            f'{partitioner_type}_time': end - start,
            # f'{partitioner_type}_partitions': partitions,
            f'buckets': num_buckets,
            # f'{partitioner_type}_mean_deviation_sum': sum(map(lambda a: abs(a - mean_value), partitions))
        })

    return results

    # result = histoptimize(data, 'total_population', 2, 30, 'bucket_', 'numpy')


def just_bad_results():
    items = np.array([705, 732, 799, 463, 69, 876, 346, 615, 404, 533, 746, 376, 782,
                      207, 782, 943, 36, 861, 712, 377])
    buckets = 10
    # These variables are just here for reference
    # slow_dividers =  np.array([2,  4,  7,  9, 11, 14, 15, 16, 18])
    # naive_dividers = np.array([2,  4,  7,  9, 11, 13, 15, 16, 18])
    slow_result = list(partitioners['naive'](items, buckets))
    naive_result = list(partitioners['slow'](items, buckets))
    pass


def regress(items, max_buckets):
    mean = sum(items) / len(items)
    results = {'items': items}
    for buckets in range(max_buckets, 1, -1):
        start = time.time()
        dividers, variance = partitioners['naive'](items, buckets, mean=mean)
        partitions = get_partition_sums(dividers, items)
        items = items[0:dividers[-1]]
        end = time.time()
        results.update({
            f'div_{buckets}': dividers,
            # f'{buckets}_std_dev': np.std(partitions),
            # f'{buckets}_max_sum': np.max(partitions),
            # f'{buckets}_time': end - start,
            # f'{buckets}_partitions': partitions,
            # f'buckets': num_buckets,
            # f'{buckets}_mean_deviation_sum': sum(map(lambda a: abs(a - mean_value), partitions))
        })
    return results


def partitioner_run():
    # interesting:
    #  [5, 1, 6, 8, 5] in 3 buckets, [3, 4] better than [2, 4]
    #  [10, 8, 3, 5, 2] in 4 buckets [1, 2, 3] better than [1, 2, 4]
    #  [6, 2, 1, 8, 8] in 3 buckets [2, 4] better than [1, 4]
    #  [5, 5, 6, 9, 7] in 3 buckets [2, 4] better than [2, 3]
    #  [7, 2, 7, 8, 5] in 4 buckets [1, 3, 4] better than [1, 3, 3]
    #  [5, 7, 2, 2, 7] in 4 buckets [1, 2, 4] better than [1, 3, 4]
    #  [3, 3, 5, 3, 4] in 4 buckets [2, 3, 4] better than [1, 3, 4]
    #  [10, 5, 5, 4, 2] in 3 buckets [1, 3] better than [2, 3]

    # Run the CUDA partitioner to pre-compile the kernels.
    m = partitioners['cuda_1']([1, 4, 6, 9], 3)
    m = partitioners['cuda_2']([1, 4, 6, 9], 3)
    m = partitioners['cuda_3']([1, 4, 6, 9], 3)

    results = []
    num_iterations = 1
    num_items = 10
    min_buckets = 5
    max_buckets = 5
    min_rand = 1
    max_rand = 10

    exclude = []  # ('slow', 'numpy_min_max_sum')
    include = ['cuda_1', 'cuda_2', 'cuda_3']
    for iteration in range(1, num_iterations+1):
        # items = (max_rand - min_rand) * np.random.random_sample((num_items,)) + min_rand
        items = np.random.randint(min_rand, max_rand + 1, size=num_items)
        # items = [10, 5, 5, 4, 2]
        # For each number of buckets
        for num_buckets in range(min_buckets, max_buckets + 1):
            results.append(run_all_partitioners(items, num_buckets, include=include, exclude=exclude))
            # new_result = regress(items, num_buckets)
            # results.append(new_result)
        print(f'Completed iteration {iteration}')

    r = pd.DataFrame(results)
    # interesting_results = r[r.dynamic_std_dev != r[f"{include[1]}_std_dev"]]
    # assert interesting_results.empty

    print('All partitioners agree on best results.\n')

    for p in include:
        time_column = f"{p}_time"
        print(f"{p} Average Time to put {num_items} items in {min_buckets} buckets: {r[time_column].mean()*1000:.2f}ms")

    pass

def single_test():
    debug_info = {}
    dividers = {}
    variance = {}
    items = np.random.randint(1, 10 + 1, size=15)
    for p in ('cuda_1', 'cuda_2', 'cuda_3'):
        debug_info[p] = {}
        dividers[p], variance[p] = partitioners[p](items, 10, debug_info=debug_info[p])
    pass


single_test()
#partitioner_run()
