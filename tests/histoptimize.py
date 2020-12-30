import numpy as np
import pandas as pd
import pytest
import time

from histoptimizer.cli import clean_and_sort
import histoptimizer
# TODO(kjoyner) Move get_partition_sums to __init__.py
from histoptimizer.pandas import get_partition_sums
from histoptimizer.enumerate import partition_generator

import histoptimizer.recursive
import histoptimizer.recursive_cache
import histoptimizer.recursive_verbose
import histoptimizer.cuda_1
import histoptimizer.cuda_2
import histoptimizer.cuda_3
import histoptimizer.dynamic
import histoptimizer.enumerate
import histoptimizer.pandas

partitioner = histoptimizer.get_partitioner_dict(
    histoptimizer.pandas,
    histoptimizer.enumerate,
    histoptimizer.dynamic,
    histoptimizer.cuda_1,
    histoptimizer.cuda_2,
    histoptimizer.cuda_3,
    histoptimizer.recursive_cache,
    histoptimizer.recursive_verbose,
    histoptimizer.recursive
)


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
        dividers = partitioner[partitioner_type](items, num_buckets, debug_info=debug_info)
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
    slow_result = list(partitioner('naive')(items, buckets))
    naive_result = list(partitioner('slow')(items, buckets))
    pass


def regress(items, max_buckets):
    mean = sum(items) / len(items)
    results = {'items': items}
    for buckets in range(max_buckets, 1, -1):
        start = time.time()
        dividers = partitioner('naive')(items, buckets, mean=mean)
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
    m = partitioner['cuda_1']([1, 4, 6, 9], 3)
    m = partitioner['cuda_2']([1, 4, 6, 9], 3)
    m = partitioner['cuda_3']([1, 4, 6, 9], 3)

    results = []
    num_iterations = 1
    num_items = 400
    min_buckets = 50
    max_buckets = 50
    min_rand = 1
    max_rand = 10

    exclude = []  # ('slow', 'numpy_min_max_sum')
    include = ['cuda_1', 'cuda_2', 'cuda_3', 'dynamic']
    for iteration in range(1, num_iterations+1):
        items = (max_rand - min_rand) * np.random.random_sample((num_items,)) + min_rand
        # items = np.random.randint(min_rand, max_rand + 1, size=num_items)
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


def holding(r):
    # for p in set(partitioners.keys() - set(exclude)):
    #    results[f'{p}_ratio'] = results[f'{p}_std_dev'] / results['naive_std_dev']
    interesting_results = r[(r.cuda_std_dev > r.naive_std_dev)][
        [
            'items',
            'buckets',
            'naive_std_dev',
            'numpy_std_dev',
            'cuda_std_dev',
            'naive_time',
            'numpy_time',
            'cuda_time',
        ]
    ]

    pass

def single_test():
    debug_info = {}
    result = {}
    #items = [8, 7, 3, 9, 8, 7, 9, 9, 3, 6]
    items = np.random.randint(1, 10 + 1, size=1000)
    for p in ('cuda_1', 'cuda_3'):
        debug_info[p] = {}
        result[p] = partitioner[p](items, 6, debug_info=debug_info[p])

    pass


#single_test()
partitioner_run()
