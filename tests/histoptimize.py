import numpy as np
import pandas as pd
import pytest
import time

from histoptimizer.cli import clean_and_sort
from histoptimizer import partitioner, partitioners
from histoptimizer.naive_partition import get_partition_sums, partition_generator


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
        dividers = partitioner(partitioner_type)(items, num_buckets)
        partitions = get_partition_sums(dividers, items)
        end = time.time()
        results.update({
            f'{partitioner_type}_dividers': dividers,
            # f'{partitioner_type}_debug_info': debug_info,
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
    # items = [7, 41, 97, 53, 67, 24]
    # num_buckets = 3
    # Produces different answers from cuda, naive, and numpy_min_max_sum
    #
    # Try five items 3-4 buckets.
    #
    # interesting_results.json contains other instances
    # data = pd.read_json('../county_narrow.json')
    # data = clean_and_sort(data, 'nonwhite_pct', True, 'total_population', 2000, True)
    # items = data['total_population'].to_numpy(dtype=np.float32)

    # Run the CUDA partitioner to compile the kernels.
    m = partitioner('cuda')([1, 4, 6, 9], 3)
    results = []
    num_iterations = 100
    num_items = 5
    min_buckets = 3
    max_buckets = 3
    min_rand = 1
    max_rand = 10

    exclude = [] # ('slow', 'numpy_min_max_sum')
    include = ['naive', 'recursive']
    for iteration in range(0, num_iterations+1):
        # Get a list of 10 random integers from ~ 0.0000005 to 100
        # items = (max_rand - min_rand) * np.random.random_sample((num_items,)) + min_rand
        items = np.random.randint(min_rand, max_rand + 1, size=num_items)
        # For each number of dividers 1-6:
        for num_buckets in range(min_buckets, max_buckets + 1):
            results.append(run_all_partitioners(items, num_buckets, include=include, exclude=exclude))
            # new_result = regress(items, num_buckets)
            # results.append(new_result)
        print(f'Completed iteration {iteration}')

    r = pd.DataFrame(results)
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


partitioner_run()
