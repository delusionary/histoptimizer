import numpy as np
from math import isclose
import pandas as pd
import json
import pytest
import time

from histoptimizer.cli import partitioners

from histoptimizer import get_partition_sums

optimal_partitioners = (
    "dynamic",
    "dynamic_numba",
    "cuda_1",
    "cuda_2",
    "cuda_3",
    "enumerate",
    "enumerate_pandas",
    "recursive",
    "recursive_cache",
    "recursive_verbose",
)

@pytest.fixture()
def expected_results():
    with open('./expected_results.json') as file:
        return json.load(file)


@pytest.mark.parametrize("partitioner", optimal_partitioners)
def test_static_correctness(expected_results, partitioner):
    for test in expected_results:
        dividers, variance = partitioners[partitioner].partition(test['items'], test['buckets'])
        test['variance'] = variance
        print(f"Items: {test['items']} Buckets: {test['buckets']}")
        assert any([list(dividers) == d for d in test['dividers']])
        assert isclose(variance, test['variance'], rel_tol=1e-04)
    pass


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
        dividers, variance = partitioners[partitioner_type].partition(items, num_buckets, debug_info=debug_info)
        partitions = get_partition_sums(dividers, items)
        end = time.time()
        results.update({
            f'{partitioner_type}_dividers': dividers,
            f'{partitioner_type}_debug_info': debug_info,
            f'{partitioner_type}_std_dev': np.std(partitions),
            f'{partitioner_type}_max_sum': np.max(partitions),
            f'{partitioner_type}_time': end - start,
            f'buckets': num_buckets,
        })

    return results

def test_random_data():
    """
    This is a test function I use for IDE Debugging
    """

    # Run the CUDA partitioner to pre-compile the kernels.
    m = partitioners['cuda_1'].partition([1, 4, 6, 9], 3)
    m = partitioners['cuda_2'].partition([1, 4, 6, 9], 3)
    m = partitioners['cuda_3'].partition([1, 4, 6, 9], 3)

    results = []
    num_iterations = 1
    item_list = range(5, 10)
    bucket_list = range(2, 4)
    min_rand = 1
    max_rand = 10

    exclude = []  # ('slow', 'numpy_min_max_sum')
    include = ['dynamic_numba', 'dynamic_numba_2']
    for num_items in item_list:
        for num_buckets in bucket_list:
            for iteration in range(1, num_iterations+1):
                # To use floating point uncomment this, but integer sizes are much better for debugging.
                # items = (max_rand - min_rand) * np.random.random_sample((num_items,)) + min_rand
                items = np.random.randint(min_rand, max_rand + 1, size=num_items)
                results.append(run_all_partitioners(items, num_buckets, include=include, exclude=exclude))
                print(f'Completed iteration {iteration}')

    r = pd.DataFrame(results)
    interesting_results = r[r.dynamic_numba_std_dev != r[f"{include[1]}_std_dev"]]
    assert interesting_results.empty

    print('All partitioners agree on best results.\n')

    pass


def test_single_test():
    # Interesting result with cuda_2 and dynamic_numba:
    # array([2, 9, 2, 3, 4, 9, 6, 1, 7, 4, 6, 6, 1, 1, 7])
    # buckets=10
    # dynamic_numba = array([ 2,  4,  5,  6,  8,  9, 10, 11, 14]) variance 43.6
    # cuda_2 = array([ 2,  4,  5,  6,  8,  9, 10, 11, 13]) variance 43.60000116
    debug_info = {}
    dividers = {}
    variance = {}
    elapsed_seconds = {}
    items = [5, 1, 6, 8, 5]
    for p in ('dynamic_numba', 'dynamic_numba_3'):
        debug_info[p] = {}
        start = time.time()
        dividers[p], variance[p] = partitioners[p].partition(items, 3, debug_info=debug_info[p])
        end = time.time()
        elapsed_seconds[p] = end - start

    # Ensure the dividers returned are all the same.
    some_dividers = list(dividers[next(iter(dividers))])
    assert all([list(dividers[d]) == some_dividers for d in dividers])


