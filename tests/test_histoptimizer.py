import numpy as np
import pandas as pd

"""Basic correctness tests for each of the optimal partitioners.
"""
import json
import pytest
import time

from math import isclose

# If Numba CUDA sim is enabled (os.environ['NUMBA_ENABLE_CUDASIM']=='1')
# then the following import will fail, but also the corresponding errors
# will not happen, so we substitute IOError.
try:
    from numba.cuda.cudadrv.error import CudaSupportError, NvvmSupportError
except ImportError:
    CudaSupportError = IOError
    NvvmSupportError = IOError

import histoptimizer
from histoptimizer import Histoptimizer
from histoptimizer.cuda import CUDAOptimizer
from histoptimizer.numba import NumbaOptimizer
from histoptimizer.historical.cuda_2 import CUDAOptimizerItemPairs
from histoptimizer.historical.enumerate import EnumeratingOptimizer
from histoptimizer.historical.recursive import RecursiveOptimizer
from histoptimizer.historical.recursive_cache import RecursiveCacheOptimizer


@pytest.fixture()
def expected_results():
    """Expected results for variance-minimizing partitioners.

    If you find a correctness bug, add the regression test case
    to this file.
    """
    with open('fixtures/expected_results.json') as file:
        return json.load(file)


@pytest.fixture()
def min_cost_divider_tests():
    """Matrix correctness test fixtures.

    These matrix correctness fixtures are generated, random
    optimization problems. They are segregated from `expected_results`
    to keep that fixture easily human-readable.
    """
    with open('fixtures/min_cost_divider_tests.json') as file:
        return json.load(file)


optimal_partitioners = (
    Histoptimizer,
    NumbaOptimizer,
    EnumeratingOptimizer,
    RecursiveOptimizer,
    RecursiveCacheOptimizer,
    CUDAOptimizer,
    CUDAOptimizerItemPairs,
)

# These dynamic-programming partitioners store and return min_cost and
# divider_location arrays that can be tested for correctness.
dynamic_partitioners = (
    Histoptimizer,
    NumbaOptimizer,
    CUDAOptimizer,
    CUDAOptimizerItemPairs
)


@pytest.mark.parametrize("value", ("dividers", "variance"))
@pytest.mark.parametrize("partitioner", optimal_partitioners)
def test_static_correctness(expected_results, partitioner, value):
    for test in expected_results:
        try:
            dividers, variance = partitioner.partition(test['item_sizes'],
                                                       test['num_buckets'])
        except (NvvmSupportError, CudaSupportError) as e:
            pytest.skip("Cuda support not available.")
        matching_dividers = [list(dividers) == d for d in test['dividers']]
        if value == "dividers":
            assert any(matching_dividers)
        elif value == "variance":
            assert isclose(variance, test['variance'], rel_tol=1e-04)
    pass


@pytest.mark.parametrize("artifact", ('min_cost', 'divider_location'))
@pytest.mark.parametrize("partitioner", dynamic_partitioners)
def test_matrix_correctness(min_cost_divider_tests, partitioner, artifact):
    for test in min_cost_divider_tests:
        debug_info = {}
        try:
            dividers, variance = partitioner.partition(test['item_sizes'],
                                                       test['num_buckets'],
                                                       debug_info)
        except (NvvmSupportError, CudaSupportError):
            pytest.skip("Cuda support not available.")

        # Column 0 and row 0 do not affect results; we remove them
        # to avoid brittleness/spurious failures.

        shared_mem = debug_info.get('shared_mem')
        test_divider_location = debug_info['divider_location'][1:, 1:]
        ref_divider_location = np.array(test['divider_location'])[1:, 1:]

        if artifact == "divider_location":
            # Remove unreferenced row/column to avoid spurious failures.
            # test_divider_location = debug_info['divider_location'][1:, 1:]
            # ref_divider_location = np.array(test['divider_location'])[1:, 1:]
            equals = np.equal(
                ref_divider_location,
                test_divider_location
            )
            divider_location_match = np.all(equals)
            if not divider_location_match:
                pass
            assert divider_location_match
        elif artifact == "min_cost":
            # Remove unreferenced row/column to avoid spurious failures.
            test_min_cost = debug_info['min_cost'][1:, 1:]
            ref_min_cost = np.array(test['min_cost'])[1:, 1:]
            min_cost_isclose = np.isclose(
                test_min_cost,
                ref_min_cost,
                atol=0.00001
            )
            infinities = np.isinf(test_min_cost)
            ref_infinities = np.isinf(ref_min_cost)
            same_infinities = np.all(np.equal(infinities, ref_infinities))

            assert same_infinities

            min_cost_match = np.all(
                np.logical_or(
                    min_cost_isclose,
                    infinities,
                )
            )

            assert min_cost_match


@pytest.fixture
def partitioner():
    class Partitioner(object):
        @classmethod
        def partition(cls, items, buckets, debug_info=None):
            if buckets == 2:
                return np.array([2]), 10
            else:
                return np.array([1, 3]), 5

    return Partitioner


@pytest.fixture
def histo_df():
    return pd.DataFrame({'id': [1, 2, 3, 4], 'sizes': [10, 20, 30, 40]})


def test_get_partition_sums():
    sums = histoptimizer.get_partition_sums([1, 3, 5], [3, 7, 4, 2, 1, 9])
    assert list(sums) == [3, 11, 3, 9]


def test_bucket_generator():
    dividers = np.array([1, 3, 5], dtype=int)
    bucket_values = histoptimizer.bucket_generator(dividers, 7)
    assert list(bucket_values) == [1, 2, 2, 3, 3, 4, 4]


def test_get_prefix_sums():
    prefix_sums = Histoptimizer.get_prefix_sums([1, 2, 3, 4])
    assert list(prefix_sums) == [0.0, 1.0, 3.0, 6.0, 10.0]


def test_partition_series(partitioner, histo_df):
    result = histoptimizer.get_partition_series(histo_df, 3, partitioner)

    s = pd.Series([1, 2, 2, 3])
    assert result.equals(s)


def test_histoptimize(partitioner, histo_df):
    result, columns = histoptimizer.histoptimize(histo_df, 'sizes', [2, 3],
                                                 'partitioner_', partitioner)

    assert result['partitioner_2'].equals(pd.Series([1, 1, 2, 2]))
    assert result['partitioner_3'].equals(pd.Series([1, 2, 2, 3]))


def test_histoptimize_optimal_only(partitioner, histo_df):
    result, columns = histoptimizer.histoptimize(histo_df, 'sizes', [2, 3],
                                                 'partitioner_', partitioner,
                                                 optimal_only=True)

    assert result['partitioner_3'].equals(pd.Series([1, 2, 2, 3]))
    assert set(result.columns) == {'id', 'sizes', 'partitioner_3'}


def test_check_parameters():
    with pytest.raises(ValueError):
        Histoptimizer.check_parameters([1, 2, 3, 4], 1, {})

    with pytest.raises(ValueError):
        Histoptimizer.check_parameters([1, 2, 3], 5, {})

    with pytest.raises(ValueError):
        Histoptimizer.check_parameters(5, 2, {})

    with pytest.raises(ValueError):
        Histoptimizer.check_parameters([1, 2], 1, {})

    with pytest.raises(ValueError):
        Histoptimizer.check_parameters([1, 2], 1, 'hi')
