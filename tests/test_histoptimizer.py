import numpy as np
import pandas as pd

"""Basic correctness tests for each of the optimal partitioners.
"""
import json
import pytest

from math import isclose

# If numba CUDA sim is enabled (os.environ['NUMBA_ENABLE_CUDASIM']='1')
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
from histoptimizer.historical.cuda_1 import CUDAOptimizerBuckets
from histoptimizer.historical.cuda_2 import CUDAOptimizerItemPairs
from histoptimizer.historical.dynamic_numba_2 import NumbaOptimizerDraft2
from histoptimizer.historical.dynamic_numba_3 import NumbaOptimizerDraft3
from histoptimizer.historical.enumerate import EnumeratingOptimizer
from histoptimizer.historical.recursive import RecursiveOptimizer
from histoptimizer.historical.recursive_cache import RecursiveCacheOptimizer
from histoptimizer.historical.recursive_verbose import RecursiveVerboseOptimizer


@pytest.fixture()
def expected_results():
    with open('fixtures/expected_results.json') as file:
        return json.load(file)


optimal_partitioners = (
    Histoptimizer,
    NumbaOptimizer,
    NumbaOptimizerDraft2,
    NumbaOptimizerDraft3,
    EnumeratingOptimizer,
    RecursiveOptimizer,
    RecursiveCacheOptimizer,
    RecursiveVerboseOptimizer,
    CUDAOptimizer,
    CUDAOptimizerBuckets,
    CUDAOptimizerItemPairs,
)


@pytest.mark.parametrize("partitioner", optimal_partitioners)
def test_static_correctness(expected_results, partitioner):
    for test in expected_results:
        try:
            dividers, variance = partitioner.partition(test['items'],
                                                       test['buckets'])
        except (NvvmSupportError, CudaSupportError):
            pytest.skip("Cuda support not available.")
        matching_dividers = [list(dividers) == d for d in test['dividers']]
        assert any(matching_dividers)
        assert isclose(variance, test['variance'], rel_tol=1e-04)
    pass

    # Ad Hoc test
    # def test_single_test():
    #     debug_info = {}
    #     dividers = {}
    #     variance = {}
    #     elapsed_seconds = {}
    #     items = [5, 1, 6, 8, 5]
    #     num_buckets = 3
    #     for pt in (Histoptimizer, CUDAOptimizer):
    #         debug_info[pt.name] = {}
    #         start = time.time()
    #         dividers[pt.name], variance[pt.name] = pt.partition(
    #             items, num_buckets,
    #             debug_info=debug_info[pt.name]
    #         )
    #         end = time.time()
    #         elapsed_seconds[pt.name] = end - start

    # Ensure the dividers returned are all the same.
    some_dividers = list(dividers[next(iter(dividers))])
    assert all([list(dividers[d]) == some_dividers for d in dividers])


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
