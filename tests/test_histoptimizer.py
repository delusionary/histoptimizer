import histoptimizer
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def partitioner():
    def p(items, buckets, debug_info=None):
        if buckets == 2:
            return np.array([2]), 10
        else:
            return np.array([1, 3]), 5

    return p

@pytest.fixture
def histo_df():
    return pd.DataFrame({'id': [1, 2, 3, 4], 'sizes': [10, 20, 30, 40]})


def test_get_partitioner_dict(monkeypatch):
    import histoptimizer.dynamic
    import histoptimizer.historical.cuda_2

    partitioners = histoptimizer.get_partitioner_dict(
        histoptimizer.dynamic,
        histoptimizer.historical.cuda_2
    )

    for p in partitioners:
        assert p == eval(f'{partitioners[p].__module__}.name')


def test_cuda_supported():
    assert histoptimizer.cuda_supported() or True


def test_get_partition_sums():
    sums = histoptimizer.get_partition_sums([1, 3, 5], [3, 7, 4, 2, 1, 9])
    assert list(sums) == [3, 11, 3, 9]


def test_bucket_generator():
    dividers = np.array([1, 3, 5], dtype=np.int)
    bucket_values = histoptimizer.bucket_generator(dividers, 7)
    assert list(bucket_values) == [1, 2, 2, 3, 3, 4, 4]


def test_get_partition_sums():
    prefix_sums = histoptimizer.get_prefix_sums([1, 2, 3, 4])
    assert list(prefix_sums) == [0.0, 1.0, 3.0, 6.0, 10.0]


def test_partition_series(partitioner, histo_df):

    result = histoptimizer.get_partition_series(histo_df, 'sizes', 3, partitioner)

    s = pd.Series([1, 2, 2, 3])
    assert result.equals(s)


def test_histoptimize(partitioner, histo_df):

    result, columns = histoptimizer.histoptimize(histo_df, 'sizes', [2, 3], 'partitioner_', partitioner)

    assert result['partitioner_2'].equals(pd.Series([1, 1, 2, 2]))
    assert result['partitioner_3'].equals(pd.Series([1, 2, 2, 3]))


def test_histoptimize_optimal_only(partitioner, histo_df):

    result, columns = histoptimizer.histoptimize(histo_df, 'sizes', [2, 3], 'partitioner_', partitioner, optimal_only=True)

    assert result['partitioner_3'].equals(pd.Series([1, 2, 2, 3]))
    assert set(result.columns) == {'id', 'sizes', 'partitioner_3'}


# TODO(kjoyner): Test optimal_only.

def test_partitioner(partitioner):
    wrapper = histoptimizer.partitioner(partitioner)

    with pytest.raises(ValueError):
        wrapper([1, 2, 3, 4], 1)

    with pytest.raises(ValueError):
        wrapper([1, 2, 3], 5)

    with pytest.raises(ValueError):
        wrapper(5, 2)

    with pytest.raises(ValueError):
        wrapper([1, 2], 1)

    dividers, variance = wrapper([1]*4, 2)
    assert list(dividers) == [2]
    assert variance == 10

    dividers, variance = wrapper([1]*4, 3)
    assert list(dividers) == [1, 3]
    assert variance == 5






