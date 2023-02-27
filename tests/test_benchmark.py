from pathlib import Path
import click.testing
import pandas as pd
import numpy as np
import pytest

import histoptimizer.benchmark as benchmark
from histoptimizer import Histoptimizer
from histoptimizer.numba_optimizer import NumbaOptimizer


@pytest.fixture
def pivot_benchmark():
    return pd.read_json('fixtures/pivot_benchmark.json')


@pytest.fixture
def specified_items_list():
    return benchmark.get_sizes_from('fixtures/sizes_only.csv')


@pytest.fixture
def partitioner_list():
    return [Histoptimizer, NumbaOptimizer]


def test_main_succeeds(tmp_path):
    report_file = tmp_path / 'report.json'
    runner = click.testing.CliRunner()
    # FILE ID_COLUMN SIZE_COLUMN PARTITIONS
    runner.invoke(benchmark.cli, ['--sizes-from', 'fixtures/sizes_only.csv',
                                  '--report', str(report_file),
                                  'numba,dynamic', '5-6', '3-4', '1'])
    filed_report = pd.read_json(str(report_file), orient='records').drop(
        ['variance', 'elapsed_seconds', 'item_set_id'], axis=1)
    expected_report = pd.read_json('fixtures/benchmark_report.json')
    assert np.array_equal(filed_report.to_numpy(), expected_report.to_numpy())


def test_benchmark(partitioner_list, specified_items_list):
    result = benchmark.benchmark(partitioner_list, (4, 5, 6), (3, 4),
                                 specified_items_sizes=specified_items_list)
    result = result.drop(['elapsed_seconds', 'variance', 'item_set_id'], axis=1)
    # Turn the dividers numpy array into a list to make it round-trip safe.
    result['dividers'] = result['dividers'].apply(list)
    expected_report = pd.read_json('fixtures/test_benchmark_report.json')
    assert np.array_equal(result.to_numpy(), expected_report.to_numpy())


def test_get_sizes_from():
    result = benchmark.get_sizes_from('fixtures/sizes_only.csv')
    assert [int(x) for x in result] == [1, 2, 5, 3, 9, 9]


def test_get_sizes_from_input_validation():
    with pytest.raises(ValueError):
        benchmark.get_sizes_from('fixtures/sortframe.csv')


def test_partitioner_pivot(pivot_benchmark):
    pivot = benchmark.partitioner_pivot(pivot_benchmark, 'dynamic')

    expected_dynamic = pd.read_json('fixtures/pivot_dynamic.json')
    assert np.array_equal(expected_dynamic.to_numpy(), pivot.to_numpy())

    pivot = benchmark.partitioner_pivot(pivot_benchmark, 'numba')
    expected_dynamic_numba = pd.read_json('fixtures/pivot_dynamic_numba.json')
    assert np.array_equal(expected_dynamic_numba.to_numpy(), pivot.to_numpy())


def test_echo_tables(pivot_benchmark, partitioner_list, capsys):
    benchmark.echo_tables(partitioner_list, pivot_benchmark)
    out, err = capsys.readouterr()
    expected = Path('fixtures/echo_table_output.txt').read_text()
    assert out == expected


def test_get_system_info():
    result = benchmark.get_system_info()
    assert result


def test_get_partitioners():
    result = {p.name for p in
              benchmark.get_partitioners(
                  'numba,custom_module_fixture.py:SampleOptimizer')}
    assert result == {'numba', 'sample'}
