from pathlib import Path
import click.testing
import pandas as pd
import numpy as np
import pytest

import histoptimizer.benchmark as benchmark

@pytest.fixture
def pivot_benchmark():
    return pd.read_json('pivot_benchmark.json')


def test_main_succeeds(tmp_path):
    report_file = tmp_path / 'report.json'
    runner = click.testing.CliRunner()
    # FILE ID_COLUMN SIZE_COLUMN PARTITIONS
    result = runner.invoke(benchmark.cli, ['--sizes-from', 'sizes_only.csv',
                                           '--report', str(report_file),
                                           'dynamic_numba,dynamic', '5-6', '3-4', '1'])
    filed_report = pd.read_json(str(report_file)).drop(['variance', 'elapsed_seconds'], axis=1)
    expected_report = pd.read_json('benchmark_report.json')
    assert np.array_equal(filed_report.to_numpy(), expected_report.to_numpy())


def test_get_sizes_from():
    result = benchmark.get_sizes_from('sizes_only.csv')
    assert [int(x) for x in result] == [1, 2, 5, 3, 9, 9]


def test_get_sizes_from_input_validation():
    with pytest.raises(ValueError):
        benchmark.get_sizes_from('sortframe.csv')


def test_partitioner_pivot(pivot_benchmark):

    pivot = benchmark.partitioner_pivot(pivot_benchmark, 'dynamic')

    expected_dynamic = pd.read_json('pivot_dynamic.json')
    assert np.array_equal(expected_dynamic.to_numpy(), pivot.to_numpy())

    pivot = benchmark.partitioner_pivot(pivot_benchmark, 'dynamic_numba')
    expected_dynamic_numba = pd.read_json('pivot_dynamic_numba.json')
    assert np.array_equal(expected_dynamic_numba.to_numpy(), pivot.to_numpy())


def test_echo_tables(pivot_benchmark, capsys):
    benchmark.echo_tables(('dynamic', 'dynamic_numba'), pivot_benchmark)
    out, err = capsys.readouterr()
    expected = Path('echo_table_output.txt').read_text()
    assert out == expected



