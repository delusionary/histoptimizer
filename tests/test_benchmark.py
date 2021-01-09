import click.testing

import histoptimizer.benchmark as benchmark


def test_main_succeeds():
    runner = click.testing.CliRunner()
    # FILE ID_COLUMN SIZE_COLUMN PARTITIONS
    result = runner.invoke(benchmark.cli, ['dynamic_numba', '3114', '14', '4'])

    assert result.exit_code == 0
