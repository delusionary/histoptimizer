import click.testing

import histoptimizer.cli


def test_main_succeeds():
    runner = click.testing.CliRunner()
    # FILE ID_COLUMN SIZE_COLUMN PARTITIONS
    result = runner.invoke(histoptimizer.cli.cli, ['sortframe.csv', 'id', 'size', '2-4'])

    assert result.exit_code == 0


def test_partitioners_dict():
    assert not {'dynamic', 'dynamic_numba'} - set(histoptimizer.cli.partitioners.keys())


def test_parse_set_spec():
    result = histoptimizer.cli.parse_set_spec('8,15-17,n-22:2', {'n': 19})
    assert result == [8, 15, 16, 17, 19, 21]
