import click.testing
from histoptimizer.cli import cli


def test_main_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0

