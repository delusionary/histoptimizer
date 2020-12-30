import click
import pandas
from time import time
import sys

from histoptimizer import histoptimize

import histoptimizer.recursive
import histoptimizer.recursive_cache
import histoptimizer.recursive_verbose
import histoptimizer.cuda_1
import histoptimizer.cuda_2
import histoptimizer.cuda_3
import histoptimizer.dynamic
import histoptimizer.enumerate
import histoptimizer.pandas

partitioner = histoptimizer.get_partitioner_dict(
    histoptimizer.pandas,
    histoptimizer.enumerate,
    histoptimizer.dynamic,
    histoptimizer.cuda_1,
    histoptimizer.cuda_2,
    histoptimizer.cuda_3,
    histoptimizer.recursive_cache,
    histoptimizer.recursive_verbose,
    histoptimizer.recursive
)

def clean_and_sort(data, sort_key, ascending, sizes, max_rows, silent_discard=False):
    """
    Performs some optional house-keeping on input files.
    """
    oldlen = len(data.index)
    if sort_key is not None:
        data = data[(data[sort_key].isna() == False)]
        if len(data.index) != oldlen and not silent_discard:
            raise ValueError('Some rows have invalid or missing sort key values.')
        oldlen = len(data.index)
        data = data.sort_values(by=sort_key, ascending=True)
    data = data[data[sizes].isna() == False].reset_index()
    if len(data.index) != oldlen and not silent_discard:
        raise ValueError('Some rows have invalid, zero, or missing size values.')

    if max_rows:
        data = data.truncate(after=max_rows - 1)

    return data


@click.command()
@click.argument('file', type=click.File('rb'))
@click.argument('id_column', type=str)
@click.argument('size_column', type=str)
@click.argument('partitions', type=click.IntRange(2, 3000), nargs=-1)
@click.option('-l', '--limit', type=int, default=None,
              help='Take the first {limit} records from the input, rather than the whole file.')
@click.option('-a/-d', '--ascending/--descending', '--asc/--desc', default=True,
              help='If a sort column is provided, ')
@click.option('--print-all/--no-print-all', '--all/--brief', default=False,
              help='Output all columns in input, or with --brief, only output the ID, size, and buckets columns.')
@click.option('-c', '--column-prefix', type=str, default=None,
              help='Partition column name prefix. The number of buckets will be appended. '
                   'Defaults to partion_{number of buckets}.')
@click.option('-s', '--sort-key', type=str, default=None,
              help='Optionally sort records by this column name before partitioning.')
@click.option('-t', '--timing/--no-timing', default=False, help='Print partitioner timing information to stderr')
@click.option('-i', '--implementation', type=str, default='numpy',
              help='Use the named partitioner implementation. Defaults to "numpy". If you have an NVidia GPU '
              'use "cuda" for better performance')
@click.option('-o', '--output', type=click.File('w'), default=sys.stdout,
              help='Send output to the given file. Defaults to stdout.')
@click.option('-f', '--output-format', type=click.Choice(['csv', 'json'], case_sensitive=False), default='csv',
              help='Specify output format. Pandas JSON or CSV. Defaults to CSV')
def cli(file, id_column, size_column, partitions, limit, ascending,
        print_all, column_prefix, sort_key, timing, implementation, output, output_format):
    """
    Given a CSV, a row name column, a size column, sort key, and a number of buckets, sort the CSV by the given key, then distribute
    the ordered keys as evenly as possible to the given number of buckets.

    > histoptimizer states.csv state_name population 10
    state_name, population, partition_10
    Wyoming, xxxxxx, 1
    California, xxxxxxxx, 10
    """
    if file.name.endswith('json'):
        data = pandas.read_json(file)
    else:
        data = pandas.read_csv(file)

    if limit:
        data = data.truncate(after=limit - 1)
    if column_prefix is None:
        column_prefix = 'partition_'
    if sort_key:
        data = data.sort_values(sort_key, ascending=ascending).reset_index()

    for num_buckets in partitions:
        start = time()
        data[f'{column_prefix}{num_buckets}'] = histoptimize(data, size_column, num_buckets, column_prefix, implementation)
        end = time()
        click.echo(f"Executed in {end-start} seconds.", err=True)

    if not print_all:
        data = data[[id_column, sort_key, size_column] + [col for col in data if col.startswith(column_prefix)]]
    if output_format == 'csv':
        data.to_csv(output, index=False)
    elif output_format == 'json':
        data.to_json(output)


if __name__ == '__main__':
    cli(sys.argv[1:])

