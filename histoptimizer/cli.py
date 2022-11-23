"""CLI wrapper for Histoptimizer functionality.

The CLI module provides `cli`, a Click-based script function that
provides the user the ability to partition an ordered set of
JSON or CSV data items into a set of "buckets" such that the variance
of the bucket sizes is minimized. Each data item must specify a size.

Copyright (C) 2022 by Kelly Joyner (de@lusion.org)

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
"""


import click
import pandas
import sys
import re

from histoptimizer import Histoptimizer, histoptimize
from histoptimizer.cuda import CUDAOptimizer
from histoptimizer.numba import NumbaOptimizer

standard_implementations = {c.name: c for c in
                            (Histoptimizer, NumbaOptimizer, CUDAOptimizer)}


def parse_set_spec(spec: str, substitute: dict = None) -> list:
    """
    Parse strings representing sets of integers, returning a tuple consisting of all integers in the specified set.

    The format is a comma-separated list of range specifications. A range specification may be a single number or two
    numbers (beginning and ending, inclusive) separated by a '-'. If two numbers, a ':' and third number may be
    supplied to provide a step. If the end number is not reachable in complete steps then the series will be truncated
    at the last valid step size.

    A dictionary may optionally be supplied that maps variable names to integer values. Variable names will be replaced
    with corresponding values before the set specification is evaluated.

    Example:
        8,15-17,19-22:2 --> (8, 15, 16, 17, 19, 21)


    """
    items = []
    if substitute is None:
        substitute = {}
    for variable, value in substitute.items():
        spec = spec.replace(variable, str(value))
    for element in spec.split(','):
        if match := re.match(r'(\d+)(?:-(\d+))?(?::(\d+))?$', element):
            g = list(map(lambda x: int(x) if x is not None else None, match.groups()))
            if g[2] is not None:
                # Range and step
                if g[1] is None:
                    raise ValueError(f'You must specify a range to specify a step. Cannot parse "{element}"')
                items.extend([x for x in range(g[0], g[1] + 1, g[2])])
            elif g[1] is not None:
                # Range
                items.extend([x for x in range(g[0], g[1] + 1)])
            else:
                # Single number
                items.extend([g[0]])
        else:
            raise ValueError(f'Could not interpret set specification "{element}" ')

    return sorted(list(set(items)))

@click.command()
@click.argument('file', type=click.File('rb'))
@click.argument('id_column', type=str)
@click.argument('size_column', type=str)
@click.argument('partitions', type=str)
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
@click.option('-i', '--implementation', type=str, default='dynamic_numba',
              help='Use the named partitioner implementation. Defaults to "dynamic_numba". If you have an NVidia GPU '
              'use "cuda" for better performance')
@click.option('-o', '--output', type=click.File('w'), default=sys.stdout,
              help='Send output to the given file. Defaults to stdout.')
@click.option('-f', '--output-format', type=click.Choice(['csv', 'json'], case_sensitive=False), default='csv',
              help='Specify output format. Pandas JSON or CSV. Defaults to CSV')
def cli(file, id_column, size_column, partitions, limit, ascending,
        print_all, column_prefix, sort_key, timing, implementation, output, output_format):
    """
    Given a CSV, a row name column, a size column, sort key, and a number of buckets, optionally sort the CSV by the
    given key, then distribute the ordered keys as evenly as possible to the given number of buckets.

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

    bucket_list = parse_set_spec(partitions)

    if implementation in standard_implementations:
        partitioner = standard_implementations[implementation]
    elif implementation in historical_implementations:
        # Do a warning
        partitioner = historical_implementations[implementation]

    data, partition_columns = histoptimize(data,
                                           size_column,
                                           bucket_list,
                                           column_prefix,
                                           partitioner.partition)

    if not print_all:
        data = data[[c for c in [id_column, sort_key, size_column] if c is not None] + partition_columns]
    if output_format == 'csv':
        data.to_csv(output, index=False)
    elif output_format == 'json':
        data.to_json(output)


if __name__ == '__main__':
    cli(sys.argv[1:])

