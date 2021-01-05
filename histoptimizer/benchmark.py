import re
import sys
import time
import os

import click
import numpy as np
import pandas as pd
from numba import cuda
from math import ceil, log10

from histoptimizer.cli import partitioners, parse_set_spec

#  os.environ['NSIGHT_CUDA_DEBUGGER'] = '1'


def partitioner_pivot(df: pd.DataFrame, partitioner: str) -> pd.DataFrame:
    """

    """
    return df[df.partitioner == partitioner].groupby(
        ['num_items', 'buckets'],
        as_index=False)\
        .mean()\
        .pivot(index=['num_items'],
                      columns='buckets',
                      values='elapsed_seconds')


def benchmark(partitioner_list: list, item_list: list, bucket_list: list, iterations: int =1,
              begin_range: int = 1, end_range: int = 10, specified_items_sizes: list = None, verbose: bool = False)\
        -> pd.DataFrame:
    """

    """
    r = pd.DataFrame(columns=(
    'partitioner', 'num_items', 'buckets', 'iteration', 'variance', 'elapsed_seconds', 'dividers', 'items'))
    for num_items in item_list:
        for num_buckets in bucket_list:
            results = []
            for i in range(1, iterations + 1):
                if specified_items_sizes is None:
                    items = np.random.randint(begin_range, end_range + 1, size=num_items)
                else:
                    items = specified_items_sizes[:num_items]
                for partitioner in partitioner_list:
                    start = time.time()
                    dividers, variance = partitioners[partitioner](items, num_buckets)
                    end = time.time()
                    results.append({
                        'partitioner': partitioner,
                        'num_items': num_items,
                        'buckets': num_buckets,
                        'iteration': i,
                        'variance': variance,
                        'elapsed_seconds': end - start,
                        'dividers': dividers,
                        'items': items
                    })
            r = r.append(results)
            mean = r[(r.num_items == num_items) & (r.buckets == num_buckets)].groupby('partitioner').mean()
            if verbose:
                click.echo(f'Items: {num_items} Buckets: {num_buckets} Mean values over {iterations} iterations:')
                click.echo(f'Partitioner\t\tTime (ms)\t\tVariance')
                for partitioner, record in mean.iterrows():
                    click.echo(f'{partitioner}\t\t\t{record.elapsed_seconds * 1000:.2f}\t\t\t{record.variance:.4f}')
    return r


def echo_tables(partitioner_list: list, r: pd.DataFrame):
    for p in partitioner_list:
        grid = partitioner_pivot(r, p)
        items_width = ceil(max(log10(grid.index.max()), 1)) + 2  # wide enough for the widest num_items value.
        width = ceil(max(log10(grid.max().max()), 1)) + 6  # Max decimal digits we have + ".000" + 2 spaces
        click.echo(f'Partitioner: {p}\n{"".rjust(items_width)}' + ''.join([str(x).rjust(width) for x in grid.columns]))
        for num_items in grid.index:
            click.echo(str(num_items).rjust(items_width) + ''.join(
                [f'{float(grid[grid.index == num_items][x]):.3f}'.rjust(width) for x in grid.columns]))
        click.echo()


def get_sizes_from(sizes_from: str) -> list:
    specified_items_sizes = None
    if sizes_from is not None:
        if '.json' in sizes_from:
            specified_items = pd.read_json(sizes_from, orient='records')
        elif '-' == sizes_from:
            specified_items = pd.read_csv(sys.stdin)
        else:
            specified_items = pd.read_csv(sizes_from)
        if len(specified_items.columns) != 1:
            raise ValueError(f'Files specified with --sizes-from must contain a CSV or JSON DataFrame with one (1)'
                             f'column. Found {len(specified_items)} columns instead.')
        try:
            specified_items_sizes = np.array(specified_items[specified_items.columns[0]], dtype=np.float32)
        except ValueError as e:
            raise ValueError(f'Files specified with --sizes-from must contain a single column of Float32-coercible'
                             f'values: {str(e)}')
    return specified_items_sizes


def write_report(r: pd.DataFrame, report: str):
    if ".json" in report.lower():
        r.to_json(report, orient="records")
    elif report == '-':
        r.to_csv(sys.stdout, index=False)
    else:
        r.to_csv(report, index=False)

@click.command()
@click.argument('partitioner_types', type=str, required=True)
@click.argument('item_spec', type=str, default="15")
@click.argument('bucket_spec', type=str, default="8")
@click.argument('iterations', type=int, default=1)
@click.argument('size_spec', type=str, default='1-10')
@click.option('--debug-info/--no-debug-info', type=bool, default=False)
@click.option('--force-jit/--no-force-jit', type=bool, default=True)
@click.option('--report', type=click.Path(writable=True, allow_dash=False))
@click.option('--sizes-from', type=click.Path(exists=True, allow_dash=True), default=None)
@click.option('--tables/--no-tables', type=bool, default=False)
@click.option('--verbose/--no-verbose', type=bool, default=False)
def cli(partitioner_types, item_spec, bucket_spec, iterations, size_spec,
        debug_info, force_jit, report, sizes_from, tables, verbose):
    """
    Histobench is a benchmarking harness for testing Histoptimizer partitioner performance.

    It uses random data, and so may not be an accurate benchmark for algorithms whose performance depends on
    the data set.

    """
    #cuda.select_device(0)
    #cuda.profile_start()
    # Parse arguments
    partitioner_list = partitioner_types.split(',')

    specified_items_sizes = get_sizes_from(sizes_from)
    item_variable_dict = {}
    if specified_items_sizes is not None:
        item_variable_dict['n'] = len(specified_items_sizes)
    bucket_list = parse_set_spec(bucket_spec)
    item_list = parse_set_spec(item_spec, item_variable_dict)
    if match := re.match(r'(\d+)-(\d+)$', size_spec):
        begin_range, end_range = map(int, match.groups())
        if end_range < begin_range:
            begin_range, end_range = end_range, begin_range
    else:
        raise ValueError("Size spec must be two numbers separated by a dash: e.g. 1-10")

    if force_jit:
        for p in {'cuda_1', 'cuda_2', 'cuda_3', 'dynamic_numba', 'dynamic_numba_2'} & set(partitioner_list):
            partitioners[p]([1, 2, 3], 2)

    r = benchmark(partitioner_list, item_list, bucket_list,
                  iterations, begin_range, end_range, specified_items_sizes, verbose)

    if tables:
        echo_tables(partitioner_list, r)

    if report is not None:
        write_report(r, report)

if __name__ == '__main__':
    cli(sys.argv[1:])
