import sys
from typing import List

import click
import pandas
import numpy
from timeit import default_timer as timer

from pandas_partition import pandas_partition
from cuda_partition import cuda_partition
from numpy_partition import numpy_partition

def histoptimize(data, key, size, partitions, ascending, print_all, column_name, timing):
    start = timer()
    data = data.sort_values(key, ascending=ascending).reset_index()
    items = data[size].astype('float64')
    dividers = numpy_partition(items, partitions)
    current_partition = 0

    def get_partition(index):
        nonlocal current_partition
        if current_partition < dividers.shape[0] and index == dividers[current_partition]:
            current_partition += 1
        return current_partition + 1

    pop = numpy.zeros(items.shape[0], dtype=int)
    for x in range(0, pop.shape[0]):
        pop = get_partition(x)
    current_partition = 0
    data[column_name] = data.index.to_series().apply(get_partition)
    end = timer()
    print(f'time consumed: {end - start}')
    return data

@click.command()
@click.argument('file', type=click.File('rb'))
@click.argument('name', type=str)
@click.argument('key', type=str)
@click.argument('size', type=str)
@click.argument('partitions', type=int)
@click.option('-n', '--number', type=int, default=None)
@click.option('-a/-d', '--ascending/--descending', '--asc/--desc', type=bool, default=True)
@click.option('--print-all', '--all', type=bool, default=False, help='Output all columns')
@click.option('-c', '--column-name', type=str, default=None,
              help='Partition column header value. Defaults to partion_n')
@click.option('-t', '--timing', type=bool, default=False)
@click.option('-i', '--implementation', type=str, default='default')
def histoptimizer_cli(file, name, key, size, partitions, number, ascending, print_all, column_name, timing, implementation):
    """
    Given a CSV, a row name column, sort key, and a number of buckets, sort the CSV by the given key, then distribute
    the ordered keys as evenly as possible to the given number of buckets.

    > histoptimizer states.csv state_name population 10
    state_name, population, partition_10
    Wyoming, xxxxxx, 1
    California, xxxxxxxx, 10
    """
    import pdb;
    pdb.set_trace()
    if file.name.endswith('json'):
        data = pandas.read_json(file)
    else:
        data = pandas.read_csv(file)
    if number:
        data = data.truncate(after=number)
        data = data.reset_index()
    if column_name is None:
        column_name = f'partition_{partitions}'
    output = histoptimize(data, key, size, partitions, ascending, print_all, column_name, timing)
    if not print_all:
        output = output[[name, key, size, column_name]]
    output.to_csv(sys.stdout)


if __name__ == '__main__':
    data = pandas.read_json('county_narrow.json')
    data = data.truncate(after=10)
    #histoptimizer_cli()
    #partitions = cuda_partition([1,2,3,4,5,6,7,8,9,10], 6)
    #data = pandas.read_csv('pop.csv')
    result = histoptimize(data, 'pct_trump16', 'total_population', 6, True, False, 'bucket', True)
    print('hi')