import re
import sys
import time
import os

import click
import numpy as np
import pandas as pd
from numba import cuda

from histoptimizer.cli import partitioners

os.environ['NSIGHT_CUDA_DEBUGGER'] = '1'

def parse_set_spec(spec: str) -> list:
    """
    Parse strings representing sets of integers.
    """
    items = []
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
@click.argument('partitioner_types', type=str, required=True)
@click.argument('item_spec', type=str, default="15")
@click.argument('bucket_spec', type=str, default="8")
@click.argument('iterations', type=int, default=1)
@click.argument('size_spec', type=str, default='1-10')
@click.option('--debug-info/--no-debug-info', type=bool, default=False)
@click.option('--force-jit/--no-force-jit', type=bool, default=True)
@click.option('--report', type=click.File('w'), default=None)
@click.option('--input', type=click.File('r'), default=None)
def cli(partitioner_types, item_spec, bucket_spec, iterations, size_spec, debug_info, force_jit, report, input):
    """
    Histobench is a minimal benchmarking harness for testing Histoptimizer partitioner performance.


    """
    cuda.select_device(0)
    cuda.profile_start()
    # Parse arguments
    partitioner_list = partitioner_types.split(',')
    bucket_list = parse_set_spec(bucket_spec)
    item_list = parse_set_spec(item_spec)
    if match := re.match(r'(\d+)-(\d+)$', size_spec):
        begin_range, end_range = map(int, match.groups())
        if end_range < begin_range:
            begin_range, end_range = end_range, begin_range
    else:
        raise ValueError("Size spec must be two numbers separated by a dash: e.g. 1-10")

    r = pd.DataFrame(columns=('partitioner', 'num_items', 'buckets', 'iteration', 'variance', 'elapsed_seconds', 'dividers', 'items'))

    if force_jit:
        for p in {'cuda_1', 'cuda_2', 'cuda_3'} & set(partitioner_list):
            partitioners[p]([1, 2, 3], 2)

    if input is not None:
        specified_items = pd.read_csv(input)


    for num_items in item_list:
        for num_buckets in bucket_list:
            results = []
            for i in range(1, iterations + 1):
                items = np.random.randint(begin_range, end_range + 1, size=num_items)
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
            print(f'Items: {num_items} Buckets: {num_buckets} Mean values over {iterations} iterations:')
            print(f'Partitioner\t\tTime (ms)\t\tVariance')
            for partitioner, record in mean.iterrows():
                print(f'{partitioner}\t\t\t{record.elapsed_seconds*1000:.2f}\t\t\t{record.variance:.4f}')

    if report is not None:
        r.to_csv(report, index=False)


if __name__ == '__main__':
    cli(sys.argv[1:])
