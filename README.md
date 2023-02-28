[![codecov](https://codecov.io/github/delusionary/histoptimizer/branch/main/graph/badge.svg?token=FCLW50JSR9)](https://codecov.io/github/delusionary/histoptimizer)

# Histoptimizer

<img src="docs/_static/histoptimizer-spirit-animal.png" width="50%"/>

## Overview

Histoptimizer ([histoptimizer.org](https://histoptimizer.org),
[documentation](https://histoptimizer.org/docs)) is a Python library and CLI that accepts
a DataFrame or ordered list of item sizes, and produces a list of "divider
locations" that partition the items as evenly as possible into a given number of
buckets, minimizing the variance and standard deviation between the bucket
sizes.

JIT compilation and GPU support through Numba provide great speed improvements
on supported hardware, enabling problem sets of a million items or more.

Histoptimizer was built in order to divide the counties of the US precisely
into intervals ordered by population density. That job was accomplished very
early on, and no other uses have been discovered. It is unclear why development
has continued to this point.

## Usage

Histoptimizer provides two APIs and two command-line tools:

### NumPY array partitioner

Several implementations of the partitioning algorithm can be called directly
with a list or array of item sizes and a number of buckets. They return an
array of divider locations (dividers come _after_ the given item in 1-based
indexing, or _before_ the given item in 0-based indexing) and the variance of
the given partition.

```python
from histoptimizer import Histoptimizer

item_sizes = [1.0, 4.5, 6.3, 2.1, 8.4, 3.7, 8.6, 0.3, 5.2, 6.9, 1.2, 2.4, 9.8, 3.7]

# Get the optimal position of two dividers that partition the list above into 3 buckets.
(dividers, variance) = Histoptimizer.partition(item_sizes, 3)

print(f"Optimal Divider Locations: {dividers} Optimal solution variance: {variance:.4}")
```

### Pandas Dataframe Partitioner

You can supply a Pandas DataFrame, the name of a size column, a list of bucket
sizes, and a column prefix to get a version of the DataFrame with added columns
where the value is the 1-based bucket number of the corresponding item 
partitioned into the number of buckets reflected in the column name.

```python
from histoptimizer import histoptimize
import pandas as pd

books = pd.read_csv('books.csv', header=0)
divisions, column_names = histoptimize(books, "Pages", [3], "assistant_", Histoptimizer)
divisions
```

|     | Title                          |   Pages | assistant_3 |
|----:|:-------------------------------|--------:|------------:|
|   0 | The Algorithm Design Manual    |     748 |           1 |
|   1 | Software Engineering at Google |     599 |           1 |
|   2 | Site Reliability Engineering   |     550 |           2 |
|  .. | ...                            |   ...   |         ... |
|  14 | Noise                          |     464 |           3 |
|  15 | Snow Crash                     |     440 |           3 |


### CLI

The CLI is a wrapper around the DataFrame functionality that can accept and
produce either CSV or Pandas JSON files.

```
Usage: histoptimizer [OPTIONS] FILE SIZE_COLUMN PARTITIONS

  Partition ordered items in a CSV into a given number of buckets,       
  evenly.

  Given a CSV or JSON Dataframe, a size column name, and a number of     
  buckets, Histoptimizer will add a column which gives the partition     
  number for each row that optimally divides the given items into the    
  buckets so as to minimize the variance from mean of the summed items   
  in each bucket.

  Additional features allow doing a list of bucket sizes in one go,      
  sorting items beforehand, and producing output with only relevant      
  columns.

  Example:

      > histoptimizer books.csv state_name population 10

      Output:

      state_name, population, partition_10     Wyoming, xxxxxx, 1        
      California, xxxxxxxx, 10

Options:
  -l, --limit INTEGER             Take the first {limit} records from    
                                  the input, rather than the whole       
                                  file.
  -a, --ascending, --asc / -d, --descending, --desc
                                  If a sort column is provided,
  --print-all, --all / --no-print-all, --brief
                                  Output all columns in input, or with   
                                  --brief, only output the ID, size,     
                                  and buckets columns.
  -c, --column-prefix TEXT        Partition column name prefix. The      
                                  number of buckets will be appended.    
                                  Defaults to partion_{number of
                                  buckets}.
  -s, --sort-key TEXT             Optionally sort records by this        
                                  column name before partitioning.       
  -i, --id-column TEXT            Optional ID column to print with       
                                  brief output.
  -p, --partitioner TEXT          Use the named partitioner
                                  implementation. Defaults to "numba".   
                                  If you have an NVidia GPU use "cuda"   
                                  for better performance
  -o, --output FILENAME           Send output to the given file.
                                  Defaults to stdout.
  -f, --output-format [csv|json]  Specify output format. Pandas JSON or  
                                  CSV. Defaults to CSV
  --help                          Show this message and exit.
```

### Benchmarking CLI

The Benchmarking CLI can be used to produce comparative performance metrics for 
various implementations of the algorithm.

```
Usage: histobench [OPTIONS] PARTITIONER_TYPES [ITEM_SPEC] [BUCKET_SPEC]
                  [ITERATIONS] [SIZE_SPEC]

  Histobench is a benchmarking harness for testing Histoptimizer partitioner
  performance.

  By Default it uses random data, and so may not be an accurate benchmark for
  algorithms whose performance depends upon the data set.

  The PARTITIONER_TYPES parameter is a comma-separated list of partitioners to
  benchmark, which can be specified as either:

  1. A standard optimizer name, or 2. filepath:classname

  To specify the standard cuda module and also a custom variant, for example,

Options:
  --debug-info / --no-debug-info
  --force-jit / --no-force-jit
  --report PATH
  --sizes-from PATH
  --tables / --no-tables
  --verbose / --no-verbose
  --help                          Show this message and exit.
```

