[![codecov](https://codecov.io/github/delusionary/histoptimizer/branch/main/graph/badge.svg?token=FCLW50JSR9)](https://codecov.io/github/delusionary/histoptimizer)

# Histoptimizer

## Overview

Histoptimizer is a Python library and CLI that accepts a DataFrame or ordered
list  of item sizes, and produces a list of "divider locations" that partition
the items as evenly as possible into a given number of buckets, minimizing the 
variance and standard deviation between the bucket sizes.

This can be useful when dividing an ordered set of ordered measurements into
equal groups for analysis or visualization.

JIT compilation and GPU support through Numba provide great speed improvements
on supported hardware.

The use case that motivated its creation was: Taking a list of the ~3117
counties in the U.S., ordering them by some attribute (voting averages,
population density, median age, etc.), and then distributing them into a number
of buckets of approximately equal population. 

## Usage

Histoptimizer provides several APIs and tools:

### NumPY array partitioner

Several implementations of the partitioning algorithm can be called directly
with a list or array of items sizes and a number of buckets. They return an
array of divider locations (dividers come _after_ the given item in 1-based
indexing, or _before_ the given item in 0-based indexing) and the variance of
the given partition.

### Pandas Dataframe Partitioner

You can supply a Pandas DataFrame, the name of a size column, a list of bucket
sizes, and a column prefix to get a version of the DataFrame with added columns
where the value is the 1-based bucket number of the corresponding item 
partitioned into the number of buckets reflected in the column name.

### CLI

The CLI is a wrapper around the DataFrame functionality that can accept and
produce either CSV or Pandas JSON files.

```
Usage: histoptimizer [OPTIONS] FILE ID_COLUMN SIZE_COLUMN PARTITIONS

  Given a CSV, a row name column, a size column, sort key, and a number of
  buckets, optionally sort the CSV by the given key, then distribute the
  ordered keys as evenly as possible to the given number of buckets.

  Example:

      > histoptimizer states.csv state_name population 10

      Output:

      state_name, population, partition_10     Wyoming, xxxxxx, 1
      California, xxxxxxxx, 10

Options:
  -l, --limit INTEGER             Take the first {limit} records from the
                                  input, rather than the whole file.
  -a, --ascending, --asc / -d, --descending, --desc
                                  If a sort column is provided,
  --print-all, --all / --no-print-all, --brief
                                  Output all columns in input, or with
                                  --brief, only output the ID, size, and
                                  buckets columns.
  -c, --column-prefix TEXT        Partition column name prefix. The number of
                                  buckets will be appended. Defaults to
                                  partion_{number of buckets}.
  -s, --sort-key TEXT             Optionally sort records by this column name
                                  before partitioning.
  -t, --timing / --no-timing      Print partitioner timing information to
                                  stderr
  -i, --implementation TEXT       Use the named partitioner implementation.
                                  Defaults to "dynamic_numba". If you have an
                                  NVidia GPU use "cuda" for better performance
  -o, --output FILENAME           Send output to the given file. Defaults to
                                  stdout.
  -f, --output-format [csv|json]  Specify output format. Pandas JSON or CSV.
                                  Defaults to CSV
  --help                          Show this message and exit.
```

### Benchmarking CLI

The Benchmarking CLI can be used to produce comparative performance metrics for
the various implementations of the algorithm.

## JIT SIMD Compilation and CUDA acceleration

Histoptimizer supports Just-in-time compilation for both CPU and NVidia CUDA
GPUs using Numba. For larger problems these implementations can be hundreds or
thousands of times faster than the Python implementation alone.
