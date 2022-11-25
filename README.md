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

### Benchmarking CLI

The Benchmarking CLI can be used to produce comparative performance metrics for
the various implementations of the algorithm.

## JIT SIMD Compilation and CUDA acceleration

Histoptimizer supports Just-in-time compilation for both CPU and NVidia CUDA
GPUs using Numba. For larger problems these implementations can be hundreds or
thousands of times faster than the Python implementation alone.
