# Histoptimizer

## Overview

Histoptimizer is a Python library and CLI for taking a DataFrame or equivalent, sorting it by a column (optional), and
optimally distributing the ordered rows into a given number of partitions/buckets as equally as possible, minimizing the
standard deviation of the sums of a specified size column.

This can be useful when dividing a set of ordered measurements into approximately equal groups for analysis or 
visualization.

The use case that motivated its creation was: Taking a list of the ~3117 counties in the U.S., ordering them by some
attribute (voting averages, population density, median age, etc.), and then distributing them into a number of buckets
of approximately equal population. 

## Usage

Histoptimizer provides several APIs and tools:

### NumPY array partitioner

Several implementations of the partitioning algorithm can be called directly with a list or array of items sizes and a
number of buckets. They return an array of divider locations (dividers come _after_ the given item in 1-based indexing,
or _before_ the given item in 0-based indexing) and the variance of the given partition.

### Pandas Dataframe Partitioner

You can supply a Pandas DataFrame, the name of a size column, a list of bucket sizes, and a column prefix to get a
version with added columns where the value is the 1-based bucket number of the corresponding item 
partitioned into the number of buckets reflected in the column name.

### CLI

### Benchmarking CLI

## JIT Compilation and CUDA acceleration

