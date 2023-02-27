#############
API Reference
#############

=============
Histoptimizer
=============

Histoptimizer provides a base class, ``Histoptimizer``, which provides a pure
Python reference implementation that will suffice for small problem sets.
However, as problem sizes grow, performance falls quickly

.. table:: Time to Solve in Seconds, Pure Python Optimizer, AMD Ryzen 9 5900 12-Core Processor

    +-------------+-------------+--------------+
    |   num_items |   5 buckets |   10 buckets |
    +=============+=============+==============+
    |         100 |       0.005 |        0.027 |
    +-------------+-------------+--------------+
    |        1000 |       0.91  |        2.07  |
    +-------------+-------------+--------------+
    |        5000 |      22.995 |       51.541 |
    +-------------+-------------+--------------+

Larger problem sets should use NumbaOptimizer or CUDAOptimizer subclasses
depending on what hardware is available

.. autoclass:: histoptimizer.Histoptimizer
   :members:

==============
NumbaOptimizer
==============

The Numba-accelerated partitioner is very practical for what could well be very
useful workload sizes, if there were any useful workloads:

.. table:: Time to Solve in Seconds, Numba Optimizer, AMD Ryzen 9 5900 12-Core Processor

    +-------------+--------------+--------------+--------------+
    |   num_items |   10 buckets |   20 buckets |   30 buckets |
    +=============+==============+==============+==============+
    |        5000 |        0.125 |        0.256 |        0.424 |
    +-------------+--------------+--------------+--------------+
    |       10000 |        0.456 |        1.053 |        1.81  |
    +-------------+--------------+--------------+--------------+
    |       15000 |        1.052 |        2.38  |        4.304 |
    +-------------+--------------+--------------+--------------+
    |       20000 |        1.854 |        4.184 |        7.742 |
    +-------------+--------------+--------------+--------------+
    |       25000 |        2.869 |        6.559 |       12.097 |
    +-------------+--------------+--------------+--------------+
    |       30000 |        4.163 |        9.369 |       17.611 |
    +-------------+--------------+--------------+--------------+

.. autoclass:: histoptimizer.numba_optimizer.NumbaOptimizer

=============
CUDAOptimizer
=============

The CUDA Optimizer is the best-performing implementation by a lot, but it is
still starting to work pretty hard by 1 million items.

.. table:: Time to Solve in Seconds, CUDA Optimizer, GeForce RTX 3080

    +-------------+-------------+--------------+--------------+
    |   num_items |   5 buckets |   10 buckets |   15 buckets |
    +=============+=============+==============+==============+
    |       10000 |       0.035 |        0.032 |        0.047 |
    +-------------+-------------+--------------+--------------+
    |      100000 |       0.472 |        0.8   |        1.179 |
    +-------------+-------------+--------------+--------------+
    |      500000 |       7.718 |       16.621 |       27.279 |
    +-------------+-------------+--------------+--------------+
    |     1000000 |      31.35  |       85.078 |      278.634 |
    +-------------+-------------+--------------+--------------+

.. autoclass:: histoptimizer.cuda.CUDAOptimizer

============
histoptimize
============

Histoptimizer provides a convenience API for dealing with Pandas DataFrames.
The ``histoptimizer`` CLI is mostly a wrapper around this function.

.. autofunction:: histoptimizer.histoptimize

=========
benchmark
=========

The ``histobench`` CLI is a wrapper around the ``benchmark`` function, which may
itself be of use to you if you were for some reason benchmarking competing
partitioner implementations.

.. autofunction:: histoptimizer.benchmark.benchmark

``partitioner_pivot`` is a small function that pivots the time results for a
particular partitioner out of the results DataFrame returned by ``benchmark``.

.. autofunction:: histoptimizer.benchmark.partitioner_pivot