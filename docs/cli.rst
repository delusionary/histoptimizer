#############
CLI Guide
#############


=============
histoptimizer
=============

Histoptimizer would be the main CLI if it did anything useful.

Set Expressions
---------------

Set expressions are used by ``histoptimizer`` and ``histobench`` to define
counts of items or buckets to be used for partitioning and benchmarking.

The format is a comma-separated list of range specifications. A range
specification may be a single number or two numbers (beginning and ending,
inclusive) separated by a '-'. If two numbers, a ':' and third number may be
supplied to provide a step. If the end number is not reachable in complete
steps then the series will be truncated at the last valid step size.

Some examples:

``10000-50000:10000`` → ``(10000, 20000, 30000, 40000, 50000)``

``3,4,7-9`` → ``(3, 4, 7, 8, 9)``

``10,20-30:5,50`` → ``(10, 20, 25, 30, 50)``

``10-25:8`` → ``(10, 18)``

Usage
-----

.. program-output:: histoptimizer --help

Examples
--------

Consider the following CSV:

.. csv-table:: books.csv
   :file: books.csv
   :widths: 80, 20
   :header-rows: 1

To sort by title, and then divide optimally into 3, 5, 6, and 7 buckets,
use this command:

.. program-output:: histoptimizer -i Title -s Title books.csv Pages 3,5-7
   :caption: histoptimizer -i Title -s Title books.csv Pages 3,5-7


=============
histobench
=============

``histobench`` is a CLI that lets you unlock Histoptimizer's most powerful
abilities: running against random data and then throwing away the results.

If you supply it with specifications of multiple item sizes and multiple
buckets, it will benchmark every possible combination of item size and buckets.

Usage
-----

.. program-output:: histobench --help

Examples
--------

.. code-block:: bash

    $ histobench --report benchymcmark.csv numba,cuda 1000,25000 3-5 1

If you supply the ``--report`` option, ``histobench`` will write a row to a CSV
(or JSON, or compressed version of either) for each benchmark test it performs.

You may request multiple iterations of each test to avoid outliers. For
interestingly large problem sets this are rarely outliers. If you supply
``--no-force-jit`` and request two iterations for a single bucket and item size,
then the different between the first and second iterations is the compile time
overhead for NumbaOptimizer and CUDAOptimizer.

.. table:: benchymcmark.csv

    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | partitioner   |   num_items |   buckets |   iteration |   variance |   elapsed_seconds | item_set_id |
    +===============+=============+===========+=============+============+===================+=============+
    | numba         |        1000 |         3 |           1 |    8       |         0         | 30...2b     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | cuda          |        1000 |         3 |           1 |    8       |         0.015     | 30...2b     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | numba         |        1000 |         4 |           1 |    2       |         0         | 20...b5     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | cuda          |        1000 |         4 |           1 |    2       |         0         | 20...b5     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | numba         |        1000 |         5 |           1 |    5.440   |         0.015     | d9...6d     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | cuda          |        1000 |         5 |           1 |    5.440   |         0         | d9...6d     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | numba         |       25000 |         3 |           1 |    2       |         0.627     | d11...02    |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | cuda          |       25000 |         3 |           1 |    2       |         0.065     | d11...02    |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | numba         |       25000 |         4 |           1 |    0.687   |         0.949     | 85...31     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | cuda          |       25000 |         4 |           1 |    0.687   |         0.062     | 85...31     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | numba         |       25000 |         5 |           1 |    4.240   |         1.255     | fe...5f     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+
    | cuda          |       25000 |         5 |           1 |    4.240   |         0.080     | fe...5f     |
    +---------------+-------------+-----------+-------------+------------+-------------------+-------------+

If you supply the ``--tables`` option then ``histobench`` will show you the average time to solve for
each problem size and each partitioner::

    $ histobench.exe --tables numba,cuda 5000-25000:5000 10-30:10 1
    Partitioner: numba
                 10      20      30
       5000   0.110   0.257   0.421
      10000   0.465   1.052   1.868
      15000   1.050   2.380   4.350
      20000   1.853   4.255   7.679
      25000   2.901   6.553  11.954

    Partitioner: cuda
                10     20     30
       5000  0.016  0.010  0.010
      10000  0.022  0.032  0.047
      15000  0.049  0.275  0.345
      20000  0.157  0.377  0.345
      25000  0.298  0.432  0.465

If you have a custom partitioner, you can reference the file path to import it::

    (venv-39) PS D:\histoptimizer\docs> histobench \
    --tables ..\old_optimizers\cuda_shfl_down.py:CUDAOptimizerShuffleDown,cuda \
    10000-50000:10000 3
    Partitioner: cuda_shfl_down
                 3
      10000  0.877
      20000  0.039
      30000  0.063
      40000  0.079
      50000  0.110

    Partitioner: cuda
                 3
      10000  0.025
      20000  0.047
      30000  0.063
      40000  0.095
      50000  0.126