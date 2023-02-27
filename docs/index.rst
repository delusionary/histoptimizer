#############
Histoptimizer
#############

"A heartbreaking work of monumental self-importance"
                                            -- My mom, probably
.. toctree::
    :maxdepth: 2

    Quickstart <Quickstart.ipynb>
    API Reference <api>
    CLI Guide <cli>
    Appendix: A Visual Guide to Dynamic Programming (draft) <Histoptimizing.ipynb>

Histoptimizer is a Python library that solves a very specific problem: Given an
ordered list of item sizes, and a number of partitions **n**, it returns a list of
divider locations that partition the given list into **n** partitions with the
lowest possible variance across the sums of the items in each partition.

Histoptimizer provides JIT- and SIMD-accelerated implementations on Intel and
AMD processors, and a CUDA implementation for NVidia graphics cards. The
implementations are in-core only and suitable for up to ~ 1 million items with
32-bit floating point sizes.

Histoptimizer provides NumPY and Pandas APIs, and a CLI that supports CSVs and
Pandas JSON. It comes with everything you could want, except a valid use case.

============
Installation
============

To get started with Histoptimizer, install the latest stable release via `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: bash
    :caption: Bash

    pip install histoptimizer

Histoptimizer currently supports Python 3.8+ and relies on the following dependencies:

- `pandas <https://pandas.pydata.org/>`_
- `numba <https://numba.pydata.org>`_
- `click <https://click.palletsprojects.com/>`_

========
Releases
========

Releases are listed at https://github.com/delusionary/histoptimizer/releases/

=======
License
=======

Histoptimizer is licensed under the 0BSD license. See the
`LICENSE <https://github.com/delusionary/histoptimizer/blob/main/LICENSE>`_
for more information.

=================
About
=================

Histoptimizer started as a simple exercise for learning CUDA programming and wound
up as an exercise for brushing up on modern Python module development tools and best
practices. At no point in between did it make contact with a practical
application or use case that could not be approximated more quickly and
easily by other methods.

There may be practical applications; if so, please don't tell me. There's a
certain platonic purity to it right now. Freeing, if you think about it. It does
nothing of value, but does it very quickly, so it makes it up in volume.


