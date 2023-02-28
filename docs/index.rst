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

Histoptimizer takes an ordered list of item sizes and a number of partitions *k*,
and returns a list of divider locations that partition the given list into *k*
partitions with the lowest possible variance across the sums of the items in
each partition.

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

    pip install histoptimizer

Histoptimizer currently supports Python 3.8+ and relies on the following fine
software projects:

- `numpy <https://numpy.org/>`_
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
nothing of value, but does it very quickly, so as to make it up in volume.

================
About the Author
================

.. image:: _static/histoptimizer-spirit-animal.png
    :width: 50%
    :alt: The LOLRus, Histoptimizer's spirit animal.
    :align: center

The author loves buckets and owns a variety of them, even one of those Yeti ones,
and makes no apologies for it.