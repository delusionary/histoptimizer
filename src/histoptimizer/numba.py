"""

Copyright (C) 2020 by Kelly Joyner (de@lusion.org)

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
"""
import numpy as np
from numba import guvectorize
from histoptimizer import Histoptimizer

# os.environ['NUMBA_DISABLE_JIT'] = '1'


@guvectorize(
    ['intc, f4[:], f4[:], f4, f4[:], f4[:]'],
    '(),(m),(n),()->(n),(n)',
    nopython=True,
    target='cpu'
)
def _get_min_cost(bucket, prefix_sum, previous_row, mean,
                  current_row_cost, current_row_dividers):  # pragma: no cover
    current_row_cost[0] = previous_row[0]
    current_row_cost[1] = previous_row[1]
    current_row_dividers[0] = 0
    current_row_dividers[1] = 0
    for item in range(2, len(prefix_sum)):
        min_cost = np.inf
        divider_location = 0
        for previous_item in range(bucket - 1, item):
            cost = previous_row[previous_item] + ((prefix_sum[item] -
                                                   prefix_sum[
                                                       previous_item]) - mean) ** 2
            if cost < min_cost:
                min_cost = cost
                divider_location = previous_item
        current_row_cost[item] = min_cost
        current_row_dividers[item] = divider_location


class NumbaOptimizer(Histoptimizer):
    name = 'numba'

    @classmethod
    def precompile(cls):
        cls.partition([1, 4, 6, 9], 3)

    @classmethod
    def get_min_cost(cls, bucket, prefix_sum, previous_row, mean):
        return _get_min_cost(bucket, prefix_sum, previous_row, mean)

    @classmethod
    def build_matrices(cls, min_cost, divider_location,
                       num_buckets, prefix_sum):
        mean = prefix_sum[-1] / num_buckets
        for bucket in range(2, min_cost.shape[1]):
            min_cost[:, bucket], divider_location[:, bucket] =\
                cls.get_min_cost(bucket, prefix_sum, min_cost[:, bucket-1],
                                 mean)

        return min_cost, divider_location
