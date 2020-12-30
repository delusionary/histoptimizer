import numpy as np
import histoptimizer

name = 'recursive_verbose'


def min_cost_partition(items, k, last_item=None, mean=None):
    #  [3, 3, 5, 3, 4] in 4 buckets [2, 3, 4] better than [1, 3, 4]
    n = len(items)
    j = k - 1
    if mean is None:
        mean = sum(items) / k
    if last_item is None:
        last_item = n - 1
    first_possible_position = j
    best_cost = np.inf

    # The base case is that we are being called to get the cost of placing 0 dividers and return the variance of
    # the first bucket.
    if j == 0:
        cost = (sum(items[0:last_item + 1]) - mean) ** 2
        print(f"  Base case: Items {items[0:last_item + 1]} has cost {cost:.2f} mean={mean:.2f}")
        return cost, []

    for current_divider_location in range(first_possible_position, last_item + 1):
        (lh_cost, previous_dividers) = min_cost_partition(items, k - 1, last_item=current_divider_location - 1,
                                                          mean=mean)
        rh_cost = (sum(items[current_divider_location:last_item + 1]) - mean) ** 2
        cost = lh_cost + rh_cost
        if cost < best_cost:
            best_cost = cost
            dividers = previous_dividers + [current_divider_location]
            print("New Best:")
        print(f"# Divider {j} after {current_divider_location} -- "
              f" RH: {items[current_divider_location:last_item + 1]} RH Cost: {rh_cost:.2f}"
              f" Prev: {lh_cost:.2f} Total: {cost:.2f} Dividers: {previous_dividers + [current_divider_location]}")
    print(f"** Best Divider {j} location: {dividers[-1]} for first {last_item} items. Cost: {best_cost:.2f} Mean: {mean:.2f} Best Divider series: {dividers}")
    return best_cost, dividers


def partition(items, k, debug_info=None):
    return min_cost_partition(items, k)[1]


