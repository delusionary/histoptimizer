import numpy as np


def min_partition(items, k, last_item_index=None, mean=None):
    n = len(items)
    j = k - 1
    if not mean:
        mean = sum(items) / k
    if not last_item_index:
        last_item_index = n-1
    first_possible_position = j
    best_cost = np.inf

    if j == 1:
        for current_divider_location in range(first_possible_position, last_item_index + 1):
            cost = (sum(items[0:current_divider_location])-mean)**2
            if cost < best_cost:
                best_cost = cost
                dividers = [current_divider_location]
        return best_cost, dividers

    for current_divider_location in range(first_possible_position, last_item_index + 1):
        for prev_divider_location in range(j - 1, current_divider_location):
            (previous_cost, previous_dividers) = min_partition(items, k-1, last_item_index=current_divider_location-1, mean=mean)
            cost = (sum(items[previous_dividers[-1]:last_item_index + 1])-mean)**2 + previous_cost
            if cost < best_cost:
                best_cost = cost
                dividers = previous_dividers + [current_divider_location]

    return best_cost, dividers


def partition(items, k, debug_info=None):
    return min_partition(items, k)[1]
