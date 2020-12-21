import numpy as np


def min_partition(items, k, last_item=None, mean=None):
    #  [3, 3, 5, 3, 4] in 4 buckets [2, 3, 4] better than [1, 3, 4]
    n = len(items)
    j = k - 1
    if not mean:
        mean = sum(items) / k
    if not last_item:
        last_item = n - 1
    first_possible_position = j
    best_cost = np.inf

    # The base case is that we are being called to find the optimum location of the first divider for a given
    # location of the second divider
    if j == 1:
        for current_divider_location in range(1, last_item + 1):
            lh_cost = (sum(items[0:current_divider_location]) - mean)**2
            rh_cost = (sum(items[current_divider_location:last_item + 1]) - mean) ** 2
            cost = lh_cost + rh_cost
            if cost < best_cost:
                best_cost = cost
                dividers = [current_divider_location]
                print("  New Best:")
            print(f" First divider after [{current_divider_location}]:"
                  f" LH {items[0:current_divider_location]} = {lh_cost}"
                  f" RH {items[current_divider_location:last_item + 1]} = {rh_cost} Total: {cost}")

        print(f"Best First Divider: {dividers}")
        return best_cost, dividers

    for current_divider_location in range(first_possible_position, last_item + 1):
        for previous_divider_location in range(j - 1, current_divider_location):
            (lh_cost, previous_dividers) = min_partition(items, k - 1, last_item=current_divider_location - 1, mean=mean)
            rh_cost = (sum(items[current_divider_location:last_item + 1]) - mean) ** 2
            cost = lh_cost + rh_cost
            if cost < best_cost:
                best_cost = cost
                dividers = previous_dividers + [current_divider_location]
                print("New Best:")
            print(f"# Divider {j} aft {current_divider_location} prev divider at {previous_divider_location}--"
                    f" RH: {items[current_divider_location:last_item + 1]} RH Cost: {rh_cost}"
                    f" Prev: {lh_cost} Total: {cost} Dividers: {previous_dividers + [current_divider_location]}")
    print(f"** Best Divider {j} location: {dividers[-1]} for first {last_item} items. Cost: {best_cost} Best Divider series: {dividers}")
    return best_cost, dividers


def partition(items, k, debug_info=None):
    return min_partition(items, k)[1]
