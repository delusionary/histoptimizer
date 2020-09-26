def reconstruct_partition(divider_location, num_items, num_buckets):
    partitions = np.zeros((num_buckets,), dtype=np.int)
    bucket = num_buckets
    while bucket > 1:
        partitions[bucket - 1] = divider_location[num_items, bucket]
        num_items = divider_location[num_items, bucket]
        bucket -= 1
    partitions[0] = num_items
    return partitions


def pandas_partition(items, buckets):
    global mp, dp
    items = [None] + items
    n = len(items) - 1
    m = pd.DataFrame(columns=range(0, buckets + 1))
    d = pd.DataFrame(columns=range(0, buckets + 1))
    p = [None] * (n + 1)  # prefix sums array
    p[0] = 0

    # Cache cumulative sums
    for i in range(1, n + 1):
        p[i] = p[i - 1] + items[i]
    for i in range(1, n + 1):
        m.at[i, 1] = p[i]
    for j in range(1, buckets + 1):
        m.at[1, j] = items[1]
    for i in range(2, n + 1):
        # print(f'i={i}, m = {pformat(m)}')
        # evaluate main recurrence
        for j in range(2, buckets + 1):
            m.at[i, j] = sys.maxsize
            for x in range(1, i):
                cost = max(m.at[x, j - 1], p[i] - p[x])
                if m.at[i, j] > cost:
                    m.at[i, j] = cost
                    d.at[i, j] = x

    mp = m
    dp = d
    return reconstruct_partition(items, d, n, buckets, [])   # print book partition
