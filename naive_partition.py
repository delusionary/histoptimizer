pop = [1,2,3,4,5,6,7,8,9,10]


def get_buckets(pop):
    bucket_list = []
    if len(pop)==1:
        return [[pop]]
    for x in range(1, len(pop)):
        bucket = [pop[0:x]]
        if x < len(pop):
            result = get_buckets(pop[x:])
            for y in result:
                row = bucket + y
                bucket_list.append(row)
        else:
            bucket_list.append(pop)
    return bucket_list


#%%

result = get_buckets(pop)
df = pd.DataFrame(result)

#%%

df.sum(axis=1)

#%%

df2 = df.copy()

#%%

for x in range(0,10):
    df2[10+x] = df2[x].apply(lambda x: None if x is None else sum(x))

#%%

df2

#%%

df2['buckets'] = df2.apply(lambda row: row[10:20].count(), axis=1)

#%%

df2['capacity'] = 55 / df2['buckets']

#%%

df2['std'] = df2.apply(lambda row: row[10:20].std(), axis=1)

#%%

df3 = df2[df2.buckets==6].sort_values('std', ascending=True)

#%%

sum(pop)

#%%

dp = pd.DataFrame({'pop': pop})

#%%

dp['cumpop'] = dp['pop'].cumsum()

#%%

dp['cumpop'].plot()

#%%

dp['cumcum'] = dp['cumpop'].cumsum()

#%%

dp['cumcum'].plot()

#%%

df2.sort_values('std', ascending=True).groupby('buckets').head(1)