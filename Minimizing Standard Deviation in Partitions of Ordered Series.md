Let

j be a number of dividers

n be a number of items

z[1..n] be an ordered series of item sizes

c[j..1] be a vector giving j partition locations between elements of z in reverse lexicographic order.

m be the sum of the sizes in i divided by (j+1)

p[0..i] be the prefix sums of items in z

s[a,b] be the size of a bucket containing items from z[a] to z[b], given by p[b-a]

Example:

Consider z = [7, 41, 97, 53, 67, 24]
and j = 2




