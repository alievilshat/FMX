import pdb
from bisect import bisect_left

cols = 3201 + 1
rows = 7

a = [[1]*cols for r in range(rows)]

# calculate pascal triangle
for j in range(1, rows):
	for i in range(cols-2, 0, -1):
		a[j][i] = a[j][i+1] + a[j-1][i]

# sum up the rows
for j in range(0, rows):
	for i in range(2, cols):
		a[j][i] = a[j][i-1] + a[j][i]		

# zero the helper column
for j in range(rows):
	a[j][0] = 0

def getcandidate(n):
	n = n + 1 # one based
	res = [0]*rows;
	k = 0
	for i in range(rows):
		r = -i-1
		o = a[r][k]
		k = bisect_left(a[r], n + o) - 1
		res[i] = k
		n = n - a[r][k] + o
	return res
