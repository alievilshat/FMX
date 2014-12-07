import numpy as np

cols = 3021
rows = 7

a = np.array([cols, rows])
a.fill(1)

for j in range(1, rows):
	for i in range(cols-1, 0):
		a[i][j] = a[i-1][j] + a[i][j-1]

a
