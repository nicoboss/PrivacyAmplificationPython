import numpy as np

#a = np.array(
#[[1, 1, 0, 1],
# [0, 1, 1, 0],
# [1, 0, 1, 1]]
#)
#b = np.array([1, 1, 0, 0])

a = np.array(
[[1, 1, 1, 1, 0, 1, 1],
 [0, 1, 1, 1, 1, 0, 1],
 [1, 0, 1, 1, 1, 1, 0],
 [0, 1, 0, 1, 1, 1, 1],
 [1, 0, 1, 0, 1, 1, 1]]
)
b = np.array([1, 1, 1, 1, 1, 0, 0])

print(a.dot(b).astype(int) % 2)