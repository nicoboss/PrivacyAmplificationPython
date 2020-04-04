import numpy as np

a = np.array(
[[1., 1.],
 [0., 1.],
 [1., 0.],
 [0., 1.],
 [1., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.],
 [0., 0.]]
)
b = np.array([1, 1])
print(a)
print(b)
print(a.dot(b))
print(a.dot(b) % 2)
print((a.dot(b).astype(np.int) % 2) ^ np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))