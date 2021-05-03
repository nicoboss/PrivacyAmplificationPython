import numpy as np

a = np.array(
[[1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1.],
 [1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0.],
 [1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
 [0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1.],
 [0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0.],
 [1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1.],
 [1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1.],
 [0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1.],
 [0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0.],
 [0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1.],
 [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1.],
 [1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
 [0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0.],
 [0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0.],
 [1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1.],
 [0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1.]]
)
b = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
print(a)
print(b)
print(a.dot(b))
print(a.dot(b) % 2)
#print((a.dot(b).astype(int) % 2) ^ np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]))
