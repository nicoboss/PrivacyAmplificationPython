from random import *
import numpy as np
import sys

n = 8
potenz = 2**n
vsize = potenz//4+potenz//8#randint(1, potenz/2-1)
hsize = potenz-vsize
ksize = potenz+1
np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth=sys.maxsize)
print(hsize, "x", vsize)
print("vertical = np." + repr(np.random.randint(0, 2, vsize)))
print("horizontal = np." + repr(np.random.randint(0, 2, hsize)))
print("key = np." + repr(np.random.randint(0, 2, ksize)))
