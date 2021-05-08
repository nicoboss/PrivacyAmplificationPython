# 16x16
# [[1. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1. 0. 1. 1. 0. 1.]
#  [1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1. 0. 1. 1. 0.]
#  [1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1. 0. 1. 1.]
#  [0. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1. 0. 1.]
#  [0. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1. 0.]
#  [1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1.]
#  [1. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 1.]
#  [0. 1. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1.]
#  [0. 0. 1. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0.]
#  [0. 0. 0. 1. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1.]
#  [0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 1. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 1. 1. 0.]
#  [1. 0. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 1. 1.]
#  [0. 1. 0. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 1.]]

import os
import configparser
import time
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import csc_matrix
np.set_printoptions(threshold=np.inf)

def permutate(toeplitz_seed, key_start):
	print("toeplitz_seed:\n", toeplitz_seed)
	print("key_start:\n", key_start)
	print("fft(toeplitz_seed):\n", np.fft.fft(toeplitz_seed))
	print("fft(key_start):\n", np.fft.fft(key_start))
	print("fft(toeplitz_seed)*fft(key_start):\n", np.fft.fft(toeplitz_seed) * np.fft.fft(key_start))
	print("np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)):\n", np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)))
	permutated_key = np.around(np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)).real).astype(int) % 2
	print("permutated_key_raw:\n", permutated_key)
	return permutated_key[:16]
		
start = time.time()

##Generate T vertical then horizonal (15 Elements):
#vertical = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])
#horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1])
#key = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])


#Changes:
#First vertical element to last (first) horizonal element
#Appended zero at end of vertical
#Insearted 0 at beginning of key
#vertical = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
#horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1])



#toeplitz_seed = np.hstack((vertical, horizontal)).astype(int)
horizontal_len = 16
toeplitz_seed = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]) # HL + VL
toeplitz_seed_length = len(toeplitz_seed)
key = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])
key_length = horizontal_len

local_seed = np.hstack((toeplitz_seed[16:24], toeplitz_seed[8:16])).astype(int)
local_key = key[0:8]
local_key_padded = np.hstack((np.zeros(1), local_key, np.zeros(7))).astype(int)
print(local_seed)
print(local_key_padded)
amp_key_T1 = permutate(local_seed, local_key_padded)

local_seed = np.hstack((toeplitz_seed[8:16], toeplitz_seed[0:8])).astype(int)
local_key = key[key_length-8:key_length]
local_key_padded = np.hstack((np.zeros(1), local_key, np.zeros(7))).astype(int)
print(local_seed)
print(local_key_padded)
amp_key_T2 = permutate(local_seed, local_key_padded)

exit(1)

#vertical = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
#v1 = np.array([1, 1, 0, 0, 1, 1, 0, 0])
#v2 = np.array([0, 1, 1, 0, 0, 0, 1, 0])
#v3 = np.array([0, 0, 1, 0, 0, 1, 0, 0])
#horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1])
#h1 = np.array([0, 1, 1, 0, 0, 0, 1, 1])
#h2 = np.array([1, 0, 1, 1, 0, 1, 1, 1])
#h3 = np.array([1, 1, 0, 0, 1, 1, 0, 0])
#key = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])
#k1 = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#k2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
#amp_key = permutate(v1, h1, k1)
#amp_key = permutate(v2, h2, k1)
#amp_key = permutate(v3, h3, k2)

key_start = np.hstack((key[:16+1], np.zeros(toeplitz_seed_length-16-1, ))).astype(int)
amp_key = permutate(toeplitz_seed, key_start)

end = time.time()
print('Amplificated Key: {}, size: {}'.format(amp_key, amp_key.size))
print('{:5.3f}s'.format(end-start))
