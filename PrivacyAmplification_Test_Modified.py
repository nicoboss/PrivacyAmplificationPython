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

def permutate(vertical, horizontal, key):

	# ceil to power of two
	desired_length = len(horizontal) + len(vertical)
	toeplitz_seed_filler_len = desired_length - vertical.size - horizontal.size
	print(toeplitz_seed_filler_len)
	toeplitz_seed = np.hstack((vertical, horizontal)).astype(int)
	key_start = key #np.hstack((key[:len(horizontal)+1], np.zeros(desired_length-len(horizontal)-1, ))).astype(int)
	#key_rest = np.hstack((key[len(horizontal)+1:], np.zeros(desired_length-(desired_length-len(horizontal)), ))).astype(int)

	print("desired_length:", desired_length)
	print("horizontal:", horizontal)
	print("vertical:\n", vertical)
	print("horizontal:\n", horizontal)
	print("toeplitz_seed:\n", toeplitz_seed)
	print("key_start:\n", key_start)
	#print("key_rest:\n", key_rest)
	print("fft(toeplitz_seed):\n", np.fft.fft(toeplitz_seed))
	print("fft(key_start):\n", np.fft.fft(key_start))
	print("fft(toeplitz_seed)*fft(key_start):\n", np.fft.fft(toeplitz_seed) * np.fft.fft(key_start))
	print("np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)):\n", np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)))
	permutated_key = np.around(np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)).real).astype(int) % 2
	print("permutated_key_raw:\n", permutated_key)
	#print("key_rest:", key_rest)
	#permutated_key ^= key_rest
	#print("permutated_key:\n", permutated_key)
	return permutated_key[:len(vertical)]
		
start = time.time()
#vertical = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
v1 = np.array([1, 1, 0, 0, 1, 1, 0, 0])
v2 = np.array([0, 1, 1, 0, 0, 0, 1, 0])
v3 = np.array([0, 0, 1, 0, 0, 1, 0, 0])
#horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1])
h1 = np.array([0, 1, 1, 0, 0, 0, 1, 1])
h2 = np.array([1, 0, 1, 1, 0, 1, 1, 1])
h3 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
#key = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])
k1 = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
k2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
amp_key = permutate(v1, h1, k1)
amp_key = permutate(v2, h2, k1)
amp_key = permutate(v3, h3, k2)
end = time.time()
print('Amplificated Key: {}, size: {}'.format(amp_key, amp_key.size))
print('{:5.3f}s'.format(end-start))
