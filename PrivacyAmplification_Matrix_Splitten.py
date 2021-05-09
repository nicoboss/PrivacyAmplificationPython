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
	#print("toeplitz_seed:\n", toeplitz_seed)
	#print("key_start:\n", key_start)
	#print("fft(toeplitz_seed):\n", np.fft.fft(toeplitz_seed))
	#print("fft(key_start):\n", np.fft.fft(key_start))
	#print("fft(toeplitz_seed)*fft(key_start):\n", np.fft.fft(toeplitz_seed) * np.fft.fft(key_start))
	#print("np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)):\n", np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)))
	permutated_key = np.around(np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)).real).astype(int) % 2
	#print("permutated_key_raw:\n", permutated_key)
	return permutated_key[:chunk_size]
		
start = time.time()

##Generate T vertical then horizonal (15 Elements):
#vertical = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])
#horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1])
#key = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])


#Changes:
#First vertical element to last (first) horizonal element
#Appended zero at end of vertical
#vertical = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
#horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1])

#vertical = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0])
#horizontal = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1])

#toeplitz_seed = np.hstack((vertical, horizontal)).astype(int)
sample_size = 64
chunk_size = 8
vertical_len = sample_size // 4 + sample_size // 8
horizontal_len = sample_size // 2 + sample_size // 8
vertical_chunks = vertical_len//chunk_size
horizontal_chunks = horizontal_len//chunk_size
amp_out_arr = np.zeros((vertical_chunks, chunk_size), dtype=int)
print(amp_out_arr)

#toeplitz_seed = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]) # HL + VL + 0

#vertical = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0]) #23 + 0
#horizontal = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
#key = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1])

toeplitz_seed = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0]) # HL + VL + 0
toeplitz_seed_length = len(toeplitz_seed)
key = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]) #64

rNr = 0
r = 0
for columnNr in range(horizontal_chunks-1, -1, -1):
    print("columnNr:", columnNr)
    currentRowNr = 0
    for keyNr in range(columnNr, columnNr+min((horizontal_chunks-1)-columnNr+1, vertical_chunks), 1):
        print(keyNr)
        local_seed = np.hstack((toeplitz_seed[r+chunk_size:r+2*chunk_size], toeplitz_seed[r:r+chunk_size])).astype(int)
        print(local_seed)
        local_key_padded = np.hstack((np.zeros(1), key[keyNr*chunk_size:keyNr*chunk_size+chunk_size], np.zeros(7))).astype(int)
        print(local_key_padded)
        amp_out = permutate(local_seed, local_key_padded)
        amp_out_arr[currentRowNr] ^= amp_out
        print(amp_out)
        currentRowNr += 1
    r += chunk_size
    rNr += 1
for rowNr in range(1, vertical_chunks, 1):
    print("rowNr:", rowNr)
    currentRowNr = rowNr
    for keyNr in range(0, min(horizontal_len//chunk_size, (vertical_chunks-rowNr)), 1):
        print(keyNr)
        local_seed = np.hstack((toeplitz_seed[r+chunk_size:r+2*chunk_size], toeplitz_seed[r:r+chunk_size])).astype(int)
        print(local_seed)
        local_key_padded = np.hstack((np.zeros(1), key[keyNr*chunk_size:keyNr*chunk_size+chunk_size], np.zeros(7))).astype(int)
        print(local_key_padded)
        amp_out = permutate(local_seed, local_key_padded)
        amp_out_arr[currentRowNr] ^= amp_out
        print(amp_out)
        currentRowNr += 1
    r += chunk_size
    rNr += 1
print()
print(amp_out_arr)
for i in range(vertical_chunks):
    key_rest_start_pos = horizontal_len+i*chunk_size
    amp_out_arr[i] ^= key[key_rest_start_pos:key_rest_start_pos+chunk_size]
print()
print(amp_out_arr)
