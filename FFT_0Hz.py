import os
import configparser
import time
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import csc_matrix
np.set_printoptions(threshold=np.inf)

class Toeplitz:
	MATRIX_SEED = np.array([0]);
	NEW_KEY_LENGTH = 0;

	@classmethod
	def permutate(cls):
		n = 2**27
		np.set_printoptions(suppress=True)
		np.set_printoptions(linewidth=np.inf)
		matrix_seed_file = np.fromfile("toeplitz_seed.bin", dtype='<i4')
		key_file = np.fromfile("keyfile.bin", dtype='<i4')
		toeplitz_seed = (((matrix_seed_file[:,None] & (0x80000000 >> np.arange(32)))) > 0).astype(int).flatten()[:n]
		key = (((key_file[:,None] & (0x80000000 >> np.arange(32)))) > 0).astype(int).flatten()[:n]
		print("toeplitz_seed:", toeplitz_seed[:100])
		print("key:", key[:100])
		vertical_len = n//4 + n//8;
		horizontal_len = n//2 + n//8;
		desired_length = horizontal_len + vertical_len
		key_start = np.hstack((key[:horizontal_len+1], np.zeros(desired_length-horizontal_len-1, )))
		key_rest = np.hstack((key[horizontal_len+1:], np.zeros(horizontal_len+1), ))
		fft_toeplitz_seed = np.fft.fft(toeplitz_seed)
		print(fft_toeplitz_seed[:100])
		
		
def generate_key_bit_string(length):
	return np.zeros(length) #np.random.randint(0, size=(length,))
		
		

#Toeplitz.MATRIX_SEED = generate_key_bit_string(2**(size+2))
#Toeplitz.NEW_KEY_LENGTH = 2**size
# So lange wie horizontal_ganz
#key = generate_key_bit_string(5)
amp_key = Toeplitz.permutate()
out_arr = np.packbits(amp_key)

