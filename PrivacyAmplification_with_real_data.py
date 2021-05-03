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
		fft_key_start = np.fft.fft(key_start)
		correction = ((fft_toeplitz_seed[0].real * fft_key_start[0].real)/n) % 2
		print("c1:", fft_toeplitz_seed[0].real)
		print("c2:", fft_key_start[0].real)
		print("correction_pre_mod:", (fft_toeplitz_seed[0].real * fft_key_start[0].real)/n)
		fft_toeplitz_seed[0] = 0+0j
		fft_key_start[0] = 0+0j
		mul1 = fft_toeplitz_seed*fft_key_start
		invOut = np.fft.ifft(mul1).real+correction
		permutated_key = np.around(invOut).astype(int) % 2
		key_rest_int = np.around(key_rest).astype(int) % 2
		
		print("correction:", correction)
		print("desired_length:", desired_length)
		print("key_start:\n", key_start[:100])
		print("key_rest_int:\n", key_rest_int[:100])
		print("key_rest_int_last:\n", key_rest_int[desired_length-horizontal_len-100:desired_length-horizontal_len+100])
		print("fft(toeplitz_seed):\n", fft_toeplitz_seed[:100])
		print("fft(key_start):\n", fft_key_start[:100])
		print("mul1:\n", mul1[:100])
		print("fft(toeplitz_seed):\n", fft_toeplitz_seed[:100]/2**11)
		print("fft(key_start):\n", fft_key_start[:100]/2**11)
		print("fft(mul1_reduced):\n", (fft_toeplitz_seed[:100]/2**11)*(fft_key_start[:100]/2**11))
		print("invOut:\n", invOut[:100])
		print("permutated_key_raw:\n", permutated_key[:100])
		permutated_key ^= key_rest_int
		print("permutated_key:\n", permutated_key[:100])
		return permutated_key[:vertical_len]
		
def generate_key_bit_string(length):
	return np.zeros(length) #np.random.randint(0, size=(length,))
		
		

#Toeplitz.MATRIX_SEED = generate_key_bit_string(2**(size+2))
#Toeplitz.NEW_KEY_LENGTH = 2**size
# So lange wie horizontal_ganz
#key = generate_key_bit_string(5)
start = time.time()
amp_key = Toeplitz.permutate()
out_arr = np.packbits(amp_key)
end = time.time()
print('Amplificated Key: {}, size: {}'.format(amp_key, amp_key.size))
print('{:5.3f}s'.format(end-start))
with open('ampout_python.bin', 'wb') as f:
	#Completes but never returns
	np.array(out_arr, dtype=np.uint8).tofile(f) 
