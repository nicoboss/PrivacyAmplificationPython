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
		vertical = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
		horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1])
		key = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])
		# ceil to power of two
		desired_length = len(horizontal) + len(vertical)
		toeplitz_seed_filler_len = desired_length - vertical.size - horizontal.size
		print(toeplitz_seed_filler_len)
		toeplitz_seed = np.hstack((vertical, horizontal)).astype(int)
		key_start = np.hstack((key[:len(horizontal)+1], np.zeros(desired_length-len(horizontal)-1, ))).astype(int)
		key_rest = np.hstack((key[len(horizontal)+1:], np.zeros(desired_length-(desired_length-len(horizontal)), ))).astype(int)

		print("desired_length:", desired_length)
		print("horizontal:", horizontal)
		print("vertical:\n", vertical)
		print("horizontal:\n", horizontal)
		print("toeplitz_seed:\n", toeplitz_seed)
		print("key_start:\n", key_start)
		print("key_rest:\n", key_rest)
		print("fft(toeplitz_seed):\n", np.fft.fft(toeplitz_seed))
		print("fft(key_start):\n", np.fft.fft(key_start))
		print("fft(toeplitz_seed)*fft(key_start):\n", np.fft.fft(toeplitz_seed) * np.fft.fft(key_start))
		print("np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)):\n", np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)))
		permutated_key = np.around(np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)).real).astype(int) % 2
		print("permutated_key_raw:\n", permutated_key)
		print("key_rest:", key_rest)
		permutated_key ^= key_rest
		print("permutated_key:\n", permutated_key)
		return permutated_key[:len(vertical)]
		
def generate_key_bit_string(length):
	return np.zeros(length) #np.random.randint(0, size=(length,))
		
		

#Toeplitz.MATRIX_SEED = generate_key_bit_string(2**(size+2))
#Toeplitz.NEW_KEY_LENGTH = 2**size
# So lange wie horizontal_ganz
# key = generate_key_bit_string(5)
start = time.time()
amp_key = Toeplitz.permutate()
end = time.time()
print('Amplificated Key: {}, size: {}'.format(amp_key, amp_key.size))
print('{:5.3f}s'.format(end-start))