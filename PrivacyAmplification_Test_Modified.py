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
	def permutate(cls, key_bit_string: np.ndarray):
		vertical = np.array([1, 0, 1, 0, 1])
		horizontal_ganz = np.array([1, 1, 1, 1, 0, 1, 1])
		horizontal = horizontal_ganz[:-len(vertical)]
		# ceil to power of two
		desired_length = 1 << (max(len(horizontal), len(vertical))*2 -1).bit_length()
		toeplitz_seed_filler_len = desired_length - vertical.size - horizontal[::-1][:-1].size
		print(toeplitz_seed_filler_len)
		toeplitz_seed = np.hstack((vertical, np.zeros(toeplitz_seed_filler_len,), horizontal[::-1][:-1])).astype(np.int)
		padded_key = np.hstack((key_bit_string, np.zeros(desired_length - key_bit_string.size, ))).astype(np.int)
		key_bit_string = padded_key[:len(horizontal)]
		key_rest = np.hstack((padded_key[len(horizontal):], np.zeros(desired_length - (desired_length - len(horizontal)), ))).astype(np.int)

		print("desired_length:", desired_length)
		print("horizontal[::-1][:-1]:", horizontal[::-1][:-1])
		print("vertical:\n", vertical)
		print("horizontal:\n", horizontal)
		print("toeplitz_seed:\n", toeplitz_seed)
		print("padded_key:\n", padded_key)
		print("fft(toeplitz_seed):\n", np.fft.fft(toeplitz_seed))
		print("fft(padded_key):\n", np.fft.fft(padded_key))
		print("fft(toeplitz_seed)*fft(padded_key):\n", np.fft.fft(toeplitz_seed) * np.fft.fft(padded_key))
		print("np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(padded_key)):\n", np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(padded_key)))
		permutated_key = np.around(np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(padded_key)).real).astype(np.int) % 2
		print("permutated_key_raw:\n", permutated_key)
		print("key_rest:", key_rest)
		permutated_key ^= key_rest
		print("permutated_key:\n", permutated_key)

		return permutated_key
		
def generate_key_bit_string(length):
	return np.zeros(length) #np.random.randint(0, size=(length,))
		
		

#Toeplitz.MATRIX_SEED = generate_key_bit_string(2**(size+2))
#Toeplitz.NEW_KEY_LENGTH = 2**size

# So lange wie horizontal_ganz
key = generate_key_bit_string(5)
key[0]=1
key[1]=1
key[2]=1
key[3]=1
key[4]=1
print('Key: {}, size: {}'.format(key, key.size))

start = time.time()
amp_key = Toeplitz.permutate(key)
end = time.time()
print('Amplificated Key: {}, size: {}'.format(amp_key, amp_key.size))
print('{:5.3f}s'.format(end-start))