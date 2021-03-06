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
		horizontal = np.array([1, 1, 1, 1, 0, 1, 1])

		# ceil to power of two
		desired_length = 1 << (key_bit_string.size*2 -1).bit_length()

		toeplitz_seed_filler_len = desired_length - vertical.size - horizontal[::-1][:-1].size
		toeplitz_seed = np.hstack((vertical, np.zeros(toeplitz_seed_filler_len,), horizontal[::-1][:-1])).astype(int) #[0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0]
		padded_key = np.hstack((key_bit_string, np.zeros(desired_length - key_bit_string.size, ))).astype(int) #[1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]

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
		permutated_key = np.around(np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(padded_key)).real).astype(int) % 2
		print("permutated_key:\n", permutated_key)

		return permutated_key
		
def generate_key_bit_string(length):
	return np.zeros(length) ##np.random.randint(2, size=(length,))
		
        

#Toeplitz.MATRIX_SEED = generate_key_bit_string(2**(size+2))
#Toeplitz.NEW_KEY_LENGTH = 2**size

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