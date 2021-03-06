
vertical = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0])
horizontal = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
key = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
desired_length = len(horizontal) + len(vertical)
toeplitz_seed_filler_len = desired_length - vertical.size - horizontal.size
toeplitz_seed = np.hstack((vertical, horizontal)).astype(int)
key_start = np.hstack((key[:len(horizontal)+1], np.zeros(desired_length-len(horizontal)-1, ))).astype(int)
key_rest = np.hstack((key[len(horizontal)+1:], np.zeros(desired_length-(desired_length-len(horizontal)), ))).astype(int)

permutated_key = np.around(np.fft.ifft(np.fft.fft(toeplitz_seed) * np.fft.fft(key_start)).real).astype(int) % 2
permutated_key ^= key_rest
return permutated_key[:len(vertical)]

