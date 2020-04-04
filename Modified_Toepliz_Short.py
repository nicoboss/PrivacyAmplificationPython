import numpy as np

m = [1, 2, 3, 4, 5]
n = [7, 7, 3, 3, 2, 1, 3]
k = [7, 7, 7, 7, 7]

padded_key = np.hstack((k, np.zeros((len(m)*(len(n)-len(m)))-len(k)))).astype(np.int)

r = n[::-1] [:-1]+m
out = np.zeros((len(m), len(n)-len(m)))
print(r)

for spalte in range(len(n)-len(m)):
    i = len(n)-1-spalte
    for zeile in range(len(m)):
        out[zeile][spalte] = r[i]
        print(spalte, zeile)
        i += 1
        
#print(np.fft.ifft(np.fft.fft(out) * np.fft.fft(padded_key)))
print(out)
print(padded_key)