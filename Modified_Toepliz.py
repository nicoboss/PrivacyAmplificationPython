import numpy as np

m = [1, 2, 3, 4, 5]
n = [7, 7, 3, 3, 2, 1, 3]
k = [7, 7, 7]

padded_key = np.hstack((k, np.zeros(len(m)*len(n) - len(k) ))).astype(np.int)

r = n[::-1][:-1]+m
#cudaMalloc(&ptr, size);
#cudaMemset(ptr, 0, size);
o = np.zeros((len(m), len(n)))
print(r)

for spalte in range(len(n)-len(m)):
    i = len(n)-1-spalte
    for zeile in range(len(m)):
        o[zeile][spalte] = r[i]
        print(spalte, zeile)
        i += 1
i=0
for spalte in range(len(n)-len(m), len(n)):
    o[i][spalte] = 1
    i += 1

print(o)
print(padded_key)