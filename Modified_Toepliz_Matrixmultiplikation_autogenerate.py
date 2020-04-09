import numpy as np

vertical = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0])
horizontal = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
key = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
r = np.hstack((horizontal, vertical)).astype(np.int)
a = np.zeros((len(vertical), len(horizontal)+1))

for spalte in range(len(horizontal)+1):
    i = len(horizontal)-spalte
    for zeile in range(len(vertical)):
        a[zeile][spalte] = r[i]
        i += 1
        

desired_length = len(horizontal) + len(vertical)
key_start = key[:len(horizontal)+1].astype(np.int)
key_rest = key[len(horizontal)+1:].astype(np.int)
print(a)
print("key_start:", key_start)
print("key_rest:", key_rest)
preXOR = (a.dot(key_start).astype(np.int) % 2)
print("preXOR:", preXOR)
print("Amplificated Key:", preXOR ^ key_rest)