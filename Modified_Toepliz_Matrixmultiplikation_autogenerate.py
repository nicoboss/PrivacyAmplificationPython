import numpy as np

#Generate T vertical then horizonal (15 Elements):
vertical = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])
horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1])
key = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])


#Changes:
#First vertical element to last (first) horizonal element
#Appended zero at end of vertical
#Insearted 0 at beginning of key
vertical = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
horizontal = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1])
key = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])
r = np.hstack((horizontal, vertical)).astype(int)
a = np.zeros((len(vertical), len(horizontal)+1))

for spalte in range(len(horizontal)+1):
    i = len(horizontal)-spalte
    for zeile in range(len(vertical)):
        a[zeile][spalte] = r[i]
        i += 1
        

desired_length = len(horizontal) + len(vertical)
key_start = key[:len(horizontal)+1].astype(int)
key_rest = key[len(horizontal)+1:].astype(int)
print(a)
print("key_start:", key_start)
print("key_rest:", key_rest)
preXOR = (a.dot(key_start).astype(int) % 2)
print("preXOR:", preXOR)
print("Amplificated Key:", preXOR ^ key_rest)
