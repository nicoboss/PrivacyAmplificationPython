import numpy as np 

in_arr = np.array([1,0,0,0,0,0,0,0, 
				   1,0,0,0,0,0,0,0, 
				   1,0,0,0,0,0,0,0, 
				   1,0,0,0,0,0,0,0]) 
print ("Input array : ", in_arr)  
out_arr = np.packbits(in_arr)
print ("Output packed array along axis 1 : ", out_arr)
with open('packed_out.bin', 'wb') as f:
	np.array(out_arr, dtype=np.uint8).tofile(f)
