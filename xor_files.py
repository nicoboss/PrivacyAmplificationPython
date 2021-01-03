with open("veracrypt_keyfile.bin", "rb") as a:
	with open("ANU_20Oct2017_100MB_7", "rb") as b:
		with open('keyfile.bin', 'wb') as o:
			while True:
				byte = bytearray(a.read(4))
				byte[0] ^= b.read(1)[0]
				byte[1] ^= b.read(1)[0]
				byte[2] ^= b.read(1)[0]
				byte[3] ^= b.read(1)[0]
				o.write(byte)
