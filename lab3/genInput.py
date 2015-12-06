import sys
import numpy as np

MAX_VALUE = 65535

if __name__=='__main__':
	array_size = int(sys.argv[1])

	array = np.random.randint(0, MAX_VALUE, array_size)

	print array_size

	for element in array:
		print element, " ",
		