import sys
import numpy as np

def printMatrix(matrix,rows,columns,row):
    print matrix[row-1][columns-1]


MAX_VALUE = 100

if __name__=='__main__':
	# print "\n".join(sys.argv)
	grid_size = int(sys.argv[1])
	kerne_size =  int(sys.argv[2])


	grid = MAX_VALUE * np.random.rand(grid_size, grid_size)
	kernel = MAX_VALUE * np.random.rand(kerne_size, kerne_size)

	print grid_size, " ", kerne_size

	for row in grid:
		for cell in row:
			print "%0.2f" % cell, " ",
		print

	for row in kernel:
		for cell in row:
			print "%0.2f" % cell , " ",
		print
    # matrix=[]

    # with open('matrix.txt','r') as f:

    #     for l in f:
    #         t=l.split(',')
    #         matrix.append(t)
    # printMatrix(matrix, 3, 3, 1)