# rotate.py
# Q2: Rotate an image by a given angle (in degrees)
# python rotate.py image.png 45
import argparse
import cv2
import numpy as numpy
from matplotlib import pyplot as plt
import copy

def rotate(imgfile, deg):
	img = cv2.imread(imgfile)
	rows, cols, ch = img.shape
	diagonal = int(numpy.sqrt(cols*cols+rows+rows))

	mat = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
	img = cv2.warpAffine(img, mat, (cols, rows))

	#Without warpAffine
	

	return img

def load_image(imgfile, deg):
	print 'Opening ', imgfile
	img = cv2.imread(imgfile)

	rotated_img = rotate(imgfile, deg)

	cv2.imshow('Original Image', img)
	cv2.imshow('Rotated image by ' + str(deg) + ' degrees', rotated_img)

	print 'Press any key to quit'
	cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='CSCI 4220U Assignment 1 Question 2')
	parser.add_argument('imgfile', help='image file')
	parser.add_argument('degrees', help='rotation in degrees')
	args = parser.parse_args()

	print args.imgfile, args.degrees
	load_image(args.imgfile, int(args.degrees))
