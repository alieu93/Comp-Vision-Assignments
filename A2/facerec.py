#facerec.py
# 1. Load n faces and perform PCA (Principle Component Analysis) analysis on these faces t
# identify the sub-space of salient "features" (think dimensionality reduction)
# https://en.wikipedia.org/wiki/Dimensionality_reduction
#
# 2. Project faces into this sub-space and store information (in a databse)
#
# 3. When a query face is presented, project it into this sub-space and find the top-k
# nearest faces in the databse
#
# 4. Check to see if your program is working as expected, i.e, how often it makes a mistake
# Construct confusion matrix to ascertain the accuracy of your system. Can use the set
# supplied faces for this pirpose

import sys
import cv2
import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt


def readfaceimgfile(imgfile):
	x = []
	if os.path.isfile(os.path.join('rawdata', imgfile)) is True:
		if imgfile == "2412" or name == "2416":
			fpath = open(os.path.join("rawdata", imgfile), 'rb')
			test = np.fromfile(fpath, dtype='uint8')
			x.append(np.asarray(test, dtype='uint8'))
	return x

def readImageFiles(path, n):
	x = []
	filenames = []
	i = 0
	for root, dirs, files in os.walk(path, topdown=False):
		for name in files:
			if name == "2412" or name == "2416":
				print
			else:
				if i > n:
					break
			#print (os.path.join(root, name))
				try:
					fpath = open(os.path.join(root, name), 'rb')
					test = np.fromfile(fpath, dtype='uint8')
					x.append(np.asarray(test, dtype='uint8'))
					filenames.append(name)
				except IOError:
					print "IO Error".format(errno, strerror)
			i += 1

	return x, filenames


# Load nn faces and perform Principle Component Analysis analysis 
# on these faces to identify the sub-space of salient "features" (think dimensionality reduction). 
def pca(mat):
	#Vectorize images
	#mat = np.array(mat)
	#mat.flatten()

	numofdata, dimension = mat.shape
	mean = mat.mean(axis=0) #Calculate mean
	#mat = mat - mean  #Subtract mean
	for i in range(numofdata):
		mat[i] -= mean
	cov = np.dot(mat, mat.T) #Calculate covariance
	[eigenvalues, eigenvectors] = np.linalg.eigh(cov) #Obtain eigenvalue and eigenvector
	eigenvectors = np.dot(mat.T, eigenvectors)

	sort_index = np.argsort(-eigenvalues)	#Sort eigenvalues
	eigenvalues = eigenvalues[sort_index]	#Sort Eigenvalues
	eigenvectors = eigenvectors[:, sort_index] #Sort eigenvectors
	#print eigenvalues, eigenvectors, mean
	return eigenvalues, eigenvectors, mean

# 2. Project faces into this sub-space and store the information (in a database).
# w is eigenvectors
# x is the matrix
def projection(w_eigenvectors, x_matrix, mean):
	# x = W^T (x - mean)
	#return np.dot(x_matrix - mean, w_eigenvectors)
	return np.dot(w_eigenvectors, (x_matrix - mean))

# Too large to save projection matrix, so just save regular matrix for later use and calculate projection
def saveToSubspace(file, filename, evalues, evectors, mean, mat):
	np.savez(file, filenames = filename, eigenvalues = evalues, eigenvectors = evectors, mean = mean)
	np.savetxt("matrix", mat)


def loadFromSubspace(file):
	subspace = np.load(file)
	matrix = np.loadtxt("matrix")
	filename = subspace['filenames']
	eigenvalues = subspace['eigenvalues']
	eigenvectors = subspace['eigenvectors']
	mean = subspace['mean']
	#proj = subspace['projection']
	return filename, eigenvalues, eigenvectors, mean, matrix

# 3. When a query face is presented, project it into this sub-space
# and find top-k nearest faces in the database
# We take an inputted filename, get the projection for it and take
# the projection from the subspace file and find the closest distance
# We use an array of filenames that correspond with it to give the name
# of the file that is closest
def closestDistance(imgfile, filenames, projection):
	x = readfaceimgfile(imgfile)
	nearest = []

	x_mat = np.array(x)
	x_mat.flatten()

	x_evalues, x_evectors, x_mean = pca(x_mat)
	#y_proj = projection(x_evectors, x_mat, x_mean)
	closest = -1
	index = 0
	counter = 0
	
	for p in projection:
		distance = np.linalg.norm(x - p[0])
		if closest > distance or distance is -1:
			closest = distance
			index = counter
			nearest.append(filenames[closest])

		counter += 1

	return filenames[index], projection[index], nearest


# 4. Check to see if your program is working as expected, 
# i.e., how often it makes a mistake. Construct confusion matrix to ascertain the accuracy of your system.
# You can use the set of supplied faces for this purpose.
# 3993 images, not practical with that many, maybe just 300 or so
# randomly choose about 2/3 of it to go through PCA and then choose another 1/3 of it to undergo query (#3)



# Delete later before submission
def testExecution(imgfile):
	x, files = readImageFiles(imgfile, 1500) #Breaks due to files being 256 kb?
	mat = np.array(x)
	mat.flatten()
	values, vectors, mean = pca(mat)
	
	y = projection(vectors, mat, mean)
	
	saveToSubspace("data", files, values, vectors, mean, mat)
	
	filename, eigenvalue, eigenvector, mean, matrix = loadFromSubspace("data.npz")

	close_file, close_projection = closestDistance("1400", filename, y)
	print close_file
	

#After files are already created for subspace and matrix
def testExecution2(imgfile):

	filename, eigenvalue, eigenvector, mean, matrix = loadFromSubspace("data.npz")

	proj = projection(eigenvector, matrix, mean)

	close_file, close_projection = closestDistance("1400", filename, proj)
	print close_file


# Number 1
def doPCA(imglocation):
	num = raw_input("How many faces?")
	x, files = readImageFiles(imglocation, num)
	mat = np.array(x)
	mat.flatten()
	eigenvalues, eigenvectors, mean = pca(mat)
	return eigenvalues, eigenvectors, mean

# Number 2
def addFace(imgfile):
	filename, eigenvalue, eigenvector, mean, matrix = loadFromSubspace("data.npz")
	x = readfaceimgfile(imgfile)
	mat = np.array(x)
	mat.flatten()
	x_values, x_vectors, x_mean = pca(mat)

	np.append(eigenvalue, x_values)
	np.append(eigenvector, x_vectors)
	np.append(mean, x_mean)
	np.append(matrix, mat)
	np.append(filename, imgfile)

	saveToSubspace("data", filename, eigenvalue, eigenvector, mean, matrix)


# Number 3
def queryFace(imgfile):
	filename, eigenvalue, eigenvector, mean, matrix = loadFromSubspace("data.npz")
	proj = projection(eigenvector, matrix, mean)

	close_file, close_projection, nearest = closestDistance(imgfile, filename, proj)
	print ("Nearest faces in database is " + close_file)

# Number 4
def confusionMatrix(imglocation):
	
	num = raw_input("How many faces? ")
	x, files = readImageFiles(imglocation, num)
	mat = np.array(x)
	mat.flatten()
	eigenvalues, eigenvectors, mean = pca(mat)
	y = projection(vectors, mat, mean)
	saveToSubspace("data", files, values, vectors, mean, mat)
	proj = projection(eigenvector, matrix, mean)

	last_third = num / 3





if __name__ == '__main__':
	arg_length = len(sys.argv)
	if(arg_length < 2 or arg_length > 3):
		print "USAGE: python facerec.py [--command] [<rawimagefolder> OR <faceimgfile>]"
		sys.exit(1)
	argument = sys.argv[1]


	if argument == '--do-pca':
		#Takes <rawimagefolder>
		print 'do-pca'
		doPCA(sys.argv[2])
	elif argument == '--add-face':
		#Takes <faceimgfile>
		print 'add-face'
		addFace(sys.argv[2])
	elif argument == '--query':
		#Takes <faceimgfile>
		print 'query'
		queryFace(sys.argv[2])
	elif argument == '--confusion-matrix':
		#Takes <rawimagefolder>
		print 'confusion-matrix'
		confusionMatrix(sys.argv[2])
	elif argument == '--test':
		print 'test ' + sys.argv[2]
		#readfaceimgfile(sys.argv[2])
		testExecution(sys.argv[2])
	elif argument == '--test2':
		testExecution2(sys.argv[2])


