# focus.py
# Question 1
import argparse
import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import time

def filter(img):
    # Complete this method according to the tasks listed in the lab handout. 
    mean = np.array([0,0])
    mean = mean.reshape(2,1)
    cov = np.array([[20,0],[0,20]])

    g2d = gaussian2d(mean, cov, 50)
    g2d_norm = g2d/np.sum(g2d)

    img_gaussian = signal.convolve2d(img, g2d_norm, mode='same', boundary='fill')
    img = img_gaussian

    return img

def gaussian2d(mu, con, n):
    #Generates an n x n gaussian kernel
    #display original image and blurred image
    #mu is mean, 2x1 vector
    #cov is covariance, 2x2 matrix
    s = int(n/2)
    x = np.linspace(-s,s,n)
    y = np.linspace(-s,s,n)
    xc, yc = np.meshgrid(x,y)
    xy = np.zeros([n,n,2])
    xy[:,:,0] = xc
    xy[:,:,1] = yc

    invcon = np.linalg.inv(con)
    results = np.ones([xy.shape[0], xy.shape[1]])
    for x in range(0, xy.shape[0]):
        for y in range(0, xy.shape[1]):
            v = xy[x,y,:].reshape(2,1) - mu
            results[x,y] = np.dot(np.dot(np.transpose(v), invcon), v)
    results = np.exp(- results /2)
    return results



def load_image(imgfile):
	start_time = time.time()
	print 'Opening ', imgfile
	img = cv2.imread(imgfile)
	''
	#f = np.fft.fft2(img)
	#fshift = np.fft.fftshift(f)
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	filtered_img = filter(img_gray)

	#Subtracting gets us the areas that are sharp
	#img3 = cv2.subtract(filtered_img, img_gray)
	img3 = filtered_img - img_gray

	elapsed_time = time.time() - start_time
	print 'Focus analysis took %s seconds' % elapsed_time

	plt.subplot(121)
	plt.imshow(img_gray, cmap='gray')
	plt.title('Input')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(122)
	plt.imshow(img3, cmap='gray')
	plt.title('Sharp regions')
	plt.xticks([])
	plt.yticks([])
	plt.show()
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="CSCI 4220U Assignment 1 Question 1")
	parser.add_argument('imgfile', help='image file')
	args = parser.parse_args()
	load_image(args.imgfile)