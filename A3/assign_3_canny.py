import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

'''
- Assignment 3:
- Design an embedded vision system for detecting tennis balls on Raspberry Pi.
- Strongly suggest going Linux + OpenCV + Python + Numpy + Scipy route
- Due: Sun March 27 at midnight (11:59 pm)
'''

#Ripped from the first in-class exercisei
#convert to hsv, use inRange, findContoiur, drawContour

#Look up how to use find and drawcontour

def process_cam():
	#cap = cv2.VideoCapture(-1)
	cap = cv2.VideoCapture(0)
	min1 = np.array([29, 86, 6])
	max1 = np.array([64, 255, 255])
	kernel = np.ones((5,5), np.uint8)


	while(True):
		ret, frame = cap.read()
		
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img_canny = cv2.Canny(frame, 100, 200)
		frame_gaussian = cv2.GaussianBlur(img_canny, (5,5), 0)

		circles =  cv2.HoughCircles(frame_gaussian, cv2.HOUGH_GRADIENT, 2, 10000, 120, 50, minRadius=100, maxRadius=0)  
		if circles is not None:
			for i in circles[0,:]:
				cv2.circle(frame, (i[0], i[1]), i[2], (0,0,255), 5)


		

			
		cv2.imshow('Frame Gaussian', frame_gaussian)		
		cv2.imshow("Frame", frame)
		

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	process_cam()
