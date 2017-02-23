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
		hue,sat,val = cv2.split(hsv)

		rangeMin = np.array([0,50,50])
		rangeMax = np.array([6,255,255])

		hueThresh = cv2.inRange(np.array(hue), np.array(0), np.array(6))
		satThresh = cv2.inRange(np.array(sat), np.array(50), np.array(240))
		valThresh = cv2.inRange(np.array(val), np.array(50), np.array(240))

		mask = cv2.bitwise_and(hueThresh, cv2.bitwise_and(satThresh, valThresh))

		#mask = cv2.inRange(hsv, min2, max2)
		dilate = cv2.dilate(mask, kernel, 1)
		#mask = cv2.erode(mask, None, iterations=3)
		morphClose = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
		frame_gaussian = cv2.GaussianBlur(morphClose, (5,5), 0)

		# May need to adjust radius values
		circles =  cv2.HoughCircles(morphClose, cv2.HOUGH_GRADIENT, 2, 1000, 120, 50, minRadius=10, maxRadius=0)  
		#circles =  cv2.HoughCircles(frame_gaussian, cv2.cv.cv_HOUGH_GRADIENT, 2, 120, 120, 50, 10, 0)  
		print circles
		if circles is not None:
			for i in circles[0,:]:
				cv2.circle(frame, (i[0], i[1]), i[2], (0,0,255), 5)


				#v2.circle(frame,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,0,255),5)
				#cv2.circle(frame, (i[0], i[1]), 2, (0,255,0), 10)
		

			
		cv2.imshow('Frame Gaussian', frame_gaussian)		
		cv2.imshow("Morph", morphClose)
		cv2.imshow("Frame", frame)
		cv2.imshow("hsv", hsv)
		

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	process_cam()
