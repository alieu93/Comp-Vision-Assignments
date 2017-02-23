import cv2
import numpy as np
import scipy as sp

'''
- Assignment 3:
- Design an embedded vision system for detecting tennis balls on Raspberry Pi.
- Strongly suggest going Linux + OpenCV + Python + Numpy + Scipy route
- Due: Sun March 27 at midnight (11:59 pm)
'''

#Ripped from the first in-class exercisei
#convert to hsv, use inRange, findContoiur, drawContour

def process_cam():
	#cap = cv2.VideoCapture(-1)
	cap = cv2.VideoCapture(0)
	kernel = np.ones((5,5), np.uint8)


	while(True):
		ret, frame = cap.read()

		print frame

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		hue,sat,val = cv2.split(hsv)



		#Defining normal colour of a tennis ball
		hueThresh = cv2.inRange(np.array(hue), np.array(28), np.array(40))
		satThresh = cv2.inRange(np.array(sat), np.array(153), np.array(255))
		valThresh = cv2.inRange(np.array(val), np.array(51), np.array(255))

		mask = cv2.bitwise_and(hueThresh, cv2.bitwise_and(satThresh, valThresh))

		#mask = cv2.inRange(hsv, min2, max2)
		dilate = cv2.dilate(mask, kernel, 1)
		#mask = cv2.erode(mask, None, iterations=3)
		morphClose = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
		frame_gaussian = cv2.GaussianBlur(morphClose, (5,5), 0)

		cnts = cv2.findContours(frame_gaussian.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		if len(cnts) > 0:
			
			c = max(cnts, key=cv2.contourArea)
			((x,y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)

			if M["m00"] != 0.0:
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# Anything below is most likely noise
			if radius > 20:
				cv2.circle(frame, (int(x), int(y)), int(radius), (0,0,255), 2)
		

			
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
