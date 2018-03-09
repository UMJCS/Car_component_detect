import cv2

import numpy as np
def de(img,kernel):
	eroded=cv2.erode(img,kernel);    
	dilated = cv2.dilate(img,kernel)      
	result = cv2.absdiff(dilated,eroded); 
	return result
def detect_blue(input_img):
	hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
	H, S, V = cv2.split(hsv)
	lower_blue=np.array([80,43,46])
	upper_blue=np.array([130,255,255])
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	#cv2.imshow('Mask', mask)
	res = cv2.bitwise_and(hsv,hsv, mask=mask)
	cv2.imshow('Result', res)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
	return res

def blue_area(res,img):
	img_h = img.shape[0]
	img_w = img.shape[1]
	H,S,V=cv2.split(res)
	lines=[0] * img_h
	for x in range(0,img_h):
		target_point = 0
		for y in H[x]:
			if y!=0:
				target_point+=1
		lines[x] = target_point
	for x in range(0,img_h):
		if lines[x]<35:
			for y in range(0,img_w):
				res[x][y] = [0,0,0]
	cv2.imshow('divide_area', w_res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return res



	

w_img = cv2.imread("./imgs/wm.bmp")
r_img = cv2.imread("./imgs/om.bmp")
w_img_h = w_img.shape[0]
w_img_w = w_img.shape[1]
r_img_h = r_img.shape[0]
r_img_w = r_img.shape[1]
kernel = np.ones((5,5),np.uint8)

cv2.imshow("w_img",w_img)
cv2.imshow("r_img",r_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


w_res=detect_blue(w_img)
r_res=detect_blue(r_img)
w_lines=blue_area(w_res,w_img)
r_lines=blue_area(r_res,r_img)


w_gradient = de(w_img,kernel)
r_gradient = de(r_img,kernel)
cv2.imshow("w_g",w_gradient)
cv2.imshow("r_g",r_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

w_gray = cv2.cvtColor(w_gradient,cv2.COLOR_BGR2GRAY) 
r_gray = cv2.cvtColor(r_gradient,cv2.COLOR_BGR2GRAY)


w_canny = cv2.Canny(w_gray.copy(), 50,200)
cv2.imshow("wrong",w_canny)
r_canny = cv2.Canny(r_gray.copy(), 50,200)
cv2.imshow("right",r_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret,thresh = cv2.threshold(w_canny,127,255,0)
_,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(w_img, contours, -1, (0,0,255), 2)
# cv2.imshow("Image", w_img)
# cv2.waitKey(0)

