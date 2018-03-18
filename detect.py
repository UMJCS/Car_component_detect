import cv2
import numpy as np

def de(img,kernel):
	eroded=cv2.erode(img,kernel);
	dilated = cv2.dilate(img,kernel)
	result = cv2.absdiff(dilated,eroded);
	return result

def detect_blue(input_img):
	img_h = input_img.shape[0]
	img_w = input_img.shape[1]
	hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
	H, S, V = cv2.split(hsv)
	lower_blue=np.array([80,43,46])
	upper_blue=np.array([130,255,255])
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	#cv2.imshow('Mask', mask)
	res = cv2.bitwise_and(hsv,hsv, mask=mask)
	left_edge_list = np.zeros(img_h,dtype=np.int)
	right_edge_list = np.zeros(img_h,dtype=np.int)
	lines=[0]*img_h
	for x in range(0,img_h):
		target_point = 0
		left_trigger = 0
		right_trigger = 0
		for y0 in res[x]:
			if y0[0]!=0:
				target_point+=1
		lines[x] = target_point
	print(lines)
	cv2.imshow('Result', res[lines.index(110):,:])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return res,left_edge_list,right_edge_list,lines

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
	left_edge = []
	for x in range(0,img_h):
		if lines[x]<110:
			left_trigger = 0
			for y in range(0,img_w):
				if res[x][y][0]!= 0 and left_trigger ==0 :
					left_trigger = 1
					left_edge[x] = y
				res[x][y] = [0,0,0]
	cv2.imshow('divide_area', w_res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return res
# def harris corner detection
def harris_corner(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,3,3,0.04)
	dst = cv2.dilate(dst,None)
	print(dst)
	img[dst>0.03*dst.max()]=[0,0,255]
	cv2.imshow('harris',img)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()
	return dst

w_img = cv2.imread("./imgs/right.bmp")
r_img = cv2.imread("./imgs/wrong_type_1.bmp")
illumination_mask = cv2.imread("./imgs/mask.png")
w_img_h = w_img.shape[0]
w_img_w = w_img.shape[1]
r_img_h = r_img.shape[0]
r_img_w = r_img.shape[1]
kernel = np.ones((5,5),np.uint8)

cv2.imshow("w_img",w_img)
cv2.imshow("r_img",r_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ill = cv2.illuminationChange(w_img,illumination_mask,1,1)
# cv2.imshow('illuminationChange',ill)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


w_res,w_left_edge_list,w_right_edge_list,w_lines=detect_blue(w_img)
#r_res,r_left_edge_list,r_right_edge_lis,r_lines=detect_blue(r_img)
#w_lines=blue_area(w_res,w_img)
#r_lines=blue_area(r_res,r_img)
