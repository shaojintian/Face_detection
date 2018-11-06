import cv2
import numpy as np

#import Haar files to train face detection model 

face_cascade=cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')

# check whether haar is done 
if face_cascade.empty():
	raise IOError('Unable to load haarcascade_frontalface_alt')

# initialize camera from PC camera 
# parameter 0 -> PC camera
camera=cv2.VideoCapture(0)

#image clearity scale 

scale_image=0.6


# infinite circle to capture images 

while True:
	#capture current image
	#frame->帧
	ret,frame=camera.read()
	#每次都调整一下帧的大小
	#interpolation（插值法）：最近邻时间复杂度最小
	frame=cv2.resize(frame,None,fx=scale_image,fy=scale_image,
					interpolation=cv2.INTER_NEAREST)
	#将图像转为灰度图
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#灰度图像上运行人脸检测器->人脸矩形框
		#1.5->乘积系数，5->最小紧邻数量	
	face_rects=face_cascade.detectMultiScale(gray,1.5,5)

	#画出橙色矩形框
	#3->框的厚度
	for(x,y,w,h) in face_rects:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,128,0),2)
	#显示帧
	cv2.namedWindow('SJT_cam',0)#0->auto size window
	cv2.imshow('SJT_cam',frame)

	# 通过esc退出
	#wait 1ms to check key response
	key=cv2.waitKey(1)
	if key==27:#27==ESC
		break;
#释放摄像头
camera.release()
#关闭所有窗体
cv2.destroyAllWindows()


