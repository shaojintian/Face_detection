import cv2

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


