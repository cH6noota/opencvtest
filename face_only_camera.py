import cv2
import numpy as np

flag=False
#カスケード分類器読み込み
hc = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
while True:
	# VideoCaptureから1フレーム読み込む
	ret, image = cap.read()
	#cv2.imshow('Raw Frame', image)
	#顔の座標xy幅高さを配列facesに
	faces = hc.detectMultiScale(image, minSize=(10, 10))

	if len(faces) == 0:
		print('no faces')
		#raise Exception('no faces')
	else :
		print('authorized!!')
		flag=True


	for x, y, w, h in faces:
		#四角を描く
		cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
		# 入力画像から窓画像を切り取り
		face_only= image[y:y+h, x:x+w]

	cv2.imshow('face',image)
	cv2.imwrite('faceonly.png',face_only)
	
	k = cv2.waitKey(1)
	
	if k == 27 or flag==True:
		break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()

