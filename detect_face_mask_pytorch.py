import cv2
import torch

# Model
# model = torch.hub.load( 'ultralytics/yolov5', 'yolov5s' )
model = torch.hub.load( 'yolov5', 'custom', path='model/model.pt', source='local' )  # local model
# Images

capture = cv2.VideoCapture( 1 )
while True:

	_, frame = capture.read()
	# frame = ImageGrab.grab( bbox=( 0, 1000, 100, 1100 ) )  #x, y, w, h

	if frame is None:
		print( "End of stream" )
		break
	frame = cv2.cvtColor( frame, cv2.COLOR_BGR2BGRA )
	# img = cv2.imread( '1.png' )  # OpenCV image (BGR to RGB)
	# img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )

	# Inference
	results = model( frame, size=640 )  # includes NMS

	# Results

	results.show()  # or .show()