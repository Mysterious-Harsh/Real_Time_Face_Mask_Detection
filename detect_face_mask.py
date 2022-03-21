import cv2
import time
import sys
import numpy as np
from centroid_tracker import CentroidTracker


class FaceMaskDetection:

	def __init__( self, model, input_width=640, input_height=640, iou_threshold=0.4, confidence_threshold=0.6 ):
		self.INPUT_WIDTH = input_width
		self.INPUT_HEIGHT = input_height
		self.iou_THRESHOLD = iou_threshold
		self.CONFIDENCE_THRESHOLD = confidence_threshold
		self.CLASSES = [ "No-Mask", "Mask", "Improper" ]
		self.COLORS = [ ( 0, 0, 255 ), ( 0, 255, 0 ), ( 0, 255, 255 ) ]
		is_cuda = len( sys.argv ) > 1 and sys.argv[ 1 ] == "cuda"
		self.net = cv2.dnn.readNet( model )
		self.tracker = CentroidTracker()
		if is_cuda:
			print( "Attempty to use CUDA" )
			self.net.setPreferableBackend( cv2.dnn.DNN_BACKEND_CUDA )
			self.net.setPreferableTarget( cv2.dnn.DNN_TARGET_CUDA_FP16 )
		else:
			print( "Running on CPU" )
			self.net.setPreferableBackend( cv2.dnn.DNN_BACKEND_OPENCV )
			self.net.setPreferableTarget( cv2.dnn.DNN_TARGET_CPU )

	def get_detection( self, image ):
		class_ids = []
		confidences = []
		boxes = []

		blob = cv2.dnn.blobFromImage(
		    image, 1 / 255.0, ( self.INPUT_WIDTH, self.INPUT_HEIGHT ), swapRB=True, crop=False
		    )
		self.net.setInput( blob )
		outs = self.net.forward()
		output_data = outs[ 0 ]

		rows = output_data.shape[ 0 ]

		image_width, image_height, _ = image.shape

		x_factor = image_width / self.INPUT_WIDTH
		y_factor = image_height / self.INPUT_HEIGHT

		for r in range( rows ):
			row = output_data[ r ]
			confidence = row[ 4 ]
			if confidence >= self.CONFIDENCE_THRESHOLD:

				classes_scores = row[ 5 : ]
				_, _, _, max_indx = cv2.minMaxLoc( classes_scores )
				class_id = max_indx[ 1 ]
				if ( classes_scores[ class_id ] > .25 ):

					confidences.append( confidence )

					class_ids.append( class_id )

					x, y, w, h = row[ 0 ].item(), row[ 1 ].item(), row[ 2 ].item(), row[ 3 ].item()
					left = int( ( x - 0.5 * w ) * x_factor )
					top = int( ( y - 0.5 * h ) * y_factor )
					width = int( w * x_factor )
					height = int( h * y_factor )
					box = np.array( [ left, top, width, height ] )
					boxes.append( box )

		indexes = cv2.dnn.NMSBoxes( boxes, confidences, self.CONFIDENCE_THRESHOLD, self.iou_THRESHOLD )

		result_class_ids = []
		result_confidences = []
		result_boxes = []
		rect = []

		for i in indexes:
			result_confidences.append( confidences[ i ] )
			result_class_ids.append( class_ids[ i ] )
			result_boxes.append( boxes[ i ] )
			rect.append(
			    [
			        boxes[ i ][ 0 ], boxes[ i ][ 1 ], boxes[ i ][ 0 ] + boxes[ i ][ 2 ],
			        boxes[ i ][ 1 ] + boxes[ i ][ 3 ]
			        ]
			    )

		return result_class_ids, result_confidences, result_boxes, rect

	def start_web( self, index ):
		self.capture = cv2.VideoCapture( index )
		self.start_detection()

	def start_detection( self ):
		start = time.time_ns()

		frame_count = 0
		total_frames = 0
		fps = -1

		while True:

			_, frame = self.capture.read()

			if frame is None:
				print( "End of stream" )
				break

			row, col, _ = frame.shape
			_max = max( col, row )
			inputImage = np.zeros( ( _max, _max, 3 ), np.uint8 )
			inputImage[ 0 : row, 0 : col ] = frame

			# outs = detect( inputImage, net )

			class_ids, confidences, boxes, rect = self.get_detection( inputImage )
			obj, did = self.tracker.update( rect )
			frame_count += 1
			total_frames += 1

			for ( classid, confidence, box, obj ) in zip( class_ids, confidences, boxes, obj.items() ):

				color = self.COLORS[ int( classid ) % len( self.COLORS ) ]
				cx, cy, x, y, x1, y1 = obj[ 1 ]
				cv2.rectangle( frame, box, color, 4 )
				cv2.rectangle( frame, ( box[ 0 ], box[ 1 ] - 40 ), ( box[ 0 ] + box[ 2 ], box[ 1 ] ), color, -1 )
				cv2.putText(
				    frame,
				    str( round( confidence, 2 ) ) + " : " + self.CLASSES[ classid ], ( box[ 0 ], box[ 1 ] - 10 ),
				    cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 0, 0 ), 2
				    )
				cv2.putText(
				    frame, "ID {}".format( obj[ 0 ] ), ( cx - 10, cy - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 255, 0 ),
				    2
				    )
				cv2.circle( frame, ( cx, cy ), 4, color, -1 )

			if frame_count >= 30:
				end = time.time_ns()
				fps = 1000000000 * frame_count / ( end - start )
				frame_count = 0
				start = time.time_ns()

			if fps > 0:
				fps_label = "FPS: %.2f" % fps
				cv2.putText( frame, fps_label, ( 10, 25 ), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 0, 255 ), 2 )

			cv2.imshow( "output", frame )

			if cv2.waitKey( 1 ) > -1:
				print( "finished by user" )
				break

		print( "Total frames: " + str( total_frames ) )


if __name__ == "__main__":
	md = FaceMaskDetection( "model/best.onnx" )
	md.start_web( 1 )
