from RFMD.static.Model.face_mask_detector import FaceMaskDetector
from flask import request, render_template, redirect, url_for, Response, make_response, jsonify
from RFMD import app
import time

VIDEO = 1


@app.route( "/user/videoFeed" )
def videoFeed():

	video = globals()[ 'VIDEO' ]
	d = FaceMaskDetector( input=video )

	return Response( d.start_detection(), mimetype='multipart/x-mixed-replace; boundary=frame' )


# @app.route( "/User/configure", methods=[ "POST" ] )
# def configure():
# 	req = request.get_json()
# 	print( req )
# 	if 'stopline' in req:
# 		detector.stopline = req[ 'stopline' ]
# 		print( req[ 'stopline' ] )
# 	elif 'midline' in req:
# 		detector.midline = req[ 'midline' ]
# 		print( req[ 'midline' ] )
# 	elif 'parkarea' in req:
# 		detector.parkarea = req[ 'parkarea' ]
# 		print( req[ 'parkarea' ] )

# 	elif 'slluc' == req[ 'sl' ]:
# 		detector.sly -= 5
# 	elif 'slruc' == req[ 'sl' ]:
# 		detector.sly1 -= 5
# 	elif 'slu' == req[ 'sl' ]:
# 		detector.sly -= 5
# 		detector.sly1 -= 5
# 	elif 'slll' == req[ 'sl' ]:
# 		detector.slx -= 5
# 	elif 'sllr' == req[ 'sl' ]:
# 		detector.slx += 5
# 	elif 'slrl' == req[ 'sl' ]:
# 		detector.slx1 -= 5
# 	elif 'slrr' == req[ 'sl' ]:
# 		detector.slx1 += 5
# 	elif 'slldc' == req[ 'sl' ]:
# 		detector.sly += 5
# 	elif 'slrdc' == req[ 'sl' ]:
# 		detector.sly1 += 5
# 	elif 'sld' == req[ 'sl' ]:
# 		detector.sly += 5
# 		detector.sly1 += 5

# 	elif 'mlluc' == req[ 'sl' ]:
# 		detector.mlx -= 5
# 	elif 'mlruc' == req[ 'sl' ]:
# 		detector.mlx1 -= 5
# 	elif 'mluu' == req[ 'sl' ]:
# 		detector.mly -= 5
# 	elif 'mll' == req[ 'sl' ]:
# 		detector.mlx -= 5
# 		detector.mlx1 -= 5
# 	elif 'mlr' == req[ 'sl' ]:
# 		detector.mlx += 5
# 		detector.mlx1 += 5

# 	elif 'mlldc' == req[ 'sl' ]:
# 		detector.mlx += 5
# 	elif 'mlrdc' == req[ 'sl' ]:
# 		detector.mlx1 += 5
# 	elif 'mldu' == req[ 'sl' ]:
# 		detector.mly += 5
# 	elif 'mldd' == req[ 'sl' ]:
# 		detector.mly1 += 5

# 	return make_response( jsonify( { "message": "json recieved" } ), 200 )
