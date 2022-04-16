from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from RFMD import app
from RFMD.com.dao.UploadvideoDAO import VideoDAO
from RFMD.com.vo.UploadvideoVO import VideoVO
import RFMD.com.controller.DetectionController as dc
from RFMD.com.controller.LoginController import LoginSession, LogoutSession


@app.route( '/user/loadVideo', methods=[ 'GET' ] )
def userLoadVideo():
	try:
		if LoginSession() == 'user':

			return render_template( 'user/uploadVideo.html' )
		else:
			return redirect( url_for( 'LogoutSession' ) )
	except Exception as ex:
		print( ex )


@app.route( '/user/insertVideo', methods=[ 'POST' ] )
def userInsertVideo():
	try:
		if LoginSession() == 'user':
			video = request.files[ 'video' ]

			# video_crossroadId = request.form[ 'video_crossroadId' ]

			videoname = secure_filename( video.filename )
			videopath = "RFMD/static/Videos/"
			video.save( videopath + videoname )
			videoVO = VideoVO()
			videoDAO = VideoDAO()
			videoVO.videoName = videoname
			videoVO.videoPath = videopath
			# videoVO.video_crossroadId = video_crossroadId
			videoDAO.insertVideo( videoVO )
			dc.VIDEO = videopath + videoname
			return render_template( 'user/detection.html' )
		else:
			return redirect( url_for( 'LogoutSession' ) )

	except:
		# video_crossroadId = request.form[ 'video_crossroadId' ]
		videopath = "RFMD/static/Videos/" + secure_filename( video.filename )
		dc.VIDEO = videopath
		return render_template( 'user/detection.html' )


@app.route( '/user/startweb', methods=[ 'POST' ] )
def startweb():
	camera_id = request.form[ 'index' ]
	dc.VIDEO = int( camera_id )
	return render_template( 'user/detection.html' )


@app.route( '/admin/viewVideo', methods=[ 'GET' ] )
def adminViewVideo():
	try:
		if LoginSession() == 'admin':
			videoDAO = VideoDAO()
			videoVOList = videoDAO.viewVideo()
			print( "__________________", videoVOList )
			return render_template( 'admin/viewVideo.html', videoVOList=videoVOList )
		else:
			return redirect( url_for( 'LogoutSession' ) )
	except Exception as ex:
		print( ex )


@app.route( '/admin/deleteVideo', methods=[ 'GET' ] )
def adminDeleteVideo():
	try:
		if LoginSession() == 'admin':
			videoVO = VideoVO()

			videoDAO = VideoDAO()

			videoId = request.args.get( 'videoId' )
			videoPath = request.args.get( 'videoPath' )
			os.remove( videoPath )
			videoVO.videoId = videoId
			videoDAO.deleteVideo( videoVO )

			return redirect( url_for( 'adminViewVideo' ) )
		else:
			return redirect( url_for( 'LogoutSession' ) )
	except Exception as ex:
		print( ex )


# @app.route('/Admin/editVideo', methods=['GET'])
# def adminEditVideo():
#    try:
#        crossroadDAO = CrossroadDAO()
#        crossroadVOList = crossroadDAO.viewCrossroad()
#        videoVO = VideoVO()
#
#        videoDAO = VideoDAO()
#
#        videoId = request.args.get('VideoId')
#
#        videoVO.VideoId = videoId
#
#        videoVOList = videoDAO.editVideo(videoVO)
#
#        print("=======videoVOList=======", videoVOList)
#
#        print("=======type of videoVOList=======", type(videoVOList))
#
#        return render_template('Admin/AddVideo.html', videoVOList=videoVOList,crossroadVOList=crossroadVOList)
#    except Exception as ex:
#        print(ex)
#
#
# @app.route('/Admin/updateVideo', methods=['POST'])
# def adminUpdateVideo():
#    try:
#        videoId = request.form['VideoId']
#        video = request.form['video']
#        video_crossroadId= request.form['video_crossroadId']
#
#        videoVO = VideoVO()
#        videoDAO = VideoDAO()
#
#        videoVO.VideoId = videoId
#        videoVO.VideoName= videoname
#        videoVO.Video_CrossroadId= video_crossroadId
#
#        videoDAO.updateVideo(videoVO)
#
#        return redirect(url_for('adminViewVideo'))
#    except Exception as ex:
#        print(ex)
