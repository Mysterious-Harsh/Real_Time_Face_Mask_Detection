from RFMD import db
from RFMD.com.vo.LoginVO import LoginVO
import datetime


class RegistrationVO( db.Model ):
	__tablename__ = 'registrationmaster'
	registrationId = db.Column( 'registrationId', db.Integer, primary_key=True, autoincrement=True )

	datetime = db.Column( 'datetime', db.DateTime, default=datetime.datetime.now() )
	registration_loginId = db.Column( 'registration_loginId', db.Integer, db.ForeignKey( LoginVO.loginId ) )

	def as_dict( self ):
		return {
		    'registrationId': self.registrationId,
		    'datetime': self.datetime,
		    'registration_loginId': self.registration_loginId
		    }


db.create_all()
