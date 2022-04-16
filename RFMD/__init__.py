from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta
import warnings

warnings.filterwarnings( "ignore", category=FutureWarning )
warnings.filterwarnings( "ignore", category=DeprecationWarning )

app = Flask( __name__ )

app.secret_key = 'BlackThunder'

app.config[ 'PERMANENT_SESSION_LIFETIME' ] = timedelta( minutes=30 )

app.config[ 'TESTING' ] = True

app.config[ 'SQLALCHEMY_TRACK_MODIFICATIONS' ] = True

app.config[ 'SQLALCHEMY_ECHO' ] = True

app.config[ 'SQLALCHEMY_RECORD_QUERIES' ] = True

app.config[ 'SQLALCHEMY_DATABASE_URI' ] = 'mysql+pymysql://root@db.cs.dal.ca::3306/rfmd'

app.config[ 'SQLALCHEMY_MAX_OVERFLOW' ] = 0

app.config[ 'UPLOAD_FOLDER' ] = 'static/Dataset/Videos'

db = SQLAlchemy( app )

print( "in project" )

import RFMD.com.controller.RegistrationController
