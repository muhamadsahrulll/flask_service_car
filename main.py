import pymysql
import torch
from ultralytics import YOLO
pymysql.install_as_MySQLdb()
from flask import Flask, after_this_request, make_response, jsonify, render_template, send_file,session
from flask_restx import Resource, Api, reqparse
from flask_cors import  CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import jwt, os,random
from flask_mail import Mail, Message
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import subprocess

import cv2
import os
import shutil
import subprocess
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', "mp4"}

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression




from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size,check_imshow, check_requirements,colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


app = Flask(__name__, template_folder='Templates')
api = Api(app)
CORS(app)
port = int(os.environ.get("RAILWAY_PORT", 5000))
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:@127.0.0.1:3306/kendaraan"
# app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:qBEtpDcwnoplHMkgZMbv@containers-us-west-46.railway.app:7986/railway"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'whateveryouwant'
# mail env config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")
mail = Mail(app)
# mail env config
db = SQLAlchemy(app)



class Users(db.Model):
    id       = db.Column(db.Integer(), primary_key=True, nullable=False)
    firstname     = db.Column(db.String(30), nullable=False)
    lastname     = db.Column(db.String(30), nullable=False)
    email    = db.Column(db.String(64), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    is_verified = db.Column(db.Boolean(),nullable=False)
    token = db.Column(db.String(5), nullable=False)
    createdAt = db.Column(db.Date, default=datetime.utcnow)
    updatedAt = db.Column(db.Date, default=datetime.utcnow)

# class History(db.Model):
#     id = db.Column(db.Integer(), primary_key=True, nullable=False)
#     nama = db.Column(db.String(30), nullable=False)
#     jenis_kendaraan = db.Column(db.String(30), nullable=False)
#     tanggal = db.Column(db.Date)
#     waktu = db.Column(db.Time)

class Histori(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    nama = db.Column(db.String(30), nullable=False)
    jenis_kendaraan = db.Column(db.String(30), nullable=False)
    tanggal = db.Column(db.Date)

    user = db.relationship('Users', backref=db.backref('histori', lazy=True))



# Email functions
# https://medium.com/@stevenrmonaghan/password-reset-with-flask-mail-protocol-ddcdfc190968
# https://www.youtube.com/watch?v=g_j6ILT-X0k
# https://stackoverflow.com/questions/72547853/unable-to-send-email-in-c-sharp-less-secure-app-access-not-longer-available

authParser = reqparse.RequestParser()
authParser.add_argument('Authorization', type=str, help='Authorization', location='headers', required=True)
@api.route('/user')
class DetailUser(Resource):
       @api.expect(authParser)
       def get(self):
        args = authParser.parse_args()
        bearerAuth  = args['Authorization']
        try:
            jwtToken    = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user =  db.session.execute(db.select(Users).filter_by(email=token['user_email'])).first()
            user = user[0]
            data = {
                'firstname' : user.firstname,
                'lastname' : user.lastname,
                'email' : user.email
            }
        except:
            return {
                'message' : 'Token Tidak valid,Silahkan Login Terlebih Dahulu!'
            }, 401

        return data, 200

#histori parser
historiParser = reqparse.RequestParser()
historiParser.add_argument('nama', type=str, help='Nama', location='json', required=True)
historiParser.add_argument('jenis_kendaraan', type=str, help='Jenis Kendaraan', location='json', required=True)
historiParser.add_argument('tanggal', type=str, help='Tanggal', location='json', required=True)

#membuat histori baru
@api.route('/add-histori')
class AddHistoriResource(Resource):
    @api.expect(authParser, historiParser)
    def post(self):
        args = authParser.parse_args()
        bearerAuth = args['Authorization']

        jwtToken = bearerAuth[7:]
        token = decodetoken(jwtToken)
        user_id = token['user_id']

        # Parse the request data and save it to the database
        data = api.payload
        histori = Histori(
            nama=data['nama'],
            jenis_kendaraan=data['jenis_kendaraan'],
            tanggal=datetime.strptime(data['tanggal'], '%Y-%m-%d').date(),
            user_id=user_id,
        )
        db.session.add(histori)
        db.session.commit()

        return {'message': 'Histori berhasil ditambahkan'}, 201


#menampilkan data histori bedasarkan id
@api.route('/read-histori')
class ReadHistori(Resource):
    @api.expect(authParser)
    def get(self):
        # Mendapatkan user_id dari token yang terverifikasi
        # user_id = get_jwt_identity()
        args = authParser.parse_args()
        bearerAuth = args['Authorization']

        jwtToken = bearerAuth[7:]
        token = decodetoken(jwtToken)
        user_id = token['user_id']

        # Mengambil data histori berdasarkan user_id
        histori = Histori.query.filter_by(user_id=user_id).all()
        if not histori:
            return {'message': 'Histori tidak ditemukan'}, 404

        histori_data = []
        for h in histori:
            histori_data.append({
                'id': h.id,
                'nama': h.nama,
                'jenis_kendaraan': h.jenis_kendaraan,
                'tanggal': h.tanggal.strftime('%Y-%m-%d')
            })

        return histori_data, 200


parserVideo = reqparse.RequestParser()
# parserVideo.add_argument('Authorization', type=str, location='headers', required=True)
parserVideo.add_argument('video', type=FileStorage, location='files', required=True)

ALLOWED_EXTENSIONS = {'3gp', 'mkv', 'avi', 'mp4'}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

detectParser = api.parser()
detectParser.add_argument('video', location='files', type=FileStorage, required=True)
@api.route('/detect')
class Detect(Resource):
    @api.expect(detectParser)
    def post(self):
        args = detectParser.parse_args()
        video = args['video']
        if video and allowed_file(video.filename):
            filename = secure_filename(video.filename)
            video.save(os.path.join("./video", filename))
            subprocess.run(['python', 'detect.py', '--source', f'./video/{filename}', '--weights', 'best.onnx', '--name', f'{filename}'])
            print('success predict')
            os.remove(f'./video/{filename}')
            print('success remove')
            return send_file(os.path.join(f"./runs/detect/{filename}", filename), mimetype='video/mp4', as_attachment=True, download_name=filename)
        else:
            return {'message' : 'invalid file extension'},400


#parserRegister
regParser = reqparse.RequestParser()
regParser.add_argument('firstname', type=str, help='firstname', location='json', required=True)
regParser.add_argument('lastname', type=str, help='lastname', location='json', required=True)
regParser.add_argument('email', type=str, help='Email', location='json', required=True)
regParser.add_argument('password', type=str, help='Password', location='json', required=True)
regParser.add_argument('confirm_password', type=str, help='Confirm Password', location='json', required=True)

@api.route('/register')
class Registration(Resource):
    @api.expect(regParser)
    def post(self):
        # BEGIN: Get request parameters.
        args        = regParser.parse_args()
        firstname   = args['firstname']
        lastname    = args['lastname']
        email       = args['email']
        password    = args['password']
        password2  = args['confirm_password']
        is_verified = False

        # cek confirm password
        if password != password2:
            return {
                'messege': 'Password tidak cocok'
            }, 400

        #cek email sudah terdaftar
        user = db.session.execute(db.select(Users).filter_by(email=email)).first()
        if user:
            return "Email sudah terpakai silahkan coba lagi menggunakan email lain"
        user          = Users()
        user.firstname    = firstname
        user.lastname     = lastname
        user.email    = email
        user.password = generate_password_hash(password)
        user.is_verified = is_verified
        db.session.add(user)
        msg = Message(subject='Verification OTP',sender=os.environ.get("MAIL_USERNAME"),recipients=[user.email])
        token =  random.randrange(10000,99999)
        session['email'] = user.email
        user.token = str(token)
        print("Isi session email:", session['email'])
        print("Isi session token:", user.token)
        msg.html=render_template(
        'verify_email.html', token=token)
        mail.send(msg)
        db.session.commit()
        return {'message':
            'Registrasi Berhasil. Silahkan cek email untuk verifikasi.'}, 201

otpparser = reqparse.RequestParser()
otpparser.add_argument('otp', type=str, help='otp', location='json', required=True)
otpparser.add_argument('email', type=str, help='email', location='json', required=True)

@api.route('/verifikasi')
class Verify(Resource):
    @api.expect(otpparser)
    def post(self):
        args = otpparser.parse_args()
        otp = args['otp']
        print("Kode OTP:", otp)  # Cetak kode OTP di log
        if args['email'] != '':
            email = args['email']

            user = Users.query.filter_by(email=email).first()
            if user.token == otp :
                user.is_verified = True
                user.token = None
                db.session.commit()
            # session.pop('token',None)
                print('Email berhasil diverifikasi')
                return {'message' : 'Email berhasil diverifikasi'}
        # if 'token' in args:
        #     sesion = args ['token']
        #     print("Token:", sesion)
        #     else:
        #         return {'message' : 'Kode Otp Salah'}
            else:
                return {'message' : 'OTP salah'}



#histori parser
historiParser = reqparse.RequestParser()
historiParser.add_argument('nama', type=str, help='Nama', location='json', required=True)
historiParser.add_argument('nama_gerakan', type=str, help='Nama Gerakan', location='json', required=True)
historiParser.add_argument('tanggal', type=str, help='Tanggal', location='json', required=True)


logParser = reqparse.RequestParser()
logParser.add_argument('email', type=str, help='Email', location='json', required=True)
logParser.add_argument('password', type=str, help='Password', location='json', required=True)

@api.route('/login')
class LogIn(Resource):
    @api.expect(logParser)
    def post(self):
        args        = logParser.parse_args()
        email       = args['email']
        password    = args['password']
        # cek jika kolom email dan password tidak terisi
        print(email)
        print(password)
        if not email or not password:
            return {
                'message': 'Email Dan Password Harus Diisi'
            }, 400
        #cek email sudah ada
        user = db.session.execute(
            db.select(Users).filter_by(email=email)).first()
        if not user:
            return {
                'message': 'Email / Password Salah'
            }, 400
        else:
            user = user[0]
        #cek password
        if check_password_hash(user.password, password):
            if user.is_verified == True:
                token= jwt.encode({
                        "user_id":user.id,
                        "user_email":user.email,
                        "exp": datetime.utcnow() + timedelta(days= 1)
                },app.config['SECRET_KEY'],algorithm="HS256")
                print(f'Token: {token}')
                return {'message' : 'Login Berhasil',
                        'token' : token
                        },200
                
            else:
                return {'message' : 'Email Belum Diverifikasi ,Silahka verifikasikan terlebih dahulu '},401
        else:
            return {
                'message': 'Email / Password Salah'
            }, 400
def decodetoken(jwtToken):
    decode_result = jwt.decode(
               jwtToken,
               app.config['SECRET_KEY'],
               algorithms = ['HS256'],
            )
    return decode_result



editParser = reqparse.RequestParser()
editParser.add_argument('firstname', type=str, help='Firstname', location='json', required=True)
editParser.add_argument('lastname', type=str, help='Lastname', location='json', required=True)
editParser.add_argument('Authorization', type=str, help='Authorization', location='headers', required=True)
@api.route('/edituser')
class EditUser(Resource):
       @api.expect(editParser)
       def put(self):
        args = editParser.parse_args()
        bearerAuth  = args['Authorization']
        firstname = args['firstname']
        lastname = args['lastname']
        datenow =  datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        try:
            jwtToken    = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user = Users.query.filter_by(email=token.get('user_email')).first()
            user.firstname = firstname
            user.lastname = lastname
            user.updatedAt = datenow
            db.session.commit()
        except:
            return {
                'message' : 'Token Tidak valid,Silahkan Login Terlebih Dahulu!'
            }, 401
        return {'message' : 'Update User Sukses'}, 200


verifyParser = reqparse.RequestParser()
verifyParser.add_argument(
    'otp', type=str, help='firstname', location='json', required=True)




#editpasswordParser
editPasswordParser =  reqparse.RequestParser()
editPasswordParser.add_argument('current_password', type=str, help='current_password',location='json', required=True)
editPasswordParser.add_argument('new_password', type=str, help='new_password',location='json', required=True)
@api.route('/editpassword')
class Password(Resource):
    @api.expect(authParser,editPasswordParser)
    def put(self):
        args = editPasswordParser.parse_args()
        argss = authParser.parse_args()
        bearerAuth  = argss['Authorization']
        cu_password = args['current_password']
        newpassword = args['new_password']
        try:
            jwtToken    = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user = Users.query.filter_by(id=token.get('user_id')).first()
            if check_password_hash(user.password, cu_password):
                user.password = generate_password_hash(newpassword)
                db.session.commit()
            else:
                return {'message' : 'Password Lama Salah'},400
        except:
            return {
                'message' : 'Token Tidak valid! Silahkan, Sign in!'
            }, 401
        return {'message' : 'Password Berhasil Diubah'}, 200

@app.route("/realtime")
def hello_world():
    return render_template('index.html')

@app.route("/opencam", methods=['GET'])
def opencam():
    print("here")
    subprocess.run(['python', 'detect.py', '--source', '0', '--weights', 'best.pt'])
    return "done"


if __name__ == '__main__':
    # app.run(ssl_context='adhoc', debug=True)
    app.run(host='0.0.0.0' , debug=True)
