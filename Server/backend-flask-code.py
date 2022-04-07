
from flask import Flask, jsonify, request, redirect, url_for, session
# import flask.scaffold
# flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restful import Resource, Api
import time
import os
from werkzeug.utils import secure_filename



app = Flask(__name__)
api = Api(app)

class UploadFile(Resource):
    def post(self):
        try:
            uploadedFile = request.files['file']
            file_path = os.path.join(os.getcwd(),'Images')
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            with open(os.path.join(file_path, secure_filename(uploadedFile.filename)), "wb") as f:
                f.write(uploadedFile.read())

        except Exception as e:
            print(str(e))

        else:
            return jsonify ({'status':'done'})

class DetectImage(Resource):
    def post(self):
        try:
            image_path = 'image.jpg'
            obj =DetectionApi()
            res = obj.detectImage(image_path)

        except Exception as e :
            print(str(e))

api.add_resource(UploadFile,'/uploadFile')





if __name__ == "__main__":
    app.run(debug=False, port=5000)