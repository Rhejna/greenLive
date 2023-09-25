import os

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import prediction
from werkzeug.utils import secure_filename
import os
import io
import tempfile



app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if(value):
                return {'Post Values': value}, 201

            return {"error":"Invalid format."}

        except Exception as error:
            return {'error': error}

class GetCropRecommandationOutput(Resource):
    def get(self):
        return {"error":"Invalid Method."}

    def post(self):
        try:
            data = request.get_json()
            if not data:
                return {"error": "Invalid format."}, 400  # Return a 400 Bad Request status code for invalid input

            predict = prediction.recommndant_crop(data)
            return f'predict : {predict}'

        except Exception as error:
            return f'error : {str(error)}', 500  # Return a 500 Internal Server Error status code for exceptions


class GetPredictionDisease(Resource):
    def get(self):
        return {"error":"Invalid Method."}

    def post(self):
        try:
            data = request.files.get('imagefile', '')
            if not data:
                return {"error": "Invalid format."}, 400  # Return a 400 Bad Request status code for invalid input

            with tempfile.TemporaryDirectory() as temp_directory:
                file_path = os.path.join(temp_directory, secure_filename(data.filename))
                data.save(file_path)

                predict = prediction.predict_disease(file_path)
                print(predict)

            # predict = prediction.predict_disease(data)
            return f'predict : {predict}'

        except Exception as error:
            return f'error : {str(error)}', 500  # Return a 500 Internal Server Error status code for exceptions


class GetPredictionWeed(Resource):
    def get(self):
        return {"error":"Invalid Method."}

    def post(self):
        try:
            # Get the uploaded file from the request
            uploaded_file = request.files.get('imagefile', '')

            if not uploaded_file:
                return {"error": "Invalid format."}, 400  # Return a 400 Bad Request status code for invalid input

            # # Generate a secure filename and save the uploaded file to a temporary directory
            # temp_directory = '/imgTemp'  # Change this to your desired temporary directory
            # os.makedirs(temp_directory, exist_ok=True)
            # file_path = os.path.join(temp_directory, secure_filename(uploaded_file.filename))
            # uploaded_file.save(file_path)
            #
            # # Now, call the predict_weed function with the file path
            # predict = prediction.predict_weed(file_path)
            # print(predict)

            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_directory:
                file_path = os.path.join(temp_directory, secure_filename(uploaded_file.filename))
                uploaded_file.save(file_path)

                predict = prediction.predict_weed(file_path)
                print(predict)

            # Return the prediction result
            return f'predict : {predict}'

        except Exception as error:
            return f'error : {str(error)}', 500  # Return a 500 Internal Server Error status code for exceptions


class GetCropRecommandationOutput(Resource):
    def get(self):
        return {"error":"Invalid Method."}

    def post(self):
        try:
            data = request.get_json()
            if not data:
                return {"error": "Invalid format."}, 400  # Return a 400 Bad Request status code for invalid input

            predict = prediction.recommndant_crop(data)
            return f'predict : {predict}'

        except Exception as error:
            return f'error : {str(error)}', 500  # Return a 500 Internal Server Error status code for exceptions


api.add_resource(Test,'/')
api.add_resource(GetCropRecommandationOutput,'/getCropRecommandationOutput')
api.add_resource(GetPredictionDisease,'/getPredictionDisease')
api.add_resource(GetPredictionWeed,'/getPredictionWeed')

@app.errorhandler(Exception)
def handle_error(e):
    status_code = 500  # Vous pouvez définir le code d'état approprié ici
    response = {
        "error": "An internal server error occurred.",
        "message": str(e)  # Convertir l'exception en chaîne pour éviter les problèmes de sérialisation
    }
    return jsonify(response), status_code

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 38856))
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)