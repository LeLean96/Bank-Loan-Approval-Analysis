from flask import Flask, jsonify, request, send_from_directory
from flask_cors import cross_origin
from flasgger import Swagger
import pickle

app = Flask(__name__)

swagger = Swagger(app)

@app.route('/predict_loan_status/', methods=['POST'])
@cross_origin()  # CORS is necessary to support queries from other websites (e.g. AWP assignment).
def predict_loan_status():
    """Endpoint to predict Loan_Status for the given parameters
    ---
    parameters:
      - name: user_data
        in: body
        schema:
          $ref: '#/definitions/UserData'
        required: true
    definitions:
      LoanStatusResult:
        type: object
        properties:
          loan_status:
            type: boolean
      UserData:
        type: object
        properties:
          gender:
            type: string
            enum: ['Male', 'Female']
          married:
            type: string
            enum: ['No', 'Yes']
          dependents:
            type: integer
          education:
            type: string
            enum: ['Not Graduate', 'Graduate']
          self_employed:
            type: string
            enum: ['No', 'Yes']
          applicant_income:
            type: number
          coapplicant_income:
            type: number
          loan_amount:
            type: number
          loan_amount_term:
            type: number
          credit_history:
            type: number
          model:
            type: string
            enum: ['bayes', 'decision_trees', 'knn', 'logistic_regression', 'svm']
    responses:
      200:
        description: The predicted loan status
        schema:
          $ref: '#/definitions/LoanStatusResult'
        examples:
          loan_status: true
    """
    label_mapping = {
        'Male': 0,
        'Female': 1,
        'No': 0,
        'Yes': 1,
        'Not Graduate': 0,
        'Graduate': 1,
    }
    data = {key: label_mapping.get(val, val) for key, val in request.json.items()}
    result = predict_result([data['gender'], data['married'], data['dependents'], data['education'], data['self_employed'], data['applicant_income'], data['coapplicant_income'], data['loan_amount'], data['loan_amount_term'], data['credit_history']], data['model'])
    return jsonify(dict(loan_status=result))

@app.route("/")
def notebook():
    return send_from_directory('.', 'notebook.html')

@app.route("/live_prediction/")
def live_prediction():
    return send_from_directory('.', 'live_prediction.html')



def predict_result(inputvector, modelname):
    with open(f"../models/{modelname}", 'rb') as f:
        model = pickle.load(f)
    return bool(model.predict([inputvector])[0])