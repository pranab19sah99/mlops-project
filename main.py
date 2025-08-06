from flask import Flask, request, jsonify
import pickle
from app.logger import log_request

app = Flask(__name__)
model = pickle.load(open('app/model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.get_json()
    input_data = list(input_json.values())
    prediction = model.predict([input_data]).tolist()
    log_request(input_data, prediction)
    return jsonify({'prediction': prediction})

@app.route('/health')
def health():
    return "OK", 200

@app.route('/metrics')
def metrics():
    return jsonify({'requests_handled': 10})  # Stub metric
