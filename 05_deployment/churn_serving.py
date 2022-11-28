import pickle
import numpy as np

from flask import Flask, request, jsonify

def predict_single(d, dv, model):
    X = dv.transform(d)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

## OPEN THE DV and MODEL binaries and save to variable
with open('dv.bin', 'rb') as f_in: 
    dict_vectorizer = pickle.load(f_in)
f_in.close()

with open('model2.bin', 'rb') as f_in:
    model = pickle.load(f_in)
f_in.close()

## FLASK APP
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dict_vectorizer, model)
    churn = prediction >= 0.5
    
    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)