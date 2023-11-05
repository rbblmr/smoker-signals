import pickle
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify


model_file = './xgb_smoking_model.pkl'

model = pickle.load(open(model_file,'rb'))

app = Flask('smoking')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    X = pd.DataFrame([patient])
    y_pred = model.predict_proba(X)[0,1]
    smoker = y_pred >= 0.5
    
    result = {
        'smoker_probability': float(y_pred),
        'smoker?': bool(smoker)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)