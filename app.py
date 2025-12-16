from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import shap
import os

app = Flask(__name__)
CORS(app)

# Load Models
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # Try to load explainer, but handle failure gracefully as SHAP can be version-sensitive
    try:
        explainer = shap.TreeExplainer(model)
    except:
        explainer = None
    model_error = None
except Exception as e:
    model = None
    scaler = None
    explainer = None
    model_error = str(e)

cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def get_report(i):
    checks = [
        (i[3]>140, f"High BP ({i[3]})", f"Normal BP ({i[3]})"),
        (i[4]>240, f"High Chol ({i[4]})", f"Chol Normal ({i[4]})") if i[4]>200 else (False, "", ""),
        (i[5]==1, "High FBS (>120)", "Normal FBS"),
        (i[8]==1, "Exercise Angina Detected", "No Exercise Angina"),
        (i[9]>1.0, f"ST Depression ({i[9]})", "Normal ST"),
        (i[11]>0, f"{int(i[11])} Colored Vessels", "No Colored Vessels")
    ]
    return [m for c,m,_ in checks if c], [m for c,_,m in checks if not c and m]

@app.route('/health')
def health():
    return jsonify({"status": "OK" if model else "Model Error", "error": model_error})

@app.route('/predict', methods=['POST'])
def predict():
    if not model: return jsonify({"error": f"Model missing: {model_error}"}), 500
    try:
        f = request.json.get('features')
        if not f or len(f)!=13: return jsonify({"error": "Bad input"}), 400
        x = scaler.transform(pd.DataFrame([f], columns=cols))
        p, prob = int(model.predict(x)[0]), model.predict_proba(x)[0]
        
        shap_vals = []
        if explainer:
            try:
                sv = explainer.shap_values(x)
                if isinstance(sv, list): sv = sv[1][0] if len(sv)>1 else sv[0]
                elif len(sv.shape)==3: sv = sv[0,:,1]
                else: sv = sv[0]
                shap_vals = sorted([{"feature":c, "value":float(v)} for c,v in zip(cols, sv)], key=lambda x:abs(x['value']), reverse=True)
            except: pass

        risk, normal = get_report(f)
        return jsonify({
            "is_heart_disease_detected": p==1,
            "certainty_percentage": round((prob[1] if p==1 else prob[0])*100, 2),
            "detailed_report": risk, "normal_findings": normal, "shap_values": shap_vals
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

import os
if __name__ == '__main__': 
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
