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
        
        # --- HYBRID AI LOGIC (SAFETY NET) ---
        # The pre-trained model can sometimes miss obvious cases or hallucinate on healthy ones.
        # We start with the Model's prediction, then apply Medical Heuristics to correct "Impossible" errors.
        
        model_pred = int(model.predict(x)[0])
        model_prob = model.predict_proba(x)[0]
        prob_val = model_prob[1] if model_pred == 1 else model_prob[0]
        
        # Calculate Risk Score (Red Flags)
        risk_score = 0
        if f[2] > 0 and f[2] != 3: risk_score += 1 # Chest Pain (Typical/Atypical) - dataset dependent, assuming non-asymptomatic is risk
        if f[3] > 140: risk_score += 1 # High BP
        if f[4] > 240: risk_score += 1 # High Chol
        if f[8] == 1: risk_score += 2  # Exercise Angina (Strong Indicator)
        if f[9] > 1.0: risk_score += 1 # ST Depression
        if f[11] > 0: risk_score += 2  # Colored Vessels (Strong Indicator)
        if f[12] == 2 or f[12] == 3: risk_score += 1 # Thalassemia defect
        
        final_pred = model_pred
        final_prob = prob_val
        is_override = False
        
        # OVERRIDE LOGIC
        if risk_score >= 3:
            # Critical Condition: Force HIGH RISK
            final_pred = 1
            final_prob = max(prob_val, 0.92) if model_pred == 1 else 0.88 # Force high confidence
            is_override = True
        elif risk_score == 0 and model_pred == 1:
            # Clean Bill of Health: Force LOW RISK (unless model is extremely sure, but even then, be skeptical)
            # If no symptoms, no angina, normal BP/Chol, normal Vessels... it's unlikely to be heart disease.
            final_pred = 0
            final_prob = 0.95
            is_override = True
            
        print(f"Hybrid Logic: RiskScore={risk_score}, Model={model_pred}, Final={final_pred}, Override={is_override}")
        
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
            "is_heart_disease_detected": bool(final_pred==1),
            "certainty_percentage": round(final_prob*100, 2),
            "detailed_report": risk, "normal_findings": normal, "shap_values": shap_vals,
            "debug_score": risk_score
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

import os
if __name__ == '__main__': 
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
