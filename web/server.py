import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, jsonify, send_from_directory
from src.predictor import GPUGatePredictor
from src.explainer import group_shap_values, generate_explanation, feature_value_description
from src.crossover import generate_profitability_curve
from src.diagnostics import format_diagnostic

app = Flask(__name__, static_folder='.', static_url_path='')

predictor_instance = GPUGatePredictor()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/style.css')
def style():
    return app.send_static_file('style.css')

@app.route('/app.js')
def script():
    return app.send_static_file('app.js')

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_path": "models/gpugate_model.pkl",
        "loaded": predictor_instance.model is not None,
        "feature_count": len(predictor_instance.model.feature_names_in_) if hasattr(predictor_instance.model, 'feature_names_in_') else 17
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'code' not in data:
        return jsonify({"error": "Missing code field"}), 400
    
    if predictor_instance.model is None or not os.path.exists("models/gpugate_model.pkl"):
        return jsonify({"error": "Service Unavailable: ML model is not loaded. Please train or provide a model file."}), 503
    
    code_string = data['code']
    results, warnings = predictor_instance.predict(code_string)
    
    response_loops = []
    for r in results:
        # 1. Extracted loop properties
        if r.feature_vector and r.feature_vector.loop_info:
            loop_type = r.feature_vector.loop_info.loop_type
            source_snippet = r.feature_vector.loop_info.source_text
        else:
            loop_type = "unknown"
            source_snippet = ""

        # 2. Explanations & Groupings
        grouped_shap = group_shap_values(r.shap_values)
        explanation = generate_explanation(
            r.verdict, r.confidence, r.features, r.shap_values,
            r.dependency, r.confidence_limited, r.confidence_reason
        )

        # 3. Feature mapping
        feature_descriptions = {
            k: feature_value_description(k, v) 
            for k, v in r.features.items()
        }

        # 4. Profitability curve
        prof_curve = []
        if r.verdict != "BLOCKED":
            prof_curve = generate_profitability_curve(
                predictor_instance.model, 
                r.features, 
                nest_depth=int(r.features.get('nest_depth', 0)), 
                n_arrays=int(r.features.get('unique_arrays', 1)), 
                element_size=r.feature_vector.element_size if r.feature_vector else 8
            )

        # 5. Diagnostic output
        diagnostic = format_diagnostic(r, filename="<web_input>")

        # Secure serialization of fixes/blockers
        fixes = []
        for fix in r.fix_suggestions:
            if isinstance(fix, dict):
                fixes.append({
                    "problem": fix.get("problem", ""),
                    "fix": fix.get("fix", ""),
                    "expected_improvement": fix.get("expected_improvement", ""),
                    "code_example": fix.get("code_example", "")
                })
        
        blockers = []
        for b in r.blockers:
            if isinstance(b, dict):
                blockers.append({
                    "reason": b.get("reason", str(b)),
                    "line": b.get("line", r.loop_line),
                    "text": b.get("text", "")
                })

        loop_data = {
            "verdict": r.verdict,
            "confidence": r.confidence,
            "features": r.features,
            "feature_descriptions": feature_descriptions,
            "grouped_shap": grouped_shap,
            "explanation": explanation,
            "crossover_n": r.crossover_n,
            "roofline_position": r.roofline_position,
            "fix_suggestions": fixes,
            "blockers": blockers,
            "loop_line": r.loop_line,
            "loop_type": loop_type,
            "source_snippet": source_snippet,
            "diagnostic": diagnostic,
            "profitability_curve": prof_curve
        }
        response_loops.append(loop_data)
        
    return jsonify({
        "loops": response_loops,
        "warnings": warnings
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
