"""
src/predictor.py — §3.7

Purpose: End-to-end prediction pipeline. Parse → Extract → Predict → Explain.
"""
import os
from typing import List
from dataclasses import dataclass, field

from src import code_parser
from src import feature_extractor
from src import model as model_utils
from src import explainer
from src import crossover
from src import roofline
from src import fixits


@dataclass
class PredictionResult:
    verdict: str
    confidence: float
    confidence_limited: bool
    confidence_reason: str
    features: dict
    shap_values: dict
    crossover_n: int | None
    roofline_position: str
    blockers: list[dict]
    fix_suggestions: list[dict]
    loop_line: int
    source_code: str
    feature_vector: feature_extractor.FeatureVector | None = None
    dependency: feature_extractor.DependencyInfo | None = None
    parse_errors: list[str] | None = None


class GPUGatePredictor:
    def __init__(self, model_path: str = "models/gpugate_model.pkl"):
        try:
            self.model = model_utils.load_model(model_path)
        except Exception:
            self.model = None

    def predict(self, code: str, preprocess: bool = False) -> tuple[List[PredictionResult], List[str]]:
        parse_result = code_parser.parse(code, preprocess=preprocess)
        results = []
        
        # Analyze the candidate loops returned by the parser
        loops_to_analyze = parse_result.loops

        for loop_info in loops_to_analyze:
            fv = feature_extractor.extract_features(loop_info, parse_result.source)
            features_dict = fv.features
            
            verdict = "UNCERTAIN"
            prob = 0.5
            
            if fv.blockers:
                verdict = "BLOCKED"
                prob = 0.0
            elif fv.dependency and fv.dependency.category == "DEPENDENCY":
                verdict = "UNPROFITABLE"
                prob = 0.05
            else:
                features_list = [features_dict[name] for name in feature_extractor.FEATURE_NAMES]
                import numpy as np
                arr = np.array([features_list], dtype=float)
                prob = float(self.model.predict_proba(arr)[0][1])
                
                if fv.confidence_limited:
                    prob = min(prob, 0.70)
                    if prob < 0.5:
                        prob = max(prob, 0.30)
                if (fv.dependency and fv.dependency.category == "UNCERTAIN") or "early exit" in fv.confidence_reason:
                    prob = min(prob, 0.65)
                    if prob < 0.5:
                        prob = max(prob, 0.35)
                        
                verdict = "PROFITABLE" if prob > 0.5 else "UNPROFITABLE"
            
            confidence = prob if verdict == "PROFITABLE" else (1 - prob)
            if verdict == "BLOCKED":
                confidence = 1.0
                
            shap_values = model_utils.get_shap_for_single(self.model, features_dict)
            grouped_shap = explainer.group_shap_values(shap_values)
            
            # Compute crossover N
            n_cross = crossover.find_breakeven(
                self.model, features_dict, 
                nest_depth=int(features_dict['nest_depth']),
                n_arrays=int(features_dict['unique_arrays']),
                element_size=fv.element_size
            )
            
            # Compute roofline
            roof_pos = roofline.compute_position(features_dict, element_size=fv.element_size)
            
            # Fixits
            fixes = []
            if verdict != "PROFITABLE" and verdict != "BLOCKED":
                fixes = fixits.suggest_fixes(features_dict, fv.dependency)
                
            res = PredictionResult(
                verdict=verdict,
                confidence=confidence,
                confidence_limited=fv.confidence_limited,
                confidence_reason=fv.confidence_reason,
                features=features_dict,
                shap_values=shap_values,
                crossover_n=n_cross,
                roofline_position=roof_pos,
                blockers=fv.blockers,
                fix_suggestions=fixes,
                loop_line=loop_info.line,
                source_code=code,
                feature_vector=fv,
                dependency=fv.dependency,
                parse_errors=parse_result.errors
            )
            results.append(res)
            
        return results, parse_result.errors

    def predict_file(self, filepath: str) -> List[PredictionResult]:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        return self.predict(code)
