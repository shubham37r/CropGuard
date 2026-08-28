from typing import Dict, Any, List
from app.config import settings

class RiskEngineService:
    """
    Dedicated Prototype Risk Engine.
    Calculates a PROTOTYPE CONTEXTUAL RISK SCORE (0-100) based on weighted contextual inputs.
    NOT a scientifically validated disease probability prediction.
    """

    STAGE_RISK_FACTORS = {
        "Flowering": 85.0,     # Vulnerable stage
        "Fruiting": 80.0,      # High economic impact
        "Vegetative": 60.0,
        "Seedling": 50.0,
        "Maturity": 40.0
    }

    def calculate_risk(
        self,
        disease_confidence: float,  # 0 - 100 %
        weather_risk: float,        # 0 - 100
        crop_stage: str,
        outbreak_signal: float = 50.0, # 0 - 100 (from local spatial cluster count)
        condition_name: str = "Suspected Condition"
    ) -> Dict[str, Any]:

        # 1. Normalize stage risk
        stage_risk = self.STAGE_RISK_FACTORS.get(crop_stage, 50.0)

        # 2. Weighted Formula
        w_conf = settings.WEIGHT_DISEASE_CONFIDENCE
        w_weather = settings.WEIGHT_WEATHER_RISK
        w_stage = settings.WEIGHT_STAGE_RISK
        w_outbreak = settings.WEIGHT_OUTBREAK_SIGNAL

        score = (
            (w_conf * disease_confidence) +
            (w_weather * weather_risk) +
            (w_stage * stage_risk) +
            (w_outbreak * outbreak_signal)
        )
        score = round(min(max(score, 0.0), 100.0), 1)

        # 3. Risk Level Mapping
        if score < 40.0:
            risk_level = "LOW"
        elif score < 70.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # 4. Contributing Factors Generation
        factors = []
        if disease_confidence >= 80.0:
            factors.append(f"Visual symptoms strongly consistent with {condition_name} ({disease_confidence:.0f}% confidence)")
        elif disease_confidence < 70.0:
            factors.append(f"Visual symptom match is moderate/uncertain ({disease_confidence:.0f}% confidence)")
        else:
            factors.append(f"Visual symptoms consistent with {condition_name} ({disease_confidence:.0f}% confidence)")

        if weather_risk >= 70.0:
            factors.append(f"Environmental conditions (humidity/temp) highly favorable for crop pathogen growth")
        elif weather_risk >= 40.0:
            factors.append(f"Environmental conditions moderately support potential symptom spread")

        if stage_risk >= 80.0:
            factors.append(f"Crop is in vulnerable growth stage ({crop_stage}), increasing risk impact")
        else:
            factors.append(f"Crop growth stage ({crop_stage}) evaluated for susceptibility")

        if outbreak_signal >= 70.0:
            factors.append(f"Elevated local report activity detected in nearby agricultural radius")
        elif outbreak_signal >= 40.0:
            factors.append(f"Moderate local report presence noted in district")

        return {
            "score": score,
            "risk_level": risk_level,
            "component_scores": {
                "disease_confidence": disease_confidence,
                "weather_risk": weather_risk,
                "stage_risk": stage_risk,
                "outbreak_signal": outbreak_signal
            },
            "contributing_factors": factors,
            "methodology_note": settings.METHODOLOGY_NOTE
        }

risk_engine = RiskEngineService()
