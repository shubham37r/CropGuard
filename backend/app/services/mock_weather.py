from typing import Dict, Any

class BaseWeatherService:
    def get_environmental_risk(self, latitude: float, longitude: float, crop: str) -> Dict[str, Any]:
        raise NotImplementedError

class MockWeatherService(BaseWeatherService):
    """
    Mock Weather & Environmental Conditions Service.
    Calculates environmental suitability score (0-100) favoring fungal/bacterial pests based on humidity/temp.
    """

    def get_environmental_risk(self, latitude: float, longitude: float, crop: str) -> Dict[str, Any]:
        # Realistic mock environmental parameters for Nagpur region (monsoon / humid microclimate)
        humidity = 82.0  # High relative humidity %
        temperature = 28.5  # Warm degrees C
        recent_rainfall_mm = 45.0  # mm in last 48 hrs

        # High humidity + moderate temp = high fungal/pest risk (78/100)
        risk_score = 78.0 if crop in ["Tomato", "Soybean"] else 65.0

        return {
            "humidity_percent": humidity,
            "temperature_celsius": temperature,
            "recent_rainfall_mm": recent_rainfall_mm,
            "environmental_risk_score": risk_score,
            "summary": f"High humidity ({humidity}%) and warm temperature ({temperature}°C) in Nagpur region create favorable conditions for disease spread."
        }

mock_weather = MockWeatherService()
