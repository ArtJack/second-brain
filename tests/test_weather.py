"""Live weather formatting stays separate from the cited memory path."""

from secondbrain import weather


def test_format_weather_without_location_asks_for_one(monkeypatch):
    monkeypatch.setattr(weather.cfg, "weather_location", None)

    text = weather.format_weather("")

    assert "Tell me a location" in text


def test_format_weather_summarizes_current_conditions(monkeypatch):
    monkeypatch.setattr(
        weather,
        "fetch_weather",
        lambda _location: {
            "nearest_area": [
                {
                    "areaName": [{"value": "Seattle"}],
                    "region": [{"value": "Washington"}],
                    "country": [{"value": "United States"}],
                }
            ],
            "current_condition": [
                {
                    "temp_F": "62",
                    "FeelsLikeF": "61",
                    "weatherDesc": [{"value": "Partly cloudy"}],
                    "humidity": "70",
                    "windspeedMiles": "8",
                }
            ],
            "weather": [{"maxtempF": "68", "mintempF": "55"}],
        },
    )

    text = weather.format_weather("Seattle")

    assert "Weather for Seattle, Washington, United States:" in text
    assert "62°F" in text
    assert "Partly cloudy" in text
    assert "high 68°F / low 55°F" in text
