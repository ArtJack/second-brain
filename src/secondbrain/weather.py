"""Live weather lookup for utility questions in the Artjeck agent."""
from __future__ import annotations

import json
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import cfg


def _first_text(value) -> str:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("value", "")).strip()
    return ""


def fetch_weather(location: str, timeout: float = 8.0) -> dict:
    url = f"https://wttr.in/{quote(location)}?format=j1"
    request = Request(url, headers={"User-Agent": "second-brain-artjeck/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def format_weather(location: str | None = None) -> str:
    target = (location or cfg.weather_location or "").strip()
    if not target:
        return "Tell me a location, like `weather in Seattle`, or set SB_WEATHER_LOCATION for a default."

    try:
        data = fetch_weather(target)
    except Exception as exc:
        return f"Weather lookup failed for {target}: {exc}"

    current = (data.get("current_condition") or [{}])[0]
    area = (data.get("nearest_area") or [{}])[0]
    today = (data.get("weather") or [{}])[0]
    name = _first_text(area.get("areaName")) or target
    region = _first_text(area.get("region"))
    country = _first_text(area.get("country"))
    place = ", ".join(part for part in (name, region, country) if part)

    desc = _first_text(current.get("weatherDesc")) or "conditions unavailable"
    temp_f = current.get("temp_F", "?")
    feels_f = current.get("FeelsLikeF", "?")
    humidity = current.get("humidity", "?")
    wind = current.get("windspeedMiles", "?")
    high_f = today.get("maxtempF")
    low_f = today.get("mintempF")

    lines = [
        f"Weather for {place}:",
        f"  now       : {temp_f}°F, feels like {feels_f}°F, {desc}",
        f"  humidity  : {humidity}%",
        f"  wind      : {wind} mph",
    ]
    if high_f and low_f:
        lines.append(f"  today     : high {high_f}°F / low {low_f}°F")
    return "\n".join(lines)
