import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "clothing_model.pkl"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

CLOTHING_LABELS: Dict[int, str] = {
    0: "A light layer with a jacket is a good choice for mild weather.",
    1: "Wear breathable fabrics, shorts, and sunglasses for hot weather.",
    2: "Choose a warm coat, scarf, and layers for cold weather.",
    3: "Carry a waterproof jacket and wear comfortable shoes for rainy weather.",
    4: "A comfortable outfit with a windbreaker is ideal for windy weather.",
}

COLOR_LABELS: Dict[int, str] = {
    0: "Light colors like white, beige, or pastel blue.",
    1: "Bright summer shades like sky blue, mint green, or peach.",
    2: "Warm/deep tones like maroon, navy, charcoal, or olive.",
    3: "Darker practical colors like navy, black, or deep gray.",
    4: "Neutral sporty colors like gray, olive, and denim blue.",
}

WEATHER_CODE_LABELS: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Heavy rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}

CONDITION_MAP: Dict[int, str] = {
    0: "clear",
    1: "clear",
    2: "cloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "rain",
    53: "rain",
    55: "rain",
    61: "rain",
    63: "rain",
    65: "rain",
    71: "snow",
    73: "snow",
    75: "snow",
    80: "rain",
    81: "rain",
    82: "rain",
    95: "storm",
    96: "storm",
    99: "storm",
}


def load_model() -> Optional[object]:
    if not MODEL_FILE.exists() or MODEL_FILE.stat().st_size == 0:
        return None

    try:
        with MODEL_FILE.open("rb") as model_file:
            return pickle.load(model_file)
    except (EOFError, pickle.UnpicklingError, AttributeError, ImportError, IndexError):
        return None


def geocode_city(city: str) -> Tuple[float, float]:
    response = requests.get(
        OPEN_METEO_GEOCODE_URL,
        params={"name": city, "count": 1},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"Could not find coordinates for '{city}'. Try a more specific name.")

    first = results[0]
    return float(first["latitude"]), float(first["longitude"])


def _hourly_row_index(hourly_times: List[str], current_time: str) -> int:
    if current_time in hourly_times:
        return hourly_times.index(current_time)
    current_dt = datetime.fromisoformat(current_time.replace("Z", "+00:00"))
    best_i = 0
    best_sec = float("inf")
    for i, t in enumerate(hourly_times):
        try:
            other = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            continue
        sec = abs((other - current_dt).total_seconds())
        if sec < best_sec:
            best_sec = sec
            best_i = i
    return best_i


def get_weather_for_city(city: str) -> Dict[str, float]:
    latitude, longitude = geocode_city(city)
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
            "hourly": "relativehumidity_2m,pressure_msl",
            "timezone": "auto",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    current = data.get("current_weather")
    hourly = data.get("hourly", {})
    if current is None or not hourly:
        raise RuntimeError("Unable to fetch weather details from the weather provider.")

    time_index = _hourly_row_index(hourly["time"], current["time"])
    humidity = float(hourly["relativehumidity_2m"][time_index])
    pressure = float(hourly["pressure_msl"][time_index])

    weather_code = int(current.get("weathercode", 0))
    description = WEATHER_CODE_LABELS.get(weather_code, "Unknown weather")
    condition = CONDITION_MAP.get(weather_code, "mild")

    return {
        "temp": float(current.get("temperature", 0.0)),
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": float(current.get("windspeed", 0.0)),
        "description": description,
        "condition": condition,
    }


def weather_features(weather: Dict[str, float]) -> List[float]:
    return [
        float(weather.get("temp", 0.0)),
        float(weather.get("humidity", 0.0)),
        float(weather.get("wind_speed", 0.0)),
        float(weather.get("pressure", 0.0)),
        float(len(weather.get("description", ""))),
    ]


def predict_clothing_suggestion(weather: Dict[str, float]) -> str:
    # Keep backward compatibility for older frontpage code paths.
    # Always return a useful suggestion instead of exposing model-load errors to users.
    style = predict_style_recommendation(weather)
    return str(style.get("clothing", "Use a light layer and adjust for the current weather."))


def _rule_based_style(weather: Dict[str, float]) -> Dict[str, str]:
    temp = float(weather.get("temp", 0.0))
    condition = str(weather.get("condition", "mild"))
    wind = float(weather.get("wind_speed", 0.0))

    if condition in {"rain", "storm"}:
        clothing = CLOTHING_LABELS[3]
        color = COLOR_LABELS[3]
    elif temp >= 30:
        clothing = CLOTHING_LABELS[1]
        color = COLOR_LABELS[1]
    elif temp <= 14:
        clothing = CLOTHING_LABELS[2]
        color = COLOR_LABELS[2]
    elif wind >= 9:
        clothing = CLOTHING_LABELS[4]
        color = COLOR_LABELS[4]
    else:
        clothing = CLOTHING_LABELS[0]
        color = COLOR_LABELS[0]

    return {"clothing": clothing, "color": color}


def _make_fun_style_text(weather: Dict[str, float], clothing: str, color: str) -> Dict[str, str]:
    condition = str(weather.get("condition", "mild"))
    temp = float(weather.get("temp", 0.0))
    wind = float(weather.get("wind_speed", 0.0))
    humidity = float(weather.get("humidity", 0.0))

    # More distinct palettes by weather + comfort factors.
    if condition in {"rain", "storm"}:
        core_palette = "navy, charcoal, deep olive, and waterproof black"
        accent = "mustard yellow or electric teal"
    elif temp >= 34:
        core_palette = "icy white, powder blue, mint, and soft peach"
        accent = "lemon or coral"
    elif temp <= 10:
        core_palette = "burgundy, forest green, coffee brown, and deep navy"
        accent = "burnt orange"
    elif humidity >= 80:
        core_palette = "cool grays, airy blue, and stone beige"
        accent = "turquoise"
    elif wind >= 9:
        core_palette = "slate gray, denim blue, olive, and graphite"
        accent = "rust orange"
    elif condition in {"fog", "cloudy"}:
        core_palette = "lavender gray, dusty blue, taupe, and muted lilac"
        accent = "rose pink"
    else:
        core_palette = "cream, pastel blue, light sage, and warm beige"
        accent = "soft marigold"

    if condition in {"rain", "storm"}:
        clothing_text = (
            "Rain mode ON! Grab an umbrella like a movie hero, wear a waterproof jacket, "
            "and choose quick-dry comfy bottoms. Puddle-proof shoes are your best friend."
        )
        color_text = f"Monsoon-ready palette: {core_palette}. Add one pop color like {accent}."
    elif temp <= 16:
        clothing_text = (
            "Brrr... sweater time! Layer up with a cozy sweater, jacket, and full sleeves. "
            "Keep a scarf ready so the cold does not ruin your vibe."
        )
        color_text = f"Winter-friendly shades: {core_palette}. Accent pick: {accent}."
    elif temp >= 32:
        clothing_text = (
            "It is hot outside! Keep it airy and chill with cotton/linen, relaxed fits, "
            "and breathable footwear. Sunglasses on, heat stress gone."
        )
        color_text = f"Heat-wave palette: {core_palette}. Fresh highlight color: {accent}."
    elif wind >= 9:
        clothing_text = (
            "Windy vibes today! Put on a light windbreaker over a tee and go for secure, "
            "comfortable shoes. Style + stability = perfect combo."
        )
        color_text = f"Wind-friendly style tones: {core_palette}. Add a sharp accent like {accent}."
    else:
        clothing_text = (
            "Perfect day to dress smart-casual: light layers, comfy fit, and a clean look "
            "that works all day."
        )
        color_text = f"Balanced everyday palette: {core_palette}. Optional accent: {accent}."

    # Keep some personalization from model outputs while making copy more user-friendly.
    return {
        "clothing": f"{clothing_text} Pro tip: {clothing}",
        "color": f"{color_text} Style hint: {color}",
    }


def weather_emoji(weather: Dict[str, float]) -> str:
    condition = str(weather.get("condition", "mild"))
    temp = float(weather.get("temp", 0.0))
    if condition in {"storm"}:
        return "⛈️"
    if condition in {"rain"}:
        return "🌧️"
    if condition in {"snow"}:
        return "❄️"
    if condition in {"fog"}:
        return "🌫️"
    if temp >= 34:
        return "🔥"
    if temp <= 12:
        return "🧊"
    if condition in {"cloudy"}:
        return "☁️"
    return "☀️"


def style_emojis(weather: Dict[str, float]) -> Dict[str, str]:
    w = weather_emoji(weather)
    condition = str(weather.get("condition", "mild"))
    if condition in {"rain", "storm"}:
        return {"weather": w, "clothing": "🧥☔", "color": "🎨🌈"}
    if condition in {"snow"} or float(weather.get("temp", 0.0)) <= 16:
        return {"weather": w, "clothing": "🧣🧥", "color": "🎨✨"}
    if float(weather.get("temp", 0.0)) >= 32:
        return {"weather": w, "clothing": "🕶️👕", "color": "🎨🌞"}
    return {"weather": w, "clothing": "👔👌", "color": "🎨💫"}


def predict_style_recommendation(weather: Dict[str, float]) -> Dict[str, Any]:
    model = load_model()
    features = weather_features(weather)

    if model is None:
        fallback = _rule_based_style(weather)
        fun = _make_fun_style_text(weather, fallback["clothing"], fallback["color"])
        fallback["clothing"] = fun["clothing"]
        fallback["color"] = fun["color"]
        fallback["source"] = "rule_based"
        return fallback

    try:
        # New model format: dict with clothing_model/color_model + optional encoders.
        if isinstance(model, dict) and "clothing_model" in model and "color_model" in model:
            clothing_raw = model["clothing_model"].predict([features])[0]
            color_raw = model["color_model"].predict([features])[0]

            clothing_label = (
                model["clothing_encoder"].inverse_transform([clothing_raw])[0]
                if "clothing_encoder" in model
                else str(clothing_raw)
            )
            color_label = (
                model["color_encoder"].inverse_transform([color_raw])[0]
                if "color_encoder" in model
                else str(color_raw)
            )
            fun = _make_fun_style_text(weather, str(clothing_label), str(color_label))
            return {"clothing": fun["clothing"], "color": fun["color"], "source": "aiml_model"}

        # Backward compatibility with the old single-label model.
        legacy_label = model.predict([features])[0]
        legacy_clothing = CLOTHING_LABELS.get(
            legacy_label, "Use a light layer and adjust for the current weather."
        )
        legacy_color = _rule_based_style(weather)["color"]
        legacy_fun = _make_fun_style_text(weather, legacy_clothing, legacy_color)
        return {"clothing": legacy_fun["clothing"], "color": legacy_fun["color"], "source": "legacy_model"}
    except Exception:
        fallback = _rule_based_style(weather)
        fun = _make_fun_style_text(weather, fallback["clothing"], fallback["color"])
        fallback["clothing"] = fun["clothing"]
        fallback["color"] = fun["color"]
        fallback["source"] = "rule_based"
        return fallback
