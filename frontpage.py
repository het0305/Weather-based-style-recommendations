from flask import Flask, jsonify, render_template_string, request
from city_db import init_city_db, search_indian_cities
from weather_predictor import get_weather_for_city, predict_style_recommendation, style_emojis

app = Flask(__name__)
init_city_db()

PAGE_HTML = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Weather Style Coach</title>
  <style>
    :root {
      --bg-1: #f6f9ff;
      --bg-2: #eef2ff;
      --card-bg: rgba(255, 255, 255, 0.82);
      --card-border: rgba(255, 255, 255, 0.72);
      --text-main: #0f172a;
      --text-subtle: #475569;
      --accent-1: #2563eb;
      --accent-2: #7c3aed;
      --accent-soft: rgba(59, 130, 246, 0.15);
      --shadow-main: 0 24px 50px rgba(15, 23, 42, 0.16);
      --shadow-soft: 0 8px 24px rgba(37, 99, 235, 0.14);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text-main);
      background:
        radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.14), transparent 34%),
        radial-gradient(circle at 88% 14%, rgba(124, 58, 237, 0.13), transparent 30%),
        linear-gradient(145deg, var(--bg-1), var(--bg-2));
      min-height: 100vh;
      position: relative;
      overflow-x: hidden;
    }

    body::before,
    body::after {
      content: "";
      position: fixed;
      inset: auto;
      z-index: 0;
      pointer-events: none;
      opacity: 0.55;
      filter: blur(2px);
    }

    body::before {
      top: -120px;
      right: -100px;
      width: 440px;
      height: 440px;
      border-radius: 50%;
      background:
        radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.55), rgba(56, 189, 248, 0) 65%),
        radial-gradient(circle at 70% 65%, rgba(124, 58, 237, 0.45), rgba(124, 58, 237, 0) 72%);
    }

    body::after {
      left: -120px;
      bottom: -130px;
      width: 500px;
      height: 500px;
      border-radius: 50%;
      background:
        radial-gradient(circle at 40% 40%, rgba(59, 130, 246, 0.45), rgba(59, 130, 246, 0) 65%),
        radial-gradient(circle at 60% 70%, rgba(16, 185, 129, 0.3), rgba(16, 185, 129, 0) 70%);
    }

    .page {
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 28px;
      position: relative;
      z-index: 1;
    }

    .card {
      width: 100%;
      max-width: 680px;
      padding: 30px;
      border-radius: 24px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      box-shadow: var(--shadow-main);
      backdrop-filter: blur(8px);
      position: relative;
      overflow: visible;
    }

    .card::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.06), rgba(124, 58, 237, 0.05));
    }

    .card > * { position: relative; z-index: 1; }

    .card h1 {
      margin: 0;
      font-size: 42px;
      line-height: 1.08;
      letter-spacing: -0.02em;
      font-weight: 800;
      background: linear-gradient(92deg, #1e3a8a, #7c3aed 60%, #0ea5e9);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .card h1::after {
      content: "";
      display: block;
      width: 78px;
      height: 5px;
      margin-top: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, #2563eb, #7c3aed);
      box-shadow: 0 8px 18px rgba(124, 58, 237, 0.22);
    }

    .card p {
      margin: 14px 0 22px;
      line-height: 1.7;
      color: var(--text-subtle);
      font-size: 18px;
    }

    form {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: flex-start;
    }

    .input-wrap { flex: 1 1 260px; position: relative; min-width: 0; }

    input {
      width: 100%;
      padding: 14px 18px;
      border: 1px solid #c7d2fe;
      border-radius: 999px;
      font-size: 17px;
      color: var(--text-main);
      background: #ffffffd9;
      outline: none;
      transition: all 0.2s ease;
      box-shadow: inset 0 1px 1px rgba(15, 23, 42, 0.04);
    }

    input:focus {
      border-color: #60a5fa;
      box-shadow: 0 0 0 4px var(--accent-soft), inset 0 1px 1px rgba(15, 23, 42, 0.04);
      transform: translateY(-1px);
    }

    button {
      padding: 13px 22px;
      border: none;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
      color: white;
      cursor: pointer;
      font-size: 17px;
      font-weight: 600;
      letter-spacing: 0.01em;
      box-shadow: var(--shadow-soft);
      transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }

    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 14px 28px rgba(37, 99, 235, 0.28);
      filter: brightness(1.02);
    }

    .suggestions {
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      right: 0;
      margin: 0;
      padding: 8px 0;
      list-style: none;
      background: #ffffff;
      border: 1px solid #dbe4ff;
      border-radius: 14px;
      box-shadow: 0 14px 30px rgba(30, 58, 138, 0.18);
      max-height: 240px;
      overflow-y: auto;
      z-index: 200;
      display: none;
      scrollbar-width: thin;
      scrollbar-color: #a5b4fc #eef2ff;
    }

    .suggestions.visible { display: block; }

    .suggestions li {
      padding: 11px 15px;
      cursor: pointer;
      font-size: 14px;
      color: #0f172a;
      transition: background 0.15s ease;
    }

    .suggestions li:hover { background: #eef3ff; }

    .suggestions::-webkit-scrollbar { width: 8px; }
    .suggestions::-webkit-scrollbar-thumb { background: #a5b4fc; border-radius: 999px; }
    .suggestions::-webkit-scrollbar-track { background: #eef2ff; border-radius: 999px; }

    .result {
      margin-top: 20px;
      padding: 18px 18px 16px;
      border-radius: 16px;
      background: linear-gradient(145deg, #f8fbff, #f3f7ff);
      border: 1px solid #dce6ff;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.95);
    }

    .result strong {
      display: block;
      margin-bottom: 8px;
      color: #1e3a8a;
      font-size: 18px;
    }

    .result ul { margin: 8px 0 0 20px; padding: 0; color: #334155; }
    .result li { margin: 6px 0; line-height: 1.6; }

    .error {
      margin-top: 20px;
      color: #b91c1c;
      background: #fff1f2;
      border: 1px solid #fecdd3;
      padding: 12px 14px;
      border-radius: 12px;
      font-weight: 500;
    }

    @media (max-width: 640px) {
      .card { padding: 22px; max-width: 100%; border-radius: 20px; }
      .card h1 { font-size: 34px; }
      .card p { font-size: 16px; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1>Weather-Based Style Recommendations</h1>
      <p>Tell me your city, and I’ll suggest the best clothes for the weather there.</p>
      <form method="post" action="/">
        <div class="input-wrap">
          <input id="city-input" name="city" type="text" placeholder="Enter city name" value="{{ city | default('') }}" autocomplete="off" required />
          <ul id="city-suggestions" class="suggestions"></ul>
        </div>
        <button type="submit">Ask</button>
      </form>
      {% if weather_info %}
      <div class="result">
        <strong>{{ weather_icon }} Weather:</strong>
        <p>{{ weather_info }}</p>
      </div>
      {% endif %}
      {% if suggestion %}
      <div class="result">
        <strong>{{ clothing_icon }} Suggestion for {{ city }}:</strong>
        <ul>
          {% for item in suggestion_points %}
          <li>{{ item }}</li>
          {% endfor %}
        </ul>
      </div>
      {% endif %}
      {% if color_suggestion %}
      <div class="result">
        <strong>{{ color_icon }} Best Color for {{ city }} Weather:</strong>
        <ul>
          {% for item in color_points %}
          <li>{{ item }}</li>
          {% endfor %}
        </ul>
      </div>
      {% endif %}
      {% if error %}
      <div class="error">{{ error }}</div>
      {% endif %}
    </div>
  </div>
</body>
<script>
  const cityInput = document.getElementById("city-input");
  const citySuggestions = document.getElementById("city-suggestions");
  let activeRequest = null;
  let debounceTimer = null;
  const form = cityInput.closest("form");

  const hideSuggestions = () => {
    citySuggestions.classList.remove("visible");
    citySuggestions.innerHTML = "";
  };

  cityInput.addEventListener("input", async (event) => {
    const query = event.target.value.trim();
    if (query.length < 1) {
      hideSuggestions();
      return;
    }
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(async () => {
      if (activeRequest) {
        activeRequest.abort();
      }
      activeRequest = new AbortController();

      try {
        const response = await fetch(`/api/cities?q=${encodeURIComponent(query)}`, {
          signal: activeRequest.signal
        });
        if (!response.ok) {
          hideSuggestions();
          return;
        }
        const data = await response.json();
        const cities = data.cities || [];
        if (!cities.length) {
          hideSuggestions();
          return;
        }

        citySuggestions.innerHTML = "";
        cities.forEach((name) => {
          const item = document.createElement("li");
          item.textContent = name;
          item.addEventListener("mousedown", (e) => {
            e.preventDefault();
            cityInput.value = name;
            hideSuggestions();
          });
          citySuggestions.appendChild(item);
        });
        citySuggestions.classList.add("visible");
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error("City autocomplete failed", err);
        }
      }
    }, 180);
  });

  document.addEventListener("click", (event) => {
    if (!form.contains(event.target)) {
      hideSuggestions();
    }
  });

  cityInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideSuggestions();
    }
  });
</script>
</html>
'''


@app.route("/", methods=["GET", "POST"])
def home():
    city = ""
    suggestion = ""
    suggestion_points = []
    color_suggestion = ""
    color_points = []
    weather_info = ""
    weather_icon = "🌤️"
    clothing_icon = "👕"
    color_icon = "🎨"
    error = ""

    if request.method == "POST":
        city = (request.form.get("city") or "").strip()
        if city:
            try:
                weather = get_weather_for_city(city)
                weather_info = (
                    f"{weather['description']} with {weather['temp']}°C, "
                    f"humidity {weather['humidity']}%, wind {weather['wind_speed']} m/s."
                )
                style = predict_style_recommendation(weather)
                icons = style_emojis(weather)
                weather_icon = icons["weather"]
                clothing_icon = icons["clothing"]
                color_icon = icons["color"]
                suggestion = style["clothing"]
                color_suggestion = style["color"]
                suggestion_points = _to_points(suggestion)
                color_points = _to_points(color_suggestion)
            except Exception as exc:
                error = str(exc)

    return render_template_string(
        PAGE_HTML,
        city=city,
        weather_info=weather_info,
        weather_icon=weather_icon,
        clothing_icon=clothing_icon,
        color_icon=color_icon,
        suggestion=suggestion,
        suggestion_points=suggestion_points,
        color_suggestion=color_suggestion,
        color_points=color_points,
        error=error,
    )


def _to_points(text: str):
    chunks = [x.strip(" -") for x in text.replace("!", ".").split(".") if x.strip()]
    if not chunks and text.strip():
        return [text.strip()]
    return chunks


@app.route("/api/cities", methods=["GET"])
def city_suggestions():
    query = request.args.get("q", "").strip()
    return jsonify({"cities": search_indian_cities(query)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
