# =============================================================
# HDI Prediction System — Flask Web Application
# Routes: /  |  /predict  |  /result
# =============================================================

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hdi-secret-2024")

# ── Load Model ────────────────────────────────────────────────
basedir = os.path.dirname(__file__)
model = None
for candidate in ("hdi_model.pkl", "HDI.pkl"):
    path = os.path.join(basedir, candidate)
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        break

if model is None:
    # Keep application importable; return informative error on requests.
    app.logger.error("Model file not found. Expected hdi_model.pkl or HDI.pkl in %s", basedir)

# ── Country List ──────────────────────────────────────────────
COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Australia",
    "Austria", "Bahamas", "Bangladesh", "Belarus", "Belgium", "Bolivia",
    "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia",
    "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia",
    "Congo", "Croatia", "Cuba", "Cyprus", "Czechia", "Denmark", "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Estonia", "Ethiopia", "Finland", "France",
    "Gabon", "Germany", "Ghana", "Greece", "Guatemala", "Guinea", "Honduras",
    "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iraq", "Ireland",
    "Israel", "Italy", "Japan", "Jordan", "Kenya", "Kuwait", "Laos", "Latvia",
    "Libya", "Lithuania", "Mali", "Malta", "Mexico", "Montenegro", "Morocco",
    "Mozambique", "Myanmar", "Namibia", "Netherlands", "New Zealand", "Nicaragua",
    "Niger", "Norway", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
    "Qatar", "Romania", "Russia", "Saudi Arabia", "Senegal", "Sierra Leone",
    "Singapore", "Slovakia", "Slovenia", "Somalia", "South Africa", "South Korea",
    "South Sudan", "Spain", "Sweden", "Switzerland", "Tanzania", "Thailand",
    "Tunisia", "UAE", "Uganda", "United Kingdom", "United States", "Uruguay",
    "Vietnam", "Zambia", "Other"
]


def classify_hdi(score: float) -> dict:
    """Map a numeric HDI score to its category and display metadata."""
    if score >= 0.800:
        return {"category": "Very High", "badge": "🌟 Very High HDI",
            "color": "#1e40af", "bg": "#eff6ff",
            "badge_class": "hdi-very-high",
            "description": "Excellent human development — high life expectancy, education, and income."}
    elif score >= 0.700:
        return {"category": "High", "badge": "✅ High HDI",
            "color": "#2563eb", "bg": "#dbeafe",
            "badge_class": "hdi-high",
            "description": "Strong human development with good standards of living."}
    elif score >= 0.550:
        return {"category": "Medium", "badge": "⚠️ Medium HDI",
            "color": "#d97706", "bg": "#fef3c7",
            "badge_class": "hdi-medium",
            "description": "Moderate human development — room for improvement in key areas."}
    else:
        return {"category": "Low", "badge": "❗ Low HDI",
            "color": "#dc2626", "bg": "#fee2e2",
            "badge_class": "hdi-low",
            "description": "Low human development — significant challenges in health, education, and income."}


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html", countries=COUNTRIES)


@app.route("/result", methods=["POST"])
def result():
    try:
        if model is None:
            return render_template("500.html", message="Model file not found. Please ensure the trained model exists."), 500
        country                 = request.form.get("country", "").strip()
        life_expectancy         = float(request.form["life_expectancy"])
        expected_years_schooling = float(request.form["expected_years_schooling"])
        mean_years_schooling    = float(request.form["mean_years_schooling"])
        gni_per_capita          = float(request.form["gni_per_capita"])

        # Validation
        errors = []
        if not country:
            errors.append("Please select a country.")
        if not (20 <= life_expectancy <= 100):
            errors.append("Life Expectancy must be between 20 and 100.")
        if not (0 <= expected_years_schooling <= 25):
            errors.append("Expected Years of Schooling must be between 0 and 25.")
        if not (0 <= mean_years_schooling <= 20):
            errors.append("Mean Years of Schooling must be between 0 and 20.")
        if not (100 <= gni_per_capita <= 200000):
            errors.append("GNI per Capita must be between 100 and 200,000.")

        if errors:
            return render_template("predict.html", countries=COUNTRIES, errors=errors,
                                   form_data=request.form)

        features = np.array([[life_expectancy, expected_years_schooling,
                               mean_years_schooling, gni_per_capita]])
        hdi_score = float(model.predict(features)[0])
        hdi_score = round(max(0.0, min(1.0, hdi_score)), 4)  # clamp to [0, 1]

        info = classify_hdi(hdi_score)

        inputs = {
            "Country":                    country,
            "Life Expectancy (yrs)":      life_expectancy,
            "Expected Yrs of Schooling":  expected_years_schooling,
            "Mean Yrs of Schooling":      mean_years_schooling,
            "GNI per Capita (USD)":       f"${gni_per_capita:,.0f}",
        }

        return render_template("result.html",
                               hdi_score=hdi_score,
                               score_pct=round(hdi_score * 100, 2),
                               inputs=inputs,
                               **info)

    except (ValueError, KeyError):
        return render_template("predict.html", countries=COUNTRIES,
                               errors=["Invalid input. Please enter valid numeric values."],
                               form_data=request.form)
    except Exception:
        return render_template("500.html"), 500




# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    )
