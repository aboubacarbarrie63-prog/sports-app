from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy import stats
import requests
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Data Layer ---
# In production, replace these with real API calls to:
# SportRadar, NBA Stats API, ESPN, Basketball-Reference, etc.

SAMPLE_PLAYERS = [
    "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
    "Luka Doncic", "Jayson Tatum", "Joel Embiid", "Nikola Jokic",
    "Damian Lillard", "Anthony Davis", "Devin Booker", "Trae Young"
]

def fetch_player_stats(player_name, opponent_team):
    """
    In production: call SportRadar / NBA Stats API for real data.
    Here we simulate realistic NBA player data.
    """
    random.seed(hash(player_name) % 10000)
    base_pts = random.uniform(18, 32)
    base_reb = random.uniform(4, 12)
    base_ast = random.uniform(3, 10)

    # Last 15 games
    last_15 = []
    for i in range(15):
        noise = random.gauss(0, 4)
        last_15.append({
            "game": i + 1,
            "points": max(0, round(base_pts + noise, 1)),
            "rebounds": max(0, round(base_reb + random.gauss(0, 2), 1)),
            "assists": max(0, round(base_ast + random.gauss(0, 2), 1)),
            "opponent": ["LAL", "BOS", "MIA", "CHI", "PHX"][i % 5],
            "home": i % 2 == 0,
            "result": "W" if random.random() > 0.45 else "L"
        })

    # Historical vs opponent (5 years)
    random.seed(hash(player_name + opponent_team) % 10000)
    hist_vs_opp = []
    for y in range(5):
        for g in range(random.randint(2, 4)):
            hist_vs_opp.append({
                "year": 2020 + y,
                "points": max(0, round(base_pts + random.gauss(2, 5), 1)),
                "rebounds": max(0, round(base_reb + random.gauss(0, 2), 1)),
                "assists": max(0, round(base_ast + random.gauss(0, 2), 1)),
            })

    return {
        "name": player_name,
        "team": "Team A",
        "position": random.choice(["PG", "SG", "SF", "PF", "C"]),
        "season_avg": {
            "points": round(base_pts, 1),
            "rebounds": round(base_reb, 1),
            "assists": round(base_ast, 1),
            "fg_pct": round(random.uniform(0.42, 0.58), 3),
            "three_pct": round(random.uniform(0.32, 0.42), 3),
        },
        "last_15": last_15,
        "hist_vs_opp": hist_vs_opp,
    }

def fetch_qualitative_factors(player_name, opponent_team, is_home):
    """
    In production: pull from injury reports, travel APIs,
    sentiment analysis of locker room news, etc.
    """
    random.seed(hash(player_name + str(is_home)) % 9999)

    injury_statuses = ["Healthy", "Questionable (knee)", "Probable (ankle)", "Healthy", "Healthy"]
    locker_room = ["Positive", "Neutral", "Tense", "Positive", "Positive"]
    travel_load = ["Low (1 game in 3 days)", "High (3 games in 4 days)", "Medium (back-to-back)", "Low", "Medium"]

    injury = random.choice(injury_statuses)
    lr = random.choice(locker_room)
    travel = random.choice(travel_load)
    days_rest = random.randint(1, 4)

    # Impact scores (-1 to +1)
    home_impact = 0.08 if is_home else -0.05
    injury_impact = {"Healthy": 0.0, "Probable (ankle)": -0.05,
                     "Questionable (knee)": -0.15}.get(injury, 0.0)
    lr_impact = {"Positive": 0.05, "Neutral": 0.0, "Tense": -0.08}.get(lr, 0.0)
    travel_impact = -0.10 if "3 games" in travel else (-0.05 if "back" in travel else 0.02)
    rest_impact = 0.03 * min(days_rest, 3) - 0.03

    return {
        "is_home": is_home,
        "injury_status": injury,
        "locker_room": lr,
        "travel_load": travel,
        "days_rest": days_rest,
        "opponent": opponent_team,
        "impacts": {
            "home_away": home_impact,
            "injury": injury_impact,
            "locker_room": lr_impact,
            "travel": travel_impact,
            "rest": rest_impact,
        }
    }

def build_probabilistic_model(player_stats, qual_factors, stat="points"):
    """
    Builds a probabilistic model combining:
    - Historical distributions (5 years vs opponent)
    - Recent form (last 15 games)
    - Qualitative impact adjustments
    """
    # Historical values
    hist_vals = [g[stat] for g in player_stats["hist_vs_opp"]]
    recent_vals = [g[stat] for g in player_stats["last_15"]]

    # Weighted mean: 40% historical vs opp, 60% recent form
    hist_mean = np.mean(hist_vals) if hist_vals else player_stats["season_avg"][stat]
    recent_mean = np.mean(recent_vals)
    base_mean = 0.4 * hist_mean + 0.6 * recent_mean

    # Apply qualitative adjustments
    total_impact = sum(qual_factors["impacts"].values())
    adjusted_mean = base_mean * (1 + total_impact)

    # Standard deviation from recent volatility
    recent_std = np.std(recent_vals) if len(recent_vals) > 1 else base_mean * 0.25
    hist_std = np.std(hist_vals) if len(hist_vals) > 1 else base_mean * 0.25
    combined_std = 0.5 * recent_std + 0.5 * hist_std

    # Generate distribution
    x = np.linspace(max(0, adjusted_mean - 3.5 * combined_std),
                    adjusted_mean + 3.5 * combined_std, 300)
    y = stats.norm.pdf(x, adjusted_mean, combined_std)

    # Percentiles
    p10 = stats.norm.ppf(0.10, adjusted_mean, combined_std)
    p25 = stats.norm.ppf(0.25, adjusted_mean, combined_std)
    p75 = stats.norm.ppf(0.75, adjusted_mean, combined_std)
    p90 = stats.norm.ppf(0.90, adjusted_mean, combined_std)

    return {
        "mean": round(adjusted_mean, 1),
        "std": round(combined_std, 1),
        "x": [round(v, 2) for v in x.tolist()],
        "y": [round(v, 4) for v in y.tolist()],
        "p10": round(max(0, p10), 1),
        "p25": round(max(0, p25), 1),
        "p75": round(max(0, p75), 1),
        "p90": round(max(0, p90), 1),
        "hist_mean": round(hist_mean, 1),
        "recent_mean": round(recent_mean, 1),
    }

def generate_recommendation(player_name, stat_models, qual_factors):
    confidence_factors = []
    concerns = []
    positives = []

    pts_model = stat_models["points"]
    volatility = pts_model["std"] / pts_model["mean"] if pts_model["mean"] > 0 else 1
    confidence = max(0.4, 1 - volatility * 1.5)

    if qual_factors["is_home"]:
        positives.append("Home court advantage")
    else:
        concerns.append("Away game — historically players average ~3% fewer points on road")

    if "Questionable" in qual_factors["injury_status"]:
        concerns.append(f"Injury concern: {qual_factors['injury_status']}")
        confidence -= 0.1
    elif qual_factors["injury_status"] == "Healthy":
        positives.append("Fully healthy")

    if qual_factors["locker_room"] == "Tense":
        concerns.append("Locker room tension reported")
        confidence -= 0.05
    elif qual_factors["locker_room"] == "Positive":
        positives.append("Strong team chemistry")

    if "3 games" in qual_factors["travel_load"]:
        concerns.append("Heavy travel schedule — fatigue risk")
        confidence -= 0.08

    if qual_factors["days_rest"] >= 3:
        positives.append(f"{qual_factors['days_rest']} days rest — well recovered")

    trend = pts_model["recent_mean"] - pts_model["hist_mean"]
    if trend > 3:
        positives.append(f"Hot streak: +{round(trend,1)} pts above historical average recently")
    elif trend < -3:
        concerns.append(f"Cold stretch: {round(trend,1)} pts below historical average recently")

    confidence = round(min(0.95, max(0.35, confidence)), 2)

    if pts_model["mean"] > 25 and confidence > 0.65:
        verdict = "STRONG PLAY"
        verdict_color = "#00ff88"
    elif pts_model["mean"] > 18 and confidence > 0.55:
        verdict = "LEAN PLAY"
        verdict_color = "#ffcc00"
    else:
        verdict = "FADE / AVOID"
        verdict_color = "#ff4444"

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100),
        "positives": positives,
        "concerns": concerns,
        "summary": f"{player_name} is projected at {pts_model['mean']} pts | {stat_models['rebounds']['mean']} reb | {stat_models['assists']['mean']} ast with {round(confidence*100)}% model confidence."
    }

@app.route("/")
def index():
    return render_template("index.html", players=SAMPLE_PLAYERS)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    player_name = data.get("player", "LeBron James")
    opponent = data.get("opponent", "BOS")
    is_home = data.get("is_home", True)

    player_stats = fetch_player_stats(player_name, opponent)
    qual_factors = fetch_qualitative_factors(player_name, opponent, is_home)

    stat_models = {
        "points": build_probabilistic_model(player_stats, qual_factors, "points"),
        "rebounds": build_probabilistic_model(player_stats, qual_factors, "rebounds"),
        "assists": build_probabilistic_model(player_stats, qual_factors, "assists"),
    }

    recommendation = generate_recommendation(player_name, stat_models, qual_factors)

    return jsonify({
        "player": player_stats,
        "qualitative": qual_factors,
        "models": stat_models,
        "recommendation": recommendation,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5050)