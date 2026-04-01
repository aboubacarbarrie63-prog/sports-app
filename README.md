# ProbaStats — NBA Player Prediction Engine

A Flask web app that analyzes NBA player performance using quantitative + qualitative data and a probabilistic model.

## Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the app**
```bash
python app.py
```

3. **Open in browser**
```
http://localhost:5050
```

## How It Works

### Data Sources (Simulated — swap with real APIs)
- **Quantitative**: Season averages, last 15 games, 5-year history vs opponent
- **Qualitative**: Home/away, injury status, locker room, travel load, rest days

### Probabilistic Model
- Weighted mean: 60% recent form + 40% historical vs opponent
- Qualitative adjustment multiplier applied to mean
- Normal distribution fitted to combined std dev
- P10, P25, P75, P90 percentile output

### Real API Integrations (to add)
Replace the `fetch_player_stats()` and `fetch_qualitative_factors()` functions with:

| Data | API |
|------|-----|
| Live stats | NBA Stats API (stats.nba.com) |
| Historical | Basketball-Reference |
| Injuries | Rotowire / RotoGrinders |
| Betting lines | The Odds API |
| Sentiment | Twitter/X API + NLP |
| Travel | Team schedule APIs |

## File Structure
```
app.py              # Flask backend + model
templates/
  index.html        # Full frontend UI
requirements.txt
README.md
```# sports-app
