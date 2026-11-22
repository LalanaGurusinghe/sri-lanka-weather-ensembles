# Ensemble Learning on Sri Lankan Weather

Next-day rain and temperature forecasts for 30 Sri Lankan cities, comparing **bagging**,
**boosting**, **stacking** and two **hybrid** models (RF + SARIMA, RF + ANN), with a
Streamlit dashboard.

*Gurusinghe W.A.L.M (224063X), BSc (Hons) IT, University of Moratuwa*

**Live dashboard:** https://sl-weather-ensembles.streamlit.app

## Data

Kaggle [*Sri Lanka Weather Dataset*](https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset)
by `rasulmah`, file `SriLanka_Weather_Dataset.csv`. It has daily Open-Meteo values for
30 cities from 2010-01-01 to 2023-06-17 (147,480 rows × 24 columns). The raw CSV is
not committed. Put it in `data/raw/`; if it's missing, the notebook downloads a
mirror of the same file.

## Tasks

| Task | Target (day t+1) | Type |
|---|---|---|
| `tmax` | next-day `temperature_2m_max` | regression |
| `rain_amount` | `log1p` of next-day `precipitation_sum` | regression |
| `rain_flag` | next-day rain ≥ 1 mm | classification |

Time split: train 2010–2020 · validation 2021 · test 2022-01-01 → 2023-06-16.

## Project layout

```
data/raw/                      raw CSV (git-ignored)
notebooks/ensemble_weather_sri_lanka.ipynb   main submission
src/features.py                cleaning, lag/rolling features, targets, time split
src/models.py                  model builders, metrics, OOF helper
artifacts/                     metrics.json, predictions, figures, best models
app/app.py                     Streamlit dashboard
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/ensemble_weather_sri_lanka.ipynb
streamlit run app/app.py
```

The dashboard only reads `artifacts/`, so it runs without the raw data and without retraining.

## AI use

Code was written with help from Claude Code. Every AI-assisted file and notebook cell starts with
`# AI-assisted: generated with Claude Code, reviewed and run by the author`. See the last section
of the notebook for the full declaration.
