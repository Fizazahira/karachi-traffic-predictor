# Karachi Traffic Route Predictor

An AI-powered web application that predicts traffic congestion levels and travel times across three major Karachi routes, recommending the optimal route based on day, time, weather, and special event conditions.

## Overview

Traffic congestion is a daily challenge in Karachi, with routes such as Sharea Faisal and II Chundrigar Road frequently experiencing significant delays during peak hours. This project demonstrates how machine learning can be applied to predict congestion patterns and assist commuters in route planning.

The application uses a Random Forest model trained on simulated traffic data that reflects realistic patterns: morning and evening rush hours, weekend variations, weather-related delays, and disruptions from events such as road closures or VIP movements.

## Features

- Predicts congestion level (low / medium / high) and estimated travel time for three routes
- Recommends the fastest route based on current conditions
- Interactive route schematic with color-coded congestion indicators
- Weekly congestion heatmap showing historical patterns by day and hour
- Adjustable inputs for day of week, time, rainfall, and special events

## Routes Covered

| Route | Name |
|---|---|
| A | Sharea Faisal |
| B | II Chundrigar Road |
| C | University Road |

## Project Structure

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `generate_dataset.py` | Generates the synthetic traffic dataset |
| `train_model.py` | Trains the congestion classifier and travel time regressor |
| `predict.py` | Standalone prediction function for testing |
| `upload_to_huggingface.py` | Uploads trained model files to Hugging Face Hub |
| `traffic_data.csv` | Synthetic dataset (4,032 rows covering 8 weeks of hourly data across 3 routes) |
| `requirements.txt` | Python dependencies |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the application

```bash
streamlit run app.py
```

The app will open in your browser, typically at `http://localhost:8501`.

### Regenerate data and retrain models (optional)

The dataset and trained models are included, but can be regenerated:

```bash
python generate_dataset.py
python train_model.py
```

## Model Details

- **Algorithm**: Random Forest (classifier for congestion level, regressor for travel time)
- **Features**: route, day of week, hour, rainfall indicator, special event indicator
- **Performance**: approximately 88% accuracy for congestion level classification; mean absolute error of approximately 1.5 minutes for travel time prediction

## Data Source

The dataset used in this project is synthetically generated to reflect realistic Karachi traffic patterns, including peak-hour congestion (7-9 AM and 5-8 PM), reduced weekend traffic, rain-related delays, and event-driven disruptions. This approach was used in place of live traffic APIs, which typically require paid access.

## Deployment

Trained models are hosted on Hugging Face Hub and downloaded automatically at runtime. The application can be deployed using Streamlit Community Cloud or any platform supporting Python web applications.

## Future Enhancements

- Integration with live traffic data sources (e.g., Google Maps API)
- Extension to additional routes and corridors
- Smart traffic signal timing simulation using reinforcement learning
- Real-time alerts for sudden congestion changes due to accidents or closures
