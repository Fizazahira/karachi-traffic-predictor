# Karachi Traffic Route Predictor 🚗

A demo AI project that predicts traffic congestion and travel time
for 3 popular Karachi routes, and recommends the best one based on
day, time, rain, and special events (VIP movement/protests).

---

## 📁 Project Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Generates synthetic traffic data (`traffic_data.csv`) |
| `traffic_data.csv` | The dataset (4,032 rows, 8 weeks of hourly data x 3 routes) |
| `train_model.py` | Trains the ML models (congestion classifier + travel time regressor) |
| `predict.py` | Standalone prediction function (command-line test) |
| `app.py` | **Streamlit demo app** (the main thing to run for your presentation) |
| `*.pkl` | Saved trained models + encoders |
| `requirements.txt` | Python dependencies |

---

## 🚀 How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) Regenerate data and retrain models — already done, but if you want to redo it:
   ```
   python generate_dataset.py
   python train_model.py
   ```

3. Launch the demo app:
   ```
   streamlit run app.py
   ```

4. It will open in your browser (usually `http://localhost:8501`).

---

## 🎤 Demo Script / Story (for your presentation)

**1. Hook (open with the problem):**
> "Sharea Faisal and MA Jinnah Road are jammed again — people are stuck
> for an hour for what should be a 15-minute drive. This wastes time,
> fuel, and adds a lot of stress and pollution to our daily lives."

**2. Introduce your solution:**
> "Cities around the world — and even parts of Karachi like DHA — are
> starting to use AI to manage traffic smartly. I built a small AI tool
> that predicts which route will be jammed at a given time, so you can
> pick the best one before leaving home."

**3. Live Demo:**
- Open the Streamlit app.
- Pick a day + time (e.g., "Wednesday, 6 PM") — show how all 3 routes
  turn red (high congestion) — this is rush hour.
- Toggle "Rainy weather" ON — show how travel times increase further.
- Pick a quiet time (e.g., "Sunday, 3 AM") — show all routes turn green.
- Point at the recommended route badge.

**4. Show the weekly heatmap:**
- Switch between routes A/B/C and show the heatmap.
- Point out: "See how mornings 7-9 AM and evenings 5-8 PM are always
  red/orange — this matches the real rush-hour pattern in Karachi."

**5. Explain the model (briefly):**
> "Behind the scenes, I trained a Random Forest model on traffic patterns
> — it learned that certain times, days, rain, and events increase
> congestion. It's about 88% accurate at predicting congestion level,
> and within 1.5 minutes for travel time."

**6. Wrap up / future scope:**
> "This is a simplified simulation using realistic patterns. With real
> data from Google Maps API or local traffic sensors, this same model
> could power a real 'Karachi Traffic Assistant' — and could even be
> extended to control traffic signal timing automatically, like the
> AI-based signals being piloted on II Chundrigar Road."

---

## 🔧 Notes

- The dataset is **synthetic** (simulated) but designed to reflect real
  Karachi traffic patterns: rush hours (7-9 AM, 5-8 PM), weekend
  reductions, rain delays (+25%), and event-based delays (+50%).
- Routes used:
  - **A**: Sharea Faisal
  - **B**: II Chundrigar Road
  - **C**: University Road
- Be transparent if asked: "This uses simulated data designed to mirror
  real traffic patterns, since live traffic APIs require paid access."
