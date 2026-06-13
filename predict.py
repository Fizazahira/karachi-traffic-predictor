"""
Karachi Traffic Route Prediction - Prediction Function
---------------------------------------------------------
Loads the trained models + encoders and provides a function:

    predict_routes(day_of_week, hour, is_rainy, has_event)

which returns predicted congestion level + travel time for
ALL routes (A, B, C), sorted from best (fastest) to worst,
with a recommendation.
"""

import joblib
import pandas as pd

# -----------------------------
# Load models and encoders
# -----------------------------
clf = joblib.load("congestion_model.pkl")
reg = joblib.load("traveltime_model.pkl")
route_encoder = joblib.load("route_encoder.pkl")
day_encoder = joblib.load("day_encoder.pkl")
congestion_encoder = joblib.load("congestion_encoder.pkl")

# Friendly names for routes (must match generate_dataset.py)
ROUTE_NAMES = {
    "A": "Sharea Faisal",
    "B": "II Chundrigar Road",
    "C": "University Road",
}


def predict_routes(day_of_week, hour, is_rainy=0, has_event=0):
    """
    Predict congestion level + travel time for all 3 routes.

    Parameters:
        day_of_week (str): one of 'Mon','Tue','Wed','Thu','Fri','Sat','Sun'
        hour (int): 0-23
        is_rainy (int): 0 or 1
        has_event (int): 0 or 1 (VIP movement / protest / road closure)

    Returns:
        list of dicts, sorted by predicted travel time (fastest first)
    """
    results = []

    day_enc = day_encoder.transform([day_of_week])[0]

    for route_code in ["A", "B", "C"]:
        route_enc = route_encoder.transform([route_code])[0]

        # Build a single-row DataFrame matching training feature order
        X = pd.DataFrame([{
            "route_enc": route_enc,
            "day_enc": day_enc,
            "hour": hour,
            "is_rainy": is_rainy,
            "has_event": has_event,
        }])

        congestion_pred_enc = clf.predict(X)[0]
        congestion_pred = congestion_encoder.inverse_transform([congestion_pred_enc])[0]

        travel_time_pred = reg.predict(X)[0]

        results.append({
            "route": route_code,
            "route_name": ROUTE_NAMES[route_code],
            "predicted_travel_time": round(float(travel_time_pred), 1),
            "predicted_congestion": congestion_pred,
        })

    # Sort by travel time, fastest first
    results.sort(key=lambda r: r["predicted_travel_time"])

    return results


def format_recommendation(results):
    """
    Returns a human-readable summary string, e.g.:

    Route A (Sharea Faisal): 40 min (high congestion)
    Route B (II Chundrigar Road): 22 min (medium congestion)
    Route C (University Road): 18 min (low congestion)
    --> Recommended: Route C (University Road)
    """
    lines = []
    for r in results:
        lines.append(
            f"Route {r['route']} ({r['route_name']}): "
            f"{r['predicted_travel_time']} min "
            f"({r['predicted_congestion']} congestion)"
        )

    best = results[0]
    lines.append(
        f"\n--> Recommended: Route {best['route']} ({best['route_name']})"
    )

    return "\n".join(lines)


# -----------------------------
# Quick test (run directly)
# -----------------------------
if __name__ == "__main__":
    # Example: Tomorrow 6 PM (18:00), rainy, no event, Wednesday
    results = predict_routes(day_of_week="Wed", hour=18, is_rainy=1, has_event=0)
    print(format_recommendation(results))

    print("\n--- Another example: Sunday 3 AM, no rain, no event ---")
    results2 = predict_routes(day_of_week="Sun", hour=3, is_rainy=0, has_event=0)
    print(format_recommendation(results2))
