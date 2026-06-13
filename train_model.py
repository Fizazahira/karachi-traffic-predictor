"""
Karachi Traffic Route Prediction - Model Training
---------------------------------------------------
Trains two models on traffic_data.csv:

1. A CLASSIFIER -> predicts congestion_level (low / medium / high)
2. A REGRESSOR  -> predicts travel_time_minutes

Inputs (features) for both models:
    - route (A/B/C)
    - day_of_week (Mon..Sun)
    - hour (0-23)
    - is_rainy (0/1)
    - has_event (0/1)

Models are saved as .pkl files so the app (Step 3/4) can load them
without retraining every time.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import joblib

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv("traffic_data.csv")
print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Encode categorical columns
# -----------------------------
# Machine learning models need numbers, not text.
# We use LabelEncoder to turn "Mon"->0, "Tue"->1, etc.
# We SAVE these encoders so the app can convert user input
# the same way later.

route_encoder = LabelEncoder()
day_encoder = LabelEncoder()
congestion_encoder = LabelEncoder()

df["route_enc"] = route_encoder.fit_transform(df["route"])
df["day_enc"] = day_encoder.fit_transform(df["day_of_week"])
df["congestion_enc"] = congestion_encoder.fit_transform(df["congestion_level"])

print("\nRoute mapping:", dict(zip(route_encoder.classes_, route_encoder.transform(route_encoder.classes_))))
print("Day mapping:", dict(zip(day_encoder.classes_, day_encoder.transform(day_encoder.classes_))))
print("Congestion mapping:", dict(zip(congestion_encoder.classes_, congestion_encoder.transform(congestion_encoder.classes_))))

# -----------------------------
# 3. Define features (X) and targets (y)
# -----------------------------
feature_cols = ["route_enc", "day_enc", "hour", "is_rainy", "has_event"]

X = df[feature_cols]
y_congestion = df["congestion_enc"]
y_traveltime = df["travel_time_minutes"]

# -----------------------------
# 4. Train/test split
# -----------------------------
X_train, X_test, y_cong_train, y_cong_test = train_test_split(
    X, y_congestion, test_size=0.2, random_state=42
)

# Use the SAME split indices for travel time (so train/test rows match)
_, _, y_time_train, y_time_test = train_test_split(
    X, y_traveltime, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Train the Congestion Classifier
# -----------------------------
print("\n--- Training Congestion Level Classifier ---")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_cong_train)

cong_preds = clf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_cong_test, cong_preds), 3))
print(classification_report(
    y_cong_test, cong_preds,
    target_names=congestion_encoder.classes_
))

# -----------------------------
# 6. Train the Travel Time Regressor
# -----------------------------
print("\n--- Training Travel Time Regressor ---")
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_time_train)

time_preds = reg.predict(X_test)
mae = mean_absolute_error(y_time_test, time_preds)
print("Mean Absolute Error (minutes):", round(mae, 2))

# -----------------------------
# 7. Save models + encoders
# -----------------------------
joblib.dump(clf, "congestion_model.pkl")
joblib.dump(reg, "traveltime_model.pkl")
joblib.dump(route_encoder, "route_encoder.pkl")
joblib.dump(day_encoder, "day_encoder.pkl")
joblib.dump(congestion_encoder, "congestion_encoder.pkl")

print("\nSaved models:")
print(" - congestion_model.pkl")
print(" - traveltime_model.pkl")
print(" - route_encoder.pkl")
print(" - day_encoder.pkl")
print(" - congestion_encoder.pkl")
print("\nDone! Ready for Step 3 (prediction function).")
