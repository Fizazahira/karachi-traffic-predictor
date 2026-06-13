"""
Karachi Traffic Route Dataset Generator
-----------------------------------------
Generates a synthetic dataset for 3 popular routes in Karachi,
simulating congestion patterns based on time of day, day of week,
rain, and special events (VIP movement / protests).

Routes (example):
  A = Sharea Faisal (Airport -> Saddar)       -- generally busy
  B = II Chundrigar Road (Tower -> Saddar)     -- busy during work hours
  C = University Road (Gulshan -> NIPA)        -- moderate traffic

Output: traffic_data.csv
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

ROUTES = {
    "A": {"name": "Sharea Faisal", "base_time": 25, "base_congestion_bias": 0.3},
    "B": {"name": "II Chundrigar Road", "base_time": 20, "base_congestion_bias": 0.2},
    "C": {"name": "University Road", "base_time": 18, "base_congestion_bias": 0.0},
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def get_congestion_level(travel_time, base_time):
    ratio = travel_time / base_time
    if ratio < 1.3:
        return "low"
    elif ratio < 1.8:
        return "medium"
    else:
        return "high"

def generate_row(route_key, day, hour, is_rainy, has_event):
    route = ROUTES[route_key]
    base_time = route["base_time"]
    bias = route["base_congestion_bias"]

    # Rush hour multiplier (7-9 AM, 5-8 PM)
    if hour in [7, 8, 9, 17, 18, 19, 20]:
        rush_multiplier = 1.6 + bias
    elif hour in [12, 13, 14]:  # lunch time, moderate
        rush_multiplier = 1.2 + bias
    elif hour in [0, 1, 2, 3, 4, 5]:  # late night, free flow
        rush_multiplier = 0.9
    else:
        rush_multiplier = 1.1 + bias

    # Weekend reduction
    if day in ["Sat", "Sun"]:
        rush_multiplier *= 0.75

    # Rain adds delay
    if is_rainy:
        rush_multiplier *= 1.25

    # Event (VIP movement/protest) adds significant delay
    if has_event:
        rush_multiplier *= 1.5

    # Add some random noise
    noise = random.uniform(-0.1, 0.15)
    multiplier = max(0.7, rush_multiplier + noise)

    travel_time = round(base_time * multiplier, 1)
    congestion = get_congestion_level(travel_time, base_time)

    return {
        "route": route_key,
        "route_name": route["name"],
        "day_of_week": day,
        "hour": hour,
        "is_rainy": int(is_rainy),
        "has_event": int(has_event),
        "travel_time_minutes": travel_time,
        "congestion_level": congestion,
    }


def generate_dataset(num_weeks=8):
    """
    Generates data for `num_weeks` worth of days (7 days x 24 hours)
    for each of the 3 routes, with random rain/event flags sprinkled in.
    """
    rows = []

    for week in range(num_weeks):
        for day in DAYS:
            for hour in range(24):
                # Randomly decide if it's rainy this day/hour (~10% chance)
                is_rainy = random.random() < 0.10
                # Randomly decide if there's an event (~5% chance)
                has_event = random.random() < 0.05

                for route_key in ROUTES.keys():
                    row = generate_row(route_key, day, hour, is_rainy, has_event)
                    rows.append(row)

    return rows


def save_to_csv(rows, filename="traffic_data.csv"):
    fieldnames = [
        "route", "route_name", "day_of_week", "hour",
        "is_rainy", "has_event", "travel_time_minutes", "congestion_level"
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {filename}")


if __name__ == "__main__":
    data = generate_dataset(num_weeks=8)  # ~8 weeks of hourly data per route
    save_to_csv(data, "traffic_data.csv")

    # Quick sanity check: print a few sample rows
    print("\nSample rows:")
    for r in data[:5]:
        print(r)
