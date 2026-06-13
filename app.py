"""
Karachi Traffic Route Predictor - Streamlit Demo App
-------------------------------------------------------
A simple, visual demo app:

- User picks day, time, rain, event
- App shows predicted travel time + congestion for 3 routes
- Highlights the recommended (fastest) route
- Shows a simple schematic map (red/yellow/green routes)
- Shows a weekly congestion heatmap

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Karachi Traffic Route Predictor",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# Hugging Face Hub repo containing the .pkl model files
# Change this to your own repo: "your-username/karachi-traffic-models"
# -----------------------------
HF_REPO_ID = "Fizazahira/karachi-traffic-models"

# -----------------------------
# Load models + encoders
# (tries Hugging Face Hub first, falls back to local files)
# -----------------------------
@st.cache_resource
def load_models():
    filenames = [
        "congestion_model.pkl",
        "traveltime_model.pkl",
        "route_encoder.pkl",
        "day_encoder.pkl",
        "congestion_encoder.pkl",
    ]

    paths = {}
    for fname in filenames:
        try:
            paths[fname] = hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
        except Exception:
            # Fallback: use local file (useful for local dev/testing)
            paths[fname] = fname

    clf = joblib.load(paths["congestion_model.pkl"])
    reg = joblib.load(paths["traveltime_model.pkl"])
    route_encoder = joblib.load(paths["route_encoder.pkl"])
    day_encoder = joblib.load(paths["day_encoder.pkl"])
    congestion_encoder = joblib.load(paths["congestion_encoder.pkl"])

    return clf, reg, route_encoder, day_encoder, congestion_encoder

clf, reg, route_encoder, day_encoder, congestion_encoder = load_models()

@st.cache_data
def load_data():
    return pd.read_csv("traffic_data.csv")

df = load_data()

ROUTE_NAMES = {
    "A": "Sharea Faisal",
    "B": "II Chundrigar Road",
    "C": "University Road",
}

CONGESTION_COLORS = {
    "low": "#10b981",     # emerald
    "medium": "#f59e0b",  # amber
    "high": "#ef4444",    # red
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# -----------------------------
# Prediction function
# -----------------------------
def predict_routes(day_of_week, hour, is_rainy=0, has_event=0):
    results = []
    day_enc = day_encoder.transform([day_of_week])[0]

    for route_code in ["A", "B", "C"]:
        route_enc = route_encoder.transform([route_code])[0]

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

    results.sort(key=lambda r: r["predicted_travel_time"])
    return results


# -----------------------------
# Custom CSS for a cleaner, professional look
# -----------------------------
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
    }
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #155e75 100%);
        padding: 28px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 28px;
        margin: 0 0 8px 0;
        font-weight: 700;
    }
    .app-header p {
        color: #cbd5e1;
        font-size: 15px;
        margin: 0;
        line-height: 1.6;
    }
    .app-header .badge {
        display: inline-block;
        background-color: #06b6d4;
        color: #0f172a;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] .stSelectbox, 
    section[data-testid="stSidebar"] .stSlider {
        margin-bottom: 4px;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e293b;
        border-color: #334155;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] [data-testid="stTickBar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="app-header">
        <span class="badge">AI TRAFFIC INTELLIGENCE · KARACHI</span>
        <h1>Smart Route Advisor</h1>
        <p>
            Predicts congestion and travel time across three major Karachi
            corridors using machine learning trained on rush-hour, weather,
            and event-based traffic patterns — helping you choose the
            fastest route before you leave home.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.markdown(
    """
    <div style="padding: 4px 0 16px 0;">
        <div style="font-size:18px; font-weight:700; color:#f8fafc; margin-bottom:2px;">
            🧭 Plan Your Trip
        </div>
        <div style="font-size:13px; color:#94a3b8;">
            Set your travel conditions to get a route recommendation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("**📅 When are you traveling?**")
day_input = st.sidebar.selectbox("Day of week", DAYS, index=2, label_visibility="collapsed")
hour_input = st.sidebar.slider("Hour of day (24h format)", 0, 23, 18)

st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)
st.sidebar.markdown("**⚠️ Conditions**")
is_rainy_input = st.sidebar.toggle("🌧️ Rainy weather", value=False)
has_event_input = st.sidebar.toggle("🚧 Road closure / VIP movement / protest", value=False)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style="
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 8px;
    ">
        <div style="font-size:12px; font-weight:700; color:#94a3b8; letter-spacing:0.5px; margin-bottom:10px;">
            ROUTES IN THIS DEMO
        </div>
        <div style="font-size:13px; color:#e2e8f0; line-height:1.9;">
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:#22d3ee; margin-right:8px;"></span>
            <b>A</b> — Sharea Faisal<br>
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:#a78bfa; margin-right:8px;"></span>
            <b>B</b> — II Chundrigar Road<br>
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:#2dd4bf; margin-right:8px;"></span>
            <b>C</b> — University Road
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# -----------------------------
# Run prediction
# -----------------------------
results = predict_routes(
    day_of_week=day_input,
    hour=hour_input,
    is_rainy=int(is_rainy_input),
    has_event=int(has_event_input)
)

best_route = results[0]

# -----------------------------
# Results section
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Predicted Routes")

    for r in results:
        color = CONGESTION_COLORS[r["predicted_congestion"]]
        is_best = (r["route"] == best_route["route"])

        badge_html = (
            f'<span style="background-color:#0d9488; color:white; '
            f'font-size:11px; font-weight:700; padding:3px 9px; '
            f'border-radius:20px; margin-left:8px; letter-spacing:0.5px;">'
            f'RECOMMENDED</span>'
        ) if is_best else ""

        st.markdown(
            f"""
            <div style="
                border: 1px solid #e2e8f0;
                border-left: 5px solid {color};
                border-radius: 8px;
                padding: 14px 18px;
                margin-bottom: 12px;
                background-color: #ffffff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size:15px; font-weight:600; color:#111827; margin-bottom:6px;">
                    Route {r['route']} — {r['route_name']} {badge_html}
                </div>
                <div style="font-size:13px; color:#6b7280;">
                    Estimated time: <b style="color:#111827;">{r['predicted_travel_time']} min</b>
                    &nbsp;·&nbsp;
                    Congestion: <span style="color:{color}; font-weight:700;">
                        {r['predicted_congestion'].upper()}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.success(
        f"**Recommended: Route {best_route['route']} ({best_route['route_name']})** "
        f"— estimated {best_route['predicted_travel_time']} min, "
        f"{best_route['predicted_congestion']} congestion."
    )

with col2:
    st.subheader("Route Schematic")

    # Simple schematic: 3 horizontal lines representing routes
    fig = go.Figure()

    y_positions = {"A": 3, "B": 2, "C": 1}

    for r in results:
        color = CONGESTION_COLORS[r["predicted_congestion"]]
        y = y_positions[r["route"]]

        fig.add_trace(go.Scatter(
            x=[0, 10],
            y=[y, y],
            mode="lines+markers+text",
            line=dict(color=color, width=10),
            marker=dict(size=14, color=color),
            text=["Start", "End"],
            textposition="top center",
            name=f"Route {r['route']} ({r['route_name']})",
            hovertemplate=(
                f"Route {r['route']}: {r['route_name']}<br>"
                f"Time: {r['predicted_travel_time']} min<br>"
                f"Congestion: {r['predicted_congestion']}<extra></extra>"
            )
        ))

    fig.update_layout(
        showlegend=True,
        xaxis=dict(visible=False, range=[-1, 11]),
        yaxis=dict(visible=False, range=[0, 4]),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Low congestion · 🟠 Medium · 🔴 High")

st.divider()

# -----------------------------
# Weekly congestion heatmap
# -----------------------------
st.subheader("Weekly Congestion Pattern")
st.caption("Based on historical data — darker red = more congestion, for the selected route.")

route_for_heatmap = st.selectbox(
    "Select route to view weekly pattern",
    options=["A", "B", "C"],
    format_func=lambda x: f"Route {x}: {ROUTE_NAMES[x]}"
)

# Average travel time by day & hour for selected route
heatmap_df = df[df["route"] == route_for_heatmap]
pivot = heatmap_df.pivot_table(
    index="day_of_week",
    columns="hour",
    values="travel_time_minutes",
    aggfunc="mean"
)

# Reorder days
pivot = pivot.reindex(DAYS)

fig2 = px.imshow(
    pivot,
    labels=dict(x="Hour of Day", y="Day of Week", color="Avg Travel Time (min)"),
    color_continuous_scale=[
        [0.0, "#10b981"],
        [0.5, "#f59e0b"],
        [1.0, "#ef4444"],
    ],
    aspect="auto"
)
fig2.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))

st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "💡 Notice how mornings (7-9 AM) and evenings (5-8 PM) are consistently "
    "darker (higher congestion) — this matches real Karachi rush-hour patterns."
)

st.divider()
st.caption(
    "Built as a demo project: predicting Karachi route congestion using "
    "machine learning (Random Forest) trained on simulated traffic patterns "
    "reflecting real rush-hour, rain, and event-based delays."
)
