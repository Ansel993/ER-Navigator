from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__, static_folder="website", static_url_path="/website")


def read_double_quoted_csv(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            parsed = next(csv.reader(io.StringIO(line), skipinitialspace=False))
            rows.append(parsed)

    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")

    header = rows[0]
    data_rows = rows[1:]
    return pd.DataFrame(data_rows, columns=header)


def to_num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def classify_congestion(delay_min: float, duration_min: float) -> str:
    if duration_min <= 0:
        return "Unknown"
    ratio = delay_min / duration_min
    if ratio >= 0.8:
        return "Gridlock"
    if ratio >= 0.45:
        return "Heavy"
    if ratio >= 0.2:
        return "Moderate"
    if ratio > 0:
        return "Light"
    return "Free Flow"


def osrm_route_summary(user_lat: float, user_lon: float, dest_lat: float, dest_lon: float) -> dict[str, Any]:
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{user_lon},{user_lat};{dest_lon},{dest_lat}"
        "?overview=false&alternatives=false"
    )
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        payload = resp.json()
        route = payload["routes"][0]
        return {
            "distance_km": route["distance"] / 1000.0,
            "duration_min": route["duration"] / 60.0,
            "source": "osrm",
        }
    except Exception:
        approx_km = haversine_km(user_lat, user_lon, dest_lat, dest_lon) * 1.25
        approx_min = (approx_km / 32.0) * 60.0
        return {
            "distance_km": approx_km,
            "duration_min": approx_min,
            "source": "fallback",
        }


def osrm_route_geometry(user_lat: float, user_lon: float, dest_lat: float, dest_lon: float) -> dict[str, Any]:
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{user_lon},{user_lat};{dest_lon},{dest_lat}"
        "?overview=full&geometries=geojson&alternatives=false"
    )
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        route = payload["routes"][0]
        coordinates = [[coord[1], coord[0]] for coord in route["geometry"]["coordinates"]]
        return {
            "distance_km": route["distance"] / 1000.0,
            "duration_min": route["duration"] / 60.0,
            "route": coordinates,
            "source": "osrm",
        }
    except Exception:
        approx_km = haversine_km(user_lat, user_lon, dest_lat, dest_lon) * 1.25
        approx_min = (approx_km / 32.0) * 60.0
        return {
            "distance_km": approx_km,
            "duration_min": approx_min,
            "route": [[user_lat, user_lon], [dest_lat, dest_lon]],
            "source": "fallback",
        }


print("Loading datasets and models...")
hospitals_raw = read_double_quoted_csv(DATA_DIR / "hospital_master_vellore.csv")
load_raw = read_double_quoted_csv(DATA_DIR / "hospital_load_timeseries_vellore.csv")
traffic_raw = read_double_quoted_csv(DATA_DIR / "traffic_dataset_vellore.csv")

traffic_model = joblib.load(MODEL_DIR / "traffic_model.pkl")
load_model = joblib.load(MODEL_DIR / "hospital_load_model.pkl")

hospitals = (
    hospitals_raw[
        [
            "hospital_id",
            "hospital_name",
            "latitude",
            "longitude",
            "hospital_type",
            "total_beds",
            "er_beds",
            "load_category",
            "er_occupancy_pct",
        ]
    ]
    .drop_duplicates(subset=["hospital_id"])
    .copy()
)
hospitals["latitude"] = pd.to_numeric(hospitals["latitude"], errors="coerce")
hospitals["longitude"] = pd.to_numeric(hospitals["longitude"], errors="coerce")

load_raw["timestamp"] = pd.to_datetime(load_raw["timestamp"], errors="coerce")
latest_load = (
    load_raw.sort_values("timestamp")
    .groupby("hospital_id", as_index=False)
    .tail(1)
    .set_index("hospital_id")
)
print("Server context loaded")


def latest_stats_for(hospital_id: str) -> pd.Series | None:
    if hospital_id in latest_load.index:
        return latest_load.loc[hospital_id]
    return None


def infer_load_class(er_load_ratio: float, fallback_category: str = "") -> str:
    category = (fallback_category or "").strip()
    if category:
        return category
    if er_load_ratio >= 1.0:
        return "Overloaded"
    if er_load_ratio >= 0.8:
        return "High"
    if er_load_ratio >= 0.5:
        return "Medium"
    return "Low"


def is_overloaded(load_class: str, er_load_ratio: float) -> bool:
    normalized = (load_class or "").lower()
    if "overloaded" in normalized or "critical" in normalized:
        return True
    return er_load_ratio >= 1.0


def fetch_rainfall_mm(user_lat: float, user_lon: float) -> float:
    """
    Fetch near-real-time rainfall (mm) to feed into the traffic delay model.
    Uses Open-Meteo free API. If unavailable, returns 0.0.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={user_lat}&longitude={user_lon}"
        "&hourly=precipitation"
        "&current_weather=true"
        "&timezone=auto"
    )
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        prec = hourly.get("precipitation") or []
        if not times or not prec or len(times) != len(prec):
            return 0.0

        now = pd.Timestamp.now()
        parsed_times = pd.to_datetime(times, errors="coerce")
        # Pick the closest hour to now
        best_idx = int((parsed_times - now).abs().argmin())
        return float(to_num(prec[best_idx], 0.0))
    except Exception:
        return 0.0


def build_traffic_features(route_distance_km: float, route_duration_min: float, rainfall_mm: float, weekday: int, hour: int) -> pd.DataFrame:
    # Feature order matters for models trained from arrays.
    vehicle_density_pct = min(100.0, max(0.0, (route_distance_km / max(route_duration_min, 1.0)) * 35.0))
    avg_speed_kmph = max(5.0, (route_distance_km / max(route_duration_min, 0.5)) * 60.0)
    is_peak_morning = 1.0 if 8 <= hour <= 10 else 0.0
    is_peak_evening = 1.0 if 17 <= hour <= 20 else 0.0
    is_weekend = 1.0 if weekday >= 5 else 0.0

    return pd.DataFrame(
        [
            {
                "vehicle_density_pct": vehicle_density_pct,
                "avg_speed_kmph": avg_speed_kmph,
                "rainfall_mm": float(rainfall_mm),
                # No external incident feed here; keep 0 and let congestion be driven by traffic model.
                "incident_active": 0.0,
                "is_peak_morning": is_peak_morning,
                "is_peak_evening": is_peak_evening,
                "is_weekend": is_weekend,
                "near_hospital": 1.0,
                "road_length_km": float(route_distance_km),
            }
        ]
    )


def predict_er_load(hospital_row: pd.Series, hospital_id: str) -> dict[str, Any]:
    stats = latest_stats_for(hospital_id)
    if stats is None:
        er_load_ratio = to_num(hospital_row.get("er_occupancy_pct"), 0.0) / 100.0
        er_class = infer_load_class(er_load_ratio, str(hospital_row.get("load_category", "")))
        overload = is_overloaded(er_class, er_load_ratio)
        return {
            "er_load_ratio": float(er_load_ratio),
            "er_load_class_next_1h": er_class,
            "will_be_overloaded_next_1h": overload,
            "waiting_count": 0,
            "patients_in_er": 0,
        }

    load_features = pd.DataFrame(
        [
            {
                "patients_in_er": to_num(stats["patients_in_er"]),
                "waiting_count": to_num(stats["waiting_count"]),
                "critical_patients": to_num(stats["critical_patients"]),
                "ambulance_arrivals": to_num(stats["ambulance_arrivals"]),
                "avg_wait_time_min": to_num(stats["avg_wait_time_min"]),
                "er_load_ratio": to_num(stats["er_load_ratio"]),
            }
        ]
    )

    pred = load_model.predict(load_features)[0]

    label_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    er_class = label_map.get(int(pred), "Medium")
    er_load_ratio = to_num(stats["er_load_ratio"])
    overload = is_overloaded(er_class, er_load_ratio)

    return {
        "er_load_ratio": float(er_load_ratio),
        "er_load_class_next_1h": er_class,
        "will_be_overloaded_next_1h": overload,
        "waiting_count": int(to_num(stats["waiting_count"])),
        "patients_in_er": int(to_num(stats["patients_in_er"])),
    }


def score_hospital(prediction: dict[str, Any]) -> float:
    """
    Smaller score is better.

    The user requirement: prefer hospitals with BOTH:
    1) lower ER load risk
    2) lower travel time (ETA, includes traffic + weather-driven delay)
    """
    er_class = (prediction.get("er_load_class_next_1h") or "").lower()
    er_risk = 0
    if "low" in er_class:
        er_risk = 0
    elif "medium" in er_class:
        er_risk = 1
    elif "high" in er_class:
        er_risk = 2
    if "overloaded" in er_class or "critical" in er_class:
        er_risk = 3

    eta_min = float(prediction.get("eta_min", 0.0) or 0.0)

    # Weighting heuristic:
    # - ETA contributes linearly.
    # - ER risk contributes strongly so an "Overloaded" hospital is strongly disfavored,
    #   but travel time still matters.
    eta_weight = 3.0
    er_weight = 45.0
    return (eta_min * eta_weight) + (er_risk * er_weight)


def predict_for_hospital(
    user_lat: float,
    user_lon: float,
    hospital_row: pd.Series,
    hospital_id: str,
    rainfall_mm: float,
    weekday: int,
    hour: int,
    include_route_geom: bool,
) -> dict[str, Any]:
    dest_lat = float(hospital_row["latitude"])
    dest_lon = float(hospital_row["longitude"])

    if include_route_geom:
        route_data = osrm_route_geometry(user_lat, user_lon, dest_lat, dest_lon)
    else:
        route_data = osrm_route_summary(user_lat, user_lon, dest_lat, dest_lon)

    traffic_features = build_traffic_features(
        route_distance_km=float(route_data["distance_km"]),
        route_duration_min=float(route_data["duration_min"]),
        rainfall_mm=rainfall_mm,
        weekday=weekday,
        hour=hour,
    )
    predicted_delay = float(traffic_model.predict(traffic_features.values)[0])
    predicted_delay = max(0.0, round(predicted_delay, 2))
    congestion = classify_congestion(predicted_delay, float(route_data["duration_min"]))

    er_pred = predict_er_load(hospital_row, hospital_id)

    response = {
        "hospital": {
            "hospital_id": hospital_id,
            "hospital_name": str(hospital_row["hospital_name"]),
            "hospital_type": str(hospital_row["hospital_type"]),
            "lat": dest_lat,
            "lon": dest_lon,
        },
        "distance_km": round(float(route_data["distance_km"]), 2),
        "eta_min": round(float(route_data["duration_min"]) + predicted_delay, 1),
        "base_duration_min": round(float(route_data["duration_min"]), 1),
        "traffic_delay_min": predicted_delay,
        "congestion_level": congestion,
        "er_load_class_next_1h": er_pred["er_load_class_next_1h"],
        "er_load_ratio": round(float(er_pred["er_load_ratio"]), 3),
        "will_be_overloaded_next_1h": bool(er_pred["will_be_overloaded_next_1h"]),
        "routing_source": route_data.get("source", "unknown"),
        "waiting_count": er_pred["waiting_count"],
        "patients_in_er": er_pred["patients_in_er"],
    }
    if include_route_geom:
        response["route"] = route_data["route"]
    return response


@app.route("/")
def home() -> Any:
    return send_from_directory("website", "index.html")


@app.route("/hospitals", methods=["GET"])
def get_hospitals() -> Any:
    user_lat = to_num(request.args.get("lat"))
    user_lon = to_num(request.args.get("lon"))
    use_distance = bool(user_lat and user_lon)

    result: list[dict[str, Any]] = []
    for _, row in hospitals.iterrows():
        hospital_id = str(row["hospital_id"])
        stats = latest_stats_for(hospital_id)
        distance_km = None
        if use_distance:
            distance_km = haversine_km(user_lat, user_lon, float(row["latitude"]), float(row["longitude"]))
        fallback_ratio = to_num(row.get("er_occupancy_pct"), 0.0) / 100.0
        er_load_ratio = to_num(stats["er_load_ratio"]) if stats is not None else fallback_ratio
        er_load_class = (
            str(stats["er_load_class_next_1h"])
            if stats is not None
            else infer_load_class(er_load_ratio, str(row.get("load_category", "")))
        )

        result.append(
            {
                "hospital_id": hospital_id,
                "hospital_name": str(row["hospital_name"]),
                "hospital_type": str(row["hospital_type"]),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "distance_km": round(distance_km, 2) if distance_km is not None else None,
                "er_load_ratio": round(er_load_ratio, 3),
                "er_load_class_next_1h": er_load_class,
                "waiting_count": int(to_num(stats["waiting_count"])) if stats is not None else 0,
                "patients_in_er": int(to_num(stats["patients_in_er"])) if stats is not None else 0,
            }
        )

    if use_distance:
        result.sort(key=lambda item: item["distance_km"] if item["distance_km"] is not None else 999999)
    return jsonify({"hospitals": result})


@app.route("/recommend", methods=["GET"])
def recommend_best_hospital() -> Any:
    user_lat = to_num(request.args.get("lat"))
    user_lon = to_num(request.args.get("lon"))
    if not user_lat or not user_lon:
        return jsonify({"error": "lat and lon are required"}), 400

    rainfall_mm = fetch_rainfall_mm(user_lat, user_lon)
    weekday = pd.Timestamp.now().weekday()
    hour = pd.Timestamp.now().hour

    best_hospital_id: str | None = None
    best_score: float = float("inf")
    best_prediction_summary: dict[str, Any] | None = None
    hospital_cards: list[dict[str, Any]] = []

    # Rank each hospital using:
    # - traffic model (includes weather-derived rainfall_mm)
    # - congestion level
    # - ER load model/fallback
    # - distance/time via eta_min
    for _, row in hospitals.iterrows():
        hospital_id = str(row["hospital_id"])
        pred_summary = predict_for_hospital(
            user_lat=user_lat,
            user_lon=user_lon,
            hospital_row=row,
            hospital_id=hospital_id,
            rainfall_mm=rainfall_mm,
            weekday=weekday,
            hour=hour,
            include_route_geom=False,
        )
        score = score_hospital(pred_summary)

        hospital_cards.append(
            {
                "hospital_id": hospital_id,
                "hospital_name": pred_summary["hospital"]["hospital_name"],
                "hospital_type": pred_summary["hospital"]["hospital_type"],
                "lat": pred_summary["hospital"]["lat"],
                "lon": pred_summary["hospital"]["lon"],
                "distance_km": pred_summary["distance_km"],
                "er_load_ratio": pred_summary.get("er_load_ratio", 0.0),
                "er_load_class_next_1h": pred_summary["er_load_class_next_1h"],
                "waiting_count": pred_summary.get("waiting_count", 0),
                "patients_in_er": pred_summary.get("patients_in_er", 0),
            }
        )

        if score < best_score:
            best_score = score
            best_hospital_id = hospital_id
            best_prediction_summary = pred_summary

    # Sort like the previous UI: closest hospitals first.
    hospital_cards.sort(key=lambda item: item.get("distance_km", 999999))

    if not best_hospital_id:
        # Extremely unlikely (there are 10 hospitals); provide deterministic fallback.
        best_hospital_id = hospital_cards[0]["hospital_id"]

    best_row = hospitals[hospitals["hospital_id"] == best_hospital_id].iloc[0]
    best_prediction = predict_for_hospital(
        user_lat=user_lat,
        user_lon=user_lon,
        hospital_row=best_row,
        hospital_id=best_hospital_id,
        rainfall_mm=rainfall_mm,
        weekday=weekday,
        hour=hour,
        include_route_geom=True,
    )
    # Keep a small indicator for UI.
    best_prediction["best_score"] = float(best_score)
    if best_prediction_summary is not None:
        best_prediction["best_score_from"] = best_prediction_summary.get("routing_source", "unknown")
    return jsonify({"best": best_prediction, "hospitals": hospital_cards})


@app.route("/predict", methods=["POST"])
def predict_route_and_load() -> Any:
    payload = request.get_json(silent=True) or {}
    user_lat = to_num(payload.get("lat"))
    user_lon = to_num(payload.get("lon"))
    hospital_id = str(payload.get("hospital_id", "")).strip()

    if not hospital_id:
        return jsonify({"error": "hospital_id is required"}), 400

    match = hospitals[hospitals["hospital_id"] == hospital_id]
    if match.empty:
        return jsonify({"error": "hospital not found"}), 404

    hospital_row = match.iloc[0]
    rainfall_mm = fetch_rainfall_mm(user_lat, user_lon)
    weekday = pd.Timestamp.now().weekday()
    hour = pd.Timestamp.now().hour

    prediction = predict_for_hospital(
        user_lat=user_lat,
        user_lon=user_lon,
        hospital_row=hospital_row,
        hospital_id=hospital_id,
        rainfall_mm=rainfall_mm,
        weekday=weekday,
        hour=hour,
        include_route_geom=True,
    )
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)