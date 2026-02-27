"""
main.py
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from data_loader import build_corridor_data, minutes_to_time
from optimizer import optimize_corridor
from predictor import load_or_train, predict_delay
from assistant import ask_assistant, reset_history

app = FastAPI(title="Railway Traffic Optimizer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load everything at startup ───────────────────────────────────────────────
print("Loading data...")
stations, trains_meta, schedules, corridor_nums, corridor_trains = build_corridor_data()
print("Training delay model...")
delay_model, label_encoder, model_features = load_or_train(corridor_trains)

active_disruptions: dict = {}
latest_optimization: dict = {}
chat_history: list = []

print(f"\n✅ Server ready! {len(corridor_trains)} corridor trains loaded.")
print(f"   Docs: http://localhost:8000/docs\n")


# ── Train position simulation ────────────────────────────────────────────────
def get_train_position(train: dict) -> Optional[dict]:
    stops = [s for s in train.get("stops", []) if s.get("lat") and s.get("lng")]
    if len(stops) < 2:
        return None

    now = datetime.now()
    sim_minutes = now.hour * 60 + now.minute
    delay = active_disruptions.get(train["number"], 0)

    for i in range(len(stops) - 1):
        dep = stops[i].get("dep_min")
        arr_next = stops[i + 1].get("arr_min")
        if dep is None or arr_next is None:
            continue
        dep_d = dep + delay
        arr_d = arr_next + delay
        if dep_d <= sim_minutes <= arr_d:
            duration = max(1, arr_d - dep_d)
            progress = (sim_minutes - dep_d) / duration
            lat = stops[i]["lat"] + (stops[i + 1]["lat"] - stops[i]["lat"]) * progress
            lng = stops[i]["lng"] + (stops[i + 1]["lng"] - stops[i]["lng"]) * progress
            status = "on_time" if delay == 0 else ("delayed" if delay <= 30 else "severely_delayed")
            return {
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "from_station": stops[i]["station_code"],
                "to_station": stops[i + 1]["station_code"],
                "progress_pct": round(progress * 100, 1),
                "delay_minutes": delay,
                "status": status,
            }
    return None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Railway Traffic Optimizer API",
        "corridor_trains": len(corridor_trains),
        "stations": len(stations),
        "docs": "/docs"
    }


@app.get("/api/trains")
def get_trains():
    return {
        "count": len(corridor_trains),
        "trains": [{
            "number": t["number"],
            "name": t["name"],
            "type": t.get("type", "Exp"),
            "from_name": t.get("from_name", ""),
            "to_name": t.get("to_name", ""),
            "distance": t.get("distance", 0),
            "stops_count": len(t.get("stops", [])),
            "coordinates": t.get("coordinates", []),
        } for t in corridor_trains]
    }


@app.get("/api/trains/{train_number}")
def get_train_detail(train_number: str):
    for t in corridor_trains:
        if t["number"] == train_number:
            return t
    raise HTTPException(404, f"Train {train_number} not found")


@app.get("/api/positions")
def get_positions():
    """Poll every 2 seconds for live train positions."""
    active = []
    for t in corridor_trains:
        pos = get_train_position(t)
        if pos:
            active.append({
                "number": t["number"],
                "name": t["name"],
                "type": t.get("type", "Exp"),
                **pos,
            })
    return {
        "timestamp": datetime.now().isoformat(),
        "active_trains": len(active),
        "positions": active
    }


@app.get("/api/stations")
def get_stations():
    result = [s for s in stations.values() if s.get("lat") and s.get("lng")]
    return {"count": len(result), "stations": result}


# ── Disruption ───────────────────────────────────────────────────────────────

class DisruptionRequest(BaseModel):
    train_number: str
    delay_minutes: int


@app.post("/api/disruption")
def inject_disruption(req: DisruptionRequest):
    global active_disruptions, latest_optimization

    if not any(t["number"] == req.train_number for t in corridor_trains):
        raise HTTPException(404, f"Train {req.train_number} not in corridor. Use /api/trains to see valid numbers.")

    active_disruptions[req.train_number] = req.delay_minutes

    print(f"[API] Optimizing with disruption: {req.train_number} +{req.delay_minutes}min")
    result = optimize_corridor(corridor_trains[:20], disruptions=active_disruptions)
    latest_optimization = result

    cascading = [
        {"number": t["number"], "name": t["name"], "added_delay": t["total_delay"]}
        for t in result.get("trains", [])
        if t["total_delay"] > 0 and t["number"] != req.train_number
    ]

    rec = "No cascading impact detected. Monitor and proceed."
    if cascading:
        worst = max(cascading, key=lambda x: x["added_delay"])
        rec = (f"Hold train {worst['number']} ({worst['name']}) to allow "
               f"delayed {req.train_number} to clear the block section. "
               f"This saves ~{worst['added_delay']} min of cascading delay.")

    return {
        "message": f"Train {req.train_number} delayed +{req.delay_minutes} min",
        "optimization_status": result["status"],
        "solve_time_ms": result["solve_time_ms"],
        "total_delay_minutes": result["total_delay_minutes"],
        "cascading_impacts": cascading[:5],
        "recommendation": rec,
    }


@app.delete("/api/disruption/{train_number}")
def clear_disruption(train_number: str):
    global latest_optimization
    active_disruptions.pop(train_number, None)
    latest_optimization = optimize_corridor(corridor_trains[:20], disruptions=active_disruptions)
    return {"message": f"Disruption cleared for {train_number}"}


@app.get("/api/disruptions")
def get_disruptions():
    return {"active_disruptions": active_disruptions}


# ── Optimization ─────────────────────────────────────────────────────────────

@app.post("/api/optimize")
def run_optimization():
    global latest_optimization
    latest_optimization = optimize_corridor(corridor_trains[:20], disruptions=active_disruptions)
    return latest_optimization


@app.get("/api/optimization/latest")
def get_latest_optimization():
    if not latest_optimization:
        return {"message": "No optimization run yet. POST /api/optimize first."}
    return latest_optimization


# ── Delay prediction ─────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    train_number: str
    current_delay_min: int
    current_station_code: str


@app.post("/api/predict-delay")
def predict_train_delay(req: PredictionRequest):
    train = next((t for t in corridor_trains if t["number"] == req.train_number), None)
    if not train:
        raise HTTPException(404, "Train not found")

    stops = train.get("stops", [])
    idx = next((i for i, s in enumerate(stops) if s["station_code"] == req.current_station_code), 0)
    stops_rem = max(1, len(stops) - idx - 1)
    dist_rem = int((train.get("distance") or 300) * stops_rem / max(1, len(stops)))

    result = predict_delay(
        delay_model, label_encoder, model_features,
        train_type=train.get("type", "Exp"),
        hour_of_day=datetime.now().hour,
        current_delay_min=req.current_delay_min,
        stops_remaining=stops_rem,
        distance_remaining_km=dist_rem,
        junction_count=min(stops_rem, 4),
    )
    return {"train_number": req.train_number, "train_name": train.get("name"), **result}


# ── KPIs ──────────────────────────────────────────────────────────────────────

@app.get("/api/kpis")
def get_kpis():
    total = len(corridor_trains)
    disrupted = len(active_disruptions)
    on_time_pct = round(((total - disrupted) / max(1, total)) * 100, 1)
    total_delay = latest_optimization.get("total_delay_minutes", 0)

    return {
        "total_trains": total,
        "active_disruptions": disrupted,
        "on_time_percentage": on_time_pct,
        "total_delay_minutes": total_delay,
        "avg_delay_per_train": round(total_delay / max(1, total), 1),
        "last_optimization_status": latest_optimization.get("status", "Not run"),
        "last_solve_ms": latest_optimization.get("solve_time_ms", 0),
    }


# ── AI Assistant ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    global chat_history
    result = ask_assistant(
        question=req.question,
        corridor_trains=corridor_trains[:15],
        optimization_result=latest_optimization or None,
        disruptions=active_disruptions,
        conversation_history=chat_history,
    )
    chat_history.append({"role": "user", "content": req.question})
    chat_history.append({"role": "assistant", "content": result["answer"]})
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
    return result


@app.delete("/api/chat/clear")
def clear_chat():
    global chat_history
    chat_history = []
    reset_history()
    return {"message": "Chat cleared"}