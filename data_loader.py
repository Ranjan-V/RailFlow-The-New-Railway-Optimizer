"""
data_loader.py
Loads stations, trains, schedules JSON files.
Filters to South India corridor: Hyderabad - Vijayawada - Chennai - Bangalore
"""
import json
from pathlib import Path

# ── Corridor definition ──────────────────────────────────────────────────────
# Any train that stops at AT LEAST 2 of these station codes is included.
CORRIDOR_STATIONS = {
    # Hyderabad cluster
    "HYB", "SC", "NZB", "KCG", "FM",
    # Vijayawada / Andhra
    "BZA", "GNT", "VSKP", "OGL", "NLR",
    # Chennai cluster
    "MAS", "MS", "MSB", "CHENNAIEGMORE",
    # Bangalore cluster
    "SBC", "YPR", "BNC", "KSR", "MYS",
    # In-between junctions
    "GDR", "RU", "GTL", "DRON", "MTM", "RJY", "NDL", "BPQ"
}

DATA_DIR = Path("data")


def load_raw():
    with open(DATA_DIR / "stations.json", encoding="utf-8") as f:
        stations_raw = json.load(f)
    with open(DATA_DIR / "trains.json", encoding="utf-8") as f:
        trains_raw = json.load(f)
    with open(DATA_DIR / "schedules.json", encoding="utf-8") as f:
        schedules_raw = json.load(f)
    return stations_raw, trains_raw, schedules_raw


def parse_stations(stations_raw):
    stations = {}
    for feature in stations_raw.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry")
        code = props.get("code", "").strip()
        if not code or not geom:
            continue
        coords = geom.get("coordinates", [None, None])
        stations[code] = {
            "code": code,
            "name": props.get("name", code),
            "lat": coords[1],
            "lng": coords[0],
            "state": props.get("state"),
            "zone": props.get("zone"),
        }
    print(f"[data_loader] Loaded {len(stations)} stations with coordinates")
    return stations


def parse_trains(trains_raw):
    trains = {}
    for feature in trains_raw.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {}) or {}
        number = str(props.get("number", "")).strip()
        if not number:
            continue
        trains[number] = {
            "number": number,
            "name": props.get("name", ""),
            "from_code": props.get("from_station_code", ""),
            "from_name": props.get("from_station_name", ""),
            "to_code": props.get("to_station_code", ""),
            "to_name": props.get("to_station_name", ""),
            "type": props.get("type", "Exp"),
            "distance": props.get("distance", 0),
            "duration_h": props.get("duration_h", 0),
            "duration_m": props.get("duration_m", 0),
            "zone": props.get("zone", ""),
            "coordinates": geom.get("coordinates", []),
            "sleeper": bool(props.get("sleeper")),
            "third_ac": bool(props.get("third_ac")),
            "second_ac": bool(props.get("second_ac")),
            "first_ac": bool(props.get("first_ac")),
        }
    print(f"[data_loader] Loaded {len(trains)} trains")
    return trains


def parse_schedules(schedules_raw):
    from collections import defaultdict
    raw_stops = defaultdict(list)
    for entry in schedules_raw:
        tn = str(entry.get("train_number", "")).strip()
        if not tn:
            continue
        arrival = entry.get("arrival", "None")
        departure = entry.get("departure", "None")
        raw_stops[tn].append({
            "station_code": entry.get("station_code", "").strip(),
            "station_name": entry.get("station_name", ""),
            "arrival": None if arrival in ("None", None, "") else arrival,
            "departure": None if departure in ("None", None, "") else departure,
            "day": int(entry.get("day") or 1),
        })

    def sort_key(stop):
        t = stop["departure"] or stop["arrival"] or "00:00:00"
        return (stop["day"], t)

    schedules = {}
    for tn, stops in raw_stops.items():
        schedules[tn] = sorted(stops, key=sort_key)

    print(f"[data_loader] Loaded schedules for {len(schedules)} trains")
    return schedules


def filter_corridor_trains(schedules, trains):
    corridor_train_numbers = []
    for tn, stops in schedules.items():
        stop_codes = {s["station_code"] for s in stops}
        overlap = stop_codes & CORRIDOR_STATIONS
        if len(overlap) >= 2:
            corridor_train_numbers.append(tn)
    print(f"[data_loader] {len(corridor_train_numbers)} trains pass through corridor")
    return corridor_train_numbers


def time_to_minutes(time_str, day=1):
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        return (day - 1) * 1440 + h * 60 + m
    except Exception:
        return None


def minutes_to_time(minutes):
    if minutes is None:
        return None
    minutes = int(minutes) % 1440
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def build_corridor_data():
    stations_raw, trains_raw, schedules_raw = load_raw()
    stations = parse_stations(stations_raw)
    trains = parse_trains(trains_raw)
    schedules = parse_schedules(schedules_raw)
    corridor_nums = filter_corridor_trains(schedules, trains)

    corridor_trains_full = []
    for tn in corridor_nums:
        train_meta = trains.get(tn, {"number": tn, "name": tn, "type": "Exp"})
        stops = schedules.get(tn, [])
        enriched_stops = []
        for stop in stops:
            sc = stop["station_code"]
            st = stations.get(sc, {})
            enriched_stops.append({
                **stop,
                "lat": st.get("lat"),
                "lng": st.get("lng"),
                "arr_min": time_to_minutes(stop["arrival"], stop["day"]),
                "dep_min": time_to_minutes(stop["departure"], stop["day"]),
            })
        corridor_trains_full.append({
            **train_meta,
            "stops": enriched_stops,
        })

    print(f"[data_loader] Built {len(corridor_trains_full)} enriched corridor trains")
    return stations, trains, schedules, corridor_nums, corridor_trains_full


if __name__ == "__main__":
    stations, trains, schedules, corridor_nums, corridor_trains = build_corridor_data()
    print("\nSample corridor train:")
    if corridor_trains:
        t = corridor_trains[0]
        print(f"  {t['number']} - {t['name']}")
        for s in t['stops'][:4]:
            print(f"    {s['station_code']:8s}  arr={s['arrival']}  dep={s['departure']}")
    else:
        print("  No corridor trains found!")
        print(f"  Check CORRIDOR_STATIONS — your data may use different codes")
        print(f"  Run this to see what codes exist in schedules:")
        print(f"    import json; d=json.load(open('data/schedules.json')); print(set(x['station_code'] for x in d[:200]))")