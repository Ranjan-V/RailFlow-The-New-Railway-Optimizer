"""
optimizer.py
CP-SAT model for train conflict resolution.
Fixed: handles zero travel times, None stops, multi-day trains correctly.
"""

from ortools.sat.python import cp_model
from data_loader import minutes_to_time
from typing import Dict, List, Optional
import time

TRAIN_PRIORITY = {
    "Raj": 1, "Shatabdi": 1, "Vande": 1,
    "SF": 2, "Exp": 3, "Pass": 4,
    "MEMU": 4, "DEMU": 4, "Hyd": 4, "Freight": 5,
}


def get_priority(train_type: str) -> int:
    for key, val in TRAIN_PRIORITY.items():
        if key.lower() in str(train_type).lower():
            return val
    return 3


def clean_stops(stops: list) -> list:
    """Remove stops where both arr and dep are None."""
    return [s for s in stops if s.get("arr_min") is not None or s.get("dep_min") is not None]


def optimize_corridor(
    corridor_trains: List[dict],
    disruptions: Optional[Dict[str, int]] = None,
    time_limit_seconds: float = 8.0,
) -> dict:
    if disruptions is None:
        disruptions = {}

    model = cp_model.CpModel()
    HORIZON = 4 * 1440  # 4 days in minutes

    # Prepare trains
    trains_data = []
    for t in corridor_trains:
        delay = disruptions.get(t["number"], 0)
        clean = clean_stops(t["stops"])
        if len(clean) < 2:
            continue

        stops = []
        for s in clean:
            a = s.get("arr_min")
            d = s.get("dep_min")
            if a is None and d is not None:
                a = d
            if d is None and a is not None:
                d = a
            stops.append({**s, "arr_min_d": a + delay, "dep_min_d": d + delay})

        trains_data.append({**t, "stops": stops, "disruption": delay})

    print(f"[optimizer] {len(trains_data)} trains with usable data")

    # Create variables
    arr_vars = {}
    dep_vars = {}

    for ti, train in enumerate(trains_data):
        arr_vars[ti] = {}
        dep_vars[ti] = {}
        for si, stop in enumerate(train["stops"]):
            a0 = stop["arr_min_d"]
            d0 = stop["dep_min_d"]
            lb = max(0, min(a0, d0))
            ub = min(HORIZON, max(a0, d0) + 180)
            arr_vars[ti][si] = model.NewIntVar(lb, ub, f"a_{ti}_{si}")
            dep_vars[ti][si] = model.NewIntVar(lb, ub, f"d_{ti}_{si}")

    conflicts_added = 0

    for ti, train in enumerate(trains_data):
        stops = train["stops"]

        for si, stop in enumerate(stops):
            a0 = stop["arr_min_d"]
            d0 = stop["dep_min_d"]

            # Departure >= Arrival
            model.Add(dep_vars[ti][si] >= arr_vars[ti][si])

            # Keep original dwell time
            dwell = max(0, d0 - a0)
            model.Add(dep_vars[ti][si] >= arr_vars[ti][si] + dwell)

            # Don't go earlier than original
            model.Add(arr_vars[ti][si] >= a0)
            model.Add(dep_vars[ti][si] >= d0)

            # Travel time to next stop
            if si + 1 < len(stops):
                ns = stops[si + 1]
                travel = ns["arr_min_d"] - stop["dep_min_d"]
                travel = max(0, travel)
                if travel > 0:
                    model.Add(arr_vars[ti][si + 1] >= dep_vars[ti][si] + travel)
                else:
                    model.Add(arr_vars[ti][si + 1] >= dep_vars[ti][si])

        # Head-on conflict constraints
        for tj in range(ti + 1, len(trains_data)):
            _add_headon_constraints(model, ti, train, tj, trains_data[tj], arr_vars, dep_vars)
            conflicts_added += 1

    # Minimize weighted delay
    delay_terms = []
    for ti, train in enumerate(trains_data):
        weight = max(1, 6 - get_priority(train.get("type", "Exp")))
        for si, stop in enumerate(train["stops"]):
            orig_dep = stop["dep_min_d"]
            dv = model.NewIntVar(0, 300, f"dl_{ti}_{si}")
            model.Add(dv >= dep_vars[ti][si] - orig_dep)
            delay_terms.append(dv * weight)

    if delay_terms:
        model.Minimize(sum(delay_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 4

    t0 = time.time()
    status = solver.Solve(model)
    solve_ms = round((time.time() - t0) * 1000, 1)

    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status_str = status_map.get(status, "UNKNOWN")
    print(f"[optimizer] Status={status_str} in {solve_ms}ms")

    result_trains = []
    total_delay_all = 0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for ti, train in enumerate(trains_data):
            result_stops = []
            train_total_delay = 0
            for si, stop in enumerate(train["stops"]):
                orig_arr = stop["arr_min_d"]
                orig_dep = stop["dep_min_d"]
                opt_arr = solver.Value(arr_vars[ti][si])
                opt_dep = solver.Value(dep_vars[ti][si])
                dep_delay = max(0, opt_dep - orig_dep)
                train_total_delay += dep_delay
                result_stops.append({
                    "station_code": stop["station_code"],
                    "station_name": stop.get("station_name", ""),
                    "lat": stop.get("lat"),
                    "lng": stop.get("lng"),
                    "original_arr": minutes_to_time(orig_arr),
                    "original_dep": minutes_to_time(orig_dep),
                    "optimized_arr": minutes_to_time(opt_arr),
                    "optimized_dep": minutes_to_time(opt_dep),
                    "dep_delay_min": dep_delay,
                })
            total_delay_all += train_total_delay
            result_trains.append({
                "number": train["number"],
                "name": train.get("name", ""),
                "type": train.get("type", "Exp"),
                "priority": get_priority(train.get("type", "Exp")),
                "disruption_injected": train.get("disruption", 0),
                "total_delay": train_total_delay,
                "stops": result_stops,
            })

    return {
        "status": status_str,
        "solve_time_ms": solve_ms,
        "trains": result_trains,
        "total_delay_minutes": total_delay_all,
        "conflicts_resolved": conflicts_added,
        "trains_optimized": len(trains_data),
    }


def _add_headon_constraints(model, ti, train_i, tj, train_j, arr_vars, dep_vars):
    stops_i = train_i["stops"]
    stops_j = train_j["stops"]

    def segments(stops):
        d = {}
        for si in range(len(stops) - 1):
            a, b = stops[si]["station_code"], stops[si + 1]["station_code"]
            if a and b and a != b:
                d[(a, b)] = si
        return d

    segs_i = segments(stops_i)
    segs_j = segments(stops_j)

    for (a, b), si in segs_i.items():
        if (b, a) not in segs_j:
            continue
        sj = segs_j[(b, a)]
        if si + 1 >= len(stops_i) or sj + 1 >= len(stops_j):
            continue
        if si not in dep_vars[ti] or si + 1 not in arr_vars[ti]:
            continue
        if sj not in dep_vars[tj] or sj + 1 not in arr_vars[tj]:
            continue
        bvar = model.NewBoolVar(f"ho_{ti}_{tj}_{a}_{b}")
        model.Add(arr_vars[ti][si + 1] <= dep_vars[tj][sj]).OnlyEnforceIf(bvar)
        model.Add(arr_vars[tj][sj + 1] <= dep_vars[ti][si]).OnlyEnforceIf(bvar.Not())


if __name__ == "__main__":
    from data_loader import build_corridor_data
    _, _, _, _, corridor_trains = build_corridor_data()

    sample = corridor_trains[:10]
    first = next((t for t in sample if any(s.get("dep_min") for s in t["stops"])), sample[0])
    disruption = {first["number"]: 30}
    print(f"\nInjecting 30-min delay on {first['number']} ({first['name']})")

    result = optimize_corridor(sample, disruptions=disruption, time_limit_seconds=10.0)
    print(f"Status      : {result['status']}")
    print(f"Solve time  : {result['solve_time_ms']}ms")
    print(f"Trains used : {result['trains_optimized']}")
    print(f"Total delay : {result['total_delay_minutes']} mins")
    for t in result.get("trains", []):
        if t["total_delay"] > 0:
            print(f"  {t['number']:6s} {t['name'][:40]:40s} +{t['total_delay']} min")