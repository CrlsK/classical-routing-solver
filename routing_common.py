"""
routing_common.py - Shared core for the Real-Time Adaptive Routing Under
Uncertainty use case (QCentroid). Both solvers import it so they parse the same
VRP-under-uncertainty schema, run the same local search (2-opt / or-opt / swap
+ multi-start), and emit an identical enriched EUR benchmark schema.

Pure standard library.
"""
import math
import time
import random
import hashlib
import json
from datetime import datetime, timezone

COMMON_VERSION = "3.2.0"

# -- Geo ----------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# -- Input parsing ------------------------------------------------------------

def parse_input(input_data):
    """Normalise the platform dataset into a single canonical problem dict."""
    depot = input_data.get("depot")
    if not depot:
        depots = input_data.get("depots") or []
        depot = depots[0] if depots else {"id": "depot", "lat": 0.0, "lon": 0.0}
    depot = {"id": depot.get("id", "depot"),
             "lat": float(depot.get("lat", 0.0)),
             "lon": float(depot.get("lon", 0.0))}

    raw_customers = input_data.get("customers") or input_data.get("stops") or []
    order_demand = {}
    for o in input_data.get("orders", []) or []:
        cid = o.get("customer_id") or o.get("id")
        if cid is not None:
            order_demand[cid] = order_demand.get(cid, 0.0) + float(o.get("demand", 0.0))

    customers = []
    seen = set()
    for c in raw_customers:
        cid = c.get("id")
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        tw = c.get("time_window")
        demand = c.get("demand")
        if demand is None:
            demand = order_demand.get(cid, 1.0)
        customers.append({
            "id": cid,
            "lat": float(c["lat"]),
            "lon": float(c["lon"]),
            "demand": float(demand),
            "time_window": tuple(tw) if tw else None,
            "service_time": float(c.get("service_time", 0.0)),
            "priority": c.get("priority", "normal"),
        })

    raw_vehicles = input_data.get("vehicles") or input_data.get("fleet") or []
    vehicles = []
    for v in raw_vehicles:
        vehicles.append({
            "id": v.get("id", f"V{len(vehicles)+1}"),
            "capacity": float(v.get("capacity", 100.0)),
            "speed_kmh": float(v.get("speed_kmh", 50.0)),
        })
    if not vehicles:
        vehicles = [{"id": "V1", "capacity": 100.0, "speed_kmh": 50.0},
                    {"id": "V2", "capacity": 100.0, "speed_kmh": 50.0}]

    disruptions = input_data.get("disruptions", []) or []

    cp = input_data.get("cost_parameters", {}) or {}
    cost_model = {
        "fuel_cost_per_km": float(cp.get("fuel_cost_per_km", 0.18)),
        "driver_cost_per_hour": float(cp.get("driver_cost_per_hour", 22.0)),
        "lateness_penalty_per_min": float(cp.get("lateness_penalty_per_min", 5.0)),
        "overtime_threshold_min": float(cp.get("overtime_threshold_min", 480.0)),
    }

    cons = input_data.get("constraints", {}) or {}
    constraints = {
        "max_route_time_min": float(cons.get("max_route_time_min", 1e9)),
        "max_vehicle_load": cons.get("max_vehicle_load"),
        "enforce_time_windows": bool(cons.get("enforce_time_windows", True)),
        "allow_split_delivery": bool(cons.get("allow_split_delivery", False)),
    }

    slt = input_data.get("service_level_targets", {}) or {}
    sla = {
        "on_time_delivery_pct": float(slt.get("on_time_delivery_pct", 0.95)),
        "max_lateness_min": float(slt.get("max_lateness_min", 15.0)),
    }

    utm = input_data.get("travel_time_uncertainty_model", {}) or {}
    base_std = float(utm.get("base_std_dev_factor", 0.15))
    traffic = input_data.get("traffic_feed", {}) or {}
    congestion = float(traffic.get("overall_congestion_index", 0.0))
    weather = input_data.get("weather_feed", {}) or {}
    weather_delay = float(weather.get("weather_delay_factor", 1.0))

    return {
        "depot": depot,
        "customers": customers,
        "vehicles": vehicles,
        "disruptions": disruptions,
        "cost_model": cost_model,
        "constraints": constraints,
        "sla": sla,
        "uncertainty_base": base_std,
        "congestion_index": congestion,
        "weather_delay_factor": weather_delay,
        "planning_horizon_minutes": float(input_data.get("planning_horizon_minutes", 480.0)),
    }


def disruption_map(disruptions):
    dm = {}
    for d in disruptions:
        for loc_id in d.get("affected_locations", []):
            dm[loc_id] = dm.get(loc_id, 0.0) + float(d.get("delay_min", 0.0))
    return dm


def effective_uncertainty(prob):
    """Blend the base uncertainty with live traffic congestion and weather."""
    base = prob["uncertainty_base"]
    congestion = prob["congestion_index"]
    weather = prob["weather_delay_factor"]
    return base * (1.0 + 0.5 * congestion) * weather


# -- Route evaluation ---------------------------------------------------------

def route_analytics(stop_ids, locs, depot_id, speed_kmh, uncertainty, disrupted):
    """Deterministic evaluation of one ordered route."""
    current_time = 0.0
    seq = [depot_id] + list(stop_ids) + [depot_id]
    stop_etas = {}
    service_results = {}
    violations = []
    total_km = 0.0
    penalty_min = 0.0
    for i in range(len(seq) - 1):
        a, b = locs[seq[i]], locs[seq[i + 1]]
        dist_km = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        travel = (dist_km / max(speed_kmh, 1e-6)) * 60.0 * (1.0 + uncertainty)
        current_time += travel
        total_km += dist_km
        bid = seq[i + 1]
        if bid != depot_id:
            current_time += disrupted.get(bid, 0.0) + float(b.get("service_time", 0.0))
            eta = round(current_time, 1)
            stop_etas[bid] = eta
            on_time = True
            tw = b.get("time_window")
            if tw:
                earliest, latest = tw
                if current_time < earliest:
                    current_time = float(earliest)
                elif current_time > latest:
                    on_time = False
                    late = current_time - latest
                    penalty_min += late
                    violations.append({"stop": bid, "lateness_min": round(late, 1)})
            service_results[bid] = {"eta_min": eta, "on_time": on_time}
    return {
        "time_min": current_time,
        "objective_time_min": current_time + penalty_min,
        "distance_km": total_km,
        "lateness_min": penalty_min,
        "stop_etas": stop_etas,
        "service_results": service_results,
        "violations": violations,
    }


# Optimisation context: economic cost model in force during local search.
_CTX = {"cost_model": None}


def _route_obj(stop_ids, locs, depot_id, speed_kmh, uncertainty, disrupted):
    """Economic route cost in EUR when a cost model is active; else route time."""
    if not stop_ids:
        return 0.0
    an = route_analytics(stop_ids, locs, depot_id, speed_kmh, uncertainty, disrupted)
    cm = _CTX["cost_model"]
    if cm is None:
        return an["objective_time_min"]
    fuel = an["distance_km"] * cm["fuel_cost_per_km"]
    driver = (an["time_min"] / 60.0) * cm["driver_cost_per_hour"]
    lateness = an["lateness_min"] * cm["lateness_penalty_per_min"]
    return fuel + driver + lateness


# -- Local search (shared by BOTH solvers) ------------------------------------

def two_opt(stop_ids, locs, depot_id, speed_kmh, uncertainty, disrupted, max_pass=8):
    """Intra-route 2-opt: reverse segments while it lowers the route objective."""
    if len(stop_ids) < 3:
        return list(stop_ids)
    best = list(stop_ids)
    best_c = _route_obj(best, locs, depot_id, speed_kmh, uncertainty, disrupted)
    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = _route_obj(cand, locs, depot_id, speed_kmh, uncertainty, disrupted)
                if c < best_c - 1e-6:
                    best, best_c = cand, c
                    improved = True
    return best


def or_opt(assignment, locs, depot_id, vehicles, uncertainty, disrupted,
           capacities, max_pass=6):
    """Inter-route relocation: move a stop to another vehicle if it lowers cost."""
    def veh_speed(k):
        return vehicles[k]["speed_kmh"]

    def load(ids):
        return sum(locs[s]["demand"] for s in ids)

    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for k1 in list(assignment.keys()):
            for stop in list(assignment[k1]):
                base = _route_obj(assignment[k1], locs, depot_id, veh_speed(k1),
                                  uncertainty, disrupted)
                for k2 in assignment.keys():
                    if k2 == k1:
                        continue
                    if load(assignment[k2]) + locs[stop]["demand"] > capacities[k2] + 1e-9:
                        continue
                    new1 = [s for s in assignment[k1] if s != stop]
                    best_pos, best_delta = None, float("inf")
                    r2 = assignment[k2]
                    for pos in range(len(r2) + 1):
                        cand2 = r2[:pos] + [stop] + r2[pos:]
                        c2 = _route_obj(cand2, locs, depot_id, veh_speed(k2),
                                        uncertainty, disrupted)
                        if c2 < best_delta:
                            best_delta, best_pos = c2, pos
                    c1 = _route_obj(new1, locs, depot_id, veh_speed(k1),
                                    uncertainty, disrupted)
                    old_total = base + _route_obj(r2, locs, depot_id, veh_speed(k2),
                                                  uncertainty, disrupted)
                    new_total = c1 + best_delta
                    if new_total < old_total - 1e-6:
                        assignment[k1] = new1
                        assignment[k2] = r2[:best_pos] + [stop] + r2[best_pos:]
                        improved = True
                        break
    return assignment


def swap_move(assignment, locs, depot_id, vehicles, uncertainty, disrupted,
              capacities, max_pass=4):
    """Inter-route exchange: swap a stop of k1 with a stop of k2 when it lowers
    the combined route objective and respects both capacities. Complements
    or-opt; swaps escape optima that relocation alone cannot."""
    def sp(k):
        return vehicles[k]["speed_kmh"]

    def load(ids):
        return sum(locs[s]["demand"] for s in ids)

    keys = list(assignment.keys())
    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for ki in range(len(keys)):
            for kj in range(ki + 1, len(keys)):
                k1, k2 = keys[ki], keys[kj]
                r1, r2 = assignment[k1], assignment[k2]
                if not r1 or not r2:
                    continue
                for ai in range(len(r1)):
                    for bi in range(len(r2)):
                        a, b = r1[ai], r2[bi]
                        l1 = load(r1) - locs[a]["demand"] + locs[b]["demand"]
                        l2 = load(r2) - locs[b]["demand"] + locs[a]["demand"]
                        if l1 > capacities[k1] + 1e-9 or l2 > capacities[k2] + 1e-9:
                            continue
                        old = (_route_obj(r1, locs, depot_id, sp(k1), uncertainty, disrupted)
                               + _route_obj(r2, locs, depot_id, sp(k2), uncertainty, disrupted))
                        n1 = list(r1); n1[ai] = b
                        n2 = list(r2); n2[bi] = a
                        new = (_route_obj(n1, locs, depot_id, sp(k1), uncertainty, disrupted)
                               + _route_obj(n2, locs, depot_id, sp(k2), uncertainty, disrupted))
                        if new < old - 1e-6:
                            assignment[k1] = n1
                            assignment[k2] = r2 = n2
                            r1 = assignment[k1]
                            improved = True
    return assignment


def nearest_neighbour_init(prob):
    """Greedy capacity-feasible construction; returns dict veh_idx -> [stop_ids]."""
    depot = prob["depot"]
    customers = prob["customers"]
    vehicles = prob["vehicles"]
    assignment = {k: [] for k in range(len(vehicles))}
    loads = {k: 0.0 for k in range(len(vehicles))}
    unvisited = list(range(len(customers)))
    k = 0
    cur = (depot["lat"], depot["lon"])
    while unvisited and k < len(vehicles):
        cap = vehicles[k]["capacity"]
        feasible = [i for i in unvisited if loads[k] + customers[i]["demand"] <= cap]
        if not feasible:
            k += 1
            cur = (depot["lat"], depot["lon"])
            continue
        nxt = min(feasible, key=lambda i: haversine(cur[0], cur[1],
                                                    customers[i]["lat"], customers[i]["lon"]))
        assignment[k].append(customers[nxt]["id"])
        loads[k] += customers[nxt]["demand"]
        cur = (customers[nxt]["lat"], customers[nxt]["lon"])
        unvisited.remove(nxt)
    for i in unvisited:
        order = sorted(range(len(vehicles)), key=lambda kk: loads[kk])
        for kk in order:
            if loads[kk] + customers[i]["demand"] <= vehicles[kk]["capacity"]:
                assignment[kk].append(customers[i]["id"])
                loads[kk] += customers[i]["demand"]
                break
        else:
            assignment[order[0]].append(customers[i]["id"])
    return assignment


def _loc_index(prob):
    locs = {prob["depot"]["id"]: prob["depot"]}
    for c in prob["customers"]:
        locs[c["id"]] = c
    return locs


def polish(assignment, prob, uncertainty, disrupted):
    """2-opt on each route, then alternate or-opt + swap across routes (v3.2)."""
    _CTX["cost_model"] = prob["cost_model"]
    depot = prob["depot"]
    vehicles = prob["vehicles"]
    locs = _loc_index(prob)
    capacities = [v["capacity"] for v in vehicles]
    for k in assignment:
        assignment[k] = two_opt(assignment[k], locs, depot["id"],
                                vehicles[k]["speed_kmh"], uncertainty, disrupted)
    for _round in range(2):
        assignment = or_opt(assignment, locs, depot["id"], vehicles, uncertainty,
                            disrupted, capacities)
        assignment = swap_move(assignment, locs, depot["id"], vehicles, uncertainty,
                               disrupted, capacities)
    for k in assignment:
        assignment[k] = two_opt(assignment[k], locs, depot["id"],
                                vehicles[k]["speed_kmh"], uncertainty, disrupted)
    return assignment


def capacity_overload(assignment, prob):
    """Total demand over capacity summed across all routes (0 = feasible)."""
    locs = _loc_index(prob)
    vehicles = prob["vehicles"]
    over = 0.0
    for k, ids in assignment.items():
        load = sum(locs[s]["demand"] for s in ids)
        cap = vehicles[k]["capacity"]
        if load > cap:
            over += load - cap
    return over


def _perturb(assignment, prob, rng):
    """Capacity-respecting random relocation kick to diversify a restart."""
    locs = {c["id"]: c for c in prob["customers"]}
    caps = {k: prob["vehicles"][k]["capacity"] for k in assignment}
    keys = [k for k in assignment if assignment[k]]
    if len(keys) < 2:
        return assignment
    k1 = rng.choice(keys)
    stop = rng.choice(assignment[k1])
    demand = locs[stop]["demand"]
    feasible_k2 = [k for k in assignment if k != k1
                   and sum(locs[s]["demand"] for s in assignment[k]) + demand <= caps[k]]
    if not feasible_k2:
        return assignment
    k2 = rng.choice(feasible_k2)
    assignment[k1] = [s for s in assignment[k1] if s != stop]
    pos = rng.randint(0, len(assignment[k2]))
    assignment[k2] = assignment[k2][:pos] + [stop] + assignment[k2][pos:]
    return assignment


def greedy_multistart(prob, uncertainty, disrupted, restarts=6, seed=42):
    """Greedy construction + full local-search polish + perturbation restarts.
    Shared by BOTH solvers so the quantum fallback is never weaker than the
    classical baseline (it keeps the better of this and its QUBO seed)."""
    rng = random.Random(seed)
    best = polish({k: list(v) for k, v in nearest_neighbour_init(prob).items()},
                  prob, uncertainty, disrupted)
    best_obj = assignment_objective(best, prob, uncertainty, disrupted)
    for _ in range(max(0, restarts - 1)):
        cand = _perturb({k: list(v) for k, v in best.items()}, prob, rng)
        cand = polish(cand, prob, uncertainty, disrupted)
        obj = assignment_objective(cand, prob, uncertainty, disrupted)
        if obj < best_obj - 1e-6:
            best, best_obj = cand, obj
    return best, best_obj


def assignment_objective(assignment, prob, uncertainty, disrupted):
    """Economic EUR objective + large penalty for capacity overload."""
    _CTX["cost_model"] = prob["cost_model"]
    locs = _loc_index(prob)
    vehicles = prob["vehicles"]
    total = 0.0
    for k, ids in assignment.items():
        if not ids:
            continue
        total += _route_obj(ids, locs, prob["depot"]["id"],
                            vehicles[k]["speed_kmh"], uncertainty, disrupted)
    total += 1e6 * capacity_overload(assignment, prob)
    return total


# -- Result assembly (identical schema for every solver) ----------------------

def build_result(assignment, prob, input_data, elapsed_s, solver_meta):
    """Standardised result dict with numeric benchmark metrics + benchmark dict."""
    depot = prob["depot"]
    vehicles = prob["vehicles"]
    customers = prob["customers"]
    cost_model = prob["cost_model"]
    sla = prob["sla"]
    locs = _loc_index(prob)
    uncertainty = effective_uncertainty(prob)
    disrupted = disruption_map(prob["disruptions"])

    routes_output = []
    all_service = {}
    all_violations = []
    total_time_min = 0.0
    total_obj_min = 0.0
    total_km = 0.0
    total_lateness = 0.0
    used = 0
    util_list = []
    max_route_time = 0.0
    capacity_violations = 0

    for k, ids in assignment.items():
        if not ids:
            continue
        used += 1
        v = vehicles[k]
        an = route_analytics(ids, locs, depot["id"], v["speed_kmh"],
                             uncertainty, disrupted)
        all_service.update(an["service_results"])
        all_violations.extend(an["violations"])
        total_time_min += an["time_min"]
        total_obj_min += an["objective_time_min"]
        total_km += an["distance_km"]
        total_lateness += an["lateness_min"]
        max_route_time = max(max_route_time, an["time_min"])
        load = sum(locs[s]["demand"] for s in ids)
        if load > v["capacity"] + 1e-6:
            capacity_violations += 1
        util_list.append(min(load / max(v["capacity"], 1e-6), 1.0))
        routes_output.append({
            "vehicle_id": v["id"],
            "stop_sequence": [depot["id"]] + list(ids) + [depot["id"]],
            "num_stops": len(ids),
            "total_load": round(load, 2),
            "capacity_utilization_pct": round(100.0 * load / max(v["capacity"], 1e-6), 1),
            "distance_km": round(an["distance_km"], 3),
            "route_time_min": round(an["time_min"], 2),
            "estimated_cost_minutes": round(an["objective_time_min"], 2),
            "lateness_min": round(an["lateness_min"], 2),
            "stop_etas": an["stop_etas"],
        })

    n_cust = len(customers)
    served = len(all_service)
    on_time = sum(1 for s in all_service.values() if s["on_time"])
    on_time_rate = round(100.0 * on_time / max(served, 1), 2)
    fulfillment = round(100.0 * served / max(n_cust, 1), 2)
    max_lateness = round(max((v["lateness_min"] for v in all_violations), default=0.0), 2)

    fuel_cost = total_km * cost_model["fuel_cost_per_km"]
    driver_cost = (total_time_min / 60.0) * cost_model["driver_cost_per_hour"]
    lateness_cost = total_lateness * cost_model["lateness_penalty_per_min"]
    total_cost_eur = fuel_cost + driver_cost + lateness_cost

    avg_util = round(100.0 * (sum(util_list) / max(len(util_list), 1)), 2)

    base_assign = nearest_neighbour_init(prob)
    _CTX["cost_model"] = cost_model
    for _k in base_assign:
        base_assign[_k] = two_opt(base_assign[_k], locs, depot["id"],
                                  vehicles[_k]["speed_kmh"], uncertainty, disrupted)
    baseline_cost_eur = round(assignment_objective(base_assign, prob, uncertainty, disrupted), 3)
    savings_vs_baseline_pct = round(100.0 * (baseline_cost_eur - total_cost_eur)
                                    / max(baseline_cost_eur, 1e-6), 2)

    l_per_100 = 9.5
    fcm = input_data.get("fuel_consumption_model", {}) or {}
    if isinstance(fcm, dict):
        l_per_100 = float(fcm.get("base_consumption_l_per_100km", l_per_100))
    fuel_litres = total_km * l_per_100 / 100.0
    co2_kg = fuel_litres * 2.68
    cost_per_stop_eur = round(total_cost_eur / max(n_cust, 1), 3)

    stress_disrupted = {kk: vv * 1.5 for kk, vv in disrupted.items()}
    stressed_obj = 0.0
    for k, ids in assignment.items():
        if not ids:
            continue
        stressed_obj += route_analytics(ids, locs, depot["id"],
                                        vehicles[k]["speed_kmh"],
                                        uncertainty * 1.5, stress_disrupted)["objective_time_min"]
    robustness = round(100.0 * total_obj_min / max(stressed_obj, 1e-6), 2)

    sla_met = on_time_rate >= sla["on_time_delivery_pct"] * 100.0 and max_lateness <= sla["max_lateness_min"]
    if capacity_violations:
        status = "infeasible"
    elif all_violations:
        status = "feasible"
    else:
        status = "optimal"

    result = {
        "objective_value": round(total_cost_eur, 3),
        "total_cost_eur": round(total_cost_eur, 3),
        "total_robust_time_min": round(total_obj_min, 3),
        "total_distance_km": round(total_km, 3),
        "total_fuel_cost_eur": round(fuel_cost, 3),
        "total_driver_cost_eur": round(driver_cost, 3),
        "lateness_penalty_eur": round(lateness_cost, 3),
        "num_vehicles_used": used,
        "on_time_delivery_rate_pct": on_time_rate,
        "demand_fulfillment_rate_pct": fulfillment,
        "avg_capacity_utilization_pct": avg_util,
        "max_lateness_min": max_lateness,
        "time_window_violations": len(all_violations),
        "capacity_violations": capacity_violations,
        "max_route_time_min": round(max_route_time, 2),
        "solution_robustness_pct": robustness,
        "disruptions_absorbed": len(prob["disruptions"]),
        "compute_time_s": round(elapsed_s, 3),
        "sla_compliant": 1 if sla_met else 0,
        "baseline_cost_eur": baseline_cost_eur,
        "savings_vs_baseline_pct": savings_vs_baseline_pct,
        "cost_per_stop_eur": cost_per_stop_eur,
        "co2_kg": round(co2_kg, 2),
        "solver_type": solver_meta["solver_type"],
        "algorithm": solver_meta["algorithm"],
        "solution_status": status,
        "routes": routes_output,
        "cost_breakdown_eur": {
            "fuel": round(fuel_cost, 2),
            "driver": round(driver_cost, 2),
            "lateness_penalty": round(lateness_cost, 2),
        },
        "risk_metrics": {
            "on_time_probability": round(on_time_rate / 100.0, 3),
            "robustness_ratio": round(robustness / 100.0, 3),
            "effective_uncertainty_factor": round(uncertainty, 4),
            "time_window_violations": len(all_violations),
            "max_lateness_min": max_lateness,
        },
        "service_level_results": all_service,
        "constraint_violations": all_violations,
        "computation_metrics": {
            "wall_time_s": round(elapsed_s, 3),
            "algorithm": solver_meta["algorithm"],
            "customers": n_cust,
            "vehicles_available": len(vehicles),
            "vehicles_used": used,
            **solver_meta.get("extra_metrics", {}),
        },
        "solver_version": solver_meta.get("solver_version", "3.2.0"),
        "common_version": COMMON_VERSION,
        "dataset_sha256": _sha256_of(input_data),
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "execution_cost": {"value": solver_meta.get("credits", 1.0), "unit": "credits"},
            "time_elapsed": f"{round(elapsed_s, 3)}s",
            "energy_consumption": round(solver_meta.get("energy_j", 0.0), 4),
        },
    }
    return result


def _sha256_of(input_data):
    try:
        blob = json.dumps(input_data, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception:
        return "unavailable"
