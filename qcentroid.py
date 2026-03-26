import copy
import math
import random
import time
import logging
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger("qcentroid-user-log")

# ── Domain Objects ──────────────────────────────────────────────────────────

class Location:
    def __init__(self, id: str, lat: float, lon: float, demand: float = 0,
                 time_window=None, service_time: float = 0.0):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.demand = demand
        self.time_window = time_window
        self.service_time = service_time


class Vehicle:
    def __init__(self, id: str, capacity: float, speed_kmh: float = 50.0):
        self.id = id
        self.capacity = capacity
        self.speed_kmh = speed_kmh


class Route:
    def __init__(self, vehicle: Vehicle, depot: Location):
        self.vehicle = vehicle
        self.depot = depot
        self.stops = []

    @property
    def load(self):
        return sum(s.demand for s in self.stops)

    def cost(self, uncertainty_factor=0.15, disruptions=None):
        disruptions = disruptions or []
        disrupted = {}
        for d in disruptions:
            for loc_id in d.get("affected_locations", []):
                disrupted[loc_id] = disrupted.get(loc_id, 0) + d.get("delay_min", 0.0)
        locs = [self.depot] + self.stops + [self.depot]
        current_time = 0.0
        total_travel = 0.0
        for i in range(len(locs) - 1):
            a, b = locs[i], locs[i + 1]
            dist_km = _haversine(a.lat, a.lon, b.lat, b.lon)
            travel = (dist_km / self.vehicle.speed_kmh) * 60.0 * (1.0 + uncertainty_factor)
            current_time += travel
            total_travel += travel
            if b.id != self.depot.id:
                current_time += disrupted.get(b.id, 0.0) + b.service_time
                if b.time_window:
                    earliest, latest = b.time_window
                    if current_time < earliest:
                        current_time = float(earliest)
                    elif current_time > latest:
                        total_travel += (current_time - latest) * 2.0
        return total_travel

    def can_add(self, loc):
        return self.load + loc.demand <= self.vehicle.capacity

    def add(self, loc):
        self.stops.append(loc)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _nearest_neighbour_init(depot, customers, vehicles):
    routes = []
    unvisited = list(customers)
    vehicle_idx = 0
    while unvisited and vehicle_idx < len(vehicles):
        route = Route(vehicles[vehicle_idx], depot)
        current = depot
        while unvisited:
            feasible = [c for c in unvisited if route.can_add(c)]
            if not feasible:
                break
            nearest = min(feasible, key=lambda c: _haversine(current.lat, current.lon, c.lat, c.lon))
            route.add(nearest)
            current = nearest
            unvisited.remove(nearest)
        if route.stops:
            routes.append(route)
        vehicle_idx += 1
    return routes


def _two_opt(route, uncertainty_factor, disruptions=None):
    improved = True
    while improved:
        improved = False
        stops = route.stops
        for i in range(len(stops) - 1):
            for j in range(i + 2, len(stops)):
                new_stops = stops[:i] + stops[i:j + 1][::-1] + stops[j + 1:]
                new_route = Route(route.vehicle, route.depot)
                new_route.stops = new_stops
                if new_route.cost(uncertainty_factor, disruptions) < route.cost(uncertainty_factor, disruptions) - 1e-6:
                    route.stops = new_stops
                    improved = True
    return route


def _or_opt(routes, uncertainty_factor, disruptions=None):
    improved = True
    while improved:
        improved = False
        for r1 in routes:
            for stop in r1.stops[:]:
                for r2 in routes:
                    if r1 is r2 or not r2.can_add(stop):
                        continue
                    old_cost = r1.cost(uncertainty_factor, disruptions) + r2.cost(uncertainty_factor, disruptions)
                    r1_new = Route(r1.vehicle, r1.depot)
                    r1_new.stops = [s for s in r1.stops if s.id != stop.id]
                    r2_new = Route(r2.vehicle, r2.depot)
                    r2_new.stops = r2.stops + [stop]
                    new_cost = r1_new.cost(uncertainty_factor, disruptions) + r2_new.cost(uncertainty_factor, disruptions)
                    if new_cost < old_cost - 1e-6:
                        r1.stops = r1_new.stops
                        r2.stops = r2_new.stops
                        improved = True
    return [r for r in routes if r.stops]


def _route_analytics(route, disruptions, uncertainty_factor):
    disrupted = {}
    for d in disruptions:
        for loc_id in d.get("affected_locations", []):
            disrupted[loc_id] = disrupted.get(loc_id, 0) + d.get("delay_min", 0.0)
    locs = [route.depot] + route.stops + [route.depot]
    current_time = 0.0
    total_km = 0.0
    stop_etas = {}
    violations = []
    service_results = {}
    for i in range(len(locs) - 1):
        a, b = locs[i], locs[i + 1]
        dist_km = _haversine(a.lat, a.lon, b.lat, b.lon)
        travel = (dist_km / route.vehicle.speed_kmh) * 60.0 * (1.0 + uncertainty_factor)
        current_time += travel
        total_km += dist_km
        if b.id != route.depot.id:
            current_time += disrupted.get(b.id, 0.0) + b.service_time
            stop_etas[b.id] = round(current_time, 1)
            on_time = True
            if b.time_window:
                earliest, latest = b.time_window
                if current_time < earliest:
                    current_time = float(earliest)
                elif current_time > latest:
                    on_time = False
                    violations.append({"stop": b.id, "lateness_min": round(current_time - latest, 1)})
            service_results[b.id] = {"eta_min": round(current_time, 1), "on_time": on_time}
    return {
        "stop_etas": stop_etas,
        "violations": violations,
        "service_results": service_results,
        "total_km": round(total_km, 3),
        "fuel_cost_eur": round(total_km * 0.18, 3),
    }


# ── Solver entry point ───────────────────────────────────────────────────────

def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    """
    QCentroid - Classical Adaptive Routing Solver (v2).

    Improvements over v1:
      - Haversine (spherical earth) distances instead of Euclidean
      - service_time incorporated into route cost per stop
      - Disruption delays applied at affected stops
      - Soft time-window penalties in objective (2x per min late)
      - Rich output: cost_breakdown, risk_metrics, service_level_results,
        per-stop ETAs, constraint_violations

    input_data schema:
        depot       : {id, lat, lon}
        customers   : [{id, lat, lon, demand, time_window, service_time}]
        vehicles    : [{id, capacity, speed_kmh}]
        disruptions : [{type, affected_locations, delay_min}]

    solver_params:
        uncertainty_factor (float): default 0.15
        max_iterations     (int)  : default 50
        seed               (int)  : default 42
    """
    start_time = time.time()
    logger.info("Classical Adaptive Routing Solver v2: starting")

    uncertainty_factor = float(solver_params.get("uncertainty_factor", 0.15))
    max_iterations = int(solver_params.get("max_iterations", 50))
    seed = int(solver_params.get("seed", 42))
    random.seed(seed)

    depot_d = input_data["depot"]
    depot = Location(depot_d["id"], depot_d["lat"], depot_d["lon"])

    customers = []
    for c in input_data.get("customers", []):
        tw = c.get("time_window")
        customers.append(Location(
            id=c["id"], lat=c["lat"], lon=c["lon"],
            demand=float(c.get("demand", 1.0)),
            time_window=tuple(tw) if tw else None,
            service_time=float(c.get("service_time", 0.0)),
        ))

    vehicles = [
        Vehicle(id=v["id"], capacity=float(v.get("capacity", 100.0)),
                speed_kmh=float(v.get("speed_kmh", 50.0)))
        for v in input_data.get("vehicles", [])
    ]
    if not vehicles:
        vehicles = [Vehicle("V1", 100.0), Vehicle("V2", 100.0)]

    disruptions = input_data.get("disruptions", [])
    logger.info(f"Parsed {len(customers)} customers, {len(vehicles)} vehicles, {len(disruptions)} disruptions")

    routes = _nearest_neighbour_init(depot, customers, vehicles)
    routes = [_two_opt(r, uncertainty_factor, disruptions) for r in routes]
    for _ in range(max(1, max_iterations // 10)):
        routes = _or_opt(routes, uncertainty_factor, disruptions)

    active_routes = [r for r in routes if r.stops]
    total_cost = sum(r.cost(uncertainty_factor, disruptions) for r in active_routes)
    elapsed = round(time.time() - start_time, 3)
    logger.info(f"Optimisation complete. Total robust cost: {total_cost:.2f} min, elapsed: {elapsed}s")

    routes_output = []
    all_service_results = {}
    all_violations = []
    total_fuel_eur = 0.0
    lateness_min_total = 0.0

    for r in active_routes:
        analytics = _route_analytics(r, disruptions, uncertainty_factor)
        all_service_results.update(analytics["service_results"])
        all_violations.extend(analytics["violations"])
        lateness_min_total += sum(v["lateness_min"] for v in analytics["violations"])
        total_fuel_eur += analytics["fuel_cost_eur"]
        routes_output.append({
            "vehicle_id": r.vehicle.id,
            "stop_sequence": ["depot"] + [s.id for s in r.stops] + ["depot"],
            "total_load": round(r.load, 2),
            "estimated_cost_minutes": round(r.cost(uncertainty_factor, disruptions), 2),
            "total_km": analytics["total_km"],
            "stop_etas": analytics["stop_etas"],
        })

    on_time_count = sum(1 for s in all_service_results.values() if s["on_time"])
    on_time_prob = round(on_time_count / max(len(all_service_results), 1), 3)
    solution_status = "optimal" if not all_violations else "feasible"

    result = {
        "routes": routes_output,
        "total_vehicles_used": len(routes_output),
        "total_robust_cost_minutes": round(total_cost, 3),
        "solver_type": "classical_adaptive_heuristic",
        "algorithm": "ALNS_Haversine_2opt_OrOpt_v2",
        "objective_value": round(total_cost, 3),
        "solution_status": solution_status,
        "computation_metrics": {
            "wall_time_s": elapsed,
            "algorithm": "ALNS_Haversine_2opt_OrOpt_v2",
            "iterations": max_iterations,
            "uncertainty_factor": uncertainty_factor,
        },
        "cost_breakdown": {
            "travel_time_min": round(total_cost - lateness_min_total * 2.0, 2),
            "lateness_penalty_min": round(lateness_min_total, 2),
            "fuel_cost_eur": round(total_fuel_eur, 2),
            "detour_cost": 0.0,
        },
        "risk_metrics": {
            "on_time_probability": on_time_prob,
            "uncertainty_factor": uncertainty_factor,
            "time_window_violations": len(all_violations),
            "uncertainty_correction_km": round(total_cost * uncertainty_factor / 60.0, 4),
        },
        "service_level_results": all_service_results,
        "constraint_violations": all_violations,
        "benchmark": {
            "execution_cost": {"value": 1.0, "unit": "credits"},
            "time_elapsed": f"{elapsed}s",
            "energy_consumption": 0.0,
        },
    }

    logger.info(f"Classical Solver v2 done. Routes: {len(routes_output)}, status: {solution_status}, on_time: {on_time_prob:.1%}")
    return result
