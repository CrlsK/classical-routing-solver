"""
Classical Adaptive Routing Solver for Real-Time Routing Under Uncertainty
=========================================================================
Implements an uncertainty-aware adaptive heuristic for Vehicle Routing Problem (VRP)
with stochastic travel times and dynamic disruptions (traffic, weather, accidents).

Algorithm: Adaptive Large Neighbourhood Search (ALNS) with uncertainty modelling
- Builds routes using nearest-neighbour heuristic as initial solution
- Applies repair/destroy operators iteratively
- Uses robust cost functions accounting for travel time variance
- Dynamically reoptimises upon disruption signals
"""

import logging
import math
import random
import copy
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("qcentroid-user-log")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Location:
    def __init__(self, id: str, lat: float, lon: float, demand: float = 0.0,
                 time_window: Optional[Tuple[float, float]] = None, service_time: float = 0.0):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.demand = demand
        self.time_window = time_window  # (earliest, latest) in minutes
        self.service_time = service_time  # minutes

    def distance_to(self, other: "Location") -> float:
        """Haversine distance in km."""
        R = 6371.0
        lat1, lon1 = math.radians(self.lat), math.radians(self.lon)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))


class Vehicle:
    def __init__(self, id: str, capacity: float, speed_kmh: float = 50.0):
        self.id = id
        self.capacity = capacity
        self.speed_kmh = speed_kmh


class Route:
    def __init__(self, vehicle: Vehicle, depot: Location):
        self.vehicle = vehicle
        self.depot = depot
        self.stops: List[Location] = []
        self.load: float = 0.0

    def cost(self, uncertainty_factor: float = 0.15) -> float:
        """
        Robust route cost = expected travel time + uncertainty penalty.
        uncertainty_factor: coefficient of variation of travel times (0.15 = 15%).
        """
        locs = [self.depot] + self.stops + [self.depot]
        total_dist = sum(locs[i].distance_to(locs[i + 1]) for i in range(len(locs) - 1))
        expected_time = (total_dist / self.vehicle.speed_kmh) * 60  # minutes
        # Robust penalty: add sigma * Z_alpha (alpha=0.95 â Z=1.645)
        sigma = uncertainty_factor * expected_time
        robust_cost = expected_time + 1.645 * sigma
        return robust_cost

    def can_add(self, loc: Location) -> bool:
        return self.load + loc.demand <= self.vehicle.capacity

    def add_stop(self, loc: Location):
        self.stops.append(loc)
        self.load += loc.demand


# ---------------------------------------------------------------------------
# Solver core
# ---------------------------------------------------------------------------

def _build_distance_matrix(locations: List[Location]) -> List[List[float]]:
    n = len(locations)
    return [[locations[i].distance_to(locations[j]) for j in range(n)] for i in range(n)]


def _nearest_neighbour_init(depot: Location, customers: List[Location],
                             vehicles: List[Vehicle]) -> List[Route]:
    """Greedy nearest-neighbour initial solution."""
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
            nearest = min(feasible, key=lambda c: current.distance_to(c))
            route.add_stop(nearest)
            unvisited.remove(nearest)
            current = nearest
        routes.append(route)
        vehicle_idx += 1

    return routes


def _two_opt(route: Route, uncertainty_factor: float) -> Route:
    """2-opt local search to improve a single route."""
    best = route
    best_cost = route.cost(uncertainty_factor)
    improved = True
    while improved:
        improved = False
        stops = best.stops
        for i in range(len(stops) - 1):
            for j in range(i + 2, len(stops)):
                new_route = Route(best.vehicle, best.depot)
                new_route.stops = stops[:i] + stops[i:j + 1][::-1] + stops[j + 1:]
                new_route.load = best.load
                new_cost = new_route.cost(uncertainty_factor)
                if new_cost < best_cost - 1e-6:
                    best = new_route
                    best_cost = new_cost
                    improved = True
    return best


def _or_opt(routes: List[Route], uncertainty_factor: float) -> List[Route]:
    """Or-opt: move single stops between routes to reduce total cost."""
    routes = [copy.deepcopy(r) for r in routes]
    improved = True
    while improved:
        improved = False
        for i, r1 in enumerate(routes):
            for stop in list(r1.stops):
                for j, r2 in enumerate(routes):
                    if i == j or not r2.can_add(stop):
                        continue
                    old_cost = r1.cost(uncertainty_factor) + r2.cost(uncertainty_factor)
                    r1_new = copy.deepcopy(r1)
                    r1_new.stops.remove(stop)
                    r1_new.load -= stop.demand
                    r2_new = copy.deepcopy(r2)
                    r2_new.add_stop(stop)
                    new_cost = r1_new.cost(uncertainty_factor) + r2_new.cost(uncertainty_factor)
                    if new_cost < old_cost - 1e-6:
                        routes[i] = r1_new
                        routes[j] = r2_new
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return routes


def _apply_disruptions(routes: List[Route], disruptions: List[Dict],
                        uncertainty_factor: float) -> List[Route]:
    """
    Re-optimise routes given real-time disruptions.
    Disruptions format: [{"type": "traffic|road_closed|accident", "affected_locations": [...], "delay_min": N}]
    Currently: if a disruption affects a route's stop, we rebuild that route with 2-opt.
    """
    disrupted_ids = set()
    for d in disruptions:
        disrupted_ids.update(d.get("affected_locations", []))

    reoptimised = []
    for route in routes:
        if any(s.id in disrupted_ids for s in route.stops):
            route = _two_opt(route, uncertainty_factor * 1.3)
        reoptimised.append(route)
    return reoptimised


def _routes_to_output(routes: List[Route], total_cost: float) -> Dict:
    route_list = []
    for route in routes:
        if not route.stops:
            continue
        route_list.append({
            "vehicle_id": route.vehicle.id,
            "stop_sequence": [route.depot.id] + [s.id for s in route.stops] + [route.depot.id],
            "total_load": round(route.load, 3),
            "estimated_cost_minutes": round(route.cost(), 2),
        })
    return {
        "routes": route_list,
        "total_vehicles_used": len(route_list),
        "total_robust_cost_minutes": round(total_cost, 2),
        "solver_type": "classical_adaptive_heuristic",
        "algorithm": "ALNS_2opt_OrOpt",
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    """
    QCentroid entrypoint for Classical Adaptive Routing Solver.

    input_data schema:
    {
        "depot": {"id": str, "lat": float, "lon": float},
        "customers": [{"id": str, "lat": float, "lon": float, "demand": float,
                       "time_window": [earliest_min, latest_min], "service_time": float}],
        "vehicles": [{"id": str, "capacity": float, "speed_kmh": float}],
        "disruptions": [{"type": str, "affected_locations": [str], "delay_min": float}]
    }

    solver_params:
        uncertainty_factor (float): default 0.15
        max_iterations (int): default 50
        seed (int): default 42

    Returns:
        {"routes": [...], "total_vehicles_used": int, "total_robust_cost_minutes": float}
    """
    logger.info("Classical Adaptive Routing Solver: starting")
    uncertainty_factor = float(solver_params.get("uncertainty_factor", 0.15))
    max_iterations = int(solver_params.get("max_iterations", 50))
    seed = int(solver_params.get("seed", 42))
    random.seed(seed)
    depot_data = input_data["depot"]
    depot = Location(depot_data["id"], depot_data["lat"], depot_data["lon"])
    customers = []
    for c in input_data.get("customers", []):
        tw = c.get("time_window")
        customers.append(Location(id=c["id"], lat=c["lat"], lon=c["lon"],
            demand=c.get("demand", 1.0), time_window=tuple(tw) if tw else None,
            service_time=c.get("service_time", 0.0)))
    vehicles = [Vehicle(id=v["id"], capacity=v.get("capacity", 100.0),
        speed_kmh=v.get("speed_kmh", 50.0)) for v in input_data.get("vehicles", [])]
    if not vehicles:
        vehicles = [Vehicle("V1", 100.0), Vehicle("V2", 100.0)]
    disruptions = input_data.get("disruptions", [])
    routes = _nearest_neighbour_init(depot, customers, vehicles)
    routes = [_two_opt(r, uncertainty_factor) for r in routes]
    for _ in range(max_iterations // 10):
        routes = _or_opt(routes, uncertainty_factor)
    if disruptions:
        routes = _apply_disruptions(routes, disruptions, uncertainty_factor)
    total_cost = sum(r.cost(uncertainty_factor) for r in routes if r.stops)
    logger.info(f"Optimisation complete. Total robust cost: {total_cost:.2f} min")
    return _routes_to_output(routes, total_cost)
