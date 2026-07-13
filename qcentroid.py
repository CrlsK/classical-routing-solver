"""
QCentroid - Classical Adaptive Routing Solver (v3)

Metaheuristic VRP-under-uncertainty solver:
  greedy nearest-neighbour construction
  -> intra-route 2-opt + inter-route or-opt local search (shared core)
  -> multi-start restarts keep the best incumbent.

All parsing, local search, metrics and additional-output generation live in
the shared modules routing_common.py / routing_visuals.py, so this solver and
the Quantum-Inspired solver emit an identical, enriched benchmark schema
(objective_value in EUR + ~17 top-level numeric metrics + benchmark dict).

Entry point signature required by QCentroid:  run(input_data, solver_params, extra_arguments)
"""
import time
import random
import logging

import routing_common as rc
from routing_visuals import generate_additional_output

logger = logging.getLogger("qcentroid-user-log")

SOLVER_VERSION = "3.0.0"


def _perturb(assignment, prob, rng):
    """Random relocation kick to diversify a restart (capacity-respecting)."""
    locs = {c["id"]: c for c in prob["customers"]}
    caps = {k: prob["vehicles"][k]["capacity"] for k in assignment}
    keys = [k for k in assignment if assignment[k]]
    if len(keys) < 2:
        return assignment
    k1 = rng.choice(keys)
    stop = rng.choice(assignment[k1])
    demand = locs[stop]["demand"]
    # only move to a vehicle that still has room (keeps restarts feasible)
    feasible_k2 = [k for k in assignment if k != k1
                   and sum(locs[s]["demand"] for s in assignment[k]) + demand <= caps[k]]
    if not feasible_k2:
        return assignment
    k2 = rng.choice(feasible_k2)
    assignment[k1] = [s for s in assignment[k1] if s != stop]
    pos = rng.randint(0, len(assignment[k2]))
    assignment[k2] = assignment[k2][:pos] + [stop] + assignment[k2][pos:]
    return assignment


def run(input_data, solver_params, extra_arguments):
    start = time.time()
    logger.info("Classical Adaptive Routing Solver v3: starting")

    seed = int(solver_params.get("seed", 42))
    restarts = int(solver_params.get("restarts", 6))
    rng = random.Random(seed)

    prob = rc.parse_input(input_data)
    uncertainty = rc.effective_uncertainty(prob)
    disrupted = rc.disruption_map(prob["disruptions"])
    logger.info(f"Parsed {len(prob['customers'])} customers, "
                f"{len(prob['vehicles'])} vehicles, {len(prob['disruptions'])} disruptions")

    # Multi-start greedy + local search
    base = rc.nearest_neighbour_init(prob)
    best = rc.polish({k: list(v) for k, v in base.items()}, prob, uncertainty, disrupted)
    best_obj = rc.assignment_objective(best, prob, uncertainty, disrupted)

    for _ in range(max(0, restarts - 1)):
        cand = _perturb({k: list(v) for k, v in best.items()}, prob, rng)
        cand = rc.polish(cand, prob, uncertainty, disrupted)
        obj = rc.assignment_objective(cand, prob, uncertainty, disrupted)
        if obj < best_obj - 1e-6:
            best, best_obj = cand, obj

    elapsed = time.time() - start
    result = rc.build_result(
        best, prob, input_data, elapsed,
        {
            "solver_type": "classical_metaheuristic",
            "algorithm": "Greedy+2opt+OrOpt+MultiStart_v3",
            "solver_version": SOLVER_VERSION,
            "credits": 1.0,
            "extra_metrics": {"restarts": restarts, "seed": seed},
        },
    )
    logger.info(f"Classical v3 done: cost EUR {result['objective_value']}, "
                f"vehicles {result['num_vehicles_used']}, "
                f"on-time {result['on_time_delivery_rate_pct']}%, {elapsed:.2f}s")
    generate_additional_output(input_data, result, prob)
    return result
