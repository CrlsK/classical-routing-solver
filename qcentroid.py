"""
QCentroid - Classical Adaptive Routing Solver (v3.2)

Metaheuristic VRP-under-uncertainty solver: greedy nearest-neighbour
construction -> intra-route 2-opt + inter-route or-opt + inter-route swap local
search (shared core) -> multi-start restarts keep the best incumbent.

All parsing, local search, metrics and additional-output generation live in the
shared modules routing_common.py / routing_visuals.py, so this solver and the
Quantum-Inspired solver emit an identical, enriched benchmark schema.

Entry point:  run(input_data, solver_params, extra_arguments)
"""
import time
import logging

import routing_common as rc
from routing_visuals import generate_additional_output

logger = logging.getLogger("qcentroid-user-log")

SOLVER_VERSION = "3.2.0"


def run(input_data, solver_params, extra_arguments):
    start = time.time()
    logger.info("Classical Adaptive Routing Solver v3.2: starting")

    seed = int(solver_params.get("seed", 42))
    restarts = int(solver_params.get("restarts", 6))

    prob = rc.parse_input(input_data)
    uncertainty = rc.effective_uncertainty(prob)
    disrupted = rc.disruption_map(prob["disruptions"])
    logger.info(f"Parsed {len(prob['customers'])} customers, "
                f"{len(prob['vehicles'])} vehicles, {len(prob['disruptions'])} disruptions")

    # Greedy construction + 2-opt/or-opt/swap local search + multi-start restarts
    best, best_obj = rc.greedy_multistart(prob, uncertainty, disrupted, restarts, seed)

    elapsed = time.time() - start
    result = rc.build_result(
        best, prob, input_data, elapsed,
        {
            "solver_type": "classical_metaheuristic",
            "algorithm": "Greedy+2opt+OrOpt+Swap+MultiStart_v32",
            "solver_version": SOLVER_VERSION,
            "credits": 1.0,
            "extra_metrics": {"restarts": restarts, "seed": seed},
        },
    )
    logger.info(f"Classical v3.2 done: cost EUR {result['objective_value']}, "
                f"vehicles {result['num_vehicles_used']}, "
                f"on-time {result['on_time_delivery_rate_pct']}%, {elapsed:.2f}s")
    generate_additional_output(input_data, result, prob)
    return result
