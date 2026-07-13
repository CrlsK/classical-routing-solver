# Classical Adaptive Routing Solver (v3)

QCentroid solver for the **Real-Time Adaptive Routing Under Uncertainty** use case.

Greedy nearest-neighbour construction, then intra-route **2-opt** and inter-route **or-opt** local search, wrapped in **multi-start restarts**. Optimises a single economic objective in **EUR** (fuel + driver + lateness penalty) under travel-time uncertainty (blended from the base uncertainty model, live traffic congestion and weather feeds) and live disruptions.

## Files
- `qcentroid.py` - entry point `run(input_data, solver_params, extra_arguments)`
- `routing_common.py` - shared parsing, local search, metrics, benchmark schema (identical across solvers)
- `routing_visuals.py` - `additional_output/` artefact generator

## Output
~18 top-level numeric benchmark metrics (objective_value in EUR, distance, on-time rate, utilisation, robustness, capacity/TW violations, ...), a `benchmark` dict for platform charts, and 6 `additional_output` files (ops dashboard, route map, resilience report, executive summary, route_plan.json, kpis.json).

## solver_params
`seed` (int, 42), `restarts` (int, 6).
