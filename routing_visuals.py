"""
routing_visuals.py - QCentroid additional_output generator for the
Real-Time Adaptive Routing Under Uncertainty use case.

The use-case team defined ONE set of additional-output artefacts that every
solver must emit, so results are comparable across solvers and each audience
gets the view it needs:

  additional_output/ops_dashboard.html     - Dispatch control room (planners)
  additional_output/route_map.html         - Geographic route + disruption map (logistics)
  additional_output/resilience_report.html - Risk / uncertainty / SLA view (risk & SLA owners)
  additional_output/executive_summary.html - One-page business view (executives)
  additional_output/route_plan.json        - Machine-readable operational plan (WMS/ERP)
  additional_output/kpis.json              - Flat numeric KPIs (BI dashboards)

All HTML is self-contained (inline CSS/SVG, no external assets) so the
platform's inline previewer renders it. Everything is wrapped defensively -
visualisation must never break a solver run.
"""
import os
import json
import math
import logging

logger = logging.getLogger("qcentroid-user-log")

NAVY = "#0B1F3A"
BLUE = "#1E6FE8"
CYAN = "#17C3E6"
GREEN = "#23B26D"
AMBER = "#F0A020"
RED = "#E5484D"
GREY = "#8A94A6"
_PAL = [BLUE, GREEN, AMBER, "#9B59B6", CYAN, "#E67E22", "#1ABC9C", "#E74C3C"]


def generate_additional_output(input_data, result, prob=None):
    """Write every additional-output artefact. Safe: never raises."""
    try:
        os.makedirs("additional_output", exist_ok=True)
        depot = (prob or {}).get("depot") or input_data.get("depot") or {}
        customers = (prob or {}).get("customers") or input_data.get("customers") \
            or input_data.get("stops") or []
        disruptions = input_data.get("disruptions", []) or []
        routes = result.get("routes", [])

        _write("additional_output/ops_dashboard.html",
                _ops_dashboard(input_data, result))
        _write("additional_output/route_map.html",
                _route_map(depot, customers, routes, disruptions, result))
        _write("additional_output/resilience_report.html",
                _resilience(input_data, result))
        _write("additional_output/executive_summary.html",
                _executive(input_data, result))
        _write("additional_output/route_plan.json",
                json.dumps(_route_plan(result), indent=2))
        _write("additional_output/kpis.json",
                json.dumps(_kpis(result), indent=2))
        logger.info("additional_output: 6 artefacts written")
    except Exception as exc:
        logger.warning(f"additional_output generation skipped: {exc}")


def _write(path, content):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# -- Shared chrome ------------------------------------------------------------

def _head(title):
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#eef2f7;color:{NAVY}}}
.kick{{background:{NAVY};color:#fff;padding:14px 22px;display:flex;align-items:center;
 gap:12px;flex-wrap:wrap}}
.kick h1{{font-size:16px;font-weight:650}}
.kick .tag{{margin-left:auto;font-size:11px;opacity:.7}}
.badge{{padding:3px 11px;border-radius:12px;font-size:11px;font-weight:700;color:#fff}}
.grid{{display:flex;flex-wrap:wrap;gap:12px;padding:16px}}
.kpi{{background:#fff;border-radius:10px;padding:14px 18px;flex:1 1 120px;
 box-shadow:0 1px 4px rgba(0,0,0,.08);text-align:center}}
.kpi .v{{font-size:22px;font-weight:750}}
.kpi .l{{font-size:11px;color:{GREY};margin-top:3px}}
.card{{background:#fff;border-radius:10px;padding:16px;margin:0 16px 16px;
 box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.card h3{{font-size:12px;color:{GREY};text-transform:uppercase;letter-spacing:.5px;
 margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#f5f7fa;padding:8px 10px;text-align:left;font-size:11px;color:#555}}
td{{padding:7px 10px;border-bottom:1px solid #f0f2f5}}
tr:last-child td{{border:none}}
.foot{{padding:10px 22px;font-size:11px;color:{GREY}}}
</style></head><body>"""


def _kick(icon, title, badge_text=None, badge_color=BLUE, tag=""):
    b = f'<span class="badge" style="background:{badge_color}">{badge_text}</span>' if badge_text else ""
    return (f'<div class="kick"><h1>{icon} {title}</h1>{b}'
            f'<span class="tag">{tag}</span></div>')


def _foot(result):
    return (f'<div class="foot">QCentroid - Real-Time Adaptive Routing Under Uncertainty - '
            f'{result.get("algorithm","")} - v{result.get("solver_version","")} - '
            f'sha {result.get("dataset_sha256","")}</div></body></html>')


def _kpi(v, l, color=NAVY):
    return f'<div class="kpi"><div class="v" style="color:{color}">{v}</div><div class="l">{l}</div></div>'


# -- 1. Ops dashboard ---------------------------------------------------------

def _ops_dashboard(input_data, result):
    routes = result.get("routes", [])
    status = result.get("solution_status", "N/A")
    sc = GREEN if status == "optimal" else AMBER
    kpis = "".join([
        _kpi(result.get("num_vehicles_used", 0), "Vehicles", BLUE),
        _kpi(f'{result.get("total_distance_km",0):.1f}', "Total km", "#9B59B6"),
        _kpi(f'{result.get("total_robust_time_min",0):.0f}', "Robust time (min)", AMBER),
        _kpi(f'{result.get("on_time_delivery_rate_pct",0):.0f}%', "On-time", GREEN),
        _kpi(f'EUR {result.get("total_cost_eur",0):.0f}', "Total cost", CYAN),
        _kpi(f'{result.get("avg_capacity_utilization_pct",0):.0f}%', "Avg util", NAVY),
        _kpi(result.get("time_window_violations", 0), "TW breaches", RED),
        _kpi(f'{result.get("compute_time_s",0):.2f}s', "Compute", GREY),
    ])
    rows = []
    for i, r in enumerate(routes):
        col = _PAL[i % len(_PAL)]
        seq = " -> ".join(r.get("stop_sequence", []))
        rows.append(
            f'<tr><td><span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{col};margin-right:6px"></span><b>{r.get("vehicle_id")}</b></td>'
            f'<td>{r.get("num_stops",0)}</td><td>{r.get("total_load",0)}</td>'
            f'<td>{r.get("capacity_utilization_pct",0)}%</td>'
            f'<td>{r.get("distance_km",0):.2f}</td><td>{r.get("route_time_min",0):.1f}</td>'
            f'<td>{r.get("lateness_min",0):.1f}</td>'
            f'<td style="color:#888;font-size:11px;max-width:320px;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis">{seq}</td></tr>')
    return (_head("Dispatch Dashboard")
            + _kick("&#128678;", "Dispatch Control Room", status.upper(), sc,
                    result.get("algorithm", ""))
            + f'<div class="grid">{kpis}</div>'
            + '<div class="card"><h3>Route plan</h3><table><thead><tr>'
              '<th>Vehicle</th><th>Stops</th><th>Load</th><th>Util</th><th>km</th>'
              '<th>Time</th><th>Late</th><th>Sequence</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>'
            + _foot(result))


# -- 2. Route map (SVG) -------------------------------------------------------

def _route_map(depot, customers, routes, disruptions, result=None):
    result = result or result_stub()
    nodes = [{"id": depot.get("id", "depot"), "lat": float(depot.get("lat", 0)),
              "lon": float(depot.get("lon", 0)), "type": "depot", "demand": 0}]
    for c in customers:
        nodes.append({"id": c["id"], "lat": float(c["lat"]), "lon": float(c["lon"]),
                      "type": "cust", "demand": c.get("demand", 0)})
    nmap = {n["id"]: n for n in nodes}
    lats = [n["lat"] for n in nodes]
    lons = [n["lon"] for n in nodes]
    if not lats:
        return _head("Route Map") + "<p style='padding:20px'>No geography.</p></body></html>"
    W, H, pad = 820, 540, 0.12
    lat_r = max(max(lats) - min(lats), 0.01) * (1 + 2 * pad)
    lon_r = max(max(lons) - min(lons), 0.01) * (1 + 2 * pad)
    min_lat = min(lats) - lat_r * pad / (1 + 2 * pad)
    min_lon = min(lons) - lon_r * pad / (1 + 2 * pad)

    def proj(lat, lon):
        return (round((lon - min_lon) / lon_r * W, 1),
                round(H - (lat - min_lat) / lat_r * H, 1))

    disrupted = set()
    for d in disruptions:
        for lid in d.get("affected_locations", []):
            disrupted.add(lid)

    svg = []
    for i in range(1, 5):
        svg.append(f'<line x1="0" y1="{H*i//4}" x2="{W}" y2="{H*i//4}" stroke="#e2e8f0"/>')
        svg.append(f'<line x1="{W*i//4}" y1="0" x2="{W*i//4}" y2="{H}" stroke="#e2e8f0"/>')
    for i, r in enumerate(routes):
        col = _PAL[i % len(_PAL)]
        pts = []
        for sid in r.get("stop_sequence", []):
            n = nmap.get(sid)
            if n:
                pts.append("%s,%s" % proj(n["lat"], n["lon"]))
        if len(pts) > 1:
            svg.append(f'<polyline points="{" ".join(pts)}" stroke="{col}" '
                       f'stroke-width="3" fill="none" opacity=".85" stroke-linejoin="round"/>')
    for n in nodes:
        if n["type"] != "cust":
            continue
        x, y = proj(n["lat"], n["lon"])
        col = GREY
        for i, r in enumerate(routes):
            if n["id"] in r.get("stop_sequence", []):
                col = _PAL[i % len(_PAL)]
                break
        svg.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{col}" stroke="#fff" stroke-width="2.5">'
                   f'<title>{n["id"]} - demand {n["demand"]}</title></circle>')
        if n["id"] in disrupted:
            svg.append(f'<circle cx="{x}" cy="{y}" r="15" fill="none" stroke="{RED}" '
                       f'stroke-width="2.5" stroke-dasharray="4 3"><title>disruption at {n["id"]}</title></circle>')
        svg.append(f'<text x="{x}" y="{y-13}" text-anchor="middle" font-size="9" fill="#444">{n["id"]}</text>')
    dx, dy = proj(depot.get("lat", 0), depot.get("lon", 0))
    svg.append(f'<rect x="{dx-11}" y="{dy-11}" width="22" height="22" rx="4" fill="{NAVY}" '
               f'stroke="#fff" stroke-width="2.5"><title>Depot</title></rect>')
    svg.append(f'<text x="{dx}" y="{dy+5}" text-anchor="middle" font-size="10" fill="#fff" font-weight="700">D</text>')
    legend = []
    for i, r in enumerate(routes):
        col = _PAL[i % len(_PAL)]
        legend.append(f'<tr><td><span style="display:inline-block;width:20px;height:4px;'
                      f'background:{col};border-radius:2px"></span></td>'
                      f'<td style="padding:2px 8px"><b>{r.get("vehicle_id")}</b></td>'
                      f'<td style="padding:2px 8px;color:#666">{r.get("num_stops",0)} stops</td>'
                      f'<td style="padding:2px 8px;color:#666">{r.get("distance_km",0):.1f} km</td></tr>')
    return (_head("Route Map")
            + _kick("&#128506;", "Adaptive Route Map",
                    f"{len(disrupted)} disruptions", RED if disrupted else GREEN,
                    f"{len(customers)} customers - {len(routes)} routes")
            + f'<div style="padding:16px;display:flex;flex-wrap:wrap;gap:16px">'
              f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
              f'style="background:#f8fafc;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.12)">'
            + "".join(svg) + '</svg>'
            + f'<div class="card" style="margin:0;min-width:230px"><h3>Vehicle routes</h3>'
              f'<table>{"".join(legend)}</table>'
              f'<p style="margin-top:10px;font-size:11px;color:{GREY}">'
              f'dashed ring = live disruption</p></div></div>'
            + _foot(result))


def result_stub():
    return {"algorithm": "", "solver_version": "", "dataset_sha256": ""}


# -- 3. Resilience / risk report ----------------------------------------------

def _resilience(input_data, result):
    rm = result.get("risk_metrics", {})
    viol = result.get("constraint_violations", [])
    on_time = result.get("on_time_delivery_rate_pct", 0)
    robust = result.get("solution_robustness_pct", 0)
    sla_ok = result.get("sla_compliant", 0)
    bar_rows = []
    for name, val, target, good_high in [
        ("On-time delivery %", on_time, 95, True),
        ("Robustness %", robust, 90, True),
        ("Capacity util %", result.get("avg_capacity_utilization_pct", 0), 70, True),
        ("Demand fulfilment %", result.get("demand_fulfillment_rate_pct", 0), 100, True),
    ]:
        pct = max(0, min(100, val))
        col = GREEN if (val >= target) == good_high else AMBER
        bar_rows.append(
            f'<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;'
            f'font-size:12px;margin-bottom:3px"><span>{name}</span><b>{val:.1f}</b></div>'
            f'<div style="background:#eef2f7;border-radius:6px;height:14px">'
            f'<div style="width:{pct}%;background:{col};height:14px;border-radius:6px"></div></div>'
            f'<div style="font-size:10px;color:{GREY}">target {target}</div></div>')
    vrows = "".join(
        f'<tr><td>{v.get("stop")}</td><td style="color:{RED}">+{v.get("lateness_min")} min late</td></tr>'
        for v in viol) or f'<tr><td colspan="2" style="color:{GREEN}">No time-window breaches</td></tr>'
    disruptions = input_data.get("disruptions", []) or []
    drows = "".join(
        f'<tr><td>{d.get("type")}</td><td>{", ".join(d.get("affected_locations",[]))}</td>'
        f'<td>+{d.get("delay_min",0)} min</td></tr>' for d in disruptions) \
        or '<tr><td colspan="3">No active disruptions</td></tr>'
    return (_head("Resilience Report")
            + _kick("&#128737;", "Resilience & Risk Report",
                    "SLA MET" if sla_ok else "SLA AT RISK", GREEN if sla_ok else RED,
                    f'uncertainty {rm.get("effective_uncertainty_factor",0)}')
            + '<div class="card"><h3>Service-level performance</h3>' + "".join(bar_rows) + '</div>'
            + '<div class="card"><h3>Time-window breaches</h3><table><thead><tr>'
              '<th>Stop</th><th>Impact</th></tr></thead><tbody>' + vrows + '</tbody></table></div>'
            + '<div class="card"><h3>Active disruptions absorbed</h3><table><thead><tr>'
              '<th>Type</th><th>Locations</th><th>Delay</th></tr></thead><tbody>' + drows
            + '</tbody></table></div>'
            + _foot(result))


# -- 4. Executive summary -----------------------------------------------------

def _executive(input_data, result):
    cost = result.get("total_cost_eur", 0)
    km = result.get("total_distance_km", 0)
    on_time = result.get("on_time_delivery_rate_pct", 0)
    veh = result.get("num_vehicles_used", 0)
    cb = result.get("cost_breakdown_eur", {})
    parts = [("Fuel", cb.get("fuel", 0), BLUE), ("Driver", cb.get("driver", 0), GREEN),
             ("Lateness", cb.get("lateness_penalty", 0), RED)]
    tot = sum(p[1] for p in parts) or 1
    cx, cy, ro, ri = 90, 90, 74, 44
    a0 = -math.pi / 2
    segs = []
    for name, val, col in parts:
        a1 = a0 + 2 * math.pi * val / tot
        if a1 - a0 > 0.001:
            x1, y1 = cx + ro * math.cos(a0), cy + ro * math.sin(a0)
            x2, y2 = cx + ro * math.cos(a1), cy + ro * math.sin(a1)
            xi, yi = cx + ri * math.cos(a1), cy + ri * math.sin(a1)
            xj, yj = cx + ri * math.cos(a0), cy + ri * math.sin(a0)
            lg = 1 if a1 - a0 > math.pi else 0
            segs.append(f'<path d="M{x1:.1f},{y1:.1f} A{ro},{ro} 0 {lg} 1 {x2:.1f},{y2:.1f} '
                        f'L{xi:.1f},{yi:.1f} A{ri},{ri} 0 {lg} 0 {xj:.1f},{yj:.1f} Z" '
                        f'fill="{col}" opacity=".92"><title>{name}: EUR {val:.2f}</title></path>')
        a0 = a1
    legend = "".join(
        f'<div style="font-size:12px;margin:4px 0"><span style="display:inline-block;width:11px;'
        f'height:11px;border-radius:2px;background:{col};margin-right:6px"></span>{name}: '
        f'<b>EUR {val:.2f}</b></div>' for name, val, col in parts)
    kpis = "".join([
        _kpi(f'EUR {cost:,.0f}', "Total plan cost", NAVY),
        _kpi(f'{on_time:.0f}%', "On-time delivery", GREEN),
        _kpi(f'{km:.0f} km', "Distance", BLUE),
        _kpi(veh, "Vehicles used", CYAN),
        _kpi(f'{result.get("solution_robustness_pct",0):.0f}%', "Robustness", AMBER),
    ])
    return (_head("Executive Summary")
            + _kick("&#128200;", "Executive Summary", result.get("solution_status", "").upper(),
                    GREEN if result.get("solution_status") == "optimal" else AMBER,
                    result.get("algorithm", ""))
            + f'<div class="grid">{kpis}</div>'
            + '<div class="card"><h3>Cost composition</h3>'
              f'<div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap">'
              f'<svg width="180" height="180">{"".join(segs)}'
              f'<text x="{cx}" y="{cy-4}" text-anchor="middle" font-size="10" fill="{GREY}">Total</text>'
              f'<text x="{cx}" y="{cy+13}" text-anchor="middle" font-size="15" font-weight="700" '
              f'fill="{NAVY}">EUR {tot:.0f}</text></svg><div>{legend}</div></div></div>'
            + _foot(result))


# -- 5 & 6. Machine-readable --------------------------------------------------

def _route_plan(result):
    return {
        "solver": result.get("algorithm"),
        "status": result.get("solution_status"),
        "generated_at_utc": result.get("run_started_at_utc"),
        "routes": [
            {"vehicle_id": r.get("vehicle_id"),
             "stop_sequence": r.get("stop_sequence"),
             "load": r.get("total_load"),
             "distance_km": r.get("distance_km"),
             "route_time_min": r.get("route_time_min"),
             "etas": r.get("stop_etas")}
            for r in result.get("routes", [])
        ],
    }


def _kpis(result):
    keys = ["objective_value", "total_cost_eur", "total_distance_km",
            "total_robust_time_min", "total_fuel_cost_eur", "total_driver_cost_eur",
            "num_vehicles_used", "on_time_delivery_rate_pct",
            "demand_fulfillment_rate_pct", "avg_capacity_utilization_pct",
            "max_lateness_min", "time_window_violations", "max_route_time_min",
            "solution_robustness_pct", "disruptions_absorbed", "compute_time_s",
            "sla_compliant"]
    return {k: result.get(k) for k in keys}
