"""
routing_visuals.py - QCentroid additional_output generator for the
Real-Time Adaptive Routing Under Uncertainty use case (v3.1).

Team-standardised artefacts every solver emits (comparable across solvers,
one native view per audience):

  additional_output/ops_dashboard.html     - Dispatch control room (planners)
  additional_output/timeline.html          - ETA vs time-window schedule (dispatchers)  [NEW]
  additional_output/route_map.html         - Geographic routes + disruption overlay (logistics)
  additional_output/resilience_report.html - Risk / uncertainty / SLA (risk & SLA owners)
  additional_output/executive_summary.html - One-page business view w/ savings & CO2 (execs)
  additional_output/route_plan.json         - Machine-readable plan (WMS/ERP)
  additional_output/kpis.json               - Flat numeric KPIs (BI)

All HTML is self-contained (inline CSS/SVG). Generation is defensive - a
visualisation error can never fail a solver run.
"""
import os
import json
import math
import logging

logger = logging.getLogger("qcentroid-user-log")

# -- Design tokens ------------------------------------------------------------
NAVY = "#0B1F3A"
INK = "#12233b"
BLUE = "#1E6FE8"
CYAN = "#17C3E6"
GREEN = "#1FB268"
AMBER = "#F0A020"
RED = "#E5484D"
VIOLET = "#7A5AF8"
GREY = "#8A94A6"
LINE = "#E6EBF1"
BG = "#F4F7FB"
_PAL = [BLUE, GREEN, AMBER, VIOLET, CYAN, "#E8618C", "#12A594", "#E5484D"]


def generate_additional_output(input_data, result, prob=None):
    """Write every additional-output artefact. Safe: never raises."""
    try:
        os.makedirs("additional_output", exist_ok=True)
        depot = (prob or {}).get("depot") or input_data.get("depot") or {}
        customers = (prob or {}).get("customers") or input_data.get("customers") \
            or input_data.get("stops") or []
        disruptions = input_data.get("disruptions", []) or []
        routes = result.get("routes", [])
        cust_ix = _cust_index(customers)

        files = {
            "ops_dashboard.html": _ops_dashboard(input_data, result),
            "timeline.html": _timeline(input_data, result, cust_ix),
            "route_map.html": _route_map(depot, customers, routes, disruptions, result),
            "resilience_report.html": _resilience(input_data, result, cust_ix),
            "executive_summary.html": _executive(input_data, result),
            "route_plan.json": json.dumps(_route_plan(result), indent=2),
            "kpis.json": json.dumps(_kpis(result), indent=2),
        }
        for name, content in files.items():
            with open(f"additional_output/{name}", "w", encoding="utf-8") as fh:
                fh.write(content)
        logger.info(f"additional_output: {len(files)} artefacts written")
    except Exception as exc:
        logger.warning(f"additional_output generation skipped: {exc}")


def _cust_index(customers):
    ix = {}
    for c in customers:
        ix[c["id"]] = c
    return ix


# -- Shared chrome ------------------------------------------------------------

def _head(title):
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:{BG};color:{INK};
 -webkit-font-smoothing:antialiased}}
.kick{{background:linear-gradient(100deg,{NAVY},#14335c);color:#fff;padding:16px 24px;
 display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.kick .dot{{width:9px;height:9px;border-radius:50%;background:{CYAN};box-shadow:0 0 0 4px rgba(23,195,230,.25)}}
.kick h1{{font-size:16px;font-weight:650;letter-spacing:.2px}}
.kick .tag{{margin-left:auto;font-size:11px;opacity:.72;font-variant:all-small-caps;letter-spacing:.6px}}
.badge{{padding:3px 11px;border-radius:20px;font-size:11px;font-weight:700;color:#fff}}
.wrap{{padding:18px;max-width:1180px;margin:0 auto}}
.grid{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}}
.kpi{{background:#fff;border:1px solid {LINE};border-radius:12px;padding:14px 16px;flex:1 1 128px;
 box-shadow:0 1px 2px rgba(16,35,59,.04)}}
.kpi .v{{font-size:23px;font-weight:750;letter-spacing:-.4px}}
.kpi .l{{font-size:11px;color:{GREY};margin-top:3px;font-weight:600}}
.kpi .s{{font-size:10px;color:{GREY};margin-top:1px}}
.card{{background:#fff;border:1px solid {LINE};border-radius:14px;padding:18px;margin-bottom:16px;
 box-shadow:0 1px 2px rgba(16,35,59,.04)}}
.card h3{{font-size:12px;color:{GREY};text-transform:uppercase;letter-spacing:.6px;margin-bottom:14px;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{background:#f7f9fc;padding:9px 11px;text-align:left;font-size:11px;color:#5a6472;font-weight:700;
 border-bottom:1px solid {LINE}}}
td{{padding:9px 11px;border-bottom:1px solid #f1f4f8}}
tr:last-child td{{border:none}}
.pill{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}}
.bar{{background:#eef2f7;border-radius:7px;height:9px;overflow:hidden}}
.bar>i{{display:block;height:9px;border-radius:7px}}
.foot{{padding:12px 24px;font-size:11px;color:{GREY};text-align:center}}
</style></head><body>"""


def _kick(icon, title, badge=None, badge_color=BLUE, tag=""):
    b = f'<span class="badge" style="background:{badge_color}">{badge}</span>' if badge else ""
    return (f'<div class="kick"><span class="dot"></span><h1>{icon} {title}</h1>{b}'
            f'<span class="tag">{tag}</span></div>')


def _foot(result):
    return (f'<div class="foot">QCentroid &middot; Real-Time Adaptive Routing Under Uncertainty '
            f'&middot; {result.get("algorithm","")} &middot; v{result.get("solver_version","")} '
            f'&middot; sha {result.get("dataset_sha256","")}</div></body></html>')


def _kpi(v, l, sub="", color=INK):
    s = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="kpi"><div class="v" style="color:{color}">{v}</div><div class="l">{l}</div>{s}</div>'


def _status_color(result):
    st = result.get("solution_status", "")
    return GREEN if st == "optimal" else (RED if st == "infeasible" else AMBER)


# -- 1. Ops dashboard ---------------------------------------------------------

def _ops_dashboard(input_data, result):
    routes = result.get("routes", [])
    kpis = "".join([
        _kpi(result.get("num_vehicles_used", 0), "Vehicles used"),
        _kpi(f'{result.get("total_distance_km",0):.0f}', "Distance", "km", VIOLET),
        _kpi(f'{result.get("total_robust_time_min",0):.0f}', "Robust time", "min", AMBER),
        _kpi(f'{result.get("on_time_delivery_rate_pct",0):.0f}%', "On-time", "SLA &ge; 95%", GREEN),
        _kpi(f'&euro;{result.get("total_cost_eur",0):.0f}', "Plan cost", "fuel+driver+late", CYAN),
        _kpi(f'{result.get("avg_capacity_utilization_pct",0):.0f}%', "Avg util", "", BLUE),
        _kpi(result.get("time_window_violations", 0), "TW breaches", "", RED),
        _kpi(f'{result.get("compute_time_s",0):.2f}s', "Compute", ""),
    ])
    rows = []
    for i, r in enumerate(routes):
        col = _PAL[i % len(_PAL)]
        util = r.get("capacity_utilization_pct", 0)
        seq = " &rarr; ".join(r.get("stop_sequence", []))
        rows.append(
            f'<tr><td><span class="pill" style="background:{col}"></span><b>{r.get("vehicle_id")}</b></td>'
            f'<td>{r.get("num_stops",0)}</td><td>{r.get("total_load",0)}</td>'
            f'<td style="min-width:96px"><div class="bar"><i style="width:{min(util,100)}%;background:{col}"></i></div>'
            f'<span style="font-size:10px;color:{GREY}">{util}%</span></td>'
            f'<td>{r.get("distance_km",0):.1f}</td><td>{r.get("route_time_min",0):.0f}</td>'
            f'<td style="color:{RED if r.get("lateness_min",0)>0 else GREY}">{r.get("lateness_min",0):.0f}</td>'
            f'<td style="color:#8a94a6;font-size:11px;max-width:300px;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis">{seq}</td></tr>')
    return (_head("Dispatch Dashboard")
            + _kick("&#128678;", "Dispatch Control Room", result.get("solution_status", "").upper(),
                    _status_color(result), result.get("algorithm", ""))
            + '<div class="wrap"><div class="grid">' + kpis + '</div>'
            + '<div class="card"><h3>Route plan</h3><table><thead><tr>'
              '<th>Vehicle</th><th>Stops</th><th>Load</th><th>Utilisation</th><th>km</th>'
              '<th>Min</th><th>Late</th><th>Sequence</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div></div>'
            + _foot(result))


# -- 2. Timeline (ETA vs time window) - NEW dispatcher view -------------------

def _timeline(input_data, result, cust_ix):
    routes = result.get("routes", [])
    horizon = float(input_data.get("planning_horizon_minutes", 0) or 0)
    max_eta = 0.0
    for r in routes:
        for v in (r.get("stop_etas") or {}).values():
            max_eta = max(max_eta, float(v))
    for c in cust_ix.values():
        tw = c.get("time_window")
        if tw:
            max_eta = max(max_eta, float(tw[1]))
    horizon = max(horizon, max_eta) or 480.0
    horizon = math.ceil(horizon / 60.0) * 60.0

    W, rowH, padL, padR = 1080, 30, 118, 24
    plot_w = W - padL - padR

    def x(t):
        return padL + plot_w * (t / horizon)

    grid = []
    h = 0
    while h <= horizon:
        gx = x(h)
        grid.append(f'<line x1="{gx:.1f}" y1="34" x2="{gx:.1f}" y2="{{H}}" stroke="{LINE}"/>')
        grid.append(f'<text x="{gx:.1f}" y="24" text-anchor="middle" font-size="10" '
                    f'fill="{GREY}">{int(h//60):02d}:00</text>')
        h += 60

    body = []
    y = 44
    legend_late = False
    for i, r in enumerate(routes):
        col = _PAL[i % len(_PAL)]
        vid = r.get("vehicle_id", f"V{i+1}")
        etas = r.get("stop_etas") or {}
        seq = [s for s in r.get("stop_sequence", []) if s in cust_ix]
        body.append(f'<text x="10" y="{y+rowH*0.62:.0f}" font-size="12" font-weight="700" '
                    f'fill="{INK}">{vid}</text>')
        body.append(f'<text x="10" y="{y+rowH*0.62+13:.0f}" font-size="9.5" '
                    f'fill="{GREY}">{len(seq)} stops</text>')
        body.append(f'<rect x="{padL}" y="{y+rowH*0.5-1:.0f}" width="{plot_w}" height="2" fill="#eef2f7"/>')
        for sid in seq:
            c = cust_ix.get(sid, {})
            tw = c.get("time_window")
            svc = float(c.get("service_time", 0) or 0)
            eta = float(etas.get(sid, 0))
            cy = y + rowH * 0.5
            if tw:
                bx, bw = x(float(tw[0])), x(float(tw[1])) - x(float(tw[0]))
                body.append(f'<rect x="{bx:.1f}" y="{cy-9:.0f}" width="{max(bw,1):.1f}" height="18" '
                            f'rx="4" fill="{col}" opacity="0.10"/>')
            late = tw and eta > float(tw[1]) + 1e-6
            if late:
                legend_late = True
            mcol = RED if late else col
            sw = max(x(eta + svc) - x(eta), 3)
            body.append(f'<rect x="{x(eta):.1f}" y="{cy-6:.0f}" width="{sw:.1f}" height="12" rx="3" '
                        f'fill="{mcol}"><title>{sid}: ETA {eta:.0f}m'
                        + (f', window {int(tw[0])}-{int(tw[1])}' if tw else '') + f', service {svc:.0f}m'
                        + (' - LATE' if late else '') + f'</title></rect>')
            body.append(f'<text x="{x(eta):.1f}" y="{cy-10:.0f}" font-size="8.5" fill="{GREY}">{sid}</text>')
        y += rowH
    Hpx = int(y + 16)
    svg = ("".join(grid).replace("{H}", str(Hpx - 12))) + "".join(body)

    legend = (f'<span style="font-size:11px;color:{GREY}">'
              f'<span class="pill" style="background:{BLUE};opacity:.25"></span>time window &nbsp; '
              f'<span class="pill" style="background:{BLUE}"></span>service at ETA')
    if legend_late:
        legend += f' &nbsp; <span class="pill" style="background:{RED}"></span>late'
    legend += '</span>'
    return (_head("Schedule Timeline")
            + _kick("&#128339;", "Delivery Schedule Timeline", "ETA vs TIME WINDOW", VIOLET,
                    result.get("algorithm", ""))
            + '<div class="wrap"><div class="card"><h3>Per-vehicle schedule over the planning horizon</h3>'
            + f'<svg width="100%" viewBox="0 0 {W} {Hpx}" style="min-width:760px">{svg}</svg>'
            + f'<div style="margin-top:8px">{legend}</div></div></div>'
            + _foot(result))


# -- 3. Route map (SVG, aspect-corrected) -------------------------------------

def _route_map(depot, customers, routes, disruptions, result=None):
    result = result or {"algorithm": "", "solver_version": "", "dataset_sha256": ""}
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
    mean_lat = sum(lats) / len(lats)
    kx = math.cos(math.radians(mean_lat))
    xs = [n["lon"] * kx for n in nodes]
    W, H, pad = 860, 560, 0.12
    x_r = max(max(xs) - min(xs), 0.01) * (1 + 2 * pad)
    y_r = max(max(lats) - min(lats), 0.01) * (1 + 2 * pad)
    min_x = min(xs) - x_r * pad / (1 + 2 * pad)
    min_y = min(lats) - y_r * pad / (1 + 2 * pad)

    def proj(lat, lon):
        return (round((lon * kx - min_x) / x_r * W, 1),
                round(H - (lat - min_y) / y_r * H, 1))

    disrupted = set()
    for d in disruptions:
        for lid in d.get("affected_locations", []):
            disrupted.add(lid)

    svg = []
    for i in range(1, 5):
        svg.append(f'<line x1="0" y1="{H*i//4}" x2="{W}" y2="{H*i//4}" stroke="{LINE}"/>')
        svg.append(f'<line x1="{W*i//4}" y1="0" x2="{W*i//4}" y2="{H}" stroke="{LINE}"/>')
    for i, r in enumerate(routes):
        col = _PAL[i % len(_PAL)]
        pts = []
        for sid in r.get("stop_sequence", []):
            n = nmap.get(sid)
            if n:
                pts.append("%s,%s" % proj(n["lat"], n["lon"]))
        if len(pts) > 1:
            svg.append(f'<polyline points="{" ".join(pts)}" stroke="{col}" stroke-width="2.6" '
                       f'fill="none" opacity=".85" stroke-linejoin="round" stroke-linecap="round"/>')
    for n in nodes:
        if n["type"] != "cust":
            continue
        px, py = proj(n["lat"], n["lon"])
        col = GREY
        for i, r in enumerate(routes):
            if n["id"] in r.get("stop_sequence", []):
                col = _PAL[i % len(_PAL)]
                break
        svg.append(f'<circle cx="{px}" cy="{py}" r="8.5" fill="{col}" stroke="#fff" stroke-width="2.4">'
                   f'<title>{n["id"]} &middot; demand {n["demand"]}</title></circle>')
        if n["id"] in disrupted:
            svg.append(f'<circle cx="{px}" cy="{py}" r="14" fill="none" stroke="{RED}" '
                       f'stroke-width="2.4" stroke-dasharray="4 3"><title>disruption at {n["id"]}</title></circle>')
        svg.append(f'<text x="{px}" y="{py-12}" text-anchor="middle" font-size="9" fill="#4a5568">{n["id"]}</text>')
    dpx, dpy = proj(depot.get("lat", 0), depot.get("lon", 0))
    svg.append(f'<rect x="{dpx-11}" y="{dpy-11}" width="22" height="22" rx="5" fill="{NAVY}" '
               f'stroke="#fff" stroke-width="2.4"><title>Depot</title></rect>')
    svg.append(f'<text x="{dpx}" y="{dpy+4}" text-anchor="middle" font-size="10" fill="#fff" font-weight="700">D</text>')
    legend = []
    for i, r in enumerate(routes):
        col = _PAL[i % len(_PAL)]
        legend.append(f'<tr><td><span style="display:inline-block;width:20px;height:4px;'
                      f'background:{col};border-radius:2px"></span></td>'
                      f'<td style="padding:3px 8px"><b>{r.get("vehicle_id")}</b></td>'
                      f'<td style="padding:3px 8px;color:{GREY}">{r.get("num_stops",0)} stops</td>'
                      f'<td style="padding:3px 8px;color:{GREY}">{r.get("distance_km",0):.1f} km</td></tr>')
    return (_head("Route Map")
            + _kick("&#128506;", "Adaptive Route Map",
                    f"{len(disrupted)} disruptions", RED if disrupted else GREEN,
                    f"{len(customers)} customers &middot; {len(routes)} routes")
            + '<div class="wrap" style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start">'
            + f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
              f'style="background:#fff;border:1px solid {LINE};border-radius:14px;max-width:100%">'
            + "".join(svg) + '</svg>'
            + f'<div class="card" style="margin:0;min-width:236px;flex:1"><h3>Vehicle routes</h3>'
              f'<table>{"".join(legend)}</table>'
              f'<p style="margin-top:12px;font-size:11px;color:{GREY}">'
              f'<span class="pill" style="background:{RED}"></span> dashed ring = live disruption</p></div></div>'
            + _foot(result))


# -- 4. Resilience report -----------------------------------------------------

def _resilience(input_data, result, cust_ix):
    on_time = result.get("on_time_delivery_rate_pct", 0)
    robust = result.get("solution_robustness_pct", 0)
    sla_ok = result.get("sla_compliant", 0)
    bars = []
    for name, val, target in [
        ("On-time delivery", on_time, 95),
        ("Robustness (nominal/stressed)", robust, 90),
        ("Capacity utilisation", result.get("avg_capacity_utilization_pct", 0), 70),
        ("Demand fulfilment", result.get("demand_fulfillment_rate_pct", 0), 100),
    ]:
        pct = max(0, min(100, val))
        col = GREEN if val >= target else AMBER
        bars.append(
            f'<div style="margin:12px 0"><div style="display:flex;justify-content:space-between;'
            f'font-size:12.5px;margin-bottom:5px"><span>{name}</span><b>{val:.1f}</b></div>'
            f'<div class="bar" style="height:12px"><i style="width:{pct}%;background:{col};height:12px"></i></div>'
            f'<div style="font-size:10px;color:{GREY};margin-top:2px">target {target}</div></div>')

    slack = []
    for r in result.get("routes", []):
        for sid, eta in (r.get("stop_etas") or {}).items():
            c = cust_ix.get(sid, {})
            tw = c.get("time_window")
            if tw:
                slack.append((sid, float(tw[1]) - float(eta)))
    slack.sort(key=lambda t: t[1])
    srows = "".join(
        f'<tr><td>{sid}</td><td style="color:{RED if sl<0 else (AMBER if sl<15 else GREEN)}">'
        f'{sl:+.0f} min</td><td><div class="bar" style="width:120px"><i style="width:'
        f'{max(2,min(100,(sl+30)/60*100)):.0f}%;background:{RED if sl<0 else (AMBER if sl<15 else GREEN)}"></i></div></td></tr>'
        for sid, sl in slack[:8]) or '<tr><td colspan="3" style="color:#8a94a6">No time-windowed stops</td></tr>'

    disruptions = input_data.get("disruptions", []) or []
    drows = "".join(
        f'<tr><td>{d.get("type")}</td><td>{", ".join(d.get("affected_locations",[]))}</td>'
        f'<td>+{d.get("delay_min",0)} min</td></tr>' for d in disruptions) \
        or '<tr><td colspan="3" style="color:#8a94a6">No active disruptions</td></tr>'

    nominal = result.get("total_robust_time_min", 0)
    stressed = nominal / max(robust / 100.0, 1e-6)
    maxv = max(nominal, stressed, 1)
    stress_svg = (
        f'<div style="display:flex;gap:22px;align-items:flex-end;height:120px;padding:6px 0">'
        f'<div style="text-align:center"><div style="width:56px;height:{int(nominal/maxv*100)}px;'
        f'background:{BLUE};border-radius:6px 6px 0 0"></div><div style="font-size:11px;margin-top:5px">Nominal<br><b>{nominal:.0f}m</b></div></div>'
        f'<div style="text-align:center"><div style="width:56px;height:{int(stressed/maxv*100)}px;'
        f'background:{AMBER};border-radius:6px 6px 0 0"></div><div style="font-size:11px;margin-top:5px">+50% stress<br><b>{stressed:.0f}m</b></div></div>'
        f'</div>')

    return (_head("Resilience Report")
            + _kick("&#128737;", "Resilience &amp; Risk Report",
                    "SLA MET" if sla_ok else "SLA AT RISK", GREEN if sla_ok else RED,
                    result.get("algorithm", ""))
            + '<div class="wrap"><div class="grid">'
            + _kpi(f'{on_time:.0f}%', "On-time", "", GREEN)
            + _kpi(f'{robust:.0f}%', "Robustness", "vs +50% stress", AMBER)
            + _kpi(result.get("time_window_violations", 0), "TW breaches", "", RED)
            + _kpi(f'{result.get("max_lateness_min",0):.0f}', "Max lateness", "min", VIOLET)
            + _kpi(result.get("disruptions_absorbed", 0), "Disruptions", "absorbed", BLUE)
            + '</div>'
            + '<div style="display:flex;gap:16px;flex-wrap:wrap">'
            + '<div class="card" style="flex:2;min-width:320px"><h3>Service-level performance</h3>' + "".join(bars) + '</div>'
            + '<div class="card" style="flex:1;min-width:220px"><h3>Robustness stress test</h3>' + stress_svg + '</div>'
            + '</div>'
            + '<div style="display:flex;gap:16px;flex-wrap:wrap">'
            + '<div class="card" style="flex:1;min-width:280px"><h3>Tightest stops (schedule slack)</h3>'
              '<table><thead><tr><th>Stop</th><th>Slack</th><th></th></tr></thead><tbody>' + srows + '</tbody></table></div>'
            + '<div class="card" style="flex:1;min-width:280px"><h3>Disruptions absorbed</h3>'
              '<table><thead><tr><th>Type</th><th>Locations</th><th>Delay</th></tr></thead><tbody>' + drows + '</tbody></table></div>'
            + '</div></div>'
            + _foot(result))


# -- 5. Executive summary -----------------------------------------------------

def _executive(input_data, result):
    cost = result.get("total_cost_eur", 0)
    baseline = result.get("baseline_cost_eur", cost)
    savings = result.get("savings_vs_baseline_pct", 0)
    cb = result.get("cost_breakdown_eur", {})
    parts = [("Driver time", cb.get("driver", 0), BLUE), ("Fuel", cb.get("fuel", 0), GREEN),
             ("Lateness", cb.get("lateness_penalty", 0), RED)]
    tot = sum(p[1] for p in parts) or 1
    cx, cy, ro, ri = 92, 92, 76, 46
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
                        f'fill="{col}" opacity=".93"><title>{name}: &euro;{val:.2f}</title></path>')
        a0 = a1
    legend = "".join(
        f'<div style="font-size:12.5px;margin:5px 0"><span class="pill" style="background:{col}"></span>'
        f'{name}: <b>&euro;{val:.2f}</b></div>' for name, val, col in parts)

    save_col = GREEN if savings >= 0 else RED
    hero = (f'<div class="card" style="display:flex;align-items:center;gap:26px;flex-wrap:wrap;'
            f'background:linear-gradient(100deg,#fff,#f2fbf6)">'
            f'<div><div style="font-size:44px;font-weight:800;color:{save_col};letter-spacing:-1px">'
            f'{savings:+.1f}%</div><div style="font-size:12px;color:{GREY};font-weight:600">'
            f'cost saved vs. manual/greedy plan</div></div>'
            f'<div style="border-left:1px solid {LINE};padding-left:24px">'
            f'<div style="font-size:13px;color:{GREY}">Optimised plan <b style="color:{INK}">&euro;{cost:.0f}</b> '
            f'&nbsp;vs&nbsp; baseline <b style="color:{INK}">&euro;{baseline:.0f}</b></div>'
            f'<div style="font-size:13px;color:{GREY};margin-top:4px">'
            f'{result.get("num_vehicles_used",0)} vehicles &middot; '
            f'{result.get("on_time_delivery_rate_pct",0):.0f}% on-time &middot; '
            f'{result.get("solution_robustness_pct",0):.0f}% robust</div></div></div>')

    kpis = "".join([
        _kpi(f'&euro;{cost:,.0f}', "Total plan cost", "fuel+driver+lateness", NAVY),
        _kpi(f'&euro;{result.get("cost_per_stop_eur",0):.1f}', "Cost / stop", "", CYAN),
        _kpi(f'{result.get("total_distance_km",0):.0f} km', "Distance", "", BLUE),
        _kpi(f'{result.get("co2_kg",0):.0f} kg', "CO2", "diesel est.", GREEN),
        _kpi(f'{result.get("on_time_delivery_rate_pct",0):.0f}%', "On-time", "", AMBER),
    ])
    return (_head("Executive Summary")
            + _kick("&#128200;", "Executive Summary", result.get("solution_status", "").upper(),
                    _status_color(result), result.get("algorithm", ""))
            + '<div class="wrap">' + hero + '<div class="grid">' + kpis + '</div>'
            + '<div class="card"><h3>Cost composition</h3>'
              f'<div style="display:flex;gap:28px;align-items:center;flex-wrap:wrap">'
              f'<svg width="184" height="184">{"".join(segs)}'
              f'<text x="{cx}" y="{cy-4}" text-anchor="middle" font-size="10" fill="{GREY}">Total</text>'
              f'<text x="{cx}" y="{cy+14}" text-anchor="middle" font-size="16" font-weight="750" '
              f'fill="{NAVY}">&euro;{tot:.0f}</text></svg><div>{legend}'
              f'<p style="font-size:11px;color:{GREY};margin-top:10px;max-width:280px">'
              f'Driver time is the dominant cost lever; the optimiser minimises total '
              f'uncertainty-adjusted route time under the service-level constraints.</p>'
              f'</div></div></div></div>'
            + _foot(result))


# -- 6 & 7. Machine-readable --------------------------------------------------

def _route_plan(result):
    return {
        "solver": result.get("algorithm"),
        "status": result.get("solution_status"),
        "generated_at_utc": result.get("run_started_at_utc"),
        "total_cost_eur": result.get("total_cost_eur"),
        "savings_vs_baseline_pct": result.get("savings_vs_baseline_pct"),
        "routes": [
            {"vehicle_id": r.get("vehicle_id"), "stop_sequence": r.get("stop_sequence"),
             "load": r.get("total_load"), "distance_km": r.get("distance_km"),
             "route_time_min": r.get("route_time_min"), "etas": r.get("stop_etas")}
            for r in result.get("routes", [])
        ],
    }


def _kpis(result):
    keys = ["objective_value", "total_cost_eur", "baseline_cost_eur", "savings_vs_baseline_pct",
            "cost_per_stop_eur", "co2_kg", "total_distance_km", "total_robust_time_min",
            "total_fuel_cost_eur", "total_driver_cost_eur", "num_vehicles_used",
            "on_time_delivery_rate_pct", "demand_fulfillment_rate_pct",
            "avg_capacity_utilization_pct", "max_lateness_min", "time_window_violations",
            "capacity_violations", "max_route_time_min", "solution_robustness_pct",
            "disruptions_absorbed", "compute_time_s", "sla_compliant"]
    return {k: result.get(k) for k in keys}
