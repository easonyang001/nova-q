"""Result figures for the three problem classes (location, scheduling, routing).

The Modeling Agent does not draw. Its visualize(data, x) returns a small spec in one of three fixed formats, and
this module draws it, so every figure is correct and consistent whatever the model:

  {"type": "map", "title": str,
   "points": [{"x": float, "y": float, "label": str, "group": str}, ...],   # group: e.g. "selected", "candidate", "demand"
   "links":  [[i, j], ...]}                                                 # optional: lines between points i and j
  {"type": "schedule", "title": str,
   "rows": [str, ...], "cols": [str, ...], "cells": [[str, ...], ...]}       # cells[r][c]: e.g. shift name, "" = off
  {"type": "routes", "title": str,
   "nodes":  [{"x": float, "y": float, "label": str, "depot": bool}, ...],
   "routes": [[node index, ...], ...], "closed": bool}                       # closed: return to the first stop
  {"type": "graph", "title": str,
   "nodes": [{"label": str, "group": str, "x": float, "y": float}, ...],     # x/y optional: laid out automatically
   "edges": [[i, j], ...] or [[i, j, weight], ...], "highlight": [[i, j], ...]}   # e.g. cut edges, chosen links
  {"type": "selection", "title": str,
   "items": [{"label": str, "value": float, "selected": bool, "cost": float}, ...]}   # knapsack, features, assets
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

MAX_POINTS = 20000


def check_spec(spec: Any) -> list[str]:
    """Problems with a visualize() spec, as feedback for the Modeling Agent ([] = drawable)."""
    if not isinstance(spec, dict) or spec.get("type") not in ("map", "schedule", "routes", "graph", "selection"):
        return ["visualize() must return a dict whose 'type' is 'map', 'schedule', 'routes', 'graph' or 'selection'"]
    t, err = spec["type"], []

    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "map":
        pts = spec.get("points")
        if not isinstance(pts, list) or not pts:
            return ["map: 'points' must be a non-empty list"]
        if len(pts) > MAX_POINTS:
            err.append(f"map: at most {MAX_POINTS} points")
        if not all(isinstance(p, dict) and num(p.get("x")) and num(p.get("y")) for p in pts):
            err.append("map: every point needs numeric 'x' and 'y'")
        for link in spec.get("links", []) or []:
            if not (isinstance(link, (list, tuple)) and len(link) == 2 and all(isinstance(i, int) and 0 <= i < len(pts) for i in link)):
                err.append("map: each link must be [i, j] with valid point indices")
                break
    elif t == "graph":
        nodes, edges = spec.get("nodes"), spec.get("edges", [])
        if not isinstance(nodes, list) or not nodes or not all(isinstance(n, dict) for n in nodes):
            return ["graph: 'nodes' must be a non-empty list of objects"]
        for key in ("edges", "highlight"):
            for e in spec.get(key, []) or []:
                if not (isinstance(e, (list, tuple)) and len(e) in (2, 3)
                        and all(isinstance(i, int) and 0 <= i < len(nodes) for i in e[:2])):
                    err.append(f"graph: each of '{key}' must be [i, j] or [i, j, weight] with valid node indices")
                    break
    elif t == "selection":
        items = spec.get("items")
        if not isinstance(items, list) or not items or not all(isinstance(i, dict) and num(i.get("value")) for i in items):
            return ["selection: 'items' must be a non-empty list of objects with a numeric 'value'"]
    elif t == "schedule":
        rows, cols, cells = spec.get("rows"), spec.get("cols"), spec.get("cells")
        if not (isinstance(rows, list) and isinstance(cols, list) and isinstance(cells, list)):
            return ["schedule: 'rows', 'cols' and 'cells' must be lists"]
        if len(cells) != len(rows) or any(not isinstance(r, list) or len(r) != len(cols) for r in cells):
            err.append(f"schedule: 'cells' must be {len(rows)} rows x {len(cols)} columns")
    else:
        nodes, routes = spec.get("nodes"), spec.get("routes")
        if not isinstance(nodes, list) or not nodes or not all(isinstance(n, dict) and num(n.get("x")) and num(n.get("y")) for n in nodes):
            return ["routes: 'nodes' must be a non-empty list of points with numeric 'x' and 'y'"]
        if not isinstance(routes, list) or not all(isinstance(r, list) and all(isinstance(i, int) and 0 <= i < len(nodes) for i in r) for r in routes):
            err.append("routes: 'routes' must be lists of valid node indices")
    return err


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            # family list = per-glyph fallback (matplotlib >= 3.6): Latin from DejaVu, CJK from Droid
            plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=path).get_name()]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def render(spec: dict, path: Path) -> Path:
    """Draw a checked spec to a PNG and return its path."""
    plt = _setup()
    t = spec["type"]
    if t == "schedule":
        rows, cols, cells = spec["rows"], spec["cols"], spec["cells"]
        fig, ax = plt.subplots(figsize=(min(18, 2 + 0.35 * len(cols)), min(14, 1.5 + 0.35 * len(rows))))
        values = sorted({str(c) for r in cells for c in r if str(c)})
        index = {v: i + 1 for i, v in enumerate(values)}
        grid = [[index.get(str(c), 0) for c in r] for r in cells]
        from matplotlib.colors import ListedColormap
        base = plt.get_cmap("tab10")
        cmap = ListedColormap(["#eeeeee"] + [base(i % 10) for i in range(max(1, len(values)))])   # 0 = empty cell
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(values) + 0.5 if values else 1.5, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks(range(len(cols)), [str(c) for c in cols], rotation=90, fontsize=7)
        ax.set_yticks(range(len(rows)), [str(r) for r in rows], fontsize=7)
        if len(rows) * len(cols) <= 600:
            for i, r in enumerate(cells):
                for j, c in enumerate(r):
                    if str(c):
                        ax.text(j, i, str(c)[:3], ha="center", va="center", fontsize=6, color="black")
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(index[v])) for v in values]
        handles.append(plt.Rectangle((0, 0), 1, 1, color=cmap(0)))
        ax.legend(handles, values + ["（空）"], loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    elif t == "map":
        pts = spec["points"]
        fig, ax = plt.subplots(figsize=(8, 7))
        for i, j in spec.get("links", []) or []:
            ax.plot([pts[i]["x"], pts[j]["x"]], [pts[i]["y"], pts[j]["y"]], color="0.75", lw=0.6, zorder=1)
        groups: dict[str, list] = {}
        for p in pts:
            groups.setdefault(str(p.get("group", "")), []).append(p)
        order = sorted(groups, key=lambda g: g.lower() in ("selected", "open", "chosen", "選中"))   # selected drawn last
        for g in order:
            ps = groups[g]
            chosen = g.lower() in ("selected", "open", "chosen", "選中")
            ax.scatter([p["x"] for p in ps], [p["y"] for p in ps], s=70 if chosen else 14,
                       marker="*" if chosen else "o", zorder=3 if chosen else 2, label=f"{g or 'points'} ({len(ps)})",
                       edgecolors="black" if chosen else "none", linewidths=0.5)
        if len(pts) <= 60:
            for p in pts:
                if p.get("label"):
                    ax.annotate(str(p["label"])[:12], (p["x"], p["y"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    elif t == "graph":
        import math
        nodes, edges = spec["nodes"], spec.get("edges", []) or []
        hl = {tuple(sorted(e[:2])) for e in spec.get("highlight", []) or []}
        if all(isinstance(n.get("x"), (int, float)) and isinstance(n.get("y"), (int, float)) for n in nodes):
            pos = [(n["x"], n["y"]) for n in nodes]
        else:                                          # circular layout when the model gives no coordinates
            k = len(nodes)
            pos = [(math.cos(2 * math.pi * i / k), math.sin(2 * math.pi * i / k)) for i in range(k)]
        fig, ax = plt.subplots(figsize=(8, 7))
        for e in edges:
            i, j = e[0], e[1]
            on = tuple(sorted((i, j))) in hl
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="crimson" if on else "0.75",
                    lw=2.0 if on else 0.7, zorder=1)
        groups: dict[str, list[int]] = {}
        for i, n in enumerate(nodes):
            groups.setdefault(str(n.get("group", "")), []).append(i)
        for g, idx in groups.items():
            ax.scatter([pos[i][0] for i in idx], [pos[i][1] for i in idx], s=60, zorder=3, label=f"{g or 'nodes'} ({len(idx)})",
                       edgecolors="black", linewidths=0.4)
        if len(nodes) <= 60:
            for i, n in enumerate(nodes):
                ax.annotate(str(n.get("label", i))[:10], pos[i], fontsize=7, xytext=(4, 4), textcoords="offset points")
        if hl:
            ax.plot([], [], color="crimson", lw=2, label=f"highlighted ({len(hl)})")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    elif t == "selection":
        items = sorted(spec["items"], key=lambda i: -float(i["value"]))[:60]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.28 * len(items) + 1)))
        colors = ["tab:orange" if i.get("selected") else "0.75" for i in items]
        ax.barh(range(len(items)), [float(i["value"]) for i in items], color=colors)
        ax.set_yticks(range(len(items)), [str(i.get("label", ""))[:24] for i in items], fontsize=7)
        ax.invert_yaxis()
        for k, i in enumerate(items):
            if isinstance(i.get("cost"), (int, float)):
                ax.annotate(f"cost {i['cost']:.3g}", (float(i["value"]), k), fontsize=6, xytext=(3, -2), textcoords="offset points")
        n_sel = sum(1 for i in spec["items"] if i.get("selected"))
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="tab:orange", label=f"selected ({n_sel})"), Patch(color="0.75", label="not selected")],
                  fontsize=8, loc="lower right")
        ax.set_xlabel("value")
    else:
        nodes = spec["nodes"]
        fig, ax = plt.subplots(figsize=(8, 7))
        cmap = plt.get_cmap("tab10")
        for k, r in enumerate(spec["routes"]):
            seq = r + ([r[0]] if spec.get("closed", True) and len(r) > 1 else [])
            ax.plot([nodes[i]["x"] for i in seq], [nodes[i]["y"] for i in seq], "-o", color=cmap(k % 10), ms=4,
                    lw=1.5, label=f"route {k + 1} ({len(r)} stops)")
        depots = [n for n in nodes if n.get("depot")]
        if depots:
            ax.scatter([n["x"] for n in depots], [n["y"] for n in depots], marker="s", s=90, c="black", zorder=4, label="depot")
        if len(nodes) <= 60:
            for n in nodes:
                if n.get("label"):
                    ax.annotate(str(n["label"])[:12], (n["x"], n["y"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(str(spec.get("title", ""))[:80])
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
