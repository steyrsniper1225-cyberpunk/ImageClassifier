"""Local arc diagnostic extension. Requires the existing geometry module.

Fixed median-only marching-squares topology; no guessed Y/angle ordering.
Ambiguous cells, multiple components and closed contours fail explicitly.
Existing profile sites and original measurements are preserved.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np
import Util_Tip_Geometry_Diagnostics as base

INNER_PX = 3.0
OUTER_PX = 6.0
MIN_SIDE_POINTS = 2


def _open_contour(median, bounds):
    ymin, ymax, xmin, xmax = bounds
    nodes, graph = {}, {}
    # Grid-edge intersections are linked only within their actual grid cell.
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            corners = ((y, x), (y, x+1), (y+1, x+1), (y+1, x))
            hits = []
            for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
                p, q = np.array(corners[i], float), np.array(corners[j], float)
                v, w = median[corners[i]], median[corners[j]]
                if (v >= 0.5) != (w >= 0.5):
                    hit = p+(0.5-v)/(w-v)*(q-p)
                    hits.append(hit)
            if len(hits) == 4:
                raise ValueError(f"Ambiguous 0.5 contour cell at y={y}, x={x}")
            if len(hits) != 2:
                continue
            keys = [tuple(np.round(p, 10)) for p in hits]
            if keys[0] == keys[1]:
                continue
            for key, p in zip(keys, hits):
                nodes[key] = p
                graph.setdefault(key, set())
            graph[keys[0]].add(keys[1])
            graph[keys[1]].add(keys[0])
    ends = [key for key, adjacent in graph.items() if len(adjacent) == 1]
    if len(ends) != 2 or any(len(v) > 2 for v in graph.values()):
        raise ValueError("ROI must contain one unbranched open D contour; inspect ROI")
    path, previous, current = [], None, min(ends)
    seen = set()
    while current is not None:
        if current in seen:
            raise ValueError("Contour cycle detected")
        seen.add(current)
        path.append(nodes[current])
        remaining = graph[current] - ({previous} if previous is not None else set())
        previous, current = current, next(iter(remaining)) if remaining else None
    if len(seen) != len(graph):
        raise ValueError("Multiple contours in ROI; do not join disconnected boundaries")
    path = np.asarray(path)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return path, arc


@dataclass(frozen=True)
class LocalArcReference(base.TipGeometryReference):
    arc_length_px: np.ndarray
    contour_path: np.ndarray
    before_indices: tuple
    after_indices: tuple


def build_tip_geometry_references(median, **kwargs):
    references = base.build_tip_geometry_references(median, **kwargs)
    a = base._mask(median)
    result = {}
    for name, r in references.items():
        path, distance = _open_contour(a, r.bounds)
        pair_distance = np.linalg.norm(r.points[:, None, :]-path[None, :, :], axis=2)
        indices = np.argmin(pair_distance, axis=1)
        if np.any(pair_distance[np.arange(len(indices)), indices] > 1e-6):
            raise ValueError(f"{name}: reference point does not match traced contour")
        arc = distance[indices]
        # Keep original site order for old metrics and paired correspondence.
        delta = arc[None, :] - arc[:, None]
        before = tuple(np.flatnonzero((row >= -OUTER_PX) & (row <= -INNER_PX)) for row in delta)
        after = tuple(np.flatnonzero((row >= INNER_PX) & (row <= OUTER_PX)) for row in delta)
        for array in (arc, path, *before, *after):
            array.setflags(write=False)
        result[name] = LocalArcReference(**vars(r), arc_length_px=arc,
            contour_path=path, before_indices=before, after_indices=after)
    return result


def _local_fields(retreat, r):
    n = len(retreat)
    excess, baseline = np.full(n, np.nan), np.full(n, np.nan)
    before_median, after_median = np.full(n, np.nan), np.full(n, np.nan)
    before_count, after_count = np.zeros(n, int), np.zeros(n, int)
    eligible = np.array([len(b) >= MIN_SIDE_POINTS and len(a) >= MIN_SIDE_POINTS
                        for b, a in zip(r.before_indices, r.after_indices)])
    for i, (before_idx, after_idx) in enumerate(
        zip(r.before_indices, r.after_indices)
    ):
        before_values = retreat[before_idx]
        after_values = retreat[after_idx]

        before_values = before_values[
            np.isfinite(before_values)
        ]
        after_values = after_values[
            np.isfinite(after_values)
        ]

        before_count[i] = before_values.size
        after_count[i] = after_values.size

        if not np.isfinite(retreat[i]):
            continue

        if (
            before_count[i] < MIN_SIDE_POINTS
            or after_count[i] < MIN_SIDE_POINTS
        ):
            continue

        before_median[i] = np.median(before_values)
        after_median[i] = np.median(after_values)

        baseline[i] = (
            before_median[i] + after_median[i]
        ) / 2.0

        excess[i] = retreat[i] - baseline[i]
    
    if eligible.any() and not np.isfinite(excess).any():
        raise RuntimeError(
            "Local diagnostic failed: "
            f"profiles={n}, "
            f"finite_retreat={np.isfinite(retreat).sum()}, "
            f"eligible={eligible.sum()}, "
            f"both_sides_valid={np.sum(
                (before_count >= MIN_SIDE_POINTS)
                & (after_count >= MIN_SIDE_POINTS)
            )}"
        )
    
    valid = np.isfinite(excess)
    fields = {key: float("nan") for key in (
        "local_retreat_excess_max_px", "local_peak_retreat_px", "local_peak_reference_px",
        "local_peak_before_median_px", "local_peak_after_median_px", "local_peak_y", "local_peak_x")}
    fields.update(local_eligible_fraction=float(eligible.mean()),
                  local_valid_fraction=float(valid.mean()), local_valid_count=int(valid.sum()),
                  local_peak_before_count=0, local_peak_after_count=0)
    peak = None
    if valid.any():
        peak = int(np.nanargmax(excess))
        fields.update(local_retreat_excess_max_px=float(excess[peak]),
            local_peak_retreat_px=float(retreat[peak]), local_peak_reference_px=float(baseline[peak]),
            local_peak_before_median_px=float(before_median[peak]),
            local_peak_after_median_px=float(after_median[peak]),
            local_peak_y=float(r.points[peak, 0]), local_peak_x=float(r.points[peak, 1]),
            local_peak_before_count=int(before_count[peak]), local_peak_after_count=int(after_count[peak]))
    detail = [dict(arc_length_px=float(r.arc_length_px[i]),
                  local_reference_px=float(baseline[i]), local_excess_px=float(excess[i]),
                  local_before_count=int(before_count[i]), local_after_count=int(after_count[i]))
              for i in range(n)]
    return fields, detail, peak


def measure_tip_geometry(observed, reference):
    fields, rows = base.measure_tip_geometry(observed, reference)
    retreat = np.array([row["retreat_px"] for row in rows])
    local, details, _ = _local_fields(retreat, reference)
    fields.update(local)
    for row, detail in zip(rows, details):
        row.update(detail)
    return fields, rows


def tip_geometry_fields(observed, references, state):
    return {f"{state}_{name}_geometry_{key}": value
            for name, reference in references.items()
            for key, value in measure_tip_geometry(observed, reference)[0].items()}


def save_tip_local_arc_review(observed, reference, path):
    """Clean crop plus local peak (yellow), before (cyan), after (magenta).

    Gray polyline is the OPEN canonical contour; red sites lack references.
    On identical canonical input all excesses are zero: peak is a tie demo.
    """
    from PIL import Image, ImageDraw
    a, r = base._mask(observed), reference
    fields, rows = measure_tip_geometry(a, r)
    ymin, ymax, xmin, xmax = r.bounds
    y0, y1 = max(0, ymin-6), min(a.shape[0], ymax+7)
    x0, x1 = max(0, xmin-6), min(a.shape[1], xmax+7)
    scale = 8
    clean = Image.fromarray(np.rint(a[y0:y1, x0:x1]*255).astype(np.uint8)).convert("RGB")
    clean = clean.resize((clean.width*scale, clean.height*scale), Image.Resampling.NEAREST)
    overlay = clean.copy()
    draw = ImageDraw.Draw(overlay)
    def xy(p):
        return ((p[1]-x0+0.5)*scale, (p[0]-y0+0.5)*scale)
    draw.line([xy(p) for p in r.contour_path], fill=(150, 150, 150), width=1)
    def dot(p, color, radius=2):
        x, y = xy(p)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    for p, row in zip(r.points, rows):
        dot(p, (0, 200, 0) if np.isfinite(row["local_excess_px"]) else (255, 50, 50))
    if fields["local_valid_count"]:
        values = np.array([row["local_excess_px"] for row in rows])
        i = int(np.nanargmax(values))
        for j in r.before_indices[i]:
            dot(r.points[j], (0, 210, 255), 3)
        for j in r.after_indices[i]:
            dot(r.points[j], (255, 0, 200), 3)
        dot(r.points[i], (255, 220, 0), 4)
    canvas = Image.new("RGB", (clean.width*2+8, clean.height), "white")
    canvas.paste(clean, (0, 0))
    canvas.paste(overlay, (clean.width+8, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_tip_geometry_qa(median, references, output_dir):
    base.save_tip_geometry_qa(median, references, output_dir)
    root = Path(output_dir)
    config = dict(version="local_arc_v1", inner_px=INNER_PX, outer_px=OUTER_PX,
                  minimum_side_points=MIN_SIDE_POINTS, topology="open marching squares",
                  baseline="mean of two side medians", clipping=False, zones={})
    for name, r in references.items():
        save_tip_local_arc_review(median, r, root / f"{name}_local_arc_qa.png")
        config["zones"][name] = dict(bounds=list(r.bounds), canonical_sha256=r.canonical_sha256,
            arc_length_px=r.arc_length_px.tolist(), before_indices=[v.tolist() for v in r.before_indices],
            after_indices=[v.tolist() for v in r.after_indices])
    (root / "local_arc_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")