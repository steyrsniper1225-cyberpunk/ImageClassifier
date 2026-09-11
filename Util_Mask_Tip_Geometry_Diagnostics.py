"""Diagnostic-only soft contour measurements; no scoring or defect metadata.

Dependencies: numpy, Pillow (QA export only).
Coordinates are pixel centers, (y, x). Profiles run outward from metal.
0.5 is a measurement level, not an ESD threshold. No input normalization,
smoothing, registration, or nuisance subtraction is performed.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

TIP_GEOMETRY_BOUNDS = {
    "tip1": (54, 77, 150, 173),
    "tip2": (132, 163, 147, 178),
}


def _mask(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("Expected a 2D soft mask")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("Soft mask must be finite and within [0, 1]")
    return a


def _sample(a, points):
    y, x = points[..., 0], points[..., 1]
    inside = (y >= 0) & (x >= 0) & (y <= a.shape[0]-1) & (x <= a.shape[1]-1)
    yc, xc = np.clip(y, 0, a.shape[0]-1), np.clip(x, 0, a.shape[1]-1)
    y0, x0 = np.floor(yc).astype(int), np.floor(xc).astype(int)
    y1, x1 = np.minimum(y0+1, a.shape[0]-1), np.minimum(x0+1, a.shape[1]-1)
    fy, fx = yc-y0, xc-x0
    v = ((1-fy)*(1-fx)*a[y0, x0] + (1-fy)*fx*a[y0, x1]
         + fy*(1-fx)*a[y1, x0] + fy*fx*a[y1, x1])
    return np.where(inside, v, np.nan)


def _crossing(t, v, level):
    if not np.isfinite(v).all():
        return np.nan, "out_of_image"
    # A plateau exactly at the level has no unique location.
    if np.any((v[:-1] == level) & (v[1:] == level)):
        return np.nan, "plateau"
    down = np.flatnonzero((v[:-1] >= level) & (v[1:] < level))
    up = np.flatnonzero((v[:-1] < level) & (v[1:] >= level))
    if len(down) != 1 or len(up):
        return np.nan, "missing_or_multiple_crossings"
    i = down[0]
    return float(t[i] + (level-v[i])*(t[i+1]-t[i])/(v[i+1]-v[i])), "valid"


def _measure(t, profiles):
    positions = np.full((len(profiles), 3), np.nan)
    reasons = []
    for i, v in enumerate(profiles):
        row = [_crossing(t, v, level) for level in (0.8, 0.5, 0.2)]
        positions[i] = [r[0] for r in row]
        reasons.append(row[1][1])
    widths = positions[:, 2] - positions[:, 0]
    ordered = (positions[:, 0] < positions[:, 1]) & (positions[:, 1] < positions[:, 2])
    widths[~ordered] = np.nan
    return positions[:, 1], widths, reasons


@dataclass(frozen=True)
class TipGeometryReference:
    zone_name: str
    shape: tuple
    bounds: tuple
    points: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    baseline_positions: np.ndarray
    baseline_widths: np.ndarray
    candidate_count: int
    neighborhood_radius_px: float
    canonical_sha256: str


def build_tip_geometry_references(median, profile_half_length_px=4.0,
                                  sample_step_px=0.25,
                                  neighborhood_radius_px=1.5):
    """Build frozen reference sites from soft 0.5 grid-edge crossings.

    Crossings are subpixel, with outward normals from the median gradient.
    A deterministic 0.75px spatial thinning avoids duplicate grid crossings.
    All contours within the ROI are retained: QA must confirm they belong to
    the intended tip. No guessed left/right contour filter is applied.
    """
    a = _mask(median)
    parameters = (profile_half_length_px, sample_step_px, neighborhood_radius_px)
    if not all(np.isfinite(v) and v > 0 for v in parameters):
        raise ValueError("Profile parameters must be positive and finite")
    if sample_step_px > profile_half_length_px:
        raise ValueError("Sample step exceeds profile half length")
    gy, gx = np.gradient(a)
    offsets = np.linspace(-profile_half_length_px, profile_half_length_px,
                          int(np.ceil(2*profile_half_length_px/sample_step_px))+1)
    digest = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    output = {}
    for name, bounds in TIP_GEOMETRY_BOUNDS.items():
        ymin, ymax, xmin, xmax = bounds
        if not (0 <= ymin <= ymax < a.shape[0] and 0 <= xmin <= xmax < a.shape[1]):
            raise ValueError(f"{name}: bounds outside mask")
        sites = []
        for axis in (0, 1):
            first = a[:-1, :] if axis == 0 else a[:, :-1]
            second = a[1:, :] if axis == 0 else a[:, 1:]
            ys, xs = np.nonzero((first >= 0.5) != (second >= 0.5))
            for y, x in zip(ys, xs):
                fraction = (0.5-first[y, x])/(second[y, x]-first[y, x])
                p = np.array([y, x], dtype=float)
                p[axis] += fraction
                if ymin <= p[0] <= ymax and xmin <= p[1] <= xmax:
                    sites.append(p)
        chosen = []
        for p in sorted(sites, key=lambda p: (p[0], p[1])):
            if not chosen or np.min(np.linalg.norm(np.asarray(chosen)-p, axis=1)) >= 0.75:
                chosen.append(p)
        if not chosen:
            raise ValueError(f"{name}: no canonical 0.5 boundary in ROI")
        points = np.asarray(chosen)
        gradient = np.column_stack((_sample(gy, points), _sample(gx, points)))
        length = np.linalg.norm(gradient, axis=1)
        usable = length > 1e-6
        normals = -gradient[usable] / length[usable, None]
        points = points[usable]
        profiles = _sample(a, points[:, None, :] + offsets[None, :, None]*normals[:, None, :])
        positions, widths, _ = _measure(offsets, profiles)
        good = np.isfinite(positions)
        if not good.any():
            raise ValueError(f"{name}: no valid canonical profiles")
        output[name] = TipGeometryReference(
            name, a.shape, bounds, points[good], normals[good], offsets,
            positions[good], widths[good], len(chosen),
            neighborhood_radius_px, digest)
        for value in vars(output[name]).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    return output


def measure_tip_geometry(observed, reference):
    """Return fixed-schema aggregate fields and per-profile diagnostic rows.

    Positive retreat = inward. Negative values (expansion) are preserved.
    Local mean uses a Euclidean radius around the peak, NOT arc length;
    only similarly directed normals are neighbors. No automatic correction
    by the whole-boundary median is made.
    """
    a = _mask(observed)
    r = reference
    if a.shape != r.shape:
        raise ValueError("Observed shape differs from reference")
    profiles = _sample(a, r.points[:, None, :] + r.offsets[None, :, None]*r.normals[:, None, :])
    positions, widths, reasons = _measure(r.offsets, profiles)
    retreat = r.baseline_positions - positions
    width_delta = widths - r.baseline_widths
    valid = np.isfinite(retreat)
    width_valid = np.isfinite(width_delta)
    fields = {k: float("nan") for k in (
        "retreat_max_px", "retreat_median_px", "retreat_local_mean_px",
        "transition_width_delta_at_peak_px", "transition_width_delta_median_px",
        "peak_y", "peak_x")}
    fields.update(reference_profile_count=len(retreat),
                  canonical_candidate_count=r.candidate_count,
                  canonical_valid_fraction=len(retreat)/r.candidate_count,
                  valid_profile_fraction=float(valid.mean()),
                  valid_width_fraction=float(width_valid.mean()),
                  valid_profile_count=int(valid.sum()), peak_neighbor_count=0)
    if valid.any():
        peak = int(np.nanargmax(retreat))
        near = (np.linalg.norm(r.points-r.points[peak], axis=1) <= r.neighborhood_radius_px)
        near &= (r.normals @ r.normals[peak] > 0.5)
        count = int(near.sum())
        fields.update(retreat_max_px=float(retreat[peak]),
                      retreat_median_px=float(np.median(retreat[valid])),
                      peak_y=float(r.points[peak, 0]), peak_x=float(r.points[peak, 1]),
                      peak_neighbor_count=count,
                      transition_width_delta_at_peak_px=float(width_delta[peak]))
        # Do not silently average over missing neighbors or a lone peak.
        if count >= 2 and valid[near].all():
            fields["retreat_local_mean_px"] = float(retreat[near].mean())
    if width_valid.any():
        fields["transition_width_delta_median_px"] = float(np.median(width_delta[width_valid]))
    rows = [dict(profile_index=i, y=float(p[0]), x=float(p[1]),
                 outward_dy=float(r.normals[i, 0]), outward_dx=float(r.normals[i, 1]),
                 retreat_px=float(retreat[i]), normal_width_px=float(r.baseline_widths[i]),
                 observed_width_px=float(widths[i]), width_delta_px=float(width_delta[i]),
                 position_status=reasons[i], width_valid=bool(width_valid[i]))
            for i, p in enumerate(r.points)]
    return fields, rows


def tip_geometry_fields(observed, references, state):
    return {f"{state}_{name}_geometry_{key}": value
            for name, reference in references.items()
            for key, value in measure_tip_geometry(observed, reference)[0].items()}


def save_tip_geometry_qa(median, references, output_dir):
    """Export clean/annotated canonical crops and canonical per-site CSVs."""
    from PIL import Image, ImageDraw
    a = _mask(median)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"version": "tip_geometry_diagnostic_v1", "levels": [0.8, 0.5, 0.2],
                "position_level": 0.5, "minimum_site_spacing_px": 0.75,
                "normal_gradient": "numpy.gradient, no smoothing", "zones": {}}
    for name, r in references.items():
        ymin, ymax, xmin, xmax = r.bounds
        margin = int(np.ceil(abs(r.offsets).max())) + 2
        y0, y1 = max(0, ymin-margin), min(a.shape[0], ymax+margin+1)
        x0, x1 = max(0, xmin-margin), min(a.shape[1], xmax+margin+1)
        scale = 8
        im = Image.fromarray(np.rint(a[y0:y1, x0:x1]*255).astype(np.uint8)).convert("RGB")
        im = im.resize((im.width*scale, im.height*scale), Image.Resampling.NEAREST)
        im.save(root / f"{name}_canonical_clean.png")
        draw = ImageDraw.Draw(im)
        def xy(p):
            return ((p[1]-x0+0.5)*scale, (p[0]-y0+0.5)*scale)
        for p, n in zip(r.points, r.normals):
            draw.line([xy(p+r.offsets[0]*n), xy(p+r.offsets[-1]*n)], fill=(0, 160, 220), width=1)
            cx, cy = xy(p)
            draw.ellipse((cx-2, cy-2, cx+2, cy+2), fill=(255, 220, 0))
        im.save(root / f"{name}_canonical_profiles.png")
        _, rows = measure_tip_geometry(a, r)
        with (root / f"{name}_canonical_profiles.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata["zones"][name] = dict(bounds=list(r.bounds), offsets=r.offsets.tolist(),
            neighborhood_radius_px=r.neighborhood_radius_px, canonical_sha256=r.canonical_sha256,
            canonical_candidate_count=r.candidate_count, reference_profile_count=len(r.points))
    (root / "geometry_diagnostic_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
