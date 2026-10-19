"""Friendly 2D geometry input, native curves, units and nested interfaces.

Geometry and material-region labels are independent of mesh controls. Curves
remain native until a solver mesh is requested. The Maxwell solver supports nested conducting regions and finite beta.
Open Air/Vacuum exteriors are supported for beta<1, without roughness.
"""
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import copy
import csv
import json
import difflib
import io
import tempfile
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ellipe
from .interfaces import PolygonPanels

FORMAT = "arcs.geometry/1"
UNIT_SCALE = {"m": 1., "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9}
TWOPI = 2*np.pi


def _save_figure(figure, destination, dpi=180):
    """Publish a finished PNG atomically so file readers never see half a plot."""
    destination = Path(destination)
    if destination.suffix.lower() != ".png":
        figure.savefig(destination, dpi=dpi)
        return
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(buffer.getvalue())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _keys(data, allowed, where):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: use a JSON object.")
    for key in data.keys()-set(allowed):
        similar = difflib.get_close_matches(key, allowed, n=1)
        hint = f" Did you mean '{similar[0]}'?" if similar else ""
        raise ValueError(f"{where}: unknown setting '{key}'.{hint}")


def _number(value, where, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{where}: enter a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: enter a finite number.") from None
    if not np.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{where}: enter a {'positive ' if positive else ''}finite number.")
    return number


def _point(value, scale, where):
    try:
        if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
            raise ValueError()
        point = np.asarray(value, float)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: enter [x, y].") from None
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{where}: enter two finite coordinates [x, y].")
    return point*scale


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


@dataclass(frozen=True)
class _Curve:
    kind: str
    data: tuple

    def point(self, t):
        t = np.asarray(t)
        if self.kind == "line":
            a, b = map(np.asarray, self.data)
            return a+t[..., None]*(b-a)
        center, axes, angle, start, sweep = self.data
        theta = start+sweep*t
        local = np.stack((axes[0]*np.cos(theta), axes[1]*np.sin(theta)), axis=-1)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return local @ rotation.T+center

    def derivative(self, t):
        if self.kind == "line":
            return np.asarray(self.data[1])-self.data[0]
        _, axes, angle, start, sweep = self.data
        theta = start+sweep*t
        local = sweep*np.stack((-axes[0]*np.sin(theta), axes[1]*np.cos(theta)), axis=-1)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return local @ rotation.T

    def reversed(self):
        if self.kind == "line":
            return _Curve("line", self.data[::-1])
        center, axes, angle, start, sweep = self.data
        return _Curve(self.kind, (center, axes, angle, start+sweep, -sweep))

    def signed_area(self):
        if self.kind == "line":
            return .5*float(_cross(np.asarray(self.data[0]), np.asarray(self.data[1])))
        center, axes, _, _, sweep = self.data
        return .5*(float(_cross(np.asarray(center), self.point(1.)-self.point(0.)))
                   + axes[0]*axes[1]*sweep)

    def length(self):
        if self.kind == "line":
            return float(np.linalg.norm(self.point(1.)-self.point(0.)))
        _, axes, _, _, sweep = self.data
        if self.kind == "arc":
            return abs(sweep)*axes[0]
        large, small = max(axes), min(axes)
        return float(4*large*ellipe(1-(small/large)**2))

    def parameter_of_angle(self, theta):
        start, sweep = self.data[-2:]
        return np.mod((np.asarray(theta)-start)*np.sign(sweep), TWOPI)/abs(sweep)


def _ellipse_metric(curve):
    center, axes, angle, _, _ = curve.data
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray(center), rotation @ np.diag(1/np.asarray(axes)**2) @ rotation.T


def _implicit_range(curve, ellipse):
    """Extrema of the enclosing ellipse's quadratic form on a native segment.

    On an arc/ellipse this is a degree-two trigonometric polynomial; all
    stationary angles come from a quartic in tan(theta/2). No chord mesh is
    used to decide whether a thin layer is nested.
    """
    origin, metric = _ellipse_metric(ellipse)
    def evaluate(points):
        offset = np.asarray(points)-origin
        return np.einsum("...i,ij,...j->...", offset, metric, offset)-1
    if curve.kind == "line":
        a, b = curve.point(0.)-origin, curve.point(1.)-origin
        vector = b-a
        denominator = float(vector @ metric @ vector)
        t = np.clip(-float(vector @ metric @ a)/denominator, 0., 1.)
        values = evaluate(curve.point(np.array([0., t, 1.])))
        return float(values.min()), float(values.max())
    center, axes, angle, start, sweep = curve.data
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    u, v = rotation[:, 0]*axes[0], rotation[:, 1]*axes[1]
    offset = np.asarray(center)-origin
    a, b, c = float(u @ metric @ u), float(v @ metric @ v), 2*float(u @ metric @ v)
    d, e = 2*float(offset @ metric @ u), 2*float(offset @ metric @ v)
    coefficients = np.array([c-e, -2*d-4*(b-a), -6*c, -2*d+4*(b-a), e+c])
    magnitude = np.max(abs(coefficients))
    angles = [start, start+sweep, 0., np.pi]
    if magnitude > 1e-14:
        roots = np.roots(np.trim_zeros(coefficients/magnitude, "f"))
        angles.extend(2*np.arctan(root.real) for root in roots
                      if abs(root.imag) < 1e-7*(1+abs(root.real)))
    t = curve.parameter_of_angle(angles)
    t = np.r_[0., 1., t[t <= 1.+1e-12]]
    values = evaluate(curve.point(t))
    return float(values.min()), float(values.max())


def _intersections(first, second, tolerance):
    """Intersection/touch points for line and circular-arc segments."""
    if first.kind == "line" and second.kind == "line":
        a, b, c, d = first.point(0.), first.point(1.), second.point(0.), second.point(1.)
        u, v = b-a, d-c
        denominator = float(_cross(u, v))
        if abs(denominator) <= tolerance*max(np.linalg.norm(u), np.linalg.norm(v)):
            if abs(_cross(u, c-a)) > tolerance*np.linalg.norm(u):
                return []
            answer = []
            for point in [a, b, c, d]:
                if (np.dot(point-a, point-b) <= tolerance**2
                        and np.dot(point-c, point-d) <= tolerance**2):
                    answer.append(point)
            return answer
        t, s = float(_cross(c-a, v)/denominator), float(_cross(c-a, u)/denominator)
        epsilon = tolerance/min(np.linalg.norm(u), np.linalg.norm(v))
        return [a+t*u] if -epsilon <= t <= 1+epsilon and -epsilon <= s <= 1+epsilon else []
    if second.kind == "line":
        return _intersections(second, first, tolerance)
    if first.kind == "line":
        a, b = first.point(0.), first.point(1.)
        center, axes = second.data[:2]
        u, offset = b-a, a-center
        aa, bb, cc = float(u @ u), 2*float(offset @ u), float(offset @ offset)-axes[0]**2
        discriminant = bb*bb-4*aa*cc
        if discriminant < -16*aa*axes[0]*tolerance:
            return []
        roots = [(-bb+sign*np.sqrt(max(0., discriminant)))/(2*aa) for sign in [-1, 1]]
        points = [a+t*u for t in roots if -tolerance/np.sqrt(aa) <= t <= 1+tolerance/np.sqrt(aa)]
    else:
        c1, axes1 = first.data[:2]
        c2, axes2 = second.data[:2]
        r1, r2 = axes1[0], axes2[0]
        vector = np.asarray(c2)-c1
        distance = np.linalg.norm(vector)
        if distance < tolerance and abs(r1-r2) < tolerance:
            points = [first.point(0.), first.point(1.), second.point(0.), second.point(1.),
                      first.point(.5), second.point(.5)]
        elif distance < abs(r1-r2)-tolerance or distance > r1+r2+tolerance or distance < tolerance:
            return []
        else:
            along = (r1*r1-r2*r2+distance*distance)/(2*distance)
            height2 = r1*r1-along*along
            if height2 < -4*r1*tolerance:
                return []
            tangent = vector/distance
            normal = np.array([-tangent[1], tangent[0]])
            middle = np.asarray(c1)+along*tangent
            points = [middle+sign*np.sqrt(max(0., height2))*normal for sign in [-1, 1]]
    answer = []
    for point in points:
        valid = True
        for curve in [first, second]:
            if curve.kind == "arc":
                angle = np.arctan2(*(point-np.asarray(curve.data[0]))[::-1])
                t = curve.parameter_of_angle(angle)
                # atan2 is undefined only at a circle centre, which cannot be an arc hit.
                valid &= bool(t <= 1+1e-10 or abs(t*abs(curve.data[-1])-TWOPI) < 1e-10)
        if valid:
            answer.append(np.asarray(point))
    return answer


@dataclass(frozen=True)
class BoundaryShape:
    kind: str
    curves: tuple
    _definition: dict
    orientation_reversed: bool = False

    @property
    def definition(self):
        return copy.deepcopy(self._definition)

    @property
    def area(self):
        return sum(curve.signed_area() for curve in self.curves)

    @property
    def perimeter(self):
        return sum(curve.length() for curve in self.curves)

    def contains(self, point):
        point = np.asarray(point)
        if len(self.curves) == 1 and self.curves[0].kind in ("ellipse", "arc"):
            center, metric = _ellipse_metric(self.curves[0])
            return bool((point-center) @ metric @ (point-center) < 1-2e-12)
        winding = 0
        scale = max(curve.length() for curve in self.curves)
        tolerance = 128*np.finfo(float).eps*scale
        for curve in self.curves:
            if curve.kind == "line":
                a, b = curve.point(0.), curve.point(1.)
                direction = b-a
                t = np.clip(np.dot(point-a, direction)/np.dot(direction, direction), 0., 1.)
                if np.linalg.norm(point-(a+t*direction)) <= tolerance:
                    return False
                if (a[1] > point[1]) != (b[1] > point[1]):
                    x = a[0]+(point[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
                    if x > point[0]:
                        winding += 1 if b[1] > a[1] else -1
            else:
                center, axes, _, _, sweep = curve.data
                offset = point-np.asarray(center)
                angle = np.arctan2(offset[1], offset[0])
                parameter = float(curve.parameter_of_angle(angle))
                if parameter <= 1+1e-12 and abs(np.linalg.norm(offset)-axes[0]) <= tolerance:
                    return False
                if min(np.linalg.norm(point-curve.point(t)) for t in [0., 1.]) <= tolerance:
                    return False
                height = (point[1]-center[1])/axes[0]
                if abs(height) >= 1:
                    continue
                angles = [np.arcsin(height), np.pi-np.arcsin(height)]
                for theta in angles:
                    t = float(curve.parameter_of_angle(theta))
                    sign = np.sign(np.cos(theta)*sweep)
                    full = abs(abs(sweep)-TWOPI) < 1e-12
                    include = full or (1e-12 < t < 1-1e-12)
                    include |= abs(t) < 1e-12 and sign > 0
                    include |= abs(t-1) < 1e-12 and sign < 0
                    if include and center[0]+axes[0]*np.cos(theta) > point[0]:
                        winding += int(sign)
        return winding != 0

    @property
    def bounds(self):
        points = []
        for curve in self.curves:
            if curve.kind == "line":
                points.extend([curve.point(0.), curve.point(1.)])
            else:
                _, axes, angle, _, _ = curve.data
                theta_x = np.arctan2(-axes[1]*np.sin(angle), axes[0]*np.cos(angle))
                theta_y = np.arctan2(axes[1]*np.cos(angle), axes[0]*np.sin(angle))
                t = curve.parameter_of_angle([theta_x, theta_x+np.pi, theta_y, theta_y+np.pi])
                points.extend(curve.point(np.r_[0., 1., t[t <= 1+1e-12]]))
        points = np.asarray(points)
        return points.min(axis=0), points.max(axis=0)


def _validate_shape(kind, curves, definition):
    scale = max(curve.length() for curve in curves)
    tolerance = 128*np.finfo(float).eps*scale
    for index, curve in enumerate(curves):
        if np.linalg.norm(curve.point(1.)-curves[(index+1) % len(curves)].point(0.)) > tolerance:
            raise ValueError("The path is open. End the last segment at the starting point.")
    if len(curves) > 1:
        # Long imported polygons can have thousands of sides. A vectorized
        # bounding-box filter avoids testing every distant pair with the
        # exact intersection routine; the geometric validity criterion is
        # unchanged. Native arcs retain the existing exact pair checks.
        line_bounds = None
        if all(curve.kind == "line" for curve in curves):
            endpoints = np.array([[curve.point(0.), curve.point(1.)] for curve in curves])
            line_bounds = (endpoints.min(axis=1), endpoints.max(axis=1))
        for i, first in enumerate(curves):
            candidates = range(i+1, len(curves))
            if line_bounds is not None:
                low, high = line_bounds
                overlap = np.all(high[i]+tolerance >= low[i+1:], axis=1) & np.all(high[i+1:]+tolerance >= low[i], axis=1)
                candidates = i+1+np.flatnonzero(overlap)
            for j in candidates:
                second = curves[j]
                hits = _intersections(first, second, tolerance)
                shared = []
                if j == i+1:
                    shared.append(first.point(1.))
                if i == 0 and j == len(curves)-1:
                    shared.append(first.point(0.))
                if any(not any(np.linalg.norm(hit-point) <= 4*tolerance for point in shared) for hit in hits):
                    raise ValueError("The outline intersects or touches itself. Check the segment order.")
    area = sum(curve.signed_area() for curve in curves)
    if abs(area) <= tolerance*scale:
        raise ValueError("The outline has zero or unresolved enclosed area.")
    reverse = bool(area < 0)
    if reverse:
        curves = [curve.reversed() for curve in curves[::-1]]
    return BoundaryShape(kind, tuple(curves), definition, reverse)


def _load_points(path):
    with path.open(newline="") as stream:
        rows = [row for row in csv.reader(stream) if row and any(value.strip() for value in row)]
    if rows and [value.strip().lower() for value in rows[0]] == ["x", "y"]:
        rows.pop(0)
    try:
        points = np.asarray([[float(value) for value in row] for row in rows])
    except ValueError:
        raise ValueError(f"{path.name}: use two numeric columns, optionally headed x,y.") from None
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{path.name}: use exactly two columns x,y; units come from the geometry file.")
    return points


def _shape(data, scale, base):
    if not isinstance(data, dict) or "type" not in data:
        raise ValueError("Each shape needs 'type', for example circle, polygon or csv.")
    kind = data["type"]
    fields = {
        "circle": {"type", "radius", "center"},
        "ellipse": {"type", "semi_axes", "center", "rotation_deg"},
        "rectangle": {"type", "width", "height", "center", "rotation_deg"},
        "rounded_rectangle": {"type", "width", "height", "corner_radius", "center", "rotation_deg"},
        "polygon": {"type", "points"}, "csv": {"type", "file"},
        "path": {"type", "start", "segments"}}
    if kind not in fields:
        raise ValueError(f"Unknown shape type '{kind}'. Choose {', '.join(fields)}.")
    _keys(data, fields[kind], kind)
    definition = copy.deepcopy(data)
    curves = []
    if kind == "csv":
        if not isinstance(data.get("file"), str):
            raise ValueError("csv.file: enter a relative CSV filename.")
        path = base/data["file"]
        points = _load_points(path)
        definition = {"type": "polygon", "points": points.tolist()}
        kind = "polygon"
    if kind == "polygon":
        points = np.asarray(definition.get("points"), float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("polygon.points: enter at least three finite [x, y] points in perimeter order.")
        points = np.array([_point(row, scale, "polygon.points") for row in definition["points"]])
        if np.array_equal(points[0], points[-1]):
            points = points[:-1]
        if len(points) < 3 or np.any(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1) == 0):
            raise ValueError("Consecutive polygon points must differ; repeat the first point only at the end.")
        curves = [_Curve("line", (tuple(a), tuple(b))) for a, b in zip(points, np.roll(points, -1, axis=0))]
    elif kind == "path":
        position = _point(data.get("start"), scale, "path.start")
        segments = data.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("path.segments: enter a nonempty list of line/arc segments.")
        for index, segment in enumerate(segments):
            _keys(segment, {"type", "to", "center", "direction"}, f"segment {index+1}")
            target = _point(segment.get("to"), scale, f"segment {index+1}.to")
            if segment.get("type") == "line":
                if "center" in segment or "direction" in segment:
                    raise ValueError("A line segment needs only type and to.")
                if np.array_equal(position, target):
                    raise ValueError("A line segment must have nonzero length.")
                curve = _Curve("line", (tuple(position), tuple(target)))
            elif segment.get("type") == "arc":
                center = _point(segment.get("center"), scale, f"segment {index+1}.center")
                first, second = position-center, target-center
                radius = np.linalg.norm(first)
                if radius <= 0 or abs(np.linalg.norm(second)-radius) > 2e-11*radius:
                    raise ValueError("An arc's start and end must be the same distance from its center.")
                direction = segment.get("direction", "ccw")
                if direction not in ("ccw", "cw"):
                    raise ValueError("arc.direction must be ccw or cw.")
                start, end = np.arctan2(first[1], first[0]), np.arctan2(second[1], second[0])
                sweep = float(np.mod((end-start)*(1 if direction == "ccw" else -1), TWOPI))
                if np.array_equal(position, target):
                    sweep = TWOPI
                if sweep == 0:
                    raise ValueError("An arc must have nonzero angular extent.")
                sweep *= 1 if direction == "ccw" else -1
                curve = _Curve("arc", (tuple(center), (radius, radius), 0., start, sweep))
            else:
                raise ValueError("path segments support type line or arc.")
            curves.append(curve)
            position = target
    else:
        center = _point(data.get("center", [0., 0.]), scale, f"{kind}.center")
        angle = np.deg2rad(_number(data.get("rotation_deg", 0.), f"{kind}.rotation_deg"))
        if kind == "circle":
            radius = scale*_number(data.get("radius"), "circle.radius", positive=True)
            curves = [_Curve("arc", (tuple(center), (radius, radius), 0., 0., TWOPI))]
        elif kind == "ellipse":
            axes = _point(data.get("semi_axes"), scale, "ellipse.semi_axes")
            if np.any(axes <= 0):
                raise ValueError("ellipse.semi_axes must both be positive; they are half axes.")
            curves = [_Curve("ellipse", (tuple(center), tuple(axes), angle, 0., TWOPI))]
        else:
            w = scale*_number(data.get("width"), f"{kind}.width", positive=True)/2
            h = scale*_number(data.get("height"), f"{kind}.height", positive=True)/2
            radius = (scale*_number(data.get("corner_radius"), "rounded_rectangle.corner_radius", positive=True)
                      if kind == "rounded_rectangle" else 0.)
            if radius >= min(w, h):
                raise ValueError("corner_radius must be smaller than half the width and half the height.")
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            def transform(p):
                return tuple(np.asarray(p) @ rotation.T+center)
            vertices = [[-w+radius, -h], [w-radius, -h], [w, -h+radius],
                        [w, h-radius], [w-radius, h], [-w+radius, h],
                        [-w, h-radius], [-w, -h+radius]] if radius else [[-w,-h],[w,-h],[w,h],[-w,h]]
            arc_centers = [[w-radius,-h+radius], [w-radius,h-radius],
                           [-w+radius,h-radius], [-w+radius,-h+radius]]
            for index, a in enumerate(vertices):
                b = vertices[(index+1) % len(vertices)]
                if radius and index % 2:
                    arc_center = transform(arc_centers[index//2])
                    start = -np.pi/2+(index//2)*np.pi/2+angle
                    curves.append(_Curve("arc", (arc_center, (radius, radius), 0., start, np.pi/2)))
                else:
                    curves.append(_Curve("line", (transform(a), transform(b))))
    return _validate_shape(kind, curves, definition)


def _nested(inner, outer):
    outer_ellipse = len(outer.curves) == 1 and outer.curves[0].kind in ("arc", "ellipse")
    inner_ellipse = len(inner.curves) == 1 and inner.curves[0].kind in ("arc", "ellipse")
    if outer_ellipse:
        return all(_implicit_range(curve, outer.curves[0])[1] < -2e-12 for curve in inner.curves)
    if inner_ellipse:
        for curve in outer.curves:
            low, high = _implicit_range(curve, inner.curves[0])
            if low <= 2e-12 and high >= -2e-12:
                return False
    else:
        tolerance = 128*np.finfo(float).eps*max(inner.perimeter, outer.perimeter)
        if any(_intersections(first, second, tolerance)
               for first in inner.curves for second in outer.curves):
            return False
    return outer.contains(inner.curves[0].point(.314159))


def _segment_hits(a, b, c, d, tolerance):
    """Vectorized crossing/contact test, including collinear endpoints."""
    u, v = b-a, d-c
    ac, ad = _cross(u, c-a), _cross(u, d-a)
    ca, cb = _cross(v, a-c), _cross(v, b-c)
    eu, ev = tolerance*np.linalg.norm(u), tolerance*np.linalg.norm(v, axis=-1)
    opposite_u = ((ac > eu) & (ad < -eu)) | ((ac < -eu) & (ad > eu))
    opposite_v = ((ca > ev) & (cb < -ev)) | ((ca < -ev) & (cb > ev))
    contact = ((abs(ac) <= eu) & (np.sum((c-a)*(c-b), axis=-1) <= tolerance**2))
    contact |= (abs(ad) <= eu) & (np.sum((d-a)*(d-b), axis=-1) <= tolerance**2)
    contact |= (abs(ca) <= ev) & (np.sum((a-c)*(a-d), axis=-1) <= tolerance**2)
    contact |= (abs(cb) <= ev) & (np.sum((b-c)*(b-d), axis=-1) <= tolerance**2)
    return (opposite_u & opposite_v) | contact


def _simple_panels(starts, ends):
    tolerance = 128*np.finfo(float).eps*np.max(np.ptp(starts, axis=0))
    for i, (a, b) in enumerate(zip(starts, ends)):
        indices = np.arange(i+2, len(starts))
        if i == 0:
            indices = indices[indices != len(starts)-1]
        if len(indices) and np.any(_segment_hits(a, b, starts[indices], ends[indices], tolerance)):
            return False
    return True


def _nested_panels(inner, outer):
    tolerance = 128*np.finfo(float).eps*max(np.max(np.ptp(inner.vertices, axis=0)),
                                          np.max(np.ptp(outer.vertices, axis=0)))
    for a, b in zip(inner.panel_start, inner.panel_end):
        if np.any(_segment_hits(a, b, outer.panel_start, outer.panel_end, tolerance)):
            return False
    p = inner.panel_start[0]
    first, second = outer.panel_start-p, outer.panel_end-p
    angle = np.arctan2(_cross(first, second), np.sum(first*second, axis=1)).sum()
    return bool(abs(angle) > np.pi)


@dataclass(frozen=True)
class MeshSettings:
    max_panel_length: float = 1e-3
    curve_tolerance: float = 1e-5
    corner_grading: float = 2.
    integration_order: int = 4
    minimum_panels: int = 64

    @classmethod
    def from_dict(cls, data, scale):
        allowed = {"max_panel_length", "curve_tolerance", "corner_grading",
                   "integration_order", "minimum_panels"}
        _keys(data, allowed, "mesh")
        length = scale*_number(data.get("max_panel_length", 1e-3/scale),
                               "mesh.max_panel_length", positive=True)
        tolerance = scale*_number(data.get("curve_tolerance", 1e-5/scale),
                                  "mesh.curve_tolerance", positive=True)
        grading = _number(data.get("corner_grading", 2.), "mesh.corner_grading")
        if grading < 1:
            raise ValueError("mesh.corner_grading must be >=1; 1 gives uniform panels.")
        order, minimum = data.get("integration_order", 4), data.get("minimum_panels", 64)
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
               for value in [order, minimum]):
            raise ValueError("mesh.integration_order and minimum_panels must be positive integers.")
        return cls(length, tolerance, grading, int(order), int(minimum))

    def input_values(self, scale):
        return {"max_panel_length": self.max_panel_length/scale,
                "curve_tolerance": self.curve_tolerance/scale,
                "corner_grading": self.corner_grading,
                "integration_order": self.integration_order,
                "minimum_panels": self.minimum_panels}


@dataclass(frozen=True)
class GeometrySolver:
    """Field backend with the saved beam position and optional pipe length.

    pipe_length_m defaults to 1.0 m and returns impedance for that uniform pipe.
    A positive length multiplies every impedance component by that length.
    Explicit pipe_length_m=None returns per-length impedance.
    The underlying 2D field backend retains its per-length normalization.
    """
    backend_name: str
    backend: object
    beam_position: tuple
    default_wall: object = None
    pipe_length_m: float | None = 1.0

    def __post_init__(self):
        from .fields import validate_pipe_length
        object.__setattr__(self, "pipe_length_m", validate_pipe_length(self.pipe_length_m))
        if self.backend_name == "pec":
            if self.pipe_length_m not in (None, 1.0):
                raise ValueError("Pipe length applies to Maxwell impedance output; the PEC geometry helper has no impedance.")
            object.__setattr__(self, "pipe_length_m", None)

    @property
    def longitudinal_units(self):
        return "Ohm/m" if self.pipe_length_m is None else "Ohm"

    @property
    def transverse_coefficient_units(self):
        return "Ohm/m2" if self.pipe_length_m is None else "Ohm/m"

    @property
    def constant_units(self):
        return self.longitudinal_units

    def _longitudinal_output(self, values):
        return values if self.pipe_length_m is None else values*self.pipe_length_m

    def _transverse_output(self, response):
        return response if self.pipe_length_m is None else response.for_pipe_length(self.pipe_length_m)

    def longitudinal(self, f, wall=None, *, source=None, witness=None, **kwargs):
        if self.backend_name == "pec":
            raise NotImplementedError("The PEC backend supplies vacuum wall fields only; choose maxwell for impedance.")
        effective_wall = self.default_wall if wall is None else wall
        values = self.backend.longitudinal(f, effective_wall, source=self.beam_position if source is None else source,
                                          witness=witness, **kwargs)
        return self._longitudinal_output(values)

    def transverse(self, f, wall=None, *, reference=None, component="wall"):
        if self.backend_name != "maxwell":
            raise NotImplementedError("Choose maxwell for transverse impedance.")
        response = self.backend.transverse(f, self.default_wall if wall is None else wall,
                                          self.beam_position if reference is None else reference,
                                          component=component)
        return self._transverse_output(response)

    def longitudinal_components(self, f, wall=None, *, source=None, witness=None):
        parts = self.backend.longitudinal_components(f, self.default_wall if wall is None else wall,
                self.beam_position if source is None else source, witness)
        return {name: self._longitudinal_output(values) for name, values in parts.items()}

    def transverse_components(self, f, wall=None, *, reference=None):
        parts = self.backend.transverse_components(f, self.default_wall if wall is None else wall,
                self.beam_position if reference is None else reference)
        return {name: self._transverse_output(response) for name, response in parts.items()}



@dataclass(frozen=True)
class GeometryInput:
    name: str
    units: str
    boundaries: tuple
    regions: tuple
    beam_position: tuple
    mesh: MeshSettings
    materials: dict = field(default_factory=dict)
    beta: float = 1.0
    roughness: object = None
    boundary_meshes: dict = field(default_factory=dict)

    @property
    def unit_scale(self):
        return UNIT_SCALE[self.units]

    def boundary(self, name="aperture"):
        for label, shape in self.boundaries:
            if label == name:
                return shape
        raise ValueError(f"Unknown boundary '{name}'; choose {', '.join(label for label, _ in self.boundaries)}.")

    def to_dict(self):
        data = {"format": FORMAT, "name": self.name, "units": self.units,
                "aperture": copy.deepcopy(self.boundaries[0][1].definition),
                "interfaces": [{"name": name, "shape": copy.deepcopy(shape.definition)}
                               for name, shape in self.boundaries[1:]],
                "regions": [dict(region) for region in self.regions],
                "beam": (np.asarray(self.beam_position)/self.unit_scale).tolist(),
                "mesh": self.mesh.input_values(self.unit_scale)}
        data["beta"] = self.beta
        if self.boundary_meshes:
            data["boundary_meshes"] = copy.deepcopy(self.boundary_meshes)
        if self.roughness is not None:
            data["roughness"] = {"model": "conductivity_gradient",
                "rms_height": self.roughness.rms_height/self.unit_scale,
                "slices": self.roughness.slices, "extent_sigma": self.roughness.extent_sigma}
            if self.roughness.interface_heights:
                data["roughness"]["interface_heights"] = (np.asarray(self.roughness.interface_heights)/self.unit_scale).tolist()
        if self.materials:
            data["materials"] = copy.deepcopy(self.materials)
        return data

    def save(self, path):
        """Save a portable JSON definition; CSV shapes are embedded as points."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2)+"\n")

    def mesh_settings(self, boundary="aperture", mesh=None):
        """Resolve shared defaults, saved boundary settings, then call overrides.

        Dictionary lengths use the geometry file's declared units. Unspecified
        boundary settings continue to inherit the shared defaults.
        """
        self.boundary(boundary)
        values = {**self.mesh.input_values(self.unit_scale),
                  **self.boundary_meshes.get(boundary, {})}
        if mesh is not None:
            _keys(mesh, values, "mesh")
            values.update(mesh)
        return MeshSettings.from_dict(values, self.unit_scale)

    def to_panels(self, boundary="aperture", mesh=None):
        """Build straight solver panels; override lengths use the input's units.

        Native curves are approximated only here, within curve_tolerance.
        Original line endpoints and genuine corners are retained exactly.
        Geometry resolution does not certify field/skin-depth convergence.
        """
        shape = self.boundary(boundary)
        settings = self.mesh_settings(boundary, mesh)
        corners = []
        for index, curve in enumerate(shape.curves):
            first = shape.curves[index-1].derivative(1.)
            second = curve.derivative(0.)
            corners.append(np.linalg.norm(first/np.linalg.norm(first)-second/np.linalg.norm(second)) > 1e-8)
        starts, ends = [], []
        for index, curve in enumerate(shape.curves):
            grading = settings.corner_grading if corners[index] or corners[(index+1) % len(corners)] else 1.
            minimum = int(np.ceil(settings.minimum_panels*curve.length()/shape.perimeter))
            if curve.kind == "line":
                count = max(1, minimum, int(np.ceil(grading*curve.length()/settings.max_panel_length)))
            else:
                axes, sweep = curve.data[1], abs(curve.data[-1])
                # Linear interpolation of a parametric curve has error <=
                # max|r''|*Delta(theta)^2/8. Also bound arc/panel length.
                count = max(4, minimum,
                    int(np.ceil(grading*sweep*max(axes)/settings.max_panel_length)),
                    int(np.ceil(grading*sweep*np.sqrt(max(axes)/(8*settings.curve_tolerance)))))
            u = np.linspace(0., 1., count+1)
            t = np.where(u <= .5, .5*(2*u)**grading, 1-.5*(2*(1-u))**grading)
            positions = curve.point(t)
            starts.extend(positions[:-1])
            ends.extend(positions[1:])
        starts, ends = np.asarray(starts), np.asarray(ends)
        # Join native segments exactly to suppress trig roundoff at arc ends.
        ends = np.roll(starts, -1, axis=0).copy()
        vectors = ends-starts
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("Mesh controls exceed floating-point geometry resolution; reduce corner_grading.")
        if np.max(lengths) > settings.max_panel_length*(1+1e-10):
            raise RuntimeError("Generated panel exceeds max_panel_length.")
        if not _simple_panels(starts, ends):
            raise ValueError("The native outline is valid, but its chord mesh intersects itself. "
                             "Reduce curve_tolerance and/or max_panel_length.")
        nodes, weights = leggauss(settings.integration_order)
        points = (starts[:, None, :]+(nodes[None, :, None]+1)/2*vectors[:, None, :]).reshape(-1, 2)
        quadrature_weights = (lengths[:, None]*weights[None, :]/2).ravel()
        tangent = np.repeat(vectors/lengths[:, None], settings.integration_order, axis=0)
        normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
        index = np.repeat(np.arange(len(starts)), settings.integration_order)
        arrays = [starts.copy(), starts, ends, points, quadrature_weights, tangent, normal, index]
        for array in arrays:
            array.flags.writeable = False
        return PolygonPanels(*arrays)

    def material_panels(self, mesh=None):
        """All physical solver interfaces, including generated roughness layers."""
        if self.roughness is None:
            return [(name,self.to_panels(name,mesh)) for name,_ in self.boundaries]
        wrapper = self.make_solver(mesh=mesh)
        if wrapper.default_wall is None:
            raise ValueError("Define the wall material to construct its roughness interfaces.")
        from .layered import material_boundaries
        boundaries,_ = material_boundaries(wrapper.backend,wrapper.backend._wall(wrapper.default_wall))
        return [("aperture" if i == 0 else f"gradient_{i}",p.panels) for i,p in enumerate(boundaries)]

    def make_solver(self, backend="maxwell", mesh=None, n=128, *, pipe_length_m=1.0):
        """Full Maxwell impedance with finite beta and nested conductors.

        Return totals for a uniform pipe of pipe_length_m metres (default 1.0).
        Explicit None selects per-length output. Cross-section drawing
        units do not change the metre units of this runtime parameter.
        Entrance/exit transitions and axial geometry variations are excluded.
        """
        from .fields import validate_pipe_length
        pipe_length_m = validate_pipe_length(pipe_length_m)
        if backend not in ("maxwell", "pec"):
            raise ValueError("Choose backend maxwell for impedance or pec for vacuum geometry fields. The MQS backend has been removed.")
        if not self.boundary().contains(self.beam_position):
            raise ValueError("The beam is outside or on the aperture; place it strictly inside.")
        panels = self.to_panels(mesh=mesh)
        if backend == "pec":
            from .pec_panels import PolygonPECSolver
            if self.beta != 1:
                raise ValueError("The standalone PEC geometry helper is Laplace-only; use MaxwellSolver for finite beta.")
            return GeometrySolver(backend, PolygonPECSolver(panels), self.beam_position,
                                  pipe_length_m=pipe_length_m)
        from .maxwell import MaxwellSolver
        from .materials import Metal, PEC, Vacuum, Air, Layer, Wall
        materials = []
        for region in self.regions[1:]:
            label = region["material"]
            properties = self.materials.get(label)
            if properties is not None and properties.get("kind") == "pec":
                if region is not self.regions[-1]:
                    raise NotImplementedError("PEC is supported as the outer termination; finite layers must be conducting materials.")
                if self.roughness is not None:
                    raise NotImplementedError("Conductivity-gradient roughness needs finite conductivity; disable it for PEC.")
                materials.append(PEC(name=label))
                continue
            if label in ("air", "vacuum") or (properties is not None and properties["conductivity"] <= 0):
                if region is not self.regions[-1]:
                    raise NotImplementedError("Air/vacuum is supported only as the exterior; finite insulating gaps are not implemented.")
                if self.beta == 1:
                    raise NotImplementedError("Open Air/Vacuum exterior requires beta<1; its beta=1 limit is not implemented.")
                if self.roughness is not None:
                    raise NotImplementedError("Open Air/Vacuum exterior with conductivity-gradient roughness is not validated.")
                if properties is not None and any(properties[key] != value for key,value in
                        (("conductivity",0.),("mu_r",1.),("epsilon_r",1.),("relaxation_time",0.))):
                    raise NotImplementedError("Open lossless exterior currently requires conductivity=0, mu_r=epsilon_r=1 and relaxation_time=0.")
                materials.append(Air() if label == "air" else Vacuum(name=label))
                continue
            if properties is None:
                if len(self.boundaries) != 1:
                    raise ValueError(f"Define material properties for '{label}' before solving its interfaces.")
                materials.append(None)
            else:
                materials.append(Metal(name=label, **properties))
        interfaces = tuple(self.to_panels(name,mesh) for name,_ in self.boundaries[1:]) or None
        if interfaces and not all(_nested_panels(a, b)
                                 for a, b in zip((panels, *interfaces), interfaces)):
            raise ValueError("The material interface chord meshes cross/touch. Refine the enclosing "
                             "boundary's curve tolerance or maximum panel length, or use matching "
                             "panel partitions for a very thin coating.")
        if interfaces and self.roughness is not None:
            raise NotImplementedError("Gradient roughness currently uses offset Wall layers; explicit interfaces need a depth-coordinate definition.")
        solver = MaxwellSolver(panels, beta=self.beta, roughness=self.roughness, interfaces=interfaces)
        default_wall = None
        if materials[-1] is not None:
            default_wall = Wall(materials[-1],tuple(Layer(m,0.) for m in materials[:-1])) if interfaces else materials[-1]
        return GeometrySolver(backend, solver, self.beam_position, default_wall, pipe_length_m)

    def report(self):
        details = []
        meshes = []
        for name, shape in self.boundaries:
            panels = self.to_panels(name)
            meshes.append(panels)
            low, high = shape.bounds
            polygon_area = .5*np.sum(_cross(panels.panel_start, panels.panel_end))
            details.append({"name": name, "shape": shape.kind,
                "mesh": self.mesh_settings(name).input_values(1.),
                "uses_boundary_mesh_settings": bool(self.boundary_meshes.get(name)),
                "area_m2": float(shape.area), "perimeter_m": float(shape.perimeter),
                "bounds_m": [low.tolist(), high.tolist()],
                "orientation_normalized_to_CCW": shape.orientation_reversed,
                "panels": len(panels.panel_start), "integration_points": panels.size,
                "maximum_panel_length_m": float(np.linalg.norm(panels.panel_end-panels.panel_start, axis=1).max()),
                "chord_mesh_area_relative_error": float(abs(polygon_area/shape.area-1))})
        meshes_nested = all(_nested_panels(inner, outer) for inner, outer in zip(meshes[:-1], meshes[1:]))
        return {"format": FORMAT, "name": self.name, "input_units": self.units,
                "solver_coordinate_units": "m", "boundary_count": len(details),
                "boundaries": details, "regions_inside_to_outside": list(self.regions),
                "materials": copy.deepcopy(self.materials),
                "exterior_is_unbounded": True,
                "beam_position_m": list(self.beam_position),
                "beam_strictly_inside_aperture": self.boundary().contains(self.beam_position),
                "native_interfaces_strictly_nested": True,
                "chord_mesh_interfaces_strictly_nested": meshes_nested,
                "mesh": self.mesh.input_values(1.),
                "multiple_interface_field_backend_available": True,
                "beam_beta": self.beta,
                "roughness": self.to_dict().get("roughness"),
                "generated_roughness_interfaces": len(self.material_panels())-1 if self.roughness is not None else 0}

    def draw(self, axes, show_mesh=True):
        """Draw native geometry and optional solver panels in the declared units."""
        from matplotlib import colormaps
        palette = colormaps["tab10"]
        for index, (name, shape) in enumerate(self.boundaries):
            points = np.concatenate([curve.point(np.linspace(0., 1., 301)) for curve in shape.curves])
            points = np.vstack((points, points[0]))/self.unit_scale
            label = name
            if show_mesh:
                from matplotlib.collections import LineCollection
                panels = self.to_panels(name)
                label += f" ({len(panels.panel_start)} panels)"
                joins = panels.panel_start/self.unit_scale
                segments = np.stack((joins, panels.panel_end/self.unit_scale), axis=1)
                colors = ["#0067b1" if panel % 2 == 0 else "#d45500" for panel in range(len(joins))]
                axes.add_collection(LineCollection(segments, colors=colors, linewidths=1.8))
                axes.plot(joins[:, 0], joins[:, 1], "o", color=palette(index), ms=3,
                          markerfacecolor="white", markeredgewidth=.6, label=label)
            axes.plot(points[:, 0], points[:, 1], color=palette(index), lw=.7 if show_mesh else 1.7,
                      label=None if show_mesh else label, zorder=0)
        if show_mesh and self.roughness is not None:
            from matplotlib.collections import LineCollection
            for index,(name,panels) in enumerate(self.material_panels()[1:]):
                starts,ends=panels.panel_start/self.unit_scale,panels.panel_end/self.unit_scale
                axes.add_collection(LineCollection(np.stack((starts,ends),axis=1),
                    colors="#007f64",linewidths=.6,label="Roughness material interfaces" if index == 0 else None))
        beam = np.asarray(self.beam_position)/self.unit_scale
        axes.plot(*beam, "+", color="#17202d" if self.boundary().contains(self.beam_position) else "#bd2929",
                  ms=8, mew=1.5, label="Beam")
        axes.set(xlabel=f"x [{self.units}]", ylabel=f"y [{self.units}]", aspect="equal", title=self.name)
        axes.grid(alpha=.15)
        axes.legend(fontsize=7, loc="upper right")
        return axes

    def preview(self, path=None, show_mesh=True):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(figsize=(7., 5.5), constrained_layout=True)
        self.draw(axes, show_mesh)
        if path is not None:
            _save_figure(fig, path)
        return fig

    def export_panels(self, path):
        """Export all boundaries together; coordinates are explicitly metres."""
        meshes = self.material_panels()
        if not all(_nested_panels(inner[1], outer[1]) for inner, outer in zip(meshes[:-1], meshes[1:])):
            raise ValueError("Native interfaces are valid, but their chord meshes cross/touch. "
                             "Refine curve_tolerance/max_panel_length before exporting the layered mesh.")
        with Path(path).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["boundary", "panel", "start_x_m", "start_y_m", "end_x_m", "end_y_m", "length_m"])
            for name, panels in meshes:
                for index, (a, b) in enumerate(zip(panels.panel_start, panels.panel_end)):
                    writer.writerow([name, index, *a, *b, np.linalg.norm(b-a)])


def load_geometry(source, *, units=None):
    """Load a JSON/dict case or a bare XY CSV with explicitly supplied units."""
    base = Path.cwd()
    if isinstance(source, dict):
        data = copy.deepcopy(source)
    else:
        path = Path(source)
        base = path.resolve().parent
        if path.suffix.lower() == ".dxf":
            from .geometry_io import import_geometry
            return import_geometry(path, units=units).geometry
        if path.suffix.lower() == ".csv":
            if units is None:
                raise ValueError("CSV coordinates have no implied units. Use load_geometry(path, units='mm').")
            data = {"units": units, "name": path.stem, "aperture": {"type": "csv", "file": path.name}}
        else:
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{error.lineno}: invalid JSON: {error.msg}") from None
    _keys(data, {"$schema", "format", "name", "units", "aperture", "interfaces", "regions", "beam", "mesh", "boundary_meshes", "materials", "beta", "roughness"}, "geometry")
    if data.get("format", FORMAT) != FORMAT:
        raise ValueError(f"Use format '{FORMAT}'.")
    if not isinstance(data.get("units"), str) or data["units"] not in UNIT_SCALE:
        raise ValueError("Specify units explicitly: m, mm, um, µm or nm.")
    if units is not None and data["units"] != units:
        raise ValueError("The units argument conflicts with the geometry file's units.")
    unit, scale = data["units"], UNIT_SCALE[data["units"]]
    try:
        boundaries = [("aperture", _shape(data.get("aperture"), scale, base))]
        interfaces = data.get("interfaces", [])
        if not isinstance(interfaces, list):
            raise ValueError("interfaces: use a list, ordered from the beam aperture outward.")
        for interface in interfaces:
            _keys(interface, {"name", "shape"}, "interface")
            name = interface.get("name")
            if not isinstance(name, str) or not name.strip() or name in [label for label, _ in boundaries]:
                raise ValueError("Give each interface a unique nonempty name; aperture is reserved.")
            shape = _shape(interface.get("shape"), scale, base)
            if not _nested(boundaries[-1][1], shape):
                raise ValueError(f"Interface '{name}' must strictly enclose '{boundaries[-1][0]}' without crossings or touching.")
            boundaries.append((name, shape))
    except (TypeError, KeyError) as error:
        raise ValueError(f"Incomplete or invalid shape input: {error}") from None
    regions = data.get("regions", [{"name": "vacuum", "material": "vacuum"}]+[
        {"name": f"region_{i+1}", "material": "unspecified"} for i in range(len(boundaries))])
    if not isinstance(regions, list) or len(regions) != len(boundaries)+1:
        raise ValueError("regions: supply one region inside the aperture, one per layer, and one outside the last boundary.")
    names = []
    for region in regions:
        _keys(region, {"name", "material"}, "region")
        if any(not isinstance(region.get(key), str) or not region[key].strip() for key in ["name", "material"]):
            raise ValueError("Each region needs nonempty name and material labels.")
        if region["name"] in names:
            raise ValueError("Region names must be unique; material labels can repeat.")
        names.append(region["name"])
    if regions[0]["material"] != "vacuum":
        raise ValueError("The innermost region must be vacuum; assign material 'vacuum' inside the aperture.")
    materials = data.get("materials", {})
    if not isinstance(materials, dict):
        raise ValueError("materials: use a dictionary of material names and SI properties.")
    materials = copy.deepcopy(materials)
    if any(region["material"] == "PEC" for region in regions):
        materials.setdefault("PEC", {"kind": "pec"})
    for label, properties in materials.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Material names must be nonempty strings.")
        _keys(properties, {"kind", "conductivity", "mu_r", "epsilon_r", "relaxation_time"}, f"material {label}")
        if "kind" in properties:
            if properties["kind"] != "pec" or label in ("vacuum", "air"):
                raise ValueError("The only special material kind is 'pec'; vacuum/air cannot be redefined as PEC.")
            _keys(properties, {"kind"}, f"PEC material {label}")
            continue
        if label == "PEC":
            raise ValueError("Define PEC as {'kind': 'pec'}; it has no finite conductivity parameters.")
        properties["conductivity"] = _number(properties.get("conductivity"), f"{label}.conductivity")
        if properties["conductivity"] < 0:
            raise ValueError("Conductivity must be nonnegative, in S/m.")
        for key in ("mu_r", "epsilon_r"):
            properties[key] = _number(properties.get(key, 1.), f"{label}.{key}", positive=True)
        properties["relaxation_time"] = _number(properties.get("relaxation_time", 0.), f"{label}.relaxation_time")
        if properties["relaxation_time"] < 0:
            raise ValueError("relaxation_time must be nonnegative, in seconds.")
        if label == "vacuum" and properties != {"conductivity": 0., "mu_r": 1., "epsilon_r": 1., "relaxation_time": 0.}:
            raise ValueError("Vacuum must have conductivity=0, mu_r=epsilon_r=1 and relaxation_time=0.")
    mesh = MeshSettings.from_dict(data.get("mesh", {}), scale)
    boundary_meshes = data.get("boundary_meshes", {})
    if not isinstance(boundary_meshes, dict):
        raise ValueError("boundary_meshes: use an object keyed by physical boundary name.")
    normalized_meshes = {}
    for label, values in boundary_meshes.items():
        if label not in [name for name, _ in boundaries]:
            raise ValueError(f"boundary_meshes: unknown physical boundary '{label}'.")
        _keys(values, mesh.input_values(scale), f"boundary_meshes.{label}")
        settings = MeshSettings.from_dict({**mesh.input_values(scale), **values}, scale)
        if values:
            normalized_meshes[label] = {key: settings.input_values(scale)[key] for key in values}
    beam = tuple(_point(data.get("beam", [0., 0.]), scale, "beam"))
    name = data.get("name", "ARCS geometry")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name: enter a nonempty geometry title.")
    from .maxwell import validate_beta
    beta = validate_beta(_number(data.get("beta", 1.0), "beta"))
    roughness = None
    if data.get("roughness") is not None:
        from .roughness import Roughness
        values = data["roughness"]
        _keys(values, {"model", "rms_height", "slices", "extent_sigma", "interface_heights"}, "roughness")
        if values.get("model", "conductivity_gradient") != "conductivity_gradient":
            raise ValueError("The supported roughness model is conductivity_gradient.")
        height = _number(values.get("rms_height"), "roughness.rms_height", positive=True)*scale
        heights = tuple(_number(x,"roughness.interface_heights",positive=True)*scale
                        for x in values.get("interface_heights", []))
        roughness = Roughness(height, values.get("slices",32), values.get("extent_sigma",5.), heights)
    return GeometryInput(name, unit, tuple(boundaries), tuple(regions), beam, mesh,
                         materials, beta, roughness, normalized_meshes)


def main(argv=None):
    """Validate a case, preview it and export mesh data without running physics."""
    parser = argparse.ArgumentParser(description="Check and preview an ARCS 2D geometry file.")
    parser.add_argument("input", type=Path, help="Geometry JSON or an XY CSV")
    parser.add_argument("--units", choices=UNIT_SCALE, help="Required for a bare CSV")
    parser.add_argument("--preview", type=Path, help="Save a geometry/panel PNG")
    parser.add_argument("--export-panels", type=Path, help="Export all boundary panels as CSV in metres")
    parser.add_argument("--summary", type=Path, help="Save a machine-readable geometry report")
    args = parser.parse_args(argv)
    try:
        geometry = load_geometry(args.input, units=args.units)
        report = geometry.report()
        if args.preview:
            import matplotlib.pyplot as plt
            plt.close(geometry.preview(args.preview))
        if args.export_panels:
            geometry.export_panels(args.export_panels)
        if args.summary:
            args.summary.write_text(json.dumps(report, indent=2)+"\n")
        print(f"{geometry.name}: {geometry.units} input -> metres for the solver")
        print(f"{len(geometry.boundaries)} closed boundary/boundaries; region order: "
              + " -> ".join(region["material"] for region in geometry.regions))
        for boundary in report["boundaries"]:
            print(f"  {boundary['name']}: {boundary['panels']} panels, "
                  f"{boundary['integration_points']} integration points")
        if not report["beam_strictly_inside_aperture"]:
            print("  Beam is outside/on the aperture; set beam before requesting a field solve.")
        if len(geometry.boundaries) > 1:
            print("  Conducting interfaces use full-Maxwell field matching; open Air/Vacuum exterior requires beta<1.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    main()
