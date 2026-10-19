"""Smooth, simply connected, counterclockwise pipe contours in metres."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class Contour:
    points: np.ndarray
    first: np.ndarray
    second: np.ndarray
    label: str = "smooth contour"

    def __post_init__(self):
        arrays = [np.asarray(a, dtype=float).copy() for a in
                  (self.points, self.first, self.second)]
        n = len(arrays[0])
        if n < 16 or n % 2 or any(a.shape != (n, 2) for a in arrays):
            raise ValueError("Use an even number >=16 of equally spaced parameter samples, shape (N,2).")
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Contour contains nonfinite values.")
        if np.any(np.linalg.norm(arrays[1], axis=1) <= 0):
            raise ValueError("The contour parametrization must have nonzero speed.")
        signed_area = .5 * np.sum(arrays[0][:, 0] * np.roll(arrays[0][:, 1], -1)
                                   - arrays[0][:, 1] * np.roll(arrays[0][:, 0], -1))
        if signed_area <= 0:
            raise ValueError("Provide a counterclockwise contour.")
        # Reject sampled intersections; a heavily undersampled spline still needs inspection.
        r = arrays[0]
        def cross(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        for i in range(n):
            a, b = r[i], r[(i + 1) % n]
            j = np.arange(i + 2, n)
            if i == 0:
                j = j[j != n - 1]
            if not len(j):
                continue
            cc, dd = r[j], r[(j + 1) % n]
            if np.any((cross(b-a, cc-a) * cross(b-a, dd-a) < 0)
                      & (cross(dd-cc, a-cc) * cross(dd-cc, b-cc) < 0)):
                raise ValueError("Self-intersecting contours are unsupported.")
        for key, value in zip(("points", "first", "second"), arrays):
            value.flags.writeable = False
            object.__setattr__(self, key, value)

    @property
    def size(self):
        return len(self.points)

    @property
    def speed(self):
        return np.linalg.norm(self.first, axis=1)

    @property
    def tangent(self):
        return self.first / self.speed[:, None]

    @property
    def normal(self):
        t = self.tangent
        return np.column_stack((t[:, 1], -t[:, 0]))

    @property
    def weights(self):
        return 2 * np.pi / self.size * self.speed

    @property
    def curvature(self):
        return (self.first[:, 0] * self.second[:, 1]
                - self.first[:, 1] * self.second[:, 0]) / self.speed**3

    def contains(self, point):
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            return False
        a = self.points - p
        b = np.roll(a, -1, axis=0)
        angles = np.arctan2(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0], (a*b).sum(axis=1))
        return abs(angles.sum()) > np.pi


def circle(radius, n=128):
    return ellipse(radius, radius, n=n, label=f"circle, radius {radius:g} m")


def ellipse(a, b, n=128, label=None):
    if a <= 0 or b <= 0:
        raise ValueError("Semiaxes must be positive.")
    theta = 2*np.pi*np.arange(n)/n
    r = np.column_stack((a*np.cos(theta), b*np.sin(theta)))
    first = np.column_stack((-a*np.sin(theta), b*np.cos(theta)))
    return Contour(r, first, -r, label or f"ellipse, semiaxes {a:g}, {b:g} m")


def sampled_smooth_contour(points, n=128, label="periodic spline contour"):
    """Interpolate ordered CCW points with a periodic cubic spline.

    The spline is the geometry: it may round corners or overshoot input points.
    This is deliberately NOT a sharp-polygon or CAD importer.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 5 or not np.isfinite(p).all():
        raise ValueError("Provide at least five finite ordered (x,y) points.")
    if np.allclose(p[-1], p[0], rtol=0, atol=1e-14):
        p = p[:-1]
    closed = np.vstack((p, p[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Consecutive points must differ.")
    theta = np.r_[0., np.cumsum(lengths)] / lengths.sum() * 2*np.pi
    spline = CubicSpline(theta, closed, bc_type="periodic")
    grid = 2*np.pi*np.arange(n)/n
    return Contour(spline(grid), spline(grid, 1), spline(grid, 2), label)
