"""Corner-preserving boundary panels and optional confocal geometry helpers.

Straight panels feed PEC reference fields and full-Maxwell material matching.
Original sharp vertices are retained, including in nested material interfaces.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from .geometry import Contour, ellipse


@dataclass(frozen=True)
class PolygonPanels:
    """A true polygon with Gauss nodes inside straight panels, never at vertices.

    Each original vertex remains an exact corner. Panel lengths are graded
    toward both ends of each edge. PolygonPECSolver uses one constant field
    density per panel. Gauss nodes are integration points, not extra unknowns.
    The existing smooth periodic Nyström operator is incompatible with them.
    """
    vertices: np.ndarray
    panel_start: np.ndarray
    panel_end: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    panel_index: np.ndarray

    @property
    def size(self):
        return len(self.points)

    @property
    def perimeter(self):
        return float(self.weights.sum())


def polygon_panels(vertices, panels_per_edge=8, order=4, grading_power=2.):
    """Sample a simple CCW polygon without smoothing or changing its corners.

    panels_per_edge is a positive integer or one positive integer per edge.
    Different edge counts keep a wide rectangle resolved near the beam.
    """
    p = np.asarray(vertices, float).copy()
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Provide at least three finite CCW polygon vertices.")
    if np.array_equal(p[-1], p[0]):
        p = p[:-1]
    if len(p) < 3:
        raise ValueError("At least three distinct vertices are required.")
    edges = np.roll(p, -1, axis=0)-p
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Consecutive polygon vertices must differ.")
    area = .5*np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1))
    if area <= 0:
        raise ValueError("Use a CCW polygon with positive area.")
    if isinstance(panels_per_edge, (int, np.integer)):
        counts = np.full(len(p), panels_per_edge, dtype=int)
    else:
        counts = np.asarray(panels_per_edge)
    if (counts.shape != (len(p),) or counts.dtype.kind not in "iu" or np.any(counts < 1)
            or not isinstance(order, (int, np.integer)) or order < 1
            or not np.isfinite(grading_power) or grading_power < 1):
        raise ValueError("Use positive integer panel/order counts and grading_power >= 1.")
    scale = np.max(np.ptp(p, axis=0))
    tolerance = 64*np.finfo(float).eps*scale**2

    def cross(a, b):
        return a[0]*b[1]-a[1]*b[0]

    def on_segment(a, b, x):
        return (abs(cross(b-a, x-a)) <= tolerance
                and np.dot(x-a, x-b) <= tolerance)

    for i in range(len(p)):
        a, b = p[i], p[(i+1) % len(p)]
        for j in range(i+1, len(p)):
            if j == i+1 or (i == 0 and j == len(p)-1):
                continue
            c, d = p[j], p[(j+1) % len(p)]
            ac, ad = cross(b-a, c-a), cross(b-a, d-a)
            ca, cb = cross(d-c, a-c), cross(d-c, b-c)
            proper = ac*ad < 0 and ca*cb < 0
            touching = (on_segment(a, b, c) or on_segment(a, b, d)
                        or on_segment(c, d, a) or on_segment(c, d, b))
            if proper or touching:
                raise ValueError("Self-intersecting or touching polygon edges are unsupported.")
    starts, ends = [], []
    for vertex, edge, count in zip(p, edges, counts):
        u = np.linspace(0., 1., int(count)+1)
        fractions = np.where(u <= .5, .5*(2*u)**grading_power,
                             1-.5*(2*(1-u))**grading_power)
        starts.append(vertex+fractions[:-1, None]*edge)
        ends.append(vertex+fractions[1:, None]*edge)
    starts, ends = np.concatenate(starts), np.concatenate(ends)
    vectors = ends-starts
    panel_lengths = np.linalg.norm(vectors, axis=1)
    if np.any(panel_lengths <= 0):
        raise ValueError("Panel grading exceeds floating-point geometry resolution.")
    nodes, gauss_weights = leggauss(order)
    points = (starts[:, None, :]+(nodes[None, :, None]+1)/2*vectors[:, None, :]).reshape(-1, 2)
    weights = (panel_lengths[:, None]*gauss_weights[None, :]/2).ravel()
    tangent = np.repeat(vectors/panel_lengths[:, None], order, axis=0)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    index = np.repeat(np.arange(len(starts)), order)
    arrays = [p, starts, ends, points, weights, tangent, normal, index]
    for array in arrays:
        array.flags.writeable = False
    return PolygonPanels(*arrays)


@dataclass(frozen=True)
class ConfocalCoating:
    """Two actual confocal interfaces; thickness varies around the perimeter."""
    inner: Contour
    outer: Contour
    inner_axes: tuple
    outer_axes: tuple
    focal_distance: float
    delta_u: float

    def normal_thickness(self, theta):
        return ellipse_normal_thickness(theta, self.inner_axes, self.outer_axes)


def confocal_coating(a, b, thickness_at_y, n=128):
    """Build the confocal geometry used in the Wang multilayer benchmark family.

    a>b>0 are the inner semiaxes. thickness_at_y is the coating thickness at
    the positive minor-axis endpoint. No uniform-normal-thickness assumption
    is made. This constructs geometry only, not Wang's impedance equations.
    """
    if not (np.isfinite([a, b, thickness_at_y]).all()
            and a > b > 0 and thickness_at_y > 0):
        raise ValueError("Use finite a>b>0 and a positive minor-axis coating thickness.")
    focal = np.sqrt(a*a-b*b)
    outer_b = b+thickness_at_y
    outer_a = np.sqrt(outer_b**2+focal**2)
    du = np.arcsinh(outer_b/focal)-np.arcsinh(b/focal)
    return ConfocalCoating(ellipse(a, b, n, "inner confocal interface"),
                           ellipse(outer_a, outer_b, n, "outer confocal interface"),
                           (a, b), (outer_a, outer_b), float(focal), float(du))


def ellipse_normal_thickness(theta, inner_axes, outer_axes):
    """Intersect the outward inner-ellipse normal with an enclosing outer ellipse.

    theta parametrizes x=a*cos(theta), y=b*sin(theta); it is not the polar
    angle in Cartesian coordinates. The intersection is solved geometrically,
    rather than identified with a radial or elliptic-coordinate difference.
    """
    a, b = np.asarray(inner_axes, float)
    outer_a, outer_b = np.asarray(outer_axes, float)
    if not (np.isfinite([a, b, outer_a, outer_b]).all()
            and a > 0 and b > 0 and outer_a > a and outer_b > b):
        raise ValueError("Both outer semiaxes must exceed the positive inner semiaxes.")
    theta = np.asarray(theta, float)
    if not np.isfinite(theta).all():
        raise ValueError("Angles must be finite.")
    point = np.stack((a*np.cos(theta), b*np.sin(theta)), axis=-1)
    normal = np.stack((np.cos(theta)/a, np.sin(theta)/b), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    denominator = np.array([outer_a**2, outer_b**2])
    aa = np.sum(normal**2/denominator, axis=-1)
    bb = 2*np.sum(point*normal/denominator, axis=-1)
    cc = np.sum(point**2/denominator, axis=-1)-1
    # Stable positive root for a very thin layer: avoid subtracting two large numbers.
    return -2*cc/(bb+np.sqrt(bb*bb-4*aa*cc))
