"""PEC vacuum wall fields on exact polygons.

This is a Dirichlet Poisson boundary-element solve, with one constant density
per straight panel and exact logarithmic panel integration. It is not the
coupled Maxwell material-interface solver. It supplies geometry and the
vacuum reference fields used by the Maxwell backend.
No circle, rectangle or parallel-plate reference enters the assembly.
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import xlogy
from .interfaces import PolygonPanels


class PolygonPECSolver:
    """PEC image-current distribution for a beta=1 line beam in a simple polygon."""

    def __init__(self, panels, logarithm_reference=None):
        if not isinstance(panels, PolygonPanels):
            raise TypeError("Supply PolygonPanels from polygon_panels().")
        self.panels = panels
        self.starts, self.ends = panels.panel_start, panels.panel_end
        vectors = self.ends-self.starts
        self.lengths = np.linalg.norm(vectors, axis=1)
        self.tangent = vectors/self.lengths[:, None]
        self.normal = np.column_stack((self.tangent[:, 1], -self.tangent[:, 0]))
        self.centres = (self.starts+self.ends)/2
        self.scale = np.max(np.ptp(panels.vertices, axis=0))
        # Larger than the domain avoids the 2D single-layer capacity degeneracy.
        self.logarithm_reference = (4*self.scale if logarithm_reference is None
                                   else float(logarithm_reference))
        if not np.isfinite(self.logarithm_reference) or self.logarithm_reference <= self.scale:
            raise ValueError("The logarithm reference must exceed the polygon extent.")
        displacement = self.centres[:, None, :]-self.starts[None, :, :]
        projected = np.einsum("ijk,jk->ij", displacement, self.tangent)
        height = abs(np.einsum("ijk,jk->ij", displacement, self.normal))

        def primitive(u):
            # xlogy handles the finite u=height=0 limit exactly.
            return (.5*xlogy(u, u*u+height*height)-u
                    + height*np.arctan2(u, height))

        integral = (primitive(self.lengths[None, :]-projected)-primitive(-projected)
                    - self.lengths[None, :]*np.log(self.logarithm_reference))
        self.single_layer = -integral/(2*np.pi)
        np.fill_diagonal(self.single_layer, self.lengths/(2*np.pi)*(
            1-np.log(self.lengths/(2*self.logarithm_reference))))
        self._lu = lu_factor(self.single_layer, check_finite=True)
        self._density_cache = {}

    def _source(self, source):
        point = np.asarray(source, float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Use a finite (x, y) source point in metres.")
        vertices = self.panels.vertices
        edges = np.roll(vertices, -1, axis=0)-vertices
        local = point-vertices
        fraction = np.clip(np.sum(local*edges, axis=1)/np.sum(edges*edges, axis=1), 0., 1.)
        distance = np.linalg.norm(local-fraction[:, None]*edges, axis=1)
        if distance.min() <= 64*np.finfo(float).eps*self.scale:
            raise ValueError("Source and witness must be strictly inside the polygon.")
        first, second = vertices-point, np.roll(vertices, -1, axis=0)-point
        angle = np.arctan2(first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0],
                           np.sum(first*second, axis=1)).sum()
        if abs(angle) < np.pi:
            raise ValueError("Source and witness must be strictly inside the polygon.")
        return tuple(point)

    def panel_density(self, source=(0., 0.)):
        """Constant PEC image current / beam current on each panel, in 1/m."""
        point = self._source(source)
        if point not in self._density_cache:
            radius = np.linalg.norm(self.centres-np.asarray(point), axis=1)
            drive = -np.log(radius/self.logarithm_reference)/(2*np.pi)
            self._density_cache[point] = lu_solve(self._lu, drive, check_finite=False)
        return self._density_cache[point].copy()

    def wall_density(self, source=(0., 0.)):
        """Return the panel density at all Gauss integration points, in 1/m."""
        return self.panel_density(source)[self.panels.panel_index]

    def source_derivatives(self, source=(0., 0.)):
        """Return p, dp/dsource and d²p/dsource² at a fixed interior point.

        The logarithmic drive is differentiated analytically, then the same
        vacuum boundary matrix is solved for all six right-hand sides.
        No offset fitting or finite differences enter these derivatives.
        """
        point = self._source(source)
        delta = self.centres-np.asarray(point)
        radius2 = np.sum(delta*delta, axis=1)
        drive = -np.log(np.sqrt(radius2)/self.logarithm_reference)/(2*np.pi)
        gradient = delta/(2*np.pi*radius2[:, None])
        hessian = (2*delta[:, :, None]*delta[:, None, :]/radius2[:, None, None]**2
                   - np.eye(2)[None, :, :]/radius2[:, None, None])/(2*np.pi)
        rhs = np.column_stack((drive, gradient, hessian[:, 0, 0],
                               hessian[:, 0, 1], hessian[:, 1, 1]))
        solution = lu_solve(self._lu, rhs, check_finite=False)
        second = np.empty((len(drive), 2, 2))
        second[:, 0, 0], second[:, 0, 1] = solution[:, 3], solution[:, 4]
        second[:, 1, 0], second[:, 1, 1] = solution[:, 4], solution[:, 5]
        return solution[:, 0], solution[:, 1:3], second

    def surface_integral(self, source=(0., 0.), witness=None):
        """Return integral p_source*p_witness ds, in 1/m; no conductor model."""
        p = self.panel_density(source)
        other = p if witness is None else self.panel_density(witness)
        return float(np.sum(self.lengths*p*other))

    def diagnostics(self, source=(0., 0.)):
        p = self.panel_density(source)
        radius = np.linalg.norm(self.centres-np.asarray(source), axis=1)
        drive = -np.log(radius/self.logarithm_reference)/(2*np.pi)
        return {"panels": len(p), "integration_points": self.panels.size,
                "image_current_integral": float(self.lengths @ p),
                "relative_collocation_residual": float(np.linalg.norm(
                    self.single_layer @ p-drive)/np.linalg.norm(drive)),
                "surface_integral_per_m": self.surface_integral(source),
                "full_material_interface_solver": False,
                "short_range_operator": False}
