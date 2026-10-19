"""Numerical Green-function quadrature, without a wall physics model."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad_vec
from .special import decaying_k as kv
from .pec_panels import PolygonPECSolver


class PanelKernels:
    def __init__(self, panels, kernel_order=12, logarithm_reference=None):
        if (isinstance(kernel_order, (bool, np.bool_))
                or not isinstance(kernel_order, (int, np.integer))
                or kernel_order < 4 or kernel_order % 2):
            raise ValueError("Use an even kernel_order of at least 4.")
        self.pec = PolygonPECSolver(panels, logarithm_reference)
        self.panels = panels
        self.kernel_order = int(kernel_order)
        p = self.pec
        displacement = p.centres[:, None, :]-p.starts[None, :, :]
        projected = np.einsum("ijk,jk->ij", displacement, p.tangent)
        self._height = np.einsum("ijk,jk->ij", displacement, p.normal)
        nodes, weights = leggauss(self.kernel_order)
        positions = (p.starts[:, None, :]+(nodes[None, :, None]+1)/2
                     * (p.ends-p.starts)[:, None, :])
        self._positions = positions
        self._displacement = p.centres[:, None, None, :]-positions[None, :, :, :]
        self._distance = np.linalg.norm(
            p.centres[:, None, None, :]-positions[None, :, :, :], axis=-1)
        self._weights = p.lengths[:, None]*weights[None, :]/2
        self._identity = np.eye(len(p.lengths))

    def _conductor_operators(self, q):
        argument = q*self._distance
        single = np.sum(kv(0, argument)*self._weights[None, :, :], axis=-1)/(2*np.pi)
        double = (np.sum(q*kv(1, argument)/self._distance
                         * self._weights[None, :, :], axis=-1)
                  * self._height/(2*np.pi))
        # The logarithmic self singularity is integrated adaptively, separately
        # from smooth off-panel Gauss quadrature. Exponential tails beyond 40
        # skin depths are negligible at the stated tolerances.
        upper = np.minimum(self.pec.lengths/2, 40/q.real)
        self_integral = upper*quad_vec(
            lambda t: kv(0, q*upper*t), 0., 1., epsabs=1e-12, epsrel=1e-11)[0]/np.pi
        np.fill_diagonal(single, self_integral)
        np.fill_diagonal(double, 0.)
        return single, double

