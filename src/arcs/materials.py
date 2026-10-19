"""Material data for Maxwell fields and independent circular references."""
from dataclasses import dataclass
import numpy as np
from .constants import EPS0


def frequency_array(f):
    a = np.asarray(f, dtype=float)
    if np.any(~np.isfinite(a)) or np.any(a < 0):
        raise ValueError("Frequencies must be finite and nonnegative, in Hz.")
    return a


@dataclass(frozen=True)
class PEC:
    """Ideal conducting termination: tangential E is zero, with no bulk sigma.

    Use directly for a bare PEC pipe or as Wall.substrate behind finite
    conducting layers. PEC is an exact boundary condition, not a large
    conductivity approximation or a finite coating material.
    """
    name: str = "PEC"


@dataclass(frozen=True)
class Vacuum:
    """Unbounded lossless exterior, with sigma=0 and epsilon_r=mu_r=1.

    At finite beam beta the synchronous exterior field decays with
    kappa=omega*sqrt(1-beta**2)/(beta*c). This is an open material region,
    not a PEC condition. Only an exterior termination is supported.
    """
    name: str = "Vacuum"

    @property
    def conductivity(self): return 0.0

    @property
    def mu_r(self): return 1.0

    @property
    def epsilon_r(self): return 1.0

    @property
    def relaxation_time(self): return 0.0

    def sigma(self, f):
        return np.zeros(frequency_array(f).shape, complex)

    def admittivity(self, f):
        return 2j*np.pi*frequency_array(f)*EPS0

    def transverse_wavenumber_squared(self, f, beta=1.0):
        from .constants import C0
        omega=2*np.pi*frequency_array(f)
        return (omega/C0)**2*(1-beta)*(1+beta)/(beta*beta)


@dataclass(frozen=True)
class Air(Vacuum):
    """Air exterior approximated by vacuum; no dielectric/atmospheric model."""
    name: str = "Air"


@dataclass(frozen=True)
class Metal:
    conductivity: float
    name: str = "metal"
    mu_r: float = 1.0
    epsilon_r: float = 1.0
    relaxation_time: float = 0.0

    def __post_init__(self):
        if not (np.isfinite(self.conductivity) and self.conductivity > 0
                and np.isfinite(self.mu_r) and self.mu_r > 0
                and np.isfinite(self.epsilon_r) and self.epsilon_r > 0
                and np.isfinite(self.relaxation_time) and self.relaxation_time >= 0):
            raise ValueError("Use positive finite conductivity, mu_r, epsilon_r and nonnegative relaxation time.")

    def sigma(self, f):
        f = frequency_array(f)
        return self.conductivity / (1 + 2j*np.pi*f*self.relaxation_time)

    def admittivity(self, f):
        f = frequency_array(f)
        return self.sigma(f) + 2j*np.pi*f*EPS0*self.epsilon_r

    def transverse_wavenumber_squared(self, f, beta=1.0):
        """Synchronous Maxwell q², with stable vacuum-term cancellation.

        Admittivity in the field coupling still retains displacement
        current. This form prevents roundoff from dominating q² in a
        very weakly conducting Gaussian tail at beta=1.
        """
        from .constants import C0, MU0
        omega = 2*np.pi*frequency_array(f)
        vacuum = (1-beta)*(1+beta)/(beta*beta)
        dielectric = 1-self.mu_r*self.epsilon_r
        return (omega/C0)**2*(vacuum+dielectric)+1j*omega*MU0*self.mu_r*self.sigma(f)

@dataclass(frozen=True)
class Layer:
    metal: Metal
    thickness: float

    def __post_init__(self):
        if not isinstance(self.metal, Metal):
            raise ValueError("Finite layers require a Metal; use PEC as the outer termination.")
        if not np.isfinite(self.thickness) or self.thickness < 0:
            raise ValueError("Layer thickness must be finite and nonnegative, in metres.")


@dataclass(frozen=True)
class Wall:
    """Finite conductors followed by a conductor, PEC or open Vacuum/Air.

    Layers are ordered from vacuum outward. With MaxwellSolver alone their
    thicknesses are normal polygon offsets. Explicit solver.interfaces use
    independent enclosing PolygonPanels and ignore Layer.thickness.
    circular_exact interprets the same thicknesses as radial circular layers.
    """
    substrate: Metal | PEC | Vacuum
    layers: tuple = ()

    def __post_init__(self):
        layers = tuple(self.layers)
        if not isinstance(self.substrate, (Metal, PEC, Vacuum)) or not all(isinstance(x, Layer) for x in layers):
            raise ValueError("Use Metal, PEC, Vacuum or Air as the exterior, and conducting Layer objects.")
        object.__setattr__(self, "layers", layers)

