"""Physical impedance result containers."""
from dataclasses import dataclass, replace
from numbers import Real
import numpy as np


def validate_pipe_length(pipe_length_m):
    """None selects per-length output; a positive finite length selects totals."""
    if pipe_length_m is None:
        return None
    if (isinstance(pipe_length_m, (bool, np.bool_))
            or not isinstance(pipe_length_m, Real)
            or not np.isfinite(pipe_length_m) or pipe_length_m <= 0):
        raise ValueError("pipe_length_m must be a positive finite length in metres, or None for per-length output.")
    return float(pipe_length_m)


@dataclass(frozen=True)
class TransverseImpedance:
    """Linear transverse expansion about one source/witness reference point.

    dipolar[..., i,j] multiplies the source offset j for a kick in direction i.
    quadrupolar[..., i,j] multiplies the witness offset j. Both are Ohm/m²,
    per unit pipe length. constant is Ohm/m; longitudinal is Ohm/m.
    When pipe_length_m is set, these are totals for a uniform pipe:
    dipolar/quadrupolar are Ohm/m; constant and longitudinal are Ohm.
    Axes are (x,y). All values use ARCS's field/Fourier convention.
    """
    longitudinal: np.ndarray
    dipolar: np.ndarray
    quadrupolar: np.ndarray
    constant: np.ndarray
    reference_position_m: tuple

    beam_beta: float = 1.0
    component: str = "wall"
    field_convention: str = "exp(-i*k*z+i*omega*t)"
    pipe_length_m: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "pipe_length_m", validate_pipe_length(self.pipe_length_m))

    def for_pipe_length(self, pipe_length_m):
        """Scale a per-length response once, preserving its reference and sign.

        The 2D fields assume a uniform pipe. Entrance/exit effects are excluded.
        A previously integrated response cannot be scaled again by this method.
        """
        length = validate_pipe_length(pipe_length_m)
        if length is None:
            raise ValueError("Supply a positive pipe_length_m to integrate a per-length response.")
        if self.pipe_length_m is not None:
            raise ValueError("This response already includes pipe length; obtain a per-length response before scaling.")
        return replace(self, longitudinal=self.longitudinal*length,
                       dipolar=self.dipolar*length, quadrupolar=self.quadrupolar*length,
                       constant=self.constant*length, pipe_length_m=length)
