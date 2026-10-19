"""Stable modified-Bessel evaluation for the Maxwell Green functions."""
import numpy as np
from scipy.special import kve


def decaying_k(order, argument):
    """Return K_order(z) through its exponentially scaled representation.

    K_order(z) = kve(order, z) * exp(-z). In the right half-plane this
    avoids the faulty underflow unscaling in some older SciPy AMOS builds
    (scipy/xsf#46, scipy/scipy#23330). Values below floating-point range
    underflow normally to zero. No frequency cutoff or wall approximation
    enters this identity.
    """
    argument = np.asarray(argument)
    with np.errstate(under="ignore"):
        return kve(order, argument) * np.exp(-argument)
