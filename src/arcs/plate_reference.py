"""Independent PEC rectangle/plate vacuum fields and geometry integrals.

These separation-of-variables formulas do not compute conductor penetration
or resistive-wall impedance. They audit the vacuum boundary operator only.
"""
import numpy as np


def _axes(a, b):
    a, b = float(a), float(b)
    if not np.isfinite([a, b]).all() or a <= 0 or b <= 0:
        raise ValueError("Use finite positive horizontal/vertical half apertures.")
    return a, b


def _sech(x):
    e = np.exp(-abs(np.asarray(x)))
    return 2*e/(1+e*e)


def rectangle_surface_integral(a, b, cutoff=40.):
    """Centred PEC geometry factor integral p^2 ds, in 1/m.

    Rectangle: x in [-a,a], y in [-b,b]. Modes above the dimensionless
    hyperbolic argument cutoff are exponentially negligible.
    """
    a, b = _axes(a, b)
    if not np.isfinite(cutoff) or cutoff < 20:
        raise ValueError("Use a finite mode cutoff >= 20.")
    total = 0.
    for half_length, half_gap in [(a, b), (b, a)]:
        modes = np.arange(max(1, int(np.ceil(cutoff*half_length/(np.pi*half_gap)))))
        argument = np.pi*(modes+.5)*half_gap/half_length
        total += np.sum(_sech(argument)**2)/(2*half_length)
    return float(total)


def rectangle_top_density(x, a, b, cutoff=40.):
    """Centred PEC wall current / beam current at y=b, in 1/m."""
    a, b = _axes(a, b)
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or np.any(abs(x) > a):
        raise ValueError("x must lie on the top face [-a,a].")
    if not np.isfinite(cutoff) or cutoff < 20:
        raise ValueError("Use a finite mode cutoff >= 20.")
    modes = np.arange(max(1, int(np.ceil(cutoff*a/(np.pi*b)))))
    k = np.pi*(modes+.5)/a
    return np.sum(np.cos(x[..., None]*k)*_sech(k*b), axis=-1)/(2*a)


def parallel_plate_density(x, half_gap):
    """PEC current / beam current on either plate y=+-b, in 1/m."""
    _, b = _axes(half_gap, half_gap)
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError("x must be finite.")
    return _sech(np.pi*x/(2*b))/(4*b)


def parallel_plate_surface_integral(half_gap):
    """Integral over both infinite plates, in 1/m, for a centred beam."""
    _, b = _axes(half_gap, half_gap)
    return 1/(2*np.pi*b)

