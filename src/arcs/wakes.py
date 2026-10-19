"""DCT-I/DST-I wake conversion adapted from the user's wakeSolver_v3 method.

The convention is Z_long=int W exp(-i*omega*t) dt and
Z_trans=i*int W_trans exp(-i*omega*t) dt, matching the user's method.
convention='arcs' and 'dct' are identical in v0.6.0. 'arcs_legacy' adapts
v0.5.x complex spectra by conjugation. Signed-time inversion uses both
components, so finite-beta indirect space charge is retained.
No spectrum fitting, positivity correction or imaginary-part replacement occurs.
"""
import numpy as np
from scipy.fft import dct, dst
from scipy.integrate import trapezoid
from .constants import C0


def _data(grid, values, label, allow_negative=False):
    grid, values = np.asarray(grid, float), np.asarray(values)
    if (grid.ndim != 1 or len(grid) < 3 or values.shape != grid.shape
            or not np.isfinite(grid).all() or not np.isfinite(values).all()
            or (not allow_negative and grid[0] < 0) or np.any(np.diff(grid) <= 0)):
        raise ValueError(f"Use at least three matching, finite, nonnegative, increasing {label} samples.")
    return grid, values


def _options(plane, convention):
    if plane not in ('longitudinal', 'transverse'):
        raise ValueError("plane must be longitudinal or transverse.")
    if convention not in ('dct', 'arcs', 'arcs_legacy'):
        raise ValueError("convention must be arcs, dct or arcs_legacy.")


def _uniform(grid):
    return grid[0] == 0 and np.allclose(np.diff(grid), grid[1], rtol=1e-10, atol=0)


def _dct_impedance(wake, dt, plane, beam_loading=True):
    # Same numerical kernels and endpoint weights as the user's implementation.
    work = np.asarray(wake, float).copy()
    if plane == 'longitudinal' and beam_loading:
        work[0] *= 2.
    cosine = .5*dt*dct(work, type=1, norm='backward')
    sine = np.zeros_like(cosine)
    sine[1:-1] = .5*dt*dst(work[1:-1], type=1, norm='backward')
    return cosine-1j*sine if plane == 'longitudinal' else sine+1j*cosine


def _dct_wake(impedance, df, plane, beam_loading=True):
    real = np.asarray(impedance).real
    if plane == 'longitudinal':
        wake = 2*df*dct(real, type=1, norm='backward')
        if beam_loading:
            wake[0] *= .5
    else:
        wake = np.zeros_like(real)
        wake[1:-1] = 2*df*dst(real[1:-1], type=1, norm='backward')
    return wake


def wake_to_impedance(time, wake, *, plane='longitudinal', beam_loading=True,
                      convention='arcs', causal=True):
    """Return matched (frequency [Hz], impedance) from a causal uniform grid.

    Longitudinal stored W(0)=W(0+)/2 when beam_loading=True. Transverse wakes
    must vanish at both endpoints, as required by the DST-I representation.
    No interpolation is performed: dt fixes fmax=1/(2*dt) and T fixes df=1/(2*T).
    Wake units V/C/m -> longitudinal Ohm/m. A dipolar or quadrupolar wake in
    V/C/m² -> transverse Ohm/m². Positive t is behind the driving particle.
    """
    _options(plane, convention)
    t, values = _data(time, wake, 'time', allow_negative=not causal)
    if np.iscomplexobj(values) and np.any(values.imag != 0):
        raise ValueError("Use a real wake on a uniform time grid starting at zero.")
    values = np.asarray(values.real, float)
    if causal:
        if not _uniform(t):
            raise ValueError("Use a uniform time grid starting at zero for causal wakes.")
        if plane == 'transverse' and (values[0] != 0 or values[-1] != 0):
            raise ValueError("DST-I transverse wakes require zero endpoint values.")
        frequency = np.arange(len(t))/(2*t[-1])
        impedance = _dct_impedance(values, t[1], plane, beam_loading)
    else:
        if (len(t) % 2 != 1 or t[len(t)//2] != 0 or not np.allclose(t, -t[::-1], rtol=1e-10, atol=0)
                or not np.allclose(np.diff(t), t[1]-t[0], rtol=1e-10, atol=0)):
            raise ValueError("Signed wakes need an odd, uniform grid symmetric about zero.")
        middle = len(t)//2
        positive, negative = values[middle:], values[middle::-1]
        cosine = .5*(t[1]-t[0])*dct(positive+negative, type=1, norm='backward')
        sine = np.zeros_like(cosine)
        sine[1:-1] = .5*(t[1]-t[0])*dst((positive-negative)[1:-1], type=1, norm='backward')
        impedance = cosine-1j*sine if plane == 'longitudinal' else sine+1j*cosine
        frequency = np.arange(middle+1)/(2*t[-1])
    if convention == 'arcs_legacy':
        impedance = np.conjugate(impedance)
    return frequency, impedance


def _frequency_grid(frequency, impedance, *, dc_value=None, resample_points=None, tmax=None):
    f, z = _data(frequency, np.asarray(impedance, complex), 'frequency')
    if f[0] > 0:
        if dc_value is None or not np.isfinite(dc_value):
            raise ValueError("A spectrum starting above zero needs an explicit finite dc_value; no DC extrapolation is automatic.")
        f, z = np.r_[0., f], np.r_[complex(dc_value), z]
    if resample_points is not None and tmax is not None:
        raise ValueError("Choose resample_points or tmax to define the intrinsic grid, not both.")
    if tmax is not None:
        if not np.isfinite(tmax) or tmax <= 0:
            raise ValueError("tmax must be finite and positive.")
        intervals = 2*f[-1]*tmax
        nearest = np.rint(intervals)
        if np.isclose(intervals, nearest, rtol=1e-12, atol=1e-12):
            intervals = nearest
        resample_points = max(3, int(np.ceil(intervals))+1)
    if resample_points is not None:
        if isinstance(resample_points, (bool, np.bool_)) or int(resample_points) != resample_points or resample_points < 3:
            raise ValueError("resample_points must be an integer >=3.")
        grid = np.linspace(0, f[-1], int(resample_points))
        z = np.interp(grid, f, z.real)+1j*np.interp(grid, f, z.imag)
        f = grid
    if not _uniform(f):
        raise ValueError("A nonuniform spectrum requires explicit resample_points (or tmax). Interpolation adds no bandwidth.")
    return f, z


def impedance_to_wake(frequency, impedance, *, plane='longitudinal', beam_loading=True,
                      convention='arcs', dc_value=None, resample_points=None,
                      tmax=None, samples=None, pad=0, causal=True):
    """Return (time [s], real causal wake) using the user's DCT/DST-I inverse.

    With causal=True the inverse uses Re(Z), as in wakeSolver_v3. It does not use
    Im(Z) to repair the input or certify causal consistency. Native N frequency
    samples produce N time samples, dt=1/(2*fmax), T=1/(2*df).
    Nonuniform data are interpolated only when explicitly requested. Missing
    DC requires dc_value. samples changes only the output sampling. pad adds
    zero high-frequency bins, preserving the original cutoff's half weight;
    this is an assumed extension, not additional physical frequency content.
    With causal=False both components enter a signed-time DCT/DST inverse.
    Use this for finite-beta wall impedance including indirect space charge.
    Its value at t=0 is the Fourier average, without an extra half-weight.
    A finite-band result does not certify a broadband or short-bunch RW wake.
    """
    _options(plane, convention)
    f, z = _frequency_grid(frequency, impedance, dc_value=dc_value,
                           resample_points=resample_points, tmax=tmax)
    if convention == 'arcs_legacy':
        z = np.conjugate(z)
    if plane == 'longitudinal' and abs(z[0].imag) > 1e-12*max(1., np.max(abs(z))):
        raise ValueError("Longitudinal DC impedance must be real.")
    if isinstance(pad, (bool, np.bool_)) or int(pad) != pad or pad < 0:
        raise ValueError("pad must be an integer >=0.")
    df = f[1]
    if pad:
        z = z.copy()
        z[-1] *= .5
        z = np.pad(z, (0, int(pad)))
        f = np.arange(len(z))*df
    time = np.arange(len(f))/(2*f[-1])
    if causal:
        if np.max(abs(z.real)) == 0 and np.max(abs(z.imag)) > 0:
            raise ValueError("A purely reactive spectrum needs causal=False; a Re-only inverse would lose it.")
        wake = _dct_wake(z, df, plane, beam_loading)
    else:
        cosine_values = z.real if plane == 'longitudinal' else z.imag
        sine_values = -z.imag if plane == 'longitudinal' else z.real
        cosine = df*dct(cosine_values, type=1, norm='backward')
        sine = np.zeros_like(cosine)
        sine[1:-1] = df*dst(sine_values[1:-1], type=1, norm='backward')
        wake = np.r_[(cosine-sine)[:0:-1], cosine+sine]
        time = np.r_[-time[:0:-1], time]
    if samples is not None or tmax is not None:
        end = time[-1] if tmax is None else float(tmax)
        count = len(time) if samples is None else samples
        if isinstance(count, (bool, np.bool_)) or int(count) != count or count < 3:
            raise ValueError("samples must be an integer >=3.")
        output = np.linspace(0 if causal else -end, end, int(count))
        wake = np.interp(output, time, wake)
        time = output
    return time, wake


def gaussian_bunch_wake(frequency, impedance, sigma_z, *, convention='arcs', beta=1.0):
    """Signed distance and longitudinal Gaussian bunch wake by DCT/DST-I.

    Both real and imaginary spectrum components are used for this smooth
    signed bunch wake. This is distinct from the causal Re-only point inverse.
    Input must be uniform from DC. Output is V/C/m when Z is Ohm/m.
    """
    _options('longitudinal', convention)
    from .maxwell import validate_beta
    speed = validate_beta(beta)*C0
    f, z = _data(frequency, np.asarray(impedance, complex), 'frequency')
    if not _uniform(f) or not np.isfinite(sigma_z) or sigma_z <= 0:
        raise ValueError("Use uniform frequencies from zero and finite positive sigma_z.")
    if abs(z[0].imag) > 1e-12*max(1., np.max(abs(z))):
        raise ValueError("Longitudinal DC impedance must be real.")
    if convention == 'arcs_legacy':
        z = np.conjugate(z)
    z = z*np.exp(-.5*(2*np.pi*f*sigma_z/speed)**2)
    cosine = f[1]*dct(z.real, type=1, norm='backward')
    sine = np.zeros_like(cosine)
    sine[1:-1] = f[1]*dst(z.imag[1:-1], type=1, norm='backward')
    time = np.arange(len(f))/(2*f[-1])
    return np.r_[-time[:0:-1], time]*speed, np.r_[(cosine+sine)[:0:-1], cosine-sine]


def gaussian_loss_factor(frequency, impedance, sigma_z, *, beta=1.0):
    """Spectral longitudinal loss factor, V/C/m, with no Fourier sign ambiguity."""
    f, z = _data(frequency, np.asarray(impedance, complex), 'frequency')
    if not np.isfinite(sigma_z) or sigma_z <= 0:
        raise ValueError("sigma_z must be finite and positive.")
    from .maxwell import validate_beta
    speed = validate_beta(beta)*C0
    return 2*trapezoid(z.real*np.exp(-(2*np.pi*f*sigma_z/speed)**2), f)
