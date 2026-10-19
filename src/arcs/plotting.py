"""Shared plots for the CLI and notebook; signed values are never clipped."""
import numpy as np


def _scale(axis, values, xscale='log', yscale='auto'):
    if xscale not in ('linear', 'log') or yscale not in ('auto', 'linear', 'log', 'symlog'):
        raise ValueError("Use xscale=linear/log and yscale=auto/linear/log/symlog.")
    axis.set_xscale(xscale)
    values = np.asarray(values)
    if yscale == 'log' and not np.all(values > 0):
        raise ValueError("Log yscale needs positive data; use symlog or linear for signed values.")
    if yscale in ('linear', 'log'):
        axis.set_yscale(yscale)
    elif yscale == 'auto' and np.all(values == 0):
        # Exact zero is a valid bare-PEC result, with a useful linear axis.
        axis.set_yscale('linear')
    elif yscale == 'auto' and np.all(values > 0):
        axis.set_yscale('log')
    else:
        axis.set_yscale('symlog', linthresh=max(float(np.max(abs(values)))*1e-4, 1e-30))
        if np.all(values < 0):
            axis.set_ylim(1.25*values.min(), .8*values.max())
    axis.grid(True, which='both', alpha=.2)
    axis.set_xlabel('Frequency [Hz]')


def plot_longitudinal(frequency, impedance, *, reference=None, title='ARCS longitudinal impedance',
                      label='ARCS Maxwell', reference_label='Circular Maxwell reference',
                      xscale='log', yscale='auto', per_unit_length=True):
    """Plot supplied values; per_unit_length selects labels and does not scale data.

    Use per_unit_length=False for already integrated uniform-pipe totals.
    Supply a reference in the same normalization as the numerical values.
    """
    import matplotlib.pyplot as plt
    f, z = np.asarray(frequency), np.asarray(impedance)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    labels = (('Re(Z∥/L) [Ω/m]', 'Im(Z∥/L) [Ω/m]') if per_unit_length
              else ('Re(Z∥) [Ω]', 'Im(Z∥) [Ω]'))
    for axis, values, axis_label in zip(axes, (z.real, z.imag), labels):
        axis.plot(f, values, 'o-', ms=3, label=label)
        scale_values = values
        if reference is not None:
            component = np.asarray(reference).real if axis is axes[0] else np.asarray(reference).imag
            axis.plot(f, component, '-', color='#182d3c', lw=1.2, label=reference_label)
            scale_values = np.r_[values.ravel(), component.ravel()]
            axis.legend(fontsize=8)
        _scale(axis, scale_values, xscale=xscale, yscale=yscale)
        if np.any(scale_values < 0) and np.any(scale_values > 0):
            axis.axhline(0., color='#666666', lw=.6)
        axis.set_ylabel(axis_label)
    figure.suptitle(title, fontsize=12)
    return figure


def plot_transverse(frequency, response, *, title='ARCS transverse impedance',
                    xscale='log', yscale='auto'):
    """Plot tensor values with units inferred from response.pipe_length_m."""
    import matplotlib.pyplot as plt
    f = np.asarray(frequency)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for row, (tensor, kind) in enumerate(((response.dipolar, 'Dipolar'), (response.quadrupolar, 'Quadrupolar'))):
        for column, name in enumerate(('x', 'y')):
            axis = axes[row, column]
            z = tensor[..., column, column]
            axis.plot(f, z.real, 'o-', ms=3, label='Re')
            axis.plot(f, z.imag, 's-', ms=3, label='Im')
            _scale(axis, np.r_[z.real, z.imag], xscale=xscale, yscale=yscale)
            axis.set(title=f'{kind} {name}{name}',
                     ylabel='Z⊥/L [Ω/m²]' if response.pipe_length_m is None else 'Z⊥ [Ω/m]')
            axis.legend(fontsize=8)
    normalization = ('per unit pipe length' if response.pipe_length_m is None
                     else f'for a {response.pipe_length_m:g} m uniform pipe')
    figure.suptitle(title+f'\nRaw signed dipolar/quadrupolar coefficients {normalization}.', fontsize=11)
    return figure
