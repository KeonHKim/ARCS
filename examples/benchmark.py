"""Full Maxwell circular longitudinal and dipolar benchmarks, 10 Hz-10 THz."""
from pathlib import Path
import argparse, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from arcs import load_geometry, Metal, Wall, TransverseImpedance
from arcs.reference import circular_exact, circular_maxwell_dipolar
from arcs.plotting import plot_longitudinal, plot_transverse, _scale
from arcs.geometry_input import _save_figure


def plot_restored_response(f, response, exact, *, panels, metal=None, radius=.03):
    """Full band plus a THz close-up with signed imaginary impedance."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout='constrained')
    # Sample analytical lines densely so joining the numerical audit points
    # does not misrepresent the reference peak. Solver samples stay unchanged.
    ref_f = np.unique(np.r_[np.geomspace(f[0], f[-1], 1201),
                            np.linspace(1e11, f[-1], 601)])
    copper = metal or Metal(5.8e7)
    ref_z = circular_exact(ref_f, radius, Wall(copper))
    for row, (component, name) in enumerate(((np.real, 'Re'), (np.imag, 'Im'))):
        full, zoom = axes[row]
        for axis in (full, zoom):
            axis.plot(ref_f, component(ref_z), color='#17202d', lw=1.8, label='Independent circular Maxwell')
            axis.plot(f, component(response.longitudinal), 'o', color='#0078ac', ms=3.5, label='ARCS Maxwell')
            axis.set_ylabel(f'{name}(Z∥/L) [Ω/m]')
        _scale(full, np.r_[component(exact), component(response.longitudinal)])
        full.set_title(f'{f[0]:g} Hz–{f[-1]/1e12:g} THz' if row == 0 else 'Signed imaginary impedance')
        for line in zoom.lines:
            line.set_xdata(np.asarray(line.get_xdata())/1e12)
        zoom.set(xlim=(.05, f[-1]/1e12), xlabel='Frequency [THz]', title='High-frequency response')
        zoom.grid(alpha=.2)
        if row:
            full.axhline(0, color='#666666', lw=.6)
            zoom.axhline(0, color='#666666', lw=.6)
        zoom.legend(fontsize=7.5)
    fig.suptitle(f'30 mm bare copper circle · {panels} boundary panels · β = 1', fontsize=13)
    return fig


def run(output_dir=None, panels=512, points=100, fmin=10., fmax=1e13):
    import matplotlib.pyplot as plt
    output = Path(output_dir or ROOT/'results/circular_benchmark')
    output.mkdir(parents=True, exist_ok=True)
    case = load_geometry(ROOT/'examples/geometries/bare_copper.json')
    data = case.to_dict(); data['mesh']['minimum_panels'] = panels; case = load_geometry(data)
    solver = case.make_solver(pipe_length_m=None)
    if points < 2 or not 0 < fmin < fmax:
        raise ValueError('Use at least two points and 0 < fmin < fmax.')
    f = np.geomspace(fmin, fmax, points)
    metal = Metal(5.8e7, 'Cu')
    calculated = []
    for index, frequency in enumerate(f):
        calculated.append(solver.transverse(float(frequency), metal))
        if index == 0 or index == len(f)-1 or (index+1) % 5 == 0:
            print(f'{index+1}/{len(f)} circular Maxwell frequencies solved', flush=True)
    response = TransverseImpedance(np.array([item.longitudinal for item in calculated]),
                  np.array([item.dipolar for item in calculated]),
                  np.array([item.quadrupolar for item in calculated]),
                  np.array([item.constant for item in calculated]), case.beam_position)
    exact = circular_exact(f, .03, Wall(metal))
    dipole = circular_maxwell_dipolar(f, .03, metal)
    count = len(solver.backend.panels.panel_start)
    le, de = abs(response.longitudinal/exact-1), abs(response.dipolar[:, 0, 0]/dipole-1)
    errors = {'panels': count, 'frequency_band_Hz': [float(f[0]), float(f[-1])], 'samples': len(f),
        'longitudinal_max_complex_relative_error_percent': float(100*np.max(le)),
        'dipolar_max_complex_relative_error_percent': float(100*np.max(de)),
        'quadrupolar_max_relative_to_dipolar': float(np.max(abs(response.quadrupolar)/abs(dipole[:, None, None]))),
        'longitudinal_reference': 'Independent full cylindrical beta=1 Maxwell m=0 fields',
        'transverse_reference': 'Independent full cylindrical beta=1 Maxwell m=1 fields',
        'full_vector_maxwell': True, 'local_surface_impedance_used': False,
        'arbitrary_geometry_certification': False, 'coating_interface_fields': True,
        'selected_points': [{
            'f_Hz': float(freq), 'ARCS_Re_Ohm_per_m': float(z.real), 'ARCS_Im_Ohm_per_m': float(z.imag),
            'reference_Re_Ohm_per_m': float(ref.real), 'reference_Im_Ohm_per_m': float(ref.imag),
            'longitudinal_complex_error_percent': float(100*err), 'dipolar_complex_error_percent': float(100*derr)
        } for freq, z, ref, err, derr in zip(f, response.longitudinal, exact, le, de)
          if freq == f[0] or freq == f[-1]]}
    plots = [
        ('longitudinal.png', plot_longitudinal(f, response.longitudinal, reference=exact,
                            title=f'ARCS circular Maxwell longitudinal benchmark · {count} panels')),
        ('transverse.png', plot_transverse(f, response, title=f'ARCS circular Maxwell tensors · {count} panels')),
        ('restored_response.png', plot_restored_response(f, response, exact, panels=count)),
    ]
    for name, fig in plots:
        _save_figure(fig, output/name, dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    for axis, component, name in zip(axes, (np.real, np.imag), ('Re', 'Im')):
        reference = component(dipole); numerical = component(response.dipolar[:, 0, 0])
        axis.plot(f, reference, color='#17202d', label='Independent full Maxwell m=1')
        axis.plot(f, numerical, 'o', ms=3, label='ARCS dipolar xx')
        _scale(axis, np.r_[reference, numerical])
        axis.set_ylabel(f'{name}(Zdip/L) [Ω/m²]'); axis.legend(fontsize=8)
        if np.any(reference < 0): axis.axhline(0., color='#666666', lw=.6)
    fig.suptitle('Circular dipolar benchmark · identical x/y; exact quadrupolar coefficients are zero')
    _save_figure(fig, output/'dipolar_reference.png', dpi=180); plt.close(fig)
    columns = np.column_stack((f, response.longitudinal.real, response.longitudinal.imag, exact.real, exact.imag,
                               response.dipolar[:, 0, 0].real, response.dipolar[:, 0, 0].imag,
                               dipole.real, dipole.imag, le, de))
    np.savetxt(output/'spectrum.csv', columns, delimiter=',', comments='',
        header='f_Hz,Re_long_Ohm_m,Im_long_Ohm_m,Re_Maxwell_Ohm_m,Im_Maxwell_Ohm_m,Re_dipolar_Ohm_m2,Im_dipolar_Ohm_m2,Re_m1_Maxwell_Ohm_m2,Im_m1_Maxwell_Ohm_m2,long_complex_relative_error,dip_complex_relative_error')
    (output/'summary.json').write_text(json.dumps(errors, indent=2)+'\n')
    print(json.dumps(errors, indent=2), flush=True)
    return f, response, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', type=int, default=512)
    parser.add_argument('--points', type=int, default=100)
    parser.add_argument('--fmin', '--f-min', dest='fmin', type=float, default=10.)
    parser.add_argument('--fmax', '--f-max', dest='fmax', type=float, default=1e13)
    args = parser.parse_args(); run(panels=args.panels, points=args.points, fmin=args.fmin, fmax=args.fmax)
