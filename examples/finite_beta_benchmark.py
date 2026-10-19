"""Independent finite-beta circular Maxwell benchmarks, longitudinal and transverse."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from arcs import MaxwellSolver, Metal, Wall, polygon_panels
from arcs.reference import circular_exact, circular_maxwell_dipolar, circular_maxwell_quadrupolar
from arcs.geometry_input import _save_figure


def run(output_dir=None, panels=512, points=19):
    import matplotlib.pyplot as plt
    output = Path(output_dir or ROOT/'results/finite_beta')
    output.mkdir(parents=True, exist_ok=True)
    angle = np.arange(panels)*2*np.pi/panels
    boundary = polygon_panels(.03*np.c_[np.cos(angle), np.sin(angle)], panels_per_edge=1)
    f = np.unique(np.r_[np.geomspace(10., 3e9, points), 1e3, 1e4, 1e5, 1e6, 1e9])
    dense = np.geomspace(f[0], f[-1], 701)
    metal = Metal(5.8e7, 'Cu')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    qfig, qaxes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    rows = []
    summary = {'panels': panels, 'frequency_band_Hz': [float(f[0]), float(f[-1])],
               'reference': 'Independent circular Maxwell m=0 and m=1 field matching',
               'impedance_reference': 'Free space; direct space charge excluded', 'betas': [],
               'full_vector_maxwell': True, 'local_surface_impedance_used': False}
    start = time.time()
    colors = ['#0067b1', '#c36319', '#007f64', '#764398']
    for beta, color in zip((.3, .7, .99, 1.), colors):
        solver = MaxwellSolver(boundary, beta=beta)
        parts = []
        for i, freq in enumerate(f):
            parts.append(solver.transverse(float(freq), metal))
            if i == 0 or i == len(f)-1 or (i+1) % 5 == 0:
                print(f'Finite beta={beta:g}: {i+1}/{len(f)} frequencies', flush=True)
        z = np.array([r.longitudinal for r in parts])
        d = np.array([r.dipolar[0, 0] for r in parts])
        q = np.array([r.quadrupolar[0, 0] for r in parts])
        rz = circular_exact(f, .03, Wall(metal), beta)
        rd = circular_maxwell_dipolar(f, .03, metal, beta)
        rq = circular_maxwell_quadrupolar(f, .03, Wall(metal), beta)
        reference_lines = (circular_exact(dense, .03, Wall(metal), beta),
                           circular_maxwell_dipolar(dense, .03, metal, beta))
        for column, (values, reference) in enumerate(zip((z, d), reference_lines)):
            for row, component in enumerate((np.real, np.imag)):
                axis = axes[row, column]
                axis.loglog(dense, component(reference), color=color, lw=1.5, label=f'β = {beta:g}')
                axis.plot(f, component(values), 'o', ms=3, color=color)
                axis.set(xlabel='Frequency [Hz]', ylabel=('Re' if row == 0 else 'Im')+
                         ('(Z∥/L) [Ω/m]' if column == 0 else '(Dxx/L) [Ω/m²]'))
                axis.grid(True, which='both', alpha=.2)
        # Finite-beta quadrupole is nonzero even in a centred circle.
        mask = f >= 1e6
        if beta < 1:
            qaxes[0].loglog(f[mask], abs(rq[mask]), color=color, label=f'β = {beta:g} reference')
            qaxes[0].plot(f[mask], abs(q[mask]), 'o', ms=3, color=color)
            qaxes[1].semilogx(f[mask], 100*abs(q[mask]/rq[mask]-1), 'o-', color=color,
                             label=f'β = {beta:g}')
        le, de = abs(z/rz-1), abs(d/rd-1)
        summary['betas'].append({'beta': beta, 'samples': len(f),
            'max_longitudinal_complex_error_percent': float(100*max(le)),
            'max_dipolar_complex_error_percent': float(100*max(de)),
            'max_quadrupolar_complex_error_percent_above_1MHz':
                float(100*max(abs(q[mask]/rq[mask]-1))) if beta < 1 else None})
        rows.extend(zip(np.full(len(f), beta), f, z.real, z.imag, rz.real, rz.imag,
                        d.real, d.imag, rd.real, rd.imag, q.real, q.imag, rq.real, rq.imag, le, de))
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title('Longitudinal wall impedance (RW + indirect SC)')
    axes[0, 1].set_title('Dipolar coefficient, Dxx = Dyy')
    fig.suptitle(f'Finite beam velocity · 30 mm copper circle · {panels} BEM panels\n'
                 'Lines: independent modes. Points: boundary solver.', fontsize=12)
    qaxes[0].set(xlabel='Frequency [Hz]', ylabel='|Qxx/L| [Ω/m²]', title='Finite-β quadrupolar coefficient')
    qaxes[1].set(xlabel='Frequency [Hz]', ylabel='Complex relative error [%]', title='Independent m=0 derivative reference')
    for axis in qaxes:
        axis.grid(True, which='both', alpha=.2)
        axis.legend(fontsize=8)
    qfig.suptitle('Qxx = Qyy; Qxx + Qyy = (κ²/k) Z∥. At β = 1 these are zero.', fontsize=11)
    _save_figure(fig, output/'finite_beta.png'); plt.close(fig)
    _save_figure(qfig, output/'finite_beta_quadrupolar.png'); plt.close(qfig)
    np.savetxt(output/'spectrum.csv', np.asarray(rows), delimiter=',', comments='', header=
        'beta,f_Hz,Re_Z_Ohm_m,Im_Z_Ohm_m,Re_Z_reference,Im_Z_reference,Re_D_Ohm_m2,Im_D_Ohm_m2,'
        'Re_D_reference,Im_D_reference,Re_Q_Ohm_m2,Im_Q_Ohm_m2,Re_Q_reference,Im_Q_reference,'
        'Z_complex_relative_error,D_complex_relative_error')
    summary['seconds'] = time.time()-start
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', type=int, default=512)
    parser.add_argument('--points', type=int, default=19)
    args = parser.parse_args()
    run(panels=args.panels, points=args.points)
