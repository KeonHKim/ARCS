"""Circular full-wall benchmark: BEM RW and indirect SC, 10 Hz to 10 THz."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from arcs import MaxwellSolver, Metal, Wall, polygon_panels
from arcs.reference import circular_exact, circular_maxwell_dipolar, circular_maxwell_quadrupolar
from arcs.geometry_input import _save_figure
from wideband import sweep_components


def references(f, radius, metal, beta, component):
    return (circular_exact(f, radius, Wall(metal), beta, component=component),
            circular_maxwell_dipolar(f, radius, metal, beta, component=component),
            circular_maxwell_quadrupolar(f, radius, Wall(metal), beta, component=component))


def audit_components(f, parts, radius, metal, beta, output=None):
    """Quantify each contribution separately, retaining tiny/underflow limits."""
    summary = {'beta':beta, 'radius_m':radius, 'conductivity_S_m':metal.conductivity,
               'samples':len(f), 'frequency_band_Hz':[float(f[0]),float(f[-1])],
               'impedance_reference':'free space; direct space charge excluded',
               'field_convention':'exp(-i*k*z+i*omega*t)',
               'analytical_reference':'Independent cylindrical Maxwell modes and HB2021 MOP10 Eq. (1)',
               'components':{}}
    for component, response in parts.items():
        expected = references(f, radius, metal, beta, component)
        calculated = (response.longitudinal, response.dipolar[:,0,0], response.quadrupolar[:,0,0])
        checks = {}
        for name, actual, reference in zip(('longitudinal','dipolar','quadrupolar'), calculated, expected):
            resolved = abs(reference) > 1e-250
            if name == 'quadrupolar': resolved &= f >= 1e4
            checks[name] = {'resolved_points':int(np.count_nonzero(resolved)),
                'max_complex_error_percent':float(100*np.max(abs(actual[resolved]/reference[resolved]-1))) if np.any(resolved) else None,
                'max_absolute_error':float(np.max(abs(actual-reference))),
                'relative_check_f_min_Hz':1e4 if name == 'quadrupolar' else 10.}
        summary['components'][component] = checks
    summary['component_sum_max_absolute_error'] = max(float(np.max(abs(
        getattr(parts['wall'],field)-getattr(parts['resistive_wall'],field)-getattr(parts['indirect_space_charge'],field))))
        for field in ('longitudinal','dipolar','quadrupolar','constant'))
    summary['quadrupolar_note'] = 'Centred-circle Q tends to zero as f² log(f). Low-frequency relative errors are dominated by Cartesian cancellation; absolute errors are retained.'
    if output is not None:
        Path(output).write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def plot_components(f, parts, radius, metal, beta):
    """Separate each contribution so overlapping image/total curves stay clear.

    Black lines are independent circular analytical references. Every BEM
    frequency sample is an open coloured scatter marker, with no joining line.
    The API name ``wall`` is displayed explicitly as Total (RW + indirect SC).
    """
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 3, figsize=(13, 9.5), layout='constrained')
    labels = ('Longitudinal','Dipolar xx','Quadrupolar xx')
    units = ('Ω/m','Ω/m²','Ω/m²')
    colors = {'wall':'#173f5f','indirect_space_charge':'#c66b26','resistive_wall':'#168273'}
    component_labels = {'wall':'Total (RW + indirect SC)',
                        'resistive_wall':'Resistive wall (RW)',
                        'indirect_space_charge':'Indirect space charge'}
    markers = {'wall':'s','resistive_wall':'o','indirect_space_charge':'^'}
    for row, component in enumerate(('wall','resistive_wall','indirect_space_charge')):
        response = parts[component]
        expected = references(f, radius, metal, beta, component)
        calculated = (response.longitudinal, response.dipolar[:,0,0], response.quadrupolar[:,0,0])
        for column,(actual,reference) in enumerate(zip(calculated,expected)):
            axis = axes[row,column]
            axis.plot(f, reference.imag, color='black', lw=1.5, zorder=2,
                      label='Circular analytical reference')
            axis.scatter(f, actual.imag, marker=markers[component], s=18,
                         facecolors='none', edgecolors=colors[component],
                         linewidths=.9, zorder=3, label='ARCS BEM')
            axis.set(xscale='log',xlabel='Frequency [Hz]',
                     ylabel=f'Im(Z/L) [{units[column]}]')
            axis.set_yscale('symlog', linthresh=1e-8 if column==0 else 1e-6)
            extent = np.r_[actual.imag, reference.imag]
            lower, upper = min(0.,float(np.min(extent))), max(0.,float(np.max(extent)))
            axis.set_ylim(1.2*lower, 1.5*upper) if lower != upper else axis.set_ylim(-1.,1.)
            axis.text(.025,.96,component_labels[component],transform=axis.transAxes,
                      va='top',fontsize=9,color=colors[component],
                      bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})
            if row==0: axis.set_title(labels[column])
            if column==0: axis.legend(loc='lower left',fontsize=7.5)
            axis.grid(True,which='both',alpha=.2)
    figure.suptitle(f'ARCS v0.6.0 · {radius*1e3:g} mm Cu · β={beta:g}\n'
                   'Black curves: circular analytical references. Open coloured points: ARCS BEM.',fontsize=12)
    return figure


def plot_component_errors(f, parts, radius, metal, beta):
    """Keep the complete relative-error audit separate from impedance curves."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    colors = {'wall':'#173f5f','indirect_space_charge':'#c66b26','resistive_wall':'#168273'}
    labels = {'wall':'Total (RW + indirect SC)', 'resistive_wall':'Resistive wall (RW)',
              'indirect_space_charge':'Indirect space charge'}
    for component, response in parts.items():
        expected = references(f,radius,metal,beta,component)
        calculated = (response.longitudinal,response.dipolar[:,0,0],response.quadrupolar[:,0,0])
        for column,(actual,reference) in enumerate(zip(calculated,expected)):
            resolved = abs(reference)>1e-250
            if column == 2: resolved &= f>=1e4
            axes[column].plot(f[resolved],100*abs(actual[resolved]/reference[resolved]-1),
                              color=colors[component],label=labels[component])
    for column in range(3):
        axes[column].set(xscale='log',xlabel='Frequency [Hz]',ylabel='Complex error [%]',
                         title=('Longitudinal','Dipolar xx','Quadrupolar xx')[column])
        reference = references(f, radius, metal, beta, 'wall')[column]
        signal = abs(reference) >= 1e-12*np.max(abs(reference))
        if np.any(signal) and f[signal][-1] < f[-1]:
            boundary = f[signal][-1]
            axes[column].axvspan(boundary, f[-1], color='#64748b', alpha=.08)
            axes[column].text(.98,.95,'Shaded: below 10⁻¹² of peak',
                             transform=axes[column].transAxes,ha='right',va='top',fontsize=7)
        axes[column].grid(True,which='both',alpha=.2)
    axes[0].legend(fontsize=7.5)
    figure.suptitle(f'ARCS v0.6.0 · {radius*1e3:g} mm Cu · β={beta:g} · RW + indirect space charge\n'
                   'Relative complex error, including tiny screened signals.',fontsize=12)
    return figure


def run(panels=256, points=100, beta=.7):
    import matplotlib.pyplot as plt
    output=ROOT/'results/space_charge';output.mkdir(parents=True,exist_ok=True)
    angle=np.arange(panels)*2*np.pi/panels
    mesh=polygon_panels(.03*np.c_[np.cos(angle),np.sin(angle)],panels_per_edge=1)
    solver=MaxwellSolver(mesh,beta=beta);metal=Metal(5.8e7)
    f=np.geomspace(10.,1e13,points)
    parts=sweep_components(solver,f,metal,output=output/'spectrum.csv')
    summary=audit_components(f,parts,.03,metal,beta,output/'summary.json')
    figure=plot_components(f,parts,.03,metal,beta)
    _save_figure(figure,output/'ARCS_v0.6.0_space_charge_components.png',dpi=180)
    plt.close(figure)
    figure=plot_component_errors(f,parts,.03,metal,beta)
    _save_figure(figure,output/'ARCS_v0.6.0_space_charge_errors.png',dpi=180)
    plt.close(figure)
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels',type=int,default=256)
    parser.add_argument('--points',type=int,default=100)
    parser.add_argument('--beta',type=float,default=.7)
    args=parser.parse_args();run(args.panels,args.points,args.beta)
