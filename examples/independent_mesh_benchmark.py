"""Circular multilayer validation using independently saved interface meshes."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import matplotlib.pyplot as plt
import arcs
from arcs import load_geometry, Metal, Layer, Wall
from arcs.reference import circular_exact, circular_layered_dipolar


def run(points=100, output=None):
    output = ROOT / 'results/independent_meshes' if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    case = load_geometry(ROOT / 'examples/geometries/independent_meshes.json')
    # Save/reload the GUI-compatible input before numerical construction.
    case.save(output / 'geometry.json')
    case = load_geometry(output / 'geometry.json')
    solver = case.make_solver(pipe_length_m=None)
    meshes = case.material_panels()
    counts = [len(p.panel_start) for _, p in meshes]
    assert counts == [128, 192]
    for (_, p), q in zip(meshes, (solver.backend.panels, *solver.backend.interfaces)):
        np.testing.assert_array_equal(p.panel_start, q.panel_start)
        np.testing.assert_array_equal(p.points, q.points)
    print('Actual panels:', dict(zip([name for name, _ in meshes], counts)), flush=True)
    f = np.geomspace(10., 1e13, points)
    started = time.perf_counter()
    response = solver.transverse(f)
    wall = Wall(Metal(5.8e7), (Layer(Metal(2e5), .005),))
    reference_long = circular_exact(f, .03, wall)
    reference_dip = circular_layered_dipolar(f, .03, wall)
    z, d = response.longitudinal, response.dipolar[:, 0, 0]
    long_error, dip_error = abs(z/reference_long-1), abs(d/reference_dip-1)
    assert np.isfinite(z).all() and np.isfinite(d).all()
    assert max(long_error.max(), dip_error.max()) < .03
    np.savetxt(output / 'spectrum.csv', np.c_[f,z.real,z.imag,reference_long.real,reference_long.imag,
        d.real,d.imag,reference_dip.real,reference_dip.imag,
        response.quadrupolar[:,0,0].real,response.quadrupolar[:,0,0].imag,long_error,dip_error],
        delimiter=',',comments='',header='f_Hz,Re_long_Ohm_m,Im_long_Ohm_m,Re_analytic_long_Ohm_m,'
        'Im_analytic_long_Ohm_m,Re_dipolar_Ohm_m2,Im_dipolar_Ohm_m2,Re_analytic_dipolar_Ohm_m2,'
        'Im_analytic_dipolar_Ohm_m2,Re_quadrupolar_Ohm_m2,Im_quadrupolar_Ohm_m2,long_complex_error,dip_complex_error')
    summary = {'version':arcs.__version__, 'points':len(f), 'frequency_band_Hz':[float(f[0]),float(f[-1])],
               'panels':dict(zip([name for name,_ in meshes],counts)), 'saved_meshes_match_solver':True,
               'long_max_complex_error_percent':float(100*long_error.max()),
               'dip_max_complex_error_percent':float(100*dip_error.max()),
               'elapsed_seconds':time.perf_counter()-started,
               'reference':'Independent cylindrical full-Maxwell m=0 and m=1 layer matching',
               'model':'Uniform 5 mm NEG-labelled test layer on infinite copper, beta=1, DC conductivity',
               'scope':'A mesh/field benchmark; not a realistic NEG coating or a physical 10 THz material certification.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary,render_saved_results(output)


def render_saved_results(output):
    """Render the saved numerical data without repeating the field solve."""
    from matplotlib.ticker import FixedLocator
    output=Path(output)
    case=load_geometry(output/'geometry.json')
    data=np.genfromtxt(output/'spectrum.csv',delimiter=',',names=True)
    f=data['f_Hz']
    z=data['Re_long_Ohm_m']+1j*data['Im_long_Ohm_m']
    d=data['Re_dipolar_Ohm_m2']+1j*data['Im_dipolar_Ohm_m2']
    reference_long=data['Re_analytic_long_Ohm_m']+1j*data['Im_analytic_long_Ohm_m']
    reference_dip=data['Re_analytic_dipolar_Ohm_m2']+1j*data['Im_analytic_dipolar_Ohm_m2']
    long_error,dip_error=data['long_complex_error'],data['dip_complex_error']
    figures=[]
    mesh_figure,mesh_axes=plt.subplots(figsize=(8,6),constrained_layout=True)
    case.draw(mesh_axes)
    mesh_axes.set_title('Independent boundary meshes: aperture 128 · coating exterior 192')
    mesh_figure.savefig(output/'boundary_meshes.png',dpi=160)
    figures.append(mesh_figure)
    fig, axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for column,(actual,reference,label,unit) in enumerate(((z,reference_long,'Longitudinal','Ω/m'),
                                                        (d,reference_dip,'Dipolar xx','Ω/m²'))):
        for row,part in enumerate(('real','imag')):
            ax=axes[row,column]
            analytical=getattr(reference,part);calculated=getattr(actual,part)
            ax.plot(f,analytical,'k-',lw=1.5,label='Circular Maxwell analytical')
            ax.scatter(f,calculated,s=18,facecolors='none',edgecolors='#0876b8',label='ARCS BEM · 128 / 192 panels')
            ax.set_xscale('log')
            if row==0:ax.set_yscale('log')
            else:
                threshold=max(float(np.max(abs(analytical)))*1e-5,1e-20)
                ax.set_yscale('symlog',linthresh=threshold)
                ticks=ax.get_yticks()
                ax.yaxis.set_major_locator(FixedLocator(ticks[(abs(ticks)>=threshold)|(ticks==0)]))
            ax.set(xlabel='Frequency [Hz]',ylabel=f'{"Re" if row==0 else "Im"}(Z) [{unit}]',title=label)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Separate saved interface meshes · 10 Hz–10 THz · Im(Z) plotted directly')
    fig.savefig(output/'impedance.png',dpi=160);figures.append(fig)
    fig, axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,error,label in zip(axes,(long_error,dip_error),('Longitudinal','Dipolar xx')):
        ax.semilogx(f,100*error,'o-',ms=3,color='#0876b8')
        ax.set(xlabel='Frequency [Hz]',ylabel='Complex relative error [%]',title=label)
        ax.grid(alpha=.2)
    fig.savefig(output/'errors.png',dpi=160);figures.append(fig)
    return figures


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=100)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    _,figures=run(args.points,args.output)
    for figure in figures:plt.close(figure)
