"""Fresh full-Maxwell spectrum and references for a uniform pipe of given length."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import arcs
from arcs import load_geometry, Metal, Wall
from arcs.reference import circular_exact, circular_maxwell_dipolar
from arcs.geometry_input import _save_figure
from arcs.plotting import plot_longitudinal, plot_transverse


def run(pipe_length_m=1., points=100, output=None):
    import matplotlib.pyplot as plt
    case=load_geometry(ROOT/'examples/geometries/bare_copper.json')
    solver=case.make_solver(mesh={'minimum_panels':256},pipe_length_m=pipe_length_m)
    f=np.geomspace(10.,1e13,points)
    response=solver.transverse(f)
    longitudinal_ref=pipe_length_m*circular_exact(f,.03,Wall(Metal(5.8e7)))
    dipolar_ref=pipe_length_m*circular_maxwell_dipolar(f,.03,Metal(5.8e7))
    destination=Path(output) if output else ROOT/'results/pipe_length'
    destination.mkdir(parents=True,exist_ok=True)
    columns=[f,response.longitudinal.real,response.longitudinal.imag,
             longitudinal_ref.real,longitudinal_ref.imag,dipolar_ref.real,dipolar_ref.imag]
    names=['f_Hz','Re_long_Ohm','Im_long_Ohm','Re_reference_long_Ohm','Im_reference_long_Ohm',
           'Re_reference_dipolar_Ohm_per_m','Im_reference_dipolar_Ohm_per_m']
    for field in ('dipolar','quadrupolar'):
        tensor=getattr(response,field)
        for i,a in enumerate('xy'):
            for j,b in enumerate('xy'):
                columns.extend([tensor[:,i,j].real,tensor[:,i,j].imag])
                names.extend([f'Re_{field}_{a}{b}_Ohm_per_m',f'Im_{field}_{a}{b}_Ohm_per_m'])
    for i,a in enumerate('xy'):
        columns.extend([response.constant[:,i].real,response.constant[:,i].imag])
        names.extend([f'Re_constant_{a}_Ohm',f'Im_constant_{a}_Ohm'])
    np.savetxt(destination/'spectrum.csv',np.column_stack(columns),delimiter=',',comments='',header=','.join(names))
    figures={
        'longitudinal':plot_longitudinal(f,response.longitudinal,reference=longitudinal_ref,
            per_unit_length=False,title=f'ARCS · {pipe_length_m:g} m circular pipe · 256 panels'),
        'transverse':plot_transverse(f,response,title=f'ARCS · {pipe_length_m:g} m circular pipe · 256 panels')}
    errors={'longitudinal_max_complex_error_percent':float(100*np.max(abs(response.longitudinal/longitudinal_ref-1))),
            'dipolar_max_complex_error_percent':float(100*np.max(abs(response.dipolar[:,0,0]/dipolar_ref-1)))}
    assert max(errors.values())<.5
    for name,figure in figures.items():_save_figure(figure,destination/(name+'.png'),dpi=160)
    summary={'version':arcs.__version__,'pipe_length_m':response.pipe_length_m,
             'frequency_samples':len(f),'frequency_band_Hz':[float(f[0]),float(f[-1])],
             'panels':len(solver.backend.panels.panel_start),'longitudinal_units':solver.longitudinal_units,
             'dipolar_quadrupolar_units':solver.transverse_coefficient_units,
             'fresh_BEM_calculation':True,'reference':'Independent circular full-Maxwell m=0 and m=1',
             'entrance_exit_effects_included':False,**errors}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary,figures


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipe-length-m',type=float,default=1.)
    parser.add_argument('--points',type=int,default=100)
    args=parser.parse_args()
    summary,figures=run(args.pipe_length_m,args.points)
    print(json.dumps(summary,indent=2))
