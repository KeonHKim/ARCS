"""Legacy -5 Rq gradient BEM comparison with circular layers and supplied ODE.

The supplied surface-impedance code is loaded unchanged only in this benchmark.
Its exp(+i omega t) DC-conductor result uses the new ARCS convention directly.
Both models use the same truncated air-side reference plane. The ODE is a
reduced depth model, not an exact synchronous circular Maxwell reference.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from arcs import MaxwellSolver, Metal, Layer, Wall, Roughness, polygon_panels
from arcs.roughness import gradient_layers
from arcs.reference import circular_exact, _circular_vacuum_reaction
from arcs.geometry_input import _save_figure


def supplied_reference(f, rms_height=.3e-6, resolution=100):
    path = ROOT/'references/SurfaceImpedance/src/surface_impedance/models.py'
    spec = importlib.util.spec_from_file_location('supplied_surface_models', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    parameters = {'sigma_metal': 5.8e7, 'rq': rms_height, 'mu_r': 1.,
                  'xmin_factor': -5., 'domain_factor': 15., 'step_size': rms_height/resolution}
    return module.compute_surface_impedance('rough-single', np.asarray(f), parameters)


def circular_gradient_reference(f, rms_height, slices, beta=1.):
    metal = Metal(5.8e7)
    depth, media = gradient_layers(Wall(metal), Roughness(rms_height, slices=slices, extent_sigma=5.))
    wall = Wall(metal, tuple(Layer(m,d) for m,d in zip(media[:-1], np.diff(depth))))
    return circular_exact(f,.03,wall,beta)


def run(output_dir=None, panels=96, slices=32, points=9):
    import matplotlib.pyplot as plt
    output = Path(output_dir or ROOT/'results/roughness')
    output.mkdir(parents=True, exist_ok=True)
    f = np.geomspace(1e7,1e11,points)
    angle = np.arange(panels)*2*np.pi/panels
    boundary = polygon_panels(.03*np.c_[np.cos(angle),np.sin(angle)],panels_per_edge=1)
    rms_height = .3e-6
    solver = MaxwellSolver(boundary,roughness=Roughness(rms_height,slices=slices,extent_sigma=5.))
    metal = Metal(5.8e7, 'Cu')
    start = time.time()
    parts = []
    for i, freq in enumerate(f):
        parts.append(solver.transverse(float(freq),metal))
        print(f'Roughness BEM: {i+1}/{len(f)} frequencies, {slices} profile slices',flush=True)
    z = np.array([r.longitudinal for r in parts])
    dipole = np.array([r.dipolar[0,0] for r in parts])
    layered = circular_gradient_reference(f,rms_height,slices)
    ode = _circular_vacuum_reaction(f,.03,supplied_reference(f,rms_height))
    smooth = circular_exact(f,.03,Wall(metal))
    # Independent profile and ODE discretization checks are inexpensive modal/FD runs.
    refined_layers = circular_gradient_reference(f,rms_height,256)
    refined_ode = _circular_vacuum_reaction(f,.03,supplied_reference(f,rms_height,200))
    numerical_error = abs(z/layered-1)
    profile_error = abs(layered/refined_layers-1)
    model_difference = abs(refined_layers/refined_ode-1)
    fig, axes = plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for column, component in enumerate((np.real,np.imag)):
        axis = axes[0,column]
        for values,label,style,color in ((smooth,'Smooth copper at aperture','--','#a0a0a0'),
                (ode,'Supplied reduced depth ODE','-','#17202d'),
                (layered,'Independent circular Maxwell layers','-','#c36319'),
                (z,'ARCS roughness BEM','o','#0067b1')):
            axis.loglog(f,component(values),style,color=color,label=label,ms=4,lw=1.4)
        axis.set(xlabel='Frequency [Hz]',ylabel=('Re' if column==0 else 'Im')+'(Z∥/L) [Ω/m]')
        axis.grid(True,which='both',alpha=.2)
        axis.legend(fontsize=7.5)
    axes[1,0].loglog(f,100*numerical_error,'o-',label='BEM vs same circular layers')
    axes[1,0].loglog(f,100*profile_error,'s-',label=f'{slices} vs 256 profile slices')
    axes[1,0].loglog(f,100*model_difference,'^-',label='Full Maxwell vs supplied depth ODE')
    axes[1,0].set(xlabel='Frequency [Hz]',ylabel='Complex discrepancy [%]',title='Separate numerical and model comparisons')
    axes[1,0].legend(fontsize=7.5)
    # Real-component disagreement is also shown, so complex error cannot hide it.
    axes[1,1].semilogx(f,100*(refined_layers.real/refined_ode.real-1),'o-',label='Re(Z) model difference')
    axes[1,1].semilogx(f,100*(refined_layers.imag/refined_ode.imag-1),'s-',label='Im(Z) model difference')
    axes[1,1].set(xlabel='Frequency [Hz]',ylabel='Signed difference [%]',title='Reference discrepancy; physical attribution unresolved')
    axes[1,1].legend(fontsize=7.5)
    for axis in axes[1]:
        axis.grid(True,which='both',alpha=.2)
    fig.suptitle(f'Gaussian conductivity-gradient roughness · Rq = 0.3 μm · β = 1\n'
                 f'{panels} panels per interface; {slices} material slices. Same air-side reference plane.',fontsize=12)
    _save_figure(fig,output/'roughness_benchmark.png');plt.close(fig)
    from arcs.layered import material_boundaries
    boundaries, media = material_boundaries(solver,Wall(metal))
    depth,_ = gradient_layers(Wall(metal),solver.roughness)
    meshfig, meshaxes = plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    x = np.linspace(-5,5,601)
    from scipy.special import ndtr
    meshaxes[0].plot(x,ndtr(x),label='Continuous Gaussian profile',color='#17202d')
    meshaxes[0].stairs([m.conductivity/metal.conductivity for m in media[:-1]],
                      depth/rms_height-5,label='BEM material slices',color='#0067b1')
    meshaxes[0].set(xlabel='Depth / Rq from mean surface',ylabel='σ / σCu',title='Material profile, no imposed surface impedance')
    meshaxes[0].legend(fontsize=8)
    for p in boundaries:
        point=p.starts;inside=np.argmin(np.linalg.norm(point-np.array([.03,0]),axis=1))
        for j in ((inside-1)%len(point),inside):
            meshaxes[1].plot((np.array([p.starts[j,0],p.ends[j,0]])-.03)*1e6,
                            np.array([p.starts[j,1],p.ends[j,1]])*1e6,color='#0067b1',lw=.7)
    meshaxes[1].axvline(5*rms_height*1e6,color='#c36319',ls='--',label='Mean wall')
    meshaxes[1].set(xlim=(-.2,3.3),ylim=(-5,5),xlabel='x − aperture radius [μm]',ylabel='y [μm]',
                    title='Actual polygon material-interface mesh')
    meshaxes[1].legend(fontsize=8)
    for axis in meshaxes: axis.grid(alpha=.2)
    _save_figure(meshfig,output/'roughness_mesh.png');plt.close(meshfig)
    values=np.column_stack((f,z.real,z.imag,layered.real,layered.imag,ode.real,ode.imag,
                           smooth.real,smooth.imag,dipole.real,dipole.imag,
                           numerical_error,profile_error,model_difference))
    np.savetxt(output/'spectrum.csv',values,delimiter=',',comments='',header=
        'f_Hz,Re_BEM_Ohm_m,Im_BEM_Ohm_m,Re_circular_layers,Im_circular_layers,Re_supplied_ODE,Im_supplied_ODE,'
        'Re_smooth,Im_smooth,Re_BEM_dipolar_Ohm_m2,Im_BEM_dipolar_Ohm_m2,'
        'BEM_complex_relative_error,profile_complex_relative_error,model_complex_relative_difference')
    mid = f <= 1e10
    summary={'panels_per_interface':panels,'profile_slices':slices,'material_interfaces':len(boundaries),
        'rms_height_m':rms_height,'frequency_band_Hz':[float(f[0]),float(f[-1])],
        'max_BEM_error_to_same_circular_Maxwell_layers_percent':float(100*max(numerical_error)),
        'max_profile_error_to_256_slices_percent':float(100*max(profile_error)),
        'max_complex_discrepancy_to_supplied_ODE_10MHz_to_10GHz_percent':float(100*max(abs(z[mid]/ode[mid]-1))),
        'max_Re_discrepancy_to_supplied_ODE_10MHz_to_10GHz_percent':float(100*max(abs(z[mid].real/ode[mid].real-1))),
        'Re_model_difference_at_100GHz_percent':float(100*(refined_layers[-1].real/refined_ode[-1].real-1)),
        'reference_plane':'Aperture at -5 Rq; first mean surface +5 Rq outward',
        'supplied_reference':'Unchanged rough-single finite-difference depth ODE; DC conductivity; conjugated Fourier sign',
        'source_sha256':hashlib.sha256((ROOT/'references/SurfaceImpedance/src/surface_impedance/models.py').read_bytes()).hexdigest(),
        'roughness_scope':'Homogenized conductivity gradient; no z-dependent corrugations',
        'full_vector_maxwell':True,'local_surface_impedance_used':False,
        'independent_rough_surface_experimental_validation':False,'seconds':time.time()-start}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    add_dipolar_audit(output,rms_height,slices)
    return summary


def add_dipolar_audit(output_dir=None, rms_height=.3e-6, slices=32):
    """Validate saved BEM dipoles against exact multilayer circular m=1 fields."""
    import matplotlib.pyplot as plt
    from arcs.reference import circular_layered_dipolar
    output=Path(output_dir or ROOT/'results/roughness')
    spectrum=np.genfromtxt(output/'spectrum.csv',delimiter=',',names=True)
    f=spectrum['f_Hz']
    numerical=spectrum['Re_BEM_dipolar_Ohm_m2']+1j*spectrum['Im_BEM_dipolar_Ohm_m2']
    metal=Metal(5.8e7)
    depth,media=gradient_layers(Wall(metal),Roughness(rms_height,slices=slices,extent_sigma=5.))
    wall=Wall(metal,tuple(Layer(m,d) for m,d in zip(media[:-1],np.diff(depth))))
    reference=circular_layered_dipolar(f,.03,wall)
    error=abs(numerical/reference-1)
    figure,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    for axis,component,label in zip(axes[:2],(np.real,np.imag),('Re','Im')):
        axis.loglog(f,component(reference),color='#17202d',label='Independent circular m=1 layers')
        axis.plot(f,component(numerical),'o',color='#0067b1',label='ARCS roughness dipolar xx',ms=4)
        axis.set(xlabel='Frequency [Hz]',ylabel=f'{label}(Dxx/L) [Ω/m²]')
        axis.legend(fontsize=7)
    axes[2].semilogx(f,100*error,'o-',color='#0067b1')
    axes[2].set(xlabel='Frequency [Hz]',ylabel='Complex relative error [%]',title='Same discretized material profile')
    for axis in axes:axis.grid(True,which='both',alpha=.2)
    figure.suptitle(f'Roughness dipolar Maxwell benchmark · Rq = 0.3 μm · {slices} slices',fontsize=12)
    _save_figure(figure,output/'roughness_dipolar.png');plt.close(figure)
    np.savetxt(output/'dipolar_reference.csv',np.c_[f,numerical.real,numerical.imag,reference.real,reference.imag,error],
        delimiter=',',comments='',header='f_Hz,Re_BEM_Ohm_m2,Im_BEM_Ohm_m2,Re_m1_reference,Im_m1_reference,complex_relative_error')
    summary=json.loads((output/'summary.json').read_text())
    summary['max_dipolar_BEM_error_to_same_circular_Maxwell_layers_percent']=float(100*max(error))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Dipolar roughness maximum complex BEM error [%]:',100*max(error),flush=True)


def run_low_frequency(output_dir=None, panels=96, slices=16):
    """Low-frequency gradient BEM audit from 10 Hz; no reduced depth reference."""
    import matplotlib.pyplot as plt
    output=Path(output_dir or ROOT/'results/roughness_low_frequency')
    output.mkdir(parents=True,exist_ok=True)
    f=np.array([10.,100.,1e3,1e4,1e6])
    angle=np.arange(panels)*2*np.pi/panels
    solver=MaxwellSolver(polygon_panels(.03*np.c_[np.cos(angle),np.sin(angle)],panels_per_edge=1),
                         roughness=Roughness(.3e-6,slices=slices,extent_sigma=5.))
    metal=Metal(5.8e7)
    z=solver.longitudinal(f,metal)
    reference=circular_gradient_reference(f,.3e-6,slices)
    error=abs(z/reference-1)
    from arcs.plotting import plot_longitudinal
    figure=plot_longitudinal(f,z,reference=reference,
        reference_label='Independent circular Maxwell gradient layers',
        title=f'Low-frequency conductivity-gradient audit · {panels} panels · {slices} profile slices')
    _save_figure(figure,output/'roughness_low_frequency.png');plt.close(figure)
    np.savetxt(output/'spectrum.csv',np.c_[f,z.real,z.imag,reference.real,reference.imag,error],
        delimiter=',',comments='',header='f_Hz,Re_BEM_Ohm_m,Im_BEM_Ohm_m,Re_reference,Im_reference,complex_relative_error')
    summary={'panels_per_interface':panels,'profile_slices':slices,'frequency_band_Hz':[10.,1e6],
        'max_complex_error_to_same_circular_Maxwell_layers_percent':float(100*max(error)),
        'reference':'Full circular Maxwell field matching of the same gradient layers; not the reduced surface ODE'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels',type=int,default=96)
    parser.add_argument('--slices',type=int,default=32)
    parser.add_argument('--points',type=int,default=9)
    parser.add_argument('--low-frequency',action='store_true',help='Also audit gradient layers from 10 Hz')
    args=parser.parse_args()
    run(panels=args.panels,slices=args.slices,points=args.points)
    if args.low_frequency:run_low_frequency()
