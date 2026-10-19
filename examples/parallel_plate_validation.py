"""Audit PEC vacuum geometry against rectangle series and the infinite-plate limit.

No conductor impedance is calculated here. The full-Maxwell conductor calculation and
its transverse tensors are exercised separately by benchmark.py and the notebook.
"""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from arcs import PolygonPECSolver, polygon_panels
from arcs.plate_reference import rectangle_surface_integral,parallel_plate_surface_integral
from arcs.geometry_input import _save_figure


def run():
    output=ROOT/'results'; b=.03; rows=[]
    for ratio in (1,2,4,8):
        a=ratio*b; mesh=polygon_panels([[-a,-b],[a,-b],[a,b],[-a,b]],panels_per_edge=[64*ratio,64,64*ratio,64])
        value=PolygonPECSolver(mesh).surface_integral(); ref=rectangle_surface_integral(a,b)
        rows.append((ratio,len(mesh.panel_start),value,ref,parallel_plate_surface_integral(b)))
    data=np.asarray(rows); np.savetxt(output/'plate_geometry.csv',data,delimiter=',',comments='',header='a_over_b,panels,ARCS_PEC_integral_per_m,rectangle_series_per_m,infinite_plate_per_m')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot(data[:,0],data[:,2],'o-',label='PEC boundary calculation');axes[0].plot(data[:,0],data[:,3],'x-',label='Rectangle series');axes[0].axhline(data[0,4],color='black',ls=':',label='Infinite plates')
    axes[0].set(xlabel='Horizontal / vertical half aperture',ylabel='Vacuum geometry integral [1/m]');axes[0].legend(fontsize=8)
    axes[1].semilogy(data[:,0],100*abs(data[:,2]/data[:,3]-1),'o-');axes[1].set(xlabel='Horizontal / vertical half aperture',ylabel='Relative error [%]')
    fig.suptitle('PEC vacuum geometry audit · conductor penetration is not tested by this figure')
    _save_figure(fig,output/'plate_geometry.png',dpi=180);plt.close(fig)
    print('Saved the PEC-only geometry comparison.',flush=True)
    return data
if __name__=='__main__':run()
