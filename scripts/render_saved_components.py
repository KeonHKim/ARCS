"""Replot the fresh finite-beta BEM CSVs without another field solve."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'examples'))
from arcs import Metal, TransverseImpedance
from arcs.geometry_input import _save_figure
from space_charge_benchmark import plot_components, plot_component_errors


def main():
    parts={}
    for component in ('wall','resistive_wall','indirect_space_charge'):
        data=np.genfromtxt(ROOT/'results/notebook'/f'finite_beta_components_{component}.csv',delimiter=',',names=True)
        def complex_field(field): return data['Re_'+field]+1j*data['Im_'+field]
        tensors=[np.stack([complex_field(kind+'_'+i+j+'_Ohm_m2') for i in 'xy' for j in 'xy'],axis=-1).reshape(-1,2,2)
                 for kind in ('dipolar','quadrupolar')]
        constant=np.stack([complex_field('constant_'+i+'_Ohm_m') for i in 'xy'],axis=-1)
        parts[component]=TransverseImpedance(complex_field('long_Ohm_m'),*tensors,constant,(0.,0.),.7,component)
    figure=plot_components(data['f_Hz'],parts,.03,Metal(5.8e7),.7)
    path=ROOT/'results/notebook/ARCS_v0.6.0_space_charge_components.png'
    _save_figure(figure,path,dpi=180)
    import matplotlib.pyplot as plt
    plt.close(figure)
    figure=plot_component_errors(data['f_Hz'],parts,.03,Metal(5.8e7),.7)
    _save_figure(figure,ROOT/'results/notebook/ARCS_v0.6.0_space_charge_errors.png',dpi=180)
    plt.close(figure)
    print('Replotted existing freshly solved CSVs:',path)


if __name__=='__main__':main()
