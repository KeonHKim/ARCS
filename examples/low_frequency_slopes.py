"""Plot the low-frequency slope change of the independent cylindrical reference."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arcs import Metal,Wall
from arcs.reference import circular_exact
from arcs.geometry_input import _save_figure


def run():
    f=np.geomspace(10,1000,401);z=circular_exact(f,.03,Wall(Metal(5.8e7)))
    slopes=np.stack((np.gradient(np.log(z.real),np.log(f)),np.gradient(np.log(z.imag),np.log(f))),axis=1)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].loglog(f,z.real,label='Re');axes[0].loglog(f,z.imag,label='Im');axes[0].set(xlabel='Frequency [Hz]',ylabel='Z∥/L [Ω/m]');axes[0].legend()
    axes[1].semilogx(f,slopes[:,0],label='Re slope');axes[1].semilogx(f,slopes[:,1],label='Im slope');axes[1].set(xlabel='Frequency [Hz]',ylabel='d log(component) / d log(f)');axes[1].legend()
    fig.suptitle('Circular Maxwell fields · low-frequency curvature from 10 Hz')
    _save_figure(fig,ROOT/'results/low_frequency_slopes.png',dpi=180);plt.close(fig)
    np.savetxt(ROOT/'results/low_frequency_slopes.csv',np.column_stack((f,z.real,z.imag,slopes)),delimiter=',',comments='',header='f_Hz,Re_Ohm_per_m,Im_Ohm_per_m,slope_Re,slope_Im')
if __name__=='__main__':run()
