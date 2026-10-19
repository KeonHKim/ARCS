"""Draw exact Gauss sample locations on one 1 mm boundary segment."""
from pathlib import Path
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from arcs.geometry_input import _save_figure


def main():
    figure, axes=plt.subplots(3,1,figsize=(9,4.2),layout='constrained')
    for axis,q in zip(axes,(1,4,8)):
        nodes,weights=leggauss(q);t=(nodes+1)/2
        axis.plot([0,1],[0,0],color='#426276',lw=2)
        axis.scatter([0,1],[0,0],s=75,facecolors='white',edgecolors='#273e50',linewidths=1.5,zorder=3)
        axis.scatter(t,np.zeros(q),s=80,marker='x',color='#ca5519',linewidths=2,zorder=4)
        axis.text(0,.22,f'q = {q}: {q} weighted interior sample'+('' if q==1 else 's'),fontsize=11,color='#273e50')
        if q==4:
            for x,w in zip(t,weights/2):
                axis.text(x,-.18,f'{x:.3f} mm\nw = {w:.3f}',ha='center',va='top',fontsize=9,color='#7d3510')
        else:
            axis.text(.5,-.2,'Same panel and same two endpoint nodes',ha='center',fontsize=9,color='#596c7a')
        axis.set_xlim(-.04,1.04);axis.set_ylim(-.52,.42)
        axis.set_yticks([])
        axis.set_xticks([0,1],labels=['Start: 0 mm','End: 1 mm'])
        axis.tick_params(length=0,labelsize=9)
        for spine in axis.spines.values():spine.set_visible(False)
    figure.suptitle('Integration points inside one boundary panel',fontsize=15)
    directory=ROOT/'results/mesh_controls';directory.mkdir(exist_ok=True)
    _save_figure(figure,directory/'integration_points.png',dpi=160)
    plt.close(figure)


if __name__=='__main__':main()
