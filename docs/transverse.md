# Dipolar and quadrupolar Maxwell impedance

For offsets s,w about an interior point r0, use

\[
 Z_{\perp,i}/L=C_i+\sum_jD_{ij}s_j+\sum_jQ_{ij}w_j+O(\text{offset}^2).
\]

D is the dipolar/driving coefficient; Q is the quadrupolar/detuning coefficient. Rows identify kick direction and columns identify displacement direction. The source and witness are differentiated independently. Cross-plane entries are returned for rotated/asymmetric geometries. C is a constant steering term.

With the ARCS field convention, Panofsky-Wenzel gives

\[
 Z_{\perp,i}/L={\beta c\over\omega}\partial_{w_i}(Z_\parallel/L),\quad
 D_{ij}={1\over k}\partial_{w_i}\partial_{s_j}(Z_\parallel/L),\quad
 Q_{ij}={1\over k}\partial_{w_i}\partial_{w_j}(Z_\parallel/L).
\]

The beam-velocity factor is 1/k, not c/ω below β=1. Source derivatives of the PEC modified-Helmholtz drive and witness single-layer gradients/Hessians are analytic. Three right-hand sides share each full Maxwell system; there is no displacement fitting or monopole-to-dipole scaling.

D and Q are in Ω/m² per unit pipe length. Multiplying by length gives Ω/m, then multiplying by the corresponding offset gives a transverse impedance in Ω. C and longitudinal output are Ω/m before length multiplication.

```python
from arcs import load_geometry
case = load_geometry('examples/geometries/finite_beta_circle.json')
response = case.make_solver().transverse([10.,1e6,1e9])
print(response.dipolar[...,0,0])
print(response.quadrupolar[...,1,1])
```

For a centred circle Dxx=Dyy and cross terms vanish. At β=1, Q=0. At finite beta,

\[
 Q_{xx}=Q_{yy}={\kappa^2\over2k}Z_\parallel/L.
\]

More generally the modified-Helmholtz PDE gives tr(Q)=(κ²/k)Z∥. Do not force Q to be traceless for a non-ultrarelativistic beam. Raw tensor values are retained.

Independent circular bare and multilayer m=1 references appear in `reference.py`. These solve the angular Maxwell fields separately from BEM. Displaced source/witness calculations, PDE traces and rigid-motion checks test the derivatives. Circular and symmetry checks do not certify arbitrary corner-field accuracy; compare meshes and grading.

Default tensors include RW and indirect space charge, with the direct free-space contribution excluded. `component='resistive_wall'` and `component='indirect_space_charge'` return individual tensors; `transverse_components()` returns all three from one material solve per frequency. For the user's DCT/DST conversion, use `plane='transverse', convention='arcs'`; time-to-distance conversion uses s=βct.

The complex convention is exp(-ikz+iωt), and plots display Im directly. For finite-beta wall or indirect-SC spectra, use `impedance_to_wake(..., plane="transverse", causal=False)` to keep the two-sided image interaction. A causal Re-only inverse cannot retain a purely imaginary indirect term.
