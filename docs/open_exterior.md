# Open Air/Vacuum exterior at finite beta

ARCS v0.6.7 adds an **experimental** unbounded lossless termination for smooth finite conducting walls at positive frequencies and 0 < beta < 1. Air uses exactly the ideal Vacuum values: sigma=0, epsilon_r=mu_r=1. Actual gas refractive-index/dispersion corrections are not included.

```python
from arcs import Air, Vacuum, Metal, Layer, Wall, MaxwellSolver
wall = Wall(Air(), (
    Layer(Metal(7.7e5), 0.4e-3),
    Layer(Metal(1.0e4), 3.0e-3),
))
solver = MaxwellSolver(aperture_panels, beta=0.52)
Z = solver.transverse(frequencies_Hz, wall, component="resistive_wall")
```

In the GUI choose Air as Exterior, enter beta=0.52, Apply, and save the case. The example `examples/geometries/paper_beta052_two_layers_air.json` has all three physical boundaries and both user-defined conductor properties. Finite layers require Metal; Air/Vacuum is an exterior termination only. Roughness combined with an open exterior is not validated and is rejected.

For the synchronous dependence exp(-i*k*z+i*omega*t), k=omega/(beta*c), the exterior fields obey (Laplacian_perp-kappa^2)F=0 with kappa=omega*sqrt(1-beta^2)/(beta*c). The exterior Green function is K0(kappa*r)/(2*pi), with decay at infinity. Both longitudinal fields and both tangential matching conditions are retained. This is an evanescent open field, not an outgoing propagating wave in this regime.

Its nonlocal normal map is N_ext*S_ext^{-1}, obtained by differentiating the same exterior single-layer representation as the trace. This avoids the mismatch between separately collocated Calderon operators, amplified by 1/kappa^2 when conductivity is zero. Annuli keep their full existing Green-identity matching; no surface impedance or independent reference enters assembly.

The exact beta=1 open-exterior limit needs another stable field representation and is not implemented. Selecting Air with beta=1 gives an explicit error; changing a material name does not create a PEC termination.

The default `wall` output is relative to free space. `resistive_wall` is the correction relative to a same-beta PEC pipe; `indirect_space_charge` is that PEC image contribution. A fictitious boundary between identical vacuum regions has zero total wall field, while its two reference contributions cancel. This identity uses the existing BEM image calculation, not a cross-section formula.

The accompanying beta=.52 paper-wall study compares a circular two-conductor wall against independent cylindrical m=0/m=1 solutions and the ellipse against Migliorati equations and the published Figure 6 curve. Mesh errors and reference approximation limits must be assessed from those data. Tiny circular quadrupolar values near the numerical cancellation floor need absolute errors as well as relative errors. A small matrix residual alone does not certify accuracy.

For this particular two-layer wall, the 512-panel circular complex RW difference is at most 0.125% at eight sampled frequencies between 10 MHz and 10 GHz. The ellipse's largest difference against the equivalent circular-load formula is 1.251%, in horizontal quadrupolar impedance. This formula is an approximation and refinement is not monotonically closer to it for every component. At 10 Hz the circular total longitudinal difference is still 11.64% (23.30% with 256 panels). Its small RW and PEC-image errors are amplified by their cancellation. Very small circular quadrupolar RW results also show numerical anisotropy. These findings prevent a claim of validated open-exterior accuracy over the entire 10 Hz–100 GHz sweep.
