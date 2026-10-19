# Indirect space charge and wall impedance

ARCS v0.6.0 defaults to the complete wall contribution for a longitudinally uniform pipe:

\[
 Z_{\mathrm{wall}}=Z_{\mathrm{RW}}+Z_{\mathrm{ISC}},\quad
 Z_{\mathrm{RW}}=Z_{\mathrm{real}}-Z_{\mathrm{PEC}},\quad
 Z_{\mathrm{ISC}}=Z_{\mathrm{PEC}}-Z_{\mathrm{free}}.
\]

These are differences of fields. No singular coincident free-space field is numerically evaluated. Direct space charge is excluded. A finite beam profile is unnecessary for the regular indirect image field.

## Boundary formulation for arbitrary shapes

Use exp(-ikz+iωt), k=ω/(βc), κ=k/γ and a=(1-β²)/β. The free-space scalar Green function is Gκ=K0(κ|r-s|)/(2π). On the actual aperture panel mesh solve

\[
 S_\kappa\rho_s=G_\kappa(\Gamma,s).
\]

The regular PEC image potential is -Sκρs. In the new convention,

\[
 Z_{\parallel,\mathrm{ISC}}/L=i kZ_0 a\,(S_\kappa\rho_s)(w),\quad
 D_{ij,\mathrm{ISC}}=iZ_0 a\,\partial_{w_i}S_\kappa\partial_{s_j}\rho_s,\quad
 Q_{ij,\mathrm{ISC}}=iZ_0 a\,\partial_{w_i}\partial_{w_j}S_\kappa\rho_s.
\]

Source derivatives and witness gradients/Hessians are analytic. The existing vacuum solve supplies ρs and its two source derivatives, so the complete wall solve shares them with the RW calculation. No additional boundary mesh, volume mesh, circular formula, fitted offset or beam radius enters the solver. At β=1 the image prefactor vanishes; image impedances are exact zeros.

For coatings and roughness the aperture PEC reference is added to the RW correction defined relative to that same aperture. Outer material interfaces do not alter the PEC image calculation.

```python
parts = solver.transverse_components(f, wall)
total = parts['wall']
rw = parts['resistive_wall']
image = parts['indirect_space_charge']
image_only = solver.transverse(f, component='indirect_space_charge')
```

GeometrySolver supplies its saved wall and beam position automatically. `longitudinal_components()` returns the three complex arrays. `boundary_state()` remains an explicitly labelled RW reaction-field diagnostic.

## Independent circular check

For a centred beam in a circular PEC pipe of radius b,

\[
 Z_{\parallel,\mathrm{ISC}}/L=
 \frac{i\omega\mu_0}{2\pi\beta^2\gamma^2}
 \frac{K_0(\kappa b)}{I_0(\kappa b)},\quad
 D_{xx}=D_{yy}=
 \frac{i k^2Z_0}{4\pi\beta\gamma^4}
 \frac{K_1(\kappa b)}{I_1(\kappa b)}.
\]

The quadrupolar diagonals equal κ² Zparallel/(2k); cross terms vanish. These formulas in `reference.py` are independent validation functions and never enter BEM assembly. Source: N. Mounet et al., HB2021 MOP10 Eq. (1), https://doi.org/10.18429/JACoW-HB2021-MOP10 and https://proceedings.jacow.org/hb2021/papers/mop10.pdf.

Tests additionally compare a rectangle with an independent sine-mode Dirichlet Green function, verify mesh convergence, finite-offset derivatives and rotations/translations. These checks do not certify every arbitrary shape. Tiny centred circular quadrupoles at low frequency are limited by cancellation of large Cartesian Hessian terms; the audit records absolute errors there. Relative errors are omitted after exponential screening/underflow and for effectively zero coefficients.

## Signed-time wake conversion

Indirect space charge at finite β is purely reactive for a PEC pipe and has a two-sided interaction in particle separation. The causal Re-only inverse would discard it. Use both complex components through DCT-I/DST-I:

```python
t, wake = impedance_to_wake(f_uniform, z_wall, causal=False)
t, dip_wake = impedance_to_wake(f_uniform, dip_wall,
                                plane='transverse', causal=False)
s = beta*c*t
```

The notebook compares BEM and independent circular signed-time waveforms. They are finite-band reconstructions and do not certify bandwidth adequacy for an arbitrary beam or geometry.
