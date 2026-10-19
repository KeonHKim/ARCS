# Finite-beta Maxwell boundary formulation

Use exp(-i k z + i ω t), k=ω/(βc), 0<β≤1, Z0=μ0c and κ²=k²−(ω/c)². The normal points outward from the vacuum; the tangent is counterclockwise. All output is per unit pipe length. References in `reference.py` are independent validation functions and never enter the numerical assembly.

## PEC reference and reaction

The PEC scalar Green function φD solves (∇⊥²−κ²)φD=−δ inside the aperture and φD=0 on its boundary. The source potential gives

\[
 {\mathbf E_{\perp,PEC}\over Z_0I}=-{\nabla\phi_D\over\beta},\quad
 {E_{z,PEC}\over -i k Z_0I}=-{1-\beta^2\over\beta}\phi_D,\quad
 p_s=-\partial_n\phi_D.
\]

Define the reaction fields

\[
 u={E_z-E_{z,PEC}\over -i k Z_0I},\quad v={H_z\over -i k I},\quad
 {\mathbf E_\perp\over Z_0I}={\mathbf E_{\perp,PEC}\over Z_0I}+k^2\mathbf W.
\]

All four components obey the homogeneous modified-Helmholtz equation in vacuum. Maxwell requires

\[
 \nabla\cdot\mathbf W=u,\quad (\nabla\times\mathbf W)_z=-\beta v.
\]

At β=1, κ=0 and these become harmonic fields. At finite beta they use K0(κr)/(2π), not a rescaled beta=1 solution. Analytic Laplace splitting stabilizes small κ without removing the constant part of K0. Scaling the transverse reaction by k² avoids subtracting nearly equal low-frequency fields.

The field matrix below calculates the RW correction relative to the same-beta PEC pipe. **Default reported impedance adds PEC indirect space charge**, giving the real-pipe field relative to free space. The direct free-space singularity is excluded. The regular image field needs no beam radius; see [space_charge.md](space_charge.md). `component='resistive_wall'` selects the material correction alone. `boundary_state()` exposes these RW matching traces and labels its component explicitly.

## Homogeneous material matching

For admittivity y=σ(ω)+iωε and μ=μ0μr,

\[
 q^2=k^2+i\omega\mu y.
\]

For numerical evaluation, ARCS expands the same quantity as

\[
 q^2=\left(\frac{\omega}{c}\right)^2
 \left[\frac{(1-\beta)(1+\beta)}{\beta^2}+1-\mu_r\epsilon_r\right]
 +i\omega\mu_0\mu_r\sigma(\omega).
\]

This avoids cancellation of equal vacuum terms in weakly conducting tails at β=μr=εr=1. The admittivity y still contains displacement current; no MQS approximation is introduced.

Ez and Hz solve the modified-Helmholtz equation in each homogeneous conducting region. With normal derivatives using the global outward direction, the tangential fields are

\[
 {E_t\over k^2Z_0I}={\partial_tu-\beta\mu_r\partial_nv\over q^2},\quad
 {H_t\over I}={-i k Z_0y\partial_nu+k^2\partial_tv\over q^2}.
\]

The vacuum magnetic trace is

\[
 {H_{t,vac}\over I}=p_s+{k^2W_n-\partial_nu\over\beta}.
\]

Continuity of Ez, Hz, Et and Ht supplies full material matching. An infinite exterior has nonlocal DtN Tm=−Sq⁻¹(½I−Dq). No surface impedance is imposed and no thin-skin condition is needed.

For vacuum single-layer Sκ, normal derivative Nκ (interior jump +½), tangential derivative Tκ, and C=Tm Sκ, the bare wall uses

\[
\begin{pmatrix}
 n_xN+t_xT&n_yN+t_yT&-S&0\\
 t_xN-n_xT&t_yN-n_yT&0&\beta S\\
 t_xS&t_yS&-T/q^2&\beta\mu_r C/q^2\\
 k^2n_xS&k^2n_yS&-N+i\beta kZ_0y C/q^2&-\beta k^2T/q^2
\end{pmatrix}
\begin{pmatrix}\lambda_x\\\lambda_y\\\lambda_u\\\lambda_v\end{pmatrix}
=\begin{pmatrix}0\\0\\0\\-\beta p_s\end{pmatrix}.
\]

Coordinate components act as row multipliers. Algebraic row/column equilibration changes no physical normalization.

## Nested layers

`layered.py` forms exact-region boundary representations for each annulus. Its domain normal is negative on the inner boundary and positive on the outer one. Green's identity supplies normal derivatives from both Dirichlet traces; single-layer potentials supply tangential derivatives. The two longitudinal fields are matched at every interface. Eliminating the outer traces recursively yields a nonlocal 2N × 2N field-admittance operator at the vacuum boundary; the vacuum system still retains all four fields.

This supports independently shaped conducting interfaces and multiple smooth coatings. Automatic `Wall` coatings are mitered normal offsets of the aperture polygon. Offsets that touch, cross or collapse are rejected. An open Air/Vacuum exterior is supported for beta<1. It uses q=kappa and a decaying K0 exterior Green function, without an artificial outer boundary or conductivity. The exterior normal map differentiates its own single-layer representation directly; conducting annuli retain their existing nonlocal matching. The beta=1 exterior limit and insulating gaps are not implemented.

Closely spaced interfaces use adaptive hyperbolic panel integration: x−x0=|h|sinh(t), r=|h|cosh(t). It resolves a nanometre normal gap even when the panel is millimetres long. This integration of the finite Green kernel is not the removed local planar-wall physics. Logarithmic self integrals are adaptive; tails beyond 40 attenuation lengths are negligible at the integration tolerance.

## Impedance and checks

\[
 Z_{\parallel,\mathrm{RW}}/L=+i kZ_0u(w),\quad
 D_{ij,\mathrm{RW}}=+iZ_0\partial_{w_i}\partial_{s_j}u,\quad
 Q_{ij,\mathrm{RW}}=+iZ_0\partial_{w_i}\partial_{w_j}u.
\]

Source and witness derivatives are analytic. Three right-hand sides share the field matrix. Panofsky-Wenzel uses **βc/ω=1/k**. The vacuum PDE gives Qxx+Qyy=(κ²/k)Z∥; it vanishes only at β=1. Witness Hessians and source derivatives of K0 are differentiated before integration.

Circular m=0 and m=1 references use scaled Bessel functions with finite-beta vacuum I0/I1 modes and exterior K0/K1 modes. Coated m=1 references match analytic radial fields on each layer, without boundary discretization. At β=1 the full vacuum feedback retains the THz turnover and imaginary sign change. At small beta/high frequency the wall reaction can be exponentially screened; relative error is not meaningful after underflow.

A matrix residual checks algebraic solution only. Constant-density midpoint collocation requires mesh refinement; curve error, field resolution and integration tolerance are distinct. Corner grading preserves vertices but does not itself certify singular-field accuracy.

Primary context for arbitrary-velocity Maxwell impedance formulations: U. Niedermayer, O. Boine-Frankenheim and H. De Gersem, Phys. Rev. ST Accel. Beams 18, 032001 (2015), https://doi.org/10.1103/PhysRevSTAB.18.032001. ARCS uses the boundary equations documented here; it is not that paper's finite-element implementation.

The new material convention uses σ(ω)=σDC/(1+iωτ) and y=σ+iωε. The field equations, material-interface maps and independent references have all changed convention together; no plotting-only sign correction is applied.
