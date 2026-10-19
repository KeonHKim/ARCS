# User DCT-I/DST-I wake conversion

The causal numerical kernels are adapted from the user's wakeSolver_v3.py. The original executed fixture and source hash remain in tests/data. ARCS v0.6.0 now agrees directly with those kernels:

\[
 Z_\parallel(f)=\int W_\parallel(t)e^{-i2\pi f t}\,dt,\quad
 Z_\perp(f)=i\int W_\perp(t)e^{-i2\pi f t}\,dt.
\]

`convention='arcs'` is the default; `'dct'` is an identical alias. For v0.5.x complex spectra use `'arcs_legacy'`, which conjugates them once. Changing convention does not change the physical wake. Plotting functions show Im directly and do not apply the legacy conversion.

## Causal kernels

`causal=True` retains the original user method. Longitudinal uses cosine for Re and minus sine for Im; transverse uses sine for Re and cosine for Im. Stored longitudinal W(0)=W(0+)/2 by default. The forward doubles it before endpoint weighting; the inverse halves its first value. Use `beam_loading=False` when storing W(0+) instead. Transverse causal wakes must vanish at both endpoints for DST-I.

For unnormalised type-I transforms, C=dt/2 DCT-I(W), S[1:-1]=dt/2 DST-I(W[1:-1]), inverse_long=2df DCT-I(Re Z), and inverse_trans[1:-1]=2df DST-I(Re Z[1:-1]). The causal inverse uses Re and does not certify consistency of Im or bandwidth. A purely imaginary nonzero spectrum is rejected in this mode because its contribution would be erased.

```python
t, w = impedance_to_wake(f_uniform, z_beta1)
t, d = impedance_to_wake(f_uniform, dip_beta1, plane='transverse')
f_back, z_back = wake_to_impedance(t, w)
```

## Two-sided kernels for finite-beta space charge

Use `causal=False` for finite-beta wall spectra containing indirect space charge. Both components are retained:

\[
 W_\parallel(t)=2\int_0^\infty
 [\Re Z_\parallel\cos(2\pi f t)-\Im Z_\parallel\sin(2\pi f t)]\,df,
\]
\[
 W_\perp(t)=2\int_0^\infty
 [\Im Z_\perp\cos(2\pi f t)+\Re Z_\perp\sin(2\pi f t)]\,df.
\]

The implementation uses type-I DCT/DST with trapezoidal endpoint weights. N frequency samples produce 2N-1 time samples from -T to +T. The origin is the Fourier average with no additional half-weight. The signed forward accepts an odd uniform grid symmetric about zero and folds its even/odd parts into DCT/DST. `beam_loading` applies only in causal mode; signed wakes store their physical origin value.

```python
t, w = impedance_to_wake(f_uniform, z_wall, causal=False)
t, d = impedance_to_wake(f_uniform, dip_wall,
                         plane='transverse', causal=False)
f_back, z_back = wake_to_impedance(t, w, causal=False)
```

Analytic Gaussian Fourier pairs and direct integrals check both planes. The notebook compares finite-beta PEC image wakes from BEM and circular formulas on the same band. This does not assume that a finite-beta wall spectrum is a one-sided causal wake.

## Units and grids

Time is seconds, frequency Hz, distance s=βct. Longitudinal Ω/m transforms to V/C/m; transverse Ω/m² to V/C/m². Uniform frequency samples include DC and fmax, with dt=1/(2fmax) and T=1/(2df). Causal inversion returns N samples; signed inversion 2N-1.

Missing DC requires an explicit `dc_value`; transverse DC can be imaginary and nonzero. Nonuniform data require `resample_points` or `tmax`. Real and imaginary parts are interpolated separately, adding no bandwidth. `samples` interpolates output sampling. `pad` assumes zero high-frequency bins, halves the old cutoff before padding, and preserves its trapezoidal weight. These options supply no missing physics.

`gaussian_bunch_wake(..., beta=...)` uses both components on a signed distance grid with σt=σz/(βc), without halving its smooth origin. `gaussian_loss_factor(..., beta=...)` uses the same speed.

The notebook's 0–10 THz beta=1 demonstration has dt=0.05 ps. Every inverse example is a finite-band reconstruction; no fitting, positivity correction or phase adjustment is applied.

SciPy definitions: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html.
