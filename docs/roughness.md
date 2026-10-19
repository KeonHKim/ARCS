# Conductivity-gradient roughness

ARCS represents a bare rough wall by σ(x)=σmetal Φ(x/Rq), where Φ is the normal CDF and x=0 is the mean surface. It subdivides the transition into homogeneous conducting layers, constructs actual offset interfaces and solves nonlocal Maxwell fields across them with boundary elements. The supplied surface-impedance code is retained unchanged in `references/SurfaceImpedance/` for historical comparisons; it does not supply the numerical wall condition.

## Reference plane and cutoff

Profiles retain **32 slices and `extent_sigma=5`**. The input aperture is the truncated air-side plane x=−5Rq. The mean surface lies 5Rq outward, and the profile ends at +5Rq before the unbounded substrate. For Rq=0.3 μm, a 30 mm input radius puts the mean wall 1.5 μm farther out. To hold a circular mean radius `b_mean` fixed, enter `b_aperture=b_mean−extent_sigma*Rq`.

Saved files with explicit slice counts or extents retain them. The historical supplied reduced ODE starts at −5Rq, and the legacy comparison script therefore specifies `extent_sigma=5` explicitly. The 5-RMS profile is not a converged full Gaussian model. Extents above 5 can be constructed and meshed, but field solving now rejects them after the failed extended-tail audit.

The imaginary-part audit found that BEM agreed closely with cylindrical matching for the same discretized layers, while extending the continuous profile from 5Rq to 10Rq changed Im(Z) substantially in part of the sampled band. This identifies profile truncation as a separate issue from BEM field error. The original audit source, CSV and figure are in `examples/roughness_audit/`; those archived BEM values are v0.4 results, not a new v0.5 run.

More slices improve the conductivity profile; more panels improve the boundary fields and geometry. Neither resolves the other error. Converge the extent independently of both, keep the physical mean surface fixed, and monitor signed Im(Z) and ΔIm(Z) relative to a smooth pipe at the same mean radius. Do not use a relative imaginary error at a zero crossing; report an absolute error there.

```python
from arcs import MaxwellSolver, Metal, Roughness, load_geometry
case = load_geometry('examples/geometries/bare_copper.json')
solver = MaxwellSolver(case.to_panels(), beta=0.7,
                       roughness=Roughness(0.3e-6, slices=32, extent_sigma=5))
response = solver.transverse([1e8, 1e9], Metal(5.8e7))
```

In JSON, RMS height follows the geometry's units:

```json
"beta": 0.7,
"roughness": {"model": "conductivity_gradient", "rms_height": 0.0003,
              "slices": 32, "extent_sigma": 5}
```

This example uses millimetre geometry units. Omitting `roughness` gives the smooth case. The GUI exposes RMS height, slices and extent. Its persistent counter includes all generated meshes, and **Zoom to layer meshes** separates thin interfaces graphically. `material_panels()` and `export_panels()` include the same actual interfaces.

## Coating gradients

For offset `Wall` layers, conductivity is the sum of Gaussian-smoothed conductivity jumps. Mean-interface positions follow coating thicknesses; interfaces can have different RMS heights using `Roughness(..., interface_heights=(rq0,rq1,...))`. The first height is for vacuum to first metal. Permeability, permittivity and relaxation time must be common across the stack because this model smooths conductivity only. Nonpositive profiles and self-intersecting offsets are rejected. Independently drawn rough coating interfaces require a defined normal-depth coordinate.

## Numerical comparisons

The executed notebook solves 100 frequencies from 10 Hz to 10¹³ Hz and separates:

1. BEM versus cylindrical Maxwell matching for the **same 32 homogeneous layers**: numerical field/geometry error.
2. Those layers versus continuous cylindrical Maxwell integration at the **same cutoff and mean surface**: material-profile subdivision error; a separate 10-RMS continuous profile reveals the remaining cutoff effect.
3. Rough versus smooth full-Maxwell impedance at the **same mean radius**: the reactive increment within the conductivity-gradient model.

The continuous circular check evolves the surface ratio `z=−Ez/Hφ` through the profile, with

\[
 z'=y z^2+z/r-q^2/y,\qquad y=\sigma+i\omega\epsilon,\qquad q^2=k^2+i\omega\mu y,\qquad k=\omega/(\beta c).
\]

Both displacement current and synchronous propagation are retained. The notebook uses β=1 and μr=εr=1 for this continuous check; then the vacuum terms cancel exactly in q², but displacement current remains in y. The outer boundary is the cylindrical decaying bulk solution. The implementation and its constant-conductivity/vacuum-annulus checks are in `examples/roughness_audit/audit_imaginary.py`.

The solver evaluates material q² in an algebraically equivalent form that avoids subtracting equal large vacuum terms in weak conductivity tails. This improves numerical conditioning without introducing an MQS approximation.

`python examples/roughness_benchmark.py` reproduces the legacy −5Rq comparison with the supplied reduced depth ODE. Its discrepancy is **not** proof that the physical roughness model is correct, and the upper-band discrepancy's cause is not assigned solely to omitted terms. The supplied reference is not a universal full-frequency benchmark. Its exp(+iωt) DC result uses the new ARCS convention directly, and the surface quantity is converted to a beam impedance only within that independent comparison.

This model homogenizes wall-depth conductivity. Corrugations varying along z and their scattering/radiation require a longitudinally varying solver or a justified effective theory. Neither circular matching nor agreement with a conductivity-gradient formula experimentally validates a real rough pipe.

Original model literature and author software references are preserved in the bundled reference bibliography: G. Gold and K. Helmreich, IEEE Trans. Microwave Theory Tech. 65 (2017), DOI 10.1109/TMTT.2017.2695192; https://github.com/garrettj403/GradientModel.

## Extended-tail failure and release boundary

A proposed 64-slice, 10-RMS BEM profile was tested at 10 Hz and 10 THz, at the same fixed mean surface. The 10-THz longitudinal discrepancy against identical circular layers was about 85%, while the transverse comparison also failed. The near-insulating front layers have extremely small q²; the current two-field trace elimination contains 1/q² terms and loses accuracy in this regime. A stabilized formulation is still needed. Increasing extent is therefore not offered as a reliable numerical fix. The solver raises `NotImplementedError` for roughness extents above 5, while geometry/mesh construction remains available.

The failed trial and successful truncated-profile endpoint checks are archived in `results/notebook/roughness_endpoint_check.json` and `roughness_legacy_endpoint_check.json`. The 5-RMS default agrees with same-layer cylindrical matching at those endpoints (longitudinal errors below 0.20%, dipolar below 0.57% for 64 boundary panels), but this says nothing about the shared cutoff error. The notebook plots both continuous cutoffs over the complete requested frequency grid so this limitation stays visible.
