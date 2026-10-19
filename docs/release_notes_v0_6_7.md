# ARCS v0.6.7 — finite-beta open Air/Vacuum exterior

An experimental unbounded lossless exterior can now follow smooth finite conducting layers when 0 < beta < 1. This removes the implementation restriction that blocked the beta=0.52 two-layer example of Migliorati et al., Phys. Rev. Accel. Beams 22, 121001 (2019).

Air and Vacuum use exactly the same ideal bulk values: conductivity 0, relative permittivity/permeability 1, relaxation time 0. No PEC termination or small artificial conductivity is substituted. The exterior uses its decaying synchronous K0 Green function and the nonlocal single-layer normal derivative map. Existing conducting and PEC annulus matching is retained.

Python exports `Air` and `Vacuum`. Geometry JSON and GUI exterior selection accept them below beta=1. The GUI material catalog is still only Vacuum, PEC, Air and Annealed Cu; other conductors require explicit definitions. The paper geometry is provided in `examples/geometries/paper_beta052_two_layers_air.json`.

## Validation and limits

- Three new open-exterior tests pass, including independent circular full-Maxwell comparisons, Air/Vacuum equivalence and a fictitious vacuum-interface zero-total-field check.
- A selected regression suite runs 53 tests: 50 pass and 3 optional-dependency tests are skipped.
- The circular two-layer RW reference difference is below 0.13% at eight paper-band samples with 512 panels/interface. The ellipse's maximum equivalent-load formula difference is about 1.25%; the largest residual is horizontal quadrupolar.
- Low-frequency open-wall cancellation remains unresolved: the circular total longitudinal relative error at 10 Hz is 23.30% at 256 panels and 11.64% at 512. Tiny quadrupolar coefficients near numerical precision also need caution. Small linear-system residuals do not establish physical accuracy.
- The beta=1 open-exterior limit, finite insulating gaps and open exterior plus roughness are unsupported and explicitly rejected.

Default pipe length remains 1.0 m. `wall` still denotes the contribution relative to free space: RW plus indirect space charge, excluding direct space charge. Signed impedance conventions and DCT/DST wake conversion are unchanged. The new benchmark does not produce or revalidate wakes.

The accompanying benchmark archive supplies the raw 100-point 10 Hz–100 GHz sweeps, 128/256/512-panel studies, signed plots, published Figure 6 raster extraction, independent reference code and reproducible notebook. Quantitative ellipse validation is limited to the paper's 10 MHz–10 GHz comparison band.
