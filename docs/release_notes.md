# ARCS v0.6.6

GUI material names are now Vacuum and Air. The predefined list is Vacuum, PEC, Air and Annealed Cu. Saved cases retain the canonical identifiers vacuum/air for compatibility. The solver and material properties are unchanged.

# Previous release: v0.6.5

The GUI predefined list contains only vacuum, PEC, air and Annealed Cu. The former Cu preset is named Annealed Cu; conductivity remains 5.8e7 S/m, relative permittivity/permeability 1 and relaxation time 0. Other materials require explicit properties through Custom… or a case file. Aluminium, stainless steel and NEG have no automatically supplied GUI presets.

Opening a file exported with the complete previous catalog removes its unused legacy presets. Materials referenced by actual layers and edited/user-defined properties are preserved. The standard old Cu definition is renamed without changing its properties; a different Cu definition is retained as user-defined. The physics modules and impedance/wake conventions match v0.6.4. Notebook outputs keep their original execution versions, and the PEC coating example now defines its material explicitly.

# Previous release: v0.6.4

PEC is a distinct material type, represented by `{"kind":"pec"}`. The GUI offers PEC as the exterior termination and displays fixed vacuum/air information. Bare PEC has exact zero RW and uses the existing BEM image operators for finite-beta indirect space charge; the beta=1 uniform-pipe impedance is exactly zero. Finite conducting coatings may precede PEC. Their final boundary enforces Ez=Et=0 while allowing surface currents. Conductivity-gradient roughness with PEC and finite PEC layers before another material are rejected.

Notebooks 05 and 06 contain fresh workflow and PEC/material results. The 150 nm coating-on-PEC test near 1 THz exposes slow transverse mesh convergence: 256 panels per interface give approximately 15% dipolar error; this case is not certified. The thick-coating and bare-PEC checks have separate results. No analytical substitution, fitting or clipping enters the field solver.

Air remains an ideal vacuum-like drawing preset; an outer dielectric/radiating air region is not solved. Pipe length remains 1.0 m by default. Existing finite-conductivity calculation paths and wake conventions remain available.

# Previous release: v0.6.3

The default axial length is now **1.0 m**, in both `case.make_solver()` and
`examples/run_case.py`. Outputs use total-pipe units: longitudinal/constant
terms in Ohm and dipolar/quadrupolar coefficients in Ohm/m. Numerical values
match previous per-metre defaults because the multiplier is exactly one.
Explicit `--pipe-length-m` remains available. Select `--per-unit-length` or
`pipe_length_m=None` when per-metre output is desired.

The geometry-only PEC helper has no axial impedance normalization; its ordinary
constructor remains usable. Per-metre benchmark source now requests that mode
explicitly. Maxwell fields, sign conventions, wakes, meshing and the GUI are
unchanged. The geometry-view redesign is under discussion, not implemented.

Fresh default/API/CLI tests and wheel verification accompany this release.
Completed spectra and notebook plots retain their recorded baseline versions;
notebook 04 explicitly uses a 5 m pipe and retains its v0.6.2 calculation.

## Previous v0.6.2 release


Uniform axial pipe length is now an optional **runtime** parameter:
`case.make_solver(pipe_length_m=5.0)` or `examples/run_case.py --pipe-length-m 5`.
Omitting it preserves per-unit-length output. Supplying it scales every complex
longitudinal, dipolar, quadrupolar and constant term, including RW and indirect
space charge. CSV headers, summaries and plotting labels use total-pipe units.
A response records its length so tensor plots infer units and accidental repeat
length integration can be rejected. This is uniform-pipe normalization; entrance,
exit and longitudinal geometry transitions are outside the 2D model.

The mesh control **Max length** is renamed **Maximum panel length**. The
**Integration order** control is renamed **Gauss points / panel**, with a visible
note explaining weighted interior samples and the separate Maxwell kernel
quadrature. The right panel is wider so the complete labels remain visible.

Notebook 04 has a fresh 100-point 10 Hz–10 THz circular Maxwell calculation for
a 5 m pipe, with longitudinal and dipolar references and plotted tensors.
Notebooks 02/03 retain their labelled v0.6.1 results; notebook 01 retains v0.6.0.
The field assembly, material kernels, wake converter and sign convention are
unchanged. Fresh tests check finite-beta components, displaced beams, tensors,
length validation, CLI data/units, installed-wheel output and the rendered GUI.

## Previous v0.6.1 release


Physical boundaries now have independent persisted mesh overrides. The GUI's
**Mesh settings** tool picks boundary edges without moving geometry. The right
panel identifies the target and edits panel length, curve tolerance, minimum
panels, corner grading and geometry integration order. The boundary list,
count table and target selector also choose the editing target. Saving, unit
conversion, undo/redo, mesh export and solver construction preserve the settings.
**Use shared defaults** resets one boundary. Mesh display controls are separate.

Meshes that cross/touch at curved thin interfaces are rejected before solving.
Generated conductivity-gradient roughness interfaces retain their aperture
partition. Existing files without overrides retain their prior mesh behavior.
The electromagnetic kernels and impedance/wake conventions are unchanged.

Fresh v0.6.1 evidence includes all tests, actual GUI screenshots, installed-wheel
verification, a 100-point circular sweep and a 100-point coated-pipe sweep with
unequal 128/192-panel interfaces. The large original notebook retains its
completed v0.6.0 outputs with explicit provenance; its code is updated for v0.6.1.

## Retained v0.6.0 physics

Default impedance now includes the resistive-wall correction and indirect space charge, relative to free space. Direct free-space space charge remains excluded. Longitudinal, full dipolar/quadrupolar tensors and constant steering terms use the same definition. `component='wall'` is default; `'resistive_wall'` and `'indirect_space_charge'` select individual terms. The components methods reuse one material solve per frequency.

The indirect term is the regular PEC image field from the existing aperture single-layer boundary solve. Analytical source/witness derivatives use the same actual panel mesh. No circular approximation or extra mesh is used in production. Circular m=0/m=1 formulas and a rectangle sine-mode Green function provide independent validation.

All field equations, Drude conductivity, displacement-current admittivity, layer matching, reference formulas, outputs and CSV now use exp(-ikz+iωt). This reverses the old imaginary sign. Plots show Im(Z) directly, preserving zero crossings and negative values. The plotting functions also accept `xscale='linear'` and `yscale='linear'`.

The user's DCT/DST convention now matches ARCS directly. `convention='arcs_legacy'` handles old v0.5.x spectra. New `causal=False` inversion uses both complex components on signed time so finite-beta indirect space charge is retained. The original causal kernels and user fixture remain. Gaussian conversion uses the specified beta.

Both notebooks have fresh v0.6.0 outputs. The main notebook covers 100 frequencies from 10 Hz to 10 THz for the circular pipe, imported arbitrary shapes, independent coating interfaces, finite-beta components and conductivity-gradient roughness. It includes signed finite-beta image wakes. Historical results retain their old signs and provenance under `results/historical_v0_5_2/`; the old roughness-cutoff figure is explicitly historical.

Visible GUI mesh panels, endpoints, optional integration points, counts and panel inspection remain. The GUI status identifies the new wall definition. The scaled-Bessel underflow and general complex-solve fixes remain.

At finite beta/high frequency fields can be exponentially screened and underflow. Tiny quadrupoles near zero suffer cancellation; relative comparisons are restricted to resolved values and absolute errors are retained. These numerical benchmarks do not certify every arbitrary shape, a constant-conductivity model to 10 THz, or a physical roughness model. Air exteriors, direct finite-beam space charge and longitudinally varying geometric roughness remain unsupported.

See [space-charge equations](space_charge.md), [field equations](maxwell.md), [wake conventions](wakes.md), and [previous release notes](release_notes_v0_5_2.md).

## GUI wording update

The roughness checkbox is now labelled **Enable surface roughness** in a **Surface roughness** section. Its parameters and labels are disabled while unchecked. The model note identifies the Gaussian surface-height distribution. Apply β / roughness commits the setting; saving preserves it. The electromagnetic solver is unchanged.
