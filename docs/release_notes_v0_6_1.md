# ARCS v0.6.1

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
