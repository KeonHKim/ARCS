# ARCS v0.5.0

## GUI

- Always-visible totals for boundary meshes, panels and unique nodes above the canvas.
- Independent displayed-mesh count; hiding or restricting the overlay keeps all interface totals visible.
- Per-interface panel/node table, scrollable mesh controls and automatic thin-layer zoom.
- Editable roughness extent, with the previous 32-slice, 5-RMS-height default retained. Explicit existing JSON settings remain unchanged.

## Notebook and numerical model

- Circular, ellipse/CSV/DXF import, nonuniform coating, finite-beta and Gaussian roughness examples use 100 direct solver frequencies from 10 Hz to 10¹³ Hz, with plotted longitudinal, dipolar and quadrupolar results and signed CSV data.
- The primary circular release benchmark uses 512 panels on the same 100-point grid. Cylindrical Maxwell matching checks the solver equations using the same constitutive assumptions.
- The roughness section emphasizes signed Im(Z) and ΔIm(Z), compares identical discretized circular layers and continuous cylindrical Maxwell fields, and includes the imaginary-part cutoff audit. Extents above 5 are rejected: an attempted 10-RMS BEM profile failed numerical endpoint checks. The truncated default has a cutoff error and is not the converged full Gaussian model.
- Material transverse-wavenumber squares are evaluated without cancellation of equal vacuum terms. Full Maxwell displacement current remains in the field matching.
- The supplied reduced roughness ODE is retained as a legacy comparison with an explicit 5-RMS-height cutoff. It is not a full-band reference and no universal physical validity is claimed.
- The user's DCT-I/DST-I conversion uses a separate uniform 0–10 THz grid. The resulting wakes are finite-band reconstructions.

## Files and provenance

`examples/01_circular_benchmark.ipynb` is executed with embedded figures. `results/circular_benchmark/`, `results/notebook/`, `results/gui/ARCS_v0.5.0*` and the root `VALIDATION.json` contain the new run's outputs and checks. `results/unit_tests.log` records release tests; `results/notebook_captured_output.log` contains complete captured notebook output. The source includes `scripts/execute_notebook.py`; `--in-process` runs sequential IPython cells with rich figure capture in environments where kernel sockets are unavailable.

The following copied outputs are historical v0.4 material, retained to reproduce previous comparisons: `results/finite_beta/`, `results/roughness/`, `results/roughness_low_frequency/`, `results/cli_coating/`, `results/cli_finite_beta/`, `results/cli_roughness/`, `results/arcs04_*`, older GUI screenshots without the v0.5 name, and `examples/roughness_audit/results/`. Their scripts and metadata identify their original meshes, slice counts, cutoffs and frequency grids. They must not be read as measurements of the new default roughness profile.

## Scope

The solver computes the RW correction to a same-beta PEC pipe for a longitudinally uniform aperture with nested conductors and an infinite conducting exterior. Numerical comparisons do not establish experimental validity of a material model to 10 THz or certify all arbitrary shapes. Roughness is a wall-depth conductivity gradient; longitudinal geometric scattering and outer air/open radiation remain outside the solver's scope. Mesh, profile and cutoff refinement remain independent accuracy checks.

## Extended-tail numerical finding

The proposed 64-slice, 10-RMS default was rolled back after an independent endpoint check. At 10 THz its longitudinal discrepancy was about 85%; low-frequency transverse matching also failed. The new guard prevents publishing those finite but unreliable BEM values. The saved trial is `results/notebook/roughness_endpoint_check.json`; the reproduced 5-RMS endpoints are `roughness_legacy_endpoint_check.json`. A stable formulation for nearly insulating tails remains future work.

Frequency sweeps optionally use independent processes via `ARCS_SWEEP_WORKERS`. The default is 1; the release notebook was executed with 4, with BLAS threads limited to 1 per process. Serial and parallel arrays were checked for equality.
