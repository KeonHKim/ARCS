# ARCS v0.6.7 — Arbitrary Resistive-wall Cross-section Solver

Python boundary elements for a longitudinally uniform pipe and **0 < β ≤ 1**. ARCS calculates **resistive wall plus indirect space charge**, relative to free space, excluding the direct free-space contribution: longitudinal impedance and complete dipolar/quadrupolar transverse tensors. Full vacuum and material Maxwell fields are matched, retaining displacement current and synchronous propagation.

The GUI now supplies only **Vacuum, PEC, Air and Annealed Cu**. Annealed Cu uses conductivity 5.8×10⁷ S/m, εr=μr=1 and relaxation time 0; use **View / edit material properties** to inspect or change those values. Choose **Custom…** to define every other material explicitly. Aluminium, stainless steel and NEG are no longer built-in choices. Existing case files retain their explicit coating properties. The former standard Cu preset becomes Annealed Cu; an edited Cu definition retains its original name and values. Unused entries from the complete old GUI catalog are removed on opening.

PEC is an exact boundary condition available in the solver, JSON cases and GUI. Choose **Materials → Exterior → PEC**, then save and run normally. Bare PEC has exactly zero RW contribution; its finite-beta indirect space charge uses the boundary-element image calculation. Finite conducting coatings can end at PEC. Notebook 06 audits their mesh convergence, including a thin-coating THz case with slow transverse convergence. **v0.6.7 adds an experimental open Air/Vacuum exterior for beta<1 and smooth finite conducting layers.** The two-layer beta=0.52 paper-band circular comparison is below 0.13% at 512 panels/interface, but its total longitudinal error is still about 12% at 10 Hz because RW and indirect space charge nearly cancel. Beta=1 and insulating gaps remain unsupported. See [open exterior](docs/open_exterior.md) and [release notes](docs/release_notes_v0_6_7.md).

The default axial pipe length remains **1.0 m**. Per-boundary mesh settings, visible panel/node counts and signed imaginary plots remain available. Choose **Mesh settings**, click an interface, edit its options and click **Generate / update mesh**. The same saved settings feed the GUI, exported mesh and solver. Notebook examples cover **100 points from 10 Hz to 10¹³ Hz**. DCT/DST conventions are unchanged.

## Install

Python 3.10+; checked with Python 3.12. Extract the ZIP and enter `arcs_v0_6_7`:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[gui,cad,notebook]"
```

For the solver without optional GUI/CAD/notebook packages: `python -m pip install -e .`.
Restart an existing notebook kernel after upgrading. The notebook checks the imported version and source path. The release also includes an installable wheel in `dist/`.

## Run

```bash
arcs-gui
python launch_gui.py examples/geometries/independent_meshes.json
python examples/run_case.py --panels 256 --points 100 --f-min 10 --f-max 1e13
python examples/run_case.py --pipe-length-m 5 --output results/pipe_5m
python examples/run_case.py --beta 0.7 --f-max 1e9 --points 21 --output results/slow_beam
python examples/run_case.py --roughness-rms-um 0.3 --roughness-slices 32 --roughness-extent 5 --f-min 1e7 --f-max 1e10 --points 21 --output results/rough_wall
jupyter lab examples/01_circular_benchmark.ipynb
jupyter lab examples/03_independent_meshes.ipynb
jupyter lab examples/04_pipe_length.ipynb
jupyter lab examples/05_workflow.ipynb
jupyter lab examples/06_PEC_materials.ipynb
```

`python launch_gui.py` also launches the GUI directly from the extracted source. The runner saves signed impedance/tensor CSV, plots, geometry and settings. It defaults to the requested 100-point 10 Hz–10 THz grid. Roughness creates many interfaces and is substantially slower than a bare wall.

The GUI edits JSON, ordered XY CSV and planar DXF, preserves polygon corners, assigns materials and displays actual boundary panels and integration points. **Boundary meshes / Panels / Nodes** stay visible above the canvas. **Mesh settings** selects a boundary without moving it; the boundary list, mesh count table and **Edit mesh for** selector also choose the editing target. **Generate / update mesh** saves that target's controls. **Use shared defaults** removes only its overrides. Shared defaults remain available for unmodified settings. The display selector affects visibility only. Generated roughness interfaces retain the aperture's panel partition. Numerical frequency sweeps use Python or the runner.

The four material presets remain Vacuum, PEC, Air and Annealed Cu. Every other conductor is explicitly user-defined.

## PEC, vacuum and air

The vacuum aperture is fixed to conductivity 0 S/m, relative permittivity and permeability 1, and relaxation time 0. The air preset currently has those same ideal values; it does not model actual gas dielectric corrections. An outer Air/Vacuum region is supported for beta<1 and positive frequencies. Select Air as Exterior and Apply a beta below 1; Air uses the same ideal parameters as Vacuum. No PEC surface or artificial conductivity is added. The exact beta=1 open-exterior limit and insulating gaps remain unsupported.

PEC is an ideal boundary condition, **tangential E=0**, with no finite bulk material parameters. Its JSON definition is `"PEC": {"kind": "pec"}`. Choose PEC only as the final termination; finite layers use conducting materials. Conductivity-gradient roughness with PEC is rejected. A finite conducting coating on PEC can still have resistive loss. Very thin PEC-backed coatings near resonances require careful mesh convergence: notebook 06 records an unresolved transverse example rather than claiming universal accuracy.

```python
from arcs import load_geometry, PEC
case = load_geometry('examples/geometries/pec_circle.json')
solver = case.make_solver()  # full Maxwell, 1.0 m pipe totals
response = solver.transverse([10., 1e6, 1e13])
# This example has beta=1: all returned coefficients are exactly zero.
# Set beta<1 in the case to calculate nonzero PEC image impedances.
```

`case.make_solver("pec")` remains the older beta=1 electrostatic geometry helper. Use the ordinary Maxwell solver with a PEC **material** for impedance and finite-beta cases.

```python
import numpy as np
from arcs import MaxwellSolver, Metal, Roughness, load_geometry
from arcs.plotting import plot_longitudinal

case = load_geometry('examples/geometries/bare_copper.json')
definition = case.to_dict()
definition['beta'] = 0.7
solver = load_geometry(definition).make_solver(mesh={'minimum_panels': 256}, pipe_length_m=None)
f = np.geomspace(10., 1e13, 100)
parts = solver.transverse_components(f)
response = parts["wall"]            # RW + indirect SC (default)
rw_only = parts["resistive_wall"]
image_only = parts["indirect_space_charge"]
z_long = response.longitudinal       # Ω/m
z_dip_x = response.dipolar[:, 0, 0]   # Ω/m² per unit pipe length
z_quad_y = response.quadrupolar[:, 1, 1]

# Input aperture is the truncated air-side plane; mean surface is 5 Rq outward.
rough_solver = MaxwellSolver(case.to_panels(), beta=1.,
                             roughness=Roughness(0.3e-6, slices=32, extent_sigma=5))
z_rough = rough_solver.longitudinal([1e8, 1e9], Metal(5.8e7))
```

The API key `wall` means the **total ARCS BEM result, RW + indirect space charge**;
it does not mean an analytical reference. The component benchmark shows these
three contributions in separate rows. Black curves are circular analytical
references, and open coloured scatter points are BEM values. The complete
relative-error audit is saved separately as `ARCS_v0.6.0_space_charge_errors.png`.

## Pipe length at runtime

```python
solver = case.make_solver(pipe_length_m=5.0)  # metres, independent of drawing units
z = solver.longitudinal(f)                   # Ohm for the 5 m uniform pipe
response = solver.transverse(f)              # dipolar/quadrupolar: Ohm/m
figure = plot_longitudinal(f, z, per_unit_length=False)
```

`--pipe-length-m 5` gives the same normalization in the runner and labels its
CSV/plots accordingly. Omitting the parameter selects a **1.0 m pipe** and total-pipe units.
Choose `pipe_length_m=None` in Python or `--per-unit-length` in the runner for
explicit per-metre normalization. The 2D model is longitudinally uniform: total impedance is
length times impedance per metre. Entrance/exit and axial transitions require
a separate calculation.

**Maximum panel length** controls boundary segments in drawing units.
**Gauss points / panel** is the previous geometry `integration_order`: weighted
samples inside each panel. Four points on each of 189 panels gives 756 points;
eight gives 1,512 without changing panels or field unknowns. This GUI setting
controls displayed/exported sampling. Maxwell field integrals use separate
kernel quadrature. See [mesh controls](docs/mesh_controls.md) for the formula
and a diagram.

## Scope and meshes

One simply connected vacuum aperture and nested conducting regions, ending in an **infinite conductor, PEC, or open Air/Vacuum termination**. Smooth coating interfaces can have independent shapes, varying thickness and different panel counts. Automatic `Wall` layers use normal polygon offsets; no confocal-ellipse restriction is imposed. Open Air/Vacuum exteriors require beta<1; the synchronous exterior field is evanescent. General dielectric/radiating exteriors and finite insulating gaps remain unsupported.

Each physical interface has a boundary line mesh. Four densities per vacuum panel describe its reaction fields. Material-interface traces are eliminated by nonlocal matching. There are no volume triangles. True corners remain vertices; grading controls their resolution. Refinement is required to assess corner fields and arbitrary-geometry accuracy.

Roughness is a **homogenized conductivity profile across wall depth**, rather than resolved bumps along the beam direction. The GUI supports a rough bare wall. Python also supports smoothed offset coating stacks with common permeability, permittivity and relaxation time. Roughness on independently drawn coating interfaces needs a depth-coordinate definition and is rejected. Explicit roughness settings in saved files are preserved; extents above 5 remain meshable but are rejected during field solving. See [roughness.md](docs/roughness.md) for the reference plane and cutoff audit.

Default finite-beta output includes the regular PEC image contribution. A finite beam distribution would be needed for the excluded coincident direct space-charge term. At sufficiently high frequency for a slower beam, the wall response becomes exponentially small and can underflow; the notebook marks where relative-error comparisons remain resolved. The DCT-I/DST-I converter uses a separate uniform frequency grid and `s=βct` for distance conversion.

## Validation

```bash
OPENBLAS_NUM_THREADS=1 python examples/benchmark.py --panels 512 --points 100 --f-min 10 --f-max 1e13
OPENBLAS_NUM_THREADS=1 python scripts/execute_notebook.py
python -m unittest discover -s tests -v
```

Notebooks **05** (workflow) and **06** (PEC/materials) retain labelled v0.6.4 calculations and plots; their code supports v0.6.6, with coating properties now explicitly defined. Notebook **04** retains labelled v0.6.2 outputs for an explicitly selected 5 m pipe. Notebooks **02/03** retain labelled v0.6.1 circular and independently meshed coated-pipe sweeps. Notebook **01** retains its completed v0.6.0 outputs. This material-selection release verifies the numerical modules against v0.6.4 and runs current GUI/CAD and installed-wheel checks. `VALIDATION.json` distinguishes current checks from retained spectra. Historical v0.5.2 data keep their original convention under `results/historical_v0_5_2/`.

Restart the notebook kernel after upgrading. `examples/02_circular_reproducibility.ipynb` is a minimal independent fresh calculation. Run `python examples/space_charge_benchmark.py --beta 0.7 --panels 256 --points 100` for the complete-wall component benchmark.

This is numerical verification of the stated equations and material models. Constant DC conductivity is not a validated copper constitutive model throughout 10 THz, and circular checks do not certify arbitrary shapes or physical rough surfaces. The legacy reduced-ODE comparison is retained separately in `examples/roughness_benchmark.py`, with an explicit 5-RMS-height cutoff. Historical v0.4 data are identified in [release notes](docs/release_notes.md).

Read [space-charge equations](docs/space_charge.md), [instructions](docs/usage.md), [field equations](docs/maxwell.md), [roughness](docs/roughness.md), [transverse definitions](docs/transverse.md), and [wake conventions](docs/wakes.md).

## Sign convention and wake conversion

ARCS fields use exp(-ikz+iωt). Wake spectra use Zlong=∫Wlong exp(-iωt)dt and Ztrans=i∫Wtrans exp(-iωt)dt, matching the user's method. Old v0.5.x complex values must be conjugated once; `convention='arcs_legacy'` performs that conversion for wakes. For new spectra use the default `convention='arcs'`.

```python
from arcs.plotting import plot_longitudinal
from arcs import impedance_to_wake
figure = plot_longitudinal(f, z_long, xscale='linear', yscale='linear')
# Finite-beta total-wall spectra require the full-complex signed-time inverse:
t, w = impedance_to_wake(f_uniform, z_wall_uniform, causal=False)
```

The impedance grid for inversion must include a justified DC value and be uniform, or explicitly resampled. Signed space-charge wakes and beta=1 causal wakes are both plotted in the notebook.
