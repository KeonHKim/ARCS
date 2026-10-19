# Geometry input: a small file, a preview, then a field solve

Use dimensions for ordinary shapes, an ordered XY table for a custom polygon,
or native line/arc segments for a curved custom outline. A JSON case keeps all
closed interfaces, units, material-region names, beam position and mesh controls
together. Python dictionaries use the same format. No mesh connectivity must
be entered by the user.

## A first circular case

Save this as `pipe.json`:

```json
{
  "units": "mm",
  "aperture": {"type": "circle", "radius": 30},
  "mesh": {"max_panel_length": 1, "curve_tolerance": 0.01}
}
```

The radius, panel length, curve tolerance and all coordinates here are in mm.
Angles use degrees. The loader converts geometry to metres once; no units are
guessed from the size of an outline or a CSV header. Supported length units
are m, mm, um/µm and nm.

```python
import numpy as np
from arcs import load_geometry, Metal

geometry = load_geometry("pipe.json")
geometry.preview("pipe_mesh.png")
solver = geometry.make_solver("maxwell")
z_per_m = solver.longitudinal(np.geomspace(10., 1e6, 41), Metal(5.8e7, "Cu"))
```

The geometry file's `beam` position is used automatically by this configured
solver. It defaults to [0,0]. Supply `source`/`witness` explicitly to override it.
The Maxwell backend treats a bare infinite conductor and requires separate
field convergence; a geometric panel-size bound is not a physics certificate.
Access the raw numerical backend through `solver.backend` when needed.

From a terminal:

```bash
arcs-geometry pipe.json --preview pipe_mesh.png --export-panels pipe_panels.csv --summary pipe_summary.json
```

After installing with `pip install -e .`, `arcs-geometry` is available. Without
installation, run `PYTHONPATH=src python -m arcs.geometry_cli ...` with the same
arguments. This command checks and previews geometry without running impedance.

## Available shapes

| type | Required dimensions/data | Meaning |
| --- | --- | --- |
| circle | radius | Radius; optional center |
| ellipse | semi_axes: [a,b] | Half axes; optional center and rotation_deg |
| rectangle | width, height | Full dimensions; optional center and rotation_deg |
| rounded_rectangle | width, height, corner_radius | Native straight sides and circular corners |
| polygon | points: [[x,y],...] | Perimeter order; actual sharp vertices retained |
| csv | file | Two columns x,y, optionally headed x,y |
| path | start, segments | A closed sequence of native lines and circular arcs |

Examples for every route are under `examples/geometries`. Run
`python examples/geometry_inputs.py` to make the overview figure, portable JSON
files, all-boundary panel tables and summaries.

For a polygon, do not supply an unordered point cloud: the row order defines
the perimeter. The last point need not repeat the first. Clockwise outlines are
reordered to counterclockwise for the solver, and that change is recorded in
the report. Points are never replaced by a convex hull or silently smoothed.

## XY tables

```csv
x,y
-45,-25
20,-25
45,-10
35,25
0,35
-45,15
```

Reference it in the case file:

```json
{
  "units": "mm",
  "aperture": {"type": "csv", "file": "outline.csv"}
}
```

The CSV path is resolved relative to the JSON file. A bare CSV can also be
loaded as `load_geometry("outline.csv", units="mm")`. `geometry.save(...)`
embeds its points in the resulting JSON, making that file portable without the
CSV. Mesh exports use `_m` column names and metres regardless of the input units.

## Native lines and arcs

```json
{
  "units": "mm",
  "aperture": {
    "type": "path",
    "start": [-30,-20],
    "segments": [
      {"type": "line", "to": [30,-20]},
      {"type": "arc", "center": [30,0], "to": [30,20], "direction": "ccw"},
      {"type": "line", "to": [-30,20]},
      {"type": "arc", "center": [-30,0], "to": [-30,-20], "direction": "ccw"}
    ]
  }
}
```

An arc's center and direction define its sweep; direction can be cw or ccw.
The start and end must be the same distance from the center. End the final
segment at the starting point. Native curves remain in the case definition;
the existing straight-panel backend approximates them only when meshing.

## Multiple coatings and a finite outer wall

The aperture is the first closed boundary. `interfaces` lists subsequent actual
boundaries from inside to outside. Each has an independent shape. Their point
counts, segment counts, centres and shape types may differ. There is no
confocal-ellipse, homothetic or uniform-thickness restriction.

```json
{
  "units": "mm",
  "aperture": {"type": "rectangle", "width": 60, "height": 30},
  "interfaces": [
    {
      "name": "coating_outer",
      "shape": {"type": "polygon", "points": [[-34,-19],[32,-19],[32,18],[-34,18]]}
    }
  ],
  "regions": [
    {"name": "beam_region", "material": "vacuum"},
    {"name": "coating", "material": "NEG"},
    {"name": "substrate", "material": "Cu"}
  ]
}
```

This gives 4 mm coating on the left and bottom, 2 mm on the right and 3 mm on
top, as a visible geometry example. It is not a physical NEG thickness proposal.
The included `coated_circle_150nm.json` demonstrates nanometre-scale dimensions.

For a finite wall, add its outer boundary and a final region labelled with its
exterior material. With B boundaries there are B+1 regions. Material labels
identify the regions; their SI material properties can be stored in the materials
dictionary and supplied to the field model automatically. The last region extends to infinity.

The full-Maxwell backend matches nested conducting material interfaces, including independently shaped coatings. A finite conducting wall followed by ideal Air/Vacuum can be drawn, meshed and solved for beta<1. The beta=1 open-exterior limit and finite insulating gaps are rejected. No interface is silently discarded.

The current format supports one simply connected beam aperture and a nested
sequence of enclosing interfaces. Internal islands, disconnected apertures,
touching regions, NURBS and arbitrary region-adjacency graphs need later schema
extensions. Planar DXF import and the Python sketch editor are available;
STEP/STL and general 3D CAD projection are not implemented. See [gui.md](gui.md).


## Mesh controls and what they guarantee

- `max_panel_length`: upper bound for straight solver-panel length.
- `curve_tolerance`: bound on native-curve interpolation error within a panel.
- `corner_grading`: 1 for uniform spacing, larger for shorter panels near a
  genuine tangent discontinuity. Physical corners remain exact.
- `minimum_panels`: lower bound on boundary panel count, allocated among segments.
- `integration_order`: Gauss points per panel; these are not independent field
  unknowns in the constant-panel backends.

Defaults are 1 mm maximum length, 0.01 mm curve tolerance, grading 2, at least
64 panels per boundary and 4 integration points per panel. Defaults have the
same physical size regardless of the chosen input unit.

The report separates exact native area/perimeter from the chord mesh's area
error. It also separates native interface nesting from chord-mesh nesting.
Closely spaced native curves can be valid while independently sampled coarse
polygons cross. That condition is reported and such layered mesh export is
rejected. It requires refined/correlated or curved interface discretization,
not a change to the physical input dimensions.

Native line/arc intersections are checked before meshing. Ellipse enclosure
uses extrema of the enclosing ellipse's quadratic form; its stationary angles
are found from a quartic. This avoids deciding whether a 150 nm layer is valid
using a micrometre-scale preview mesh. Checks use finite floating-point
tolerances and cannot certify zero gaps or sub-roundoff clearances.

Run `geometry.report()` before a field solve. Inspect dimensions, beam position,
number of boundaries, material order and the mesh preview. The current input
layer does not automatically choose a mesh from frequency, conductivity or a
target impedance error; those convergence controls remain a separate task.

`schema/geometry.schema.json` supplies editor completion and static structure
validation. The Python loader additionally checks topology and geometric units.

## Desktop editor, CAD and material properties

The Python desktop editor and planar DXF conversion are described
in [gui.md](gui.md). The optional `materials` dictionary records conductivity
(S/m), mu_r, epsilon_r and relaxation_time (s) for each named material. The
innermost region must be vacuum. The last region is unbounded, and can be a
conductor, vacuum or air; supported geometry is broader than the currently
implemented field model.


## Dipolar and quadrupolar results

Use solver.transverse(f) on a supported bare-conductor case. The saved beam
position is the reference point for both source and witness. The returned
dipolar and quadrupolar arrays have shape frequency.shape+(2,2), with rows
for kick direction and columns for offset direction. See transverse.md.

## Finite beta and roughness

Save `"beta": 0.7` for a finite beam velocity. Optional `"roughness"` settings select the conductivity-gradient BEM; RMS heights use drawing units. Its aperture is the truncated air-side reference plane. Smooth coating interfaces are independent enclosing outlines and are matched by the Maxwell solver. See [roughness.md](roughness.md) for the supported roughness geometry and reference-plane definition. `material_panels()` and mesh export include generated roughness interfaces.
## Independent boundary mesh settings (v0.6.1)

The optional top-level `boundary_meshes` object stores overrides keyed by
`aperture` or an exact interface name. Each missing setting inherits the
top-level `mesh` defaults. Lengths use the geometry's declared units.

```json
"mesh": {"max_panel_length": 3, "curve_tolerance": 0.05, "minimum_panels": 64},
"boundary_meshes": {
  "aperture": {"minimum_panels": 128},
  "coat_outer": {"minimum_panels": 192, "corner_grading": 1.5}
}
```

See `examples/geometries/independent_meshes.json` for a complete file.
`case.make_solver()` and `case.material_panels()` automatically use these
settings. `case.mesh_settings("coat_outer")` returns the resolved settings.
A call-level `mesh={...}` takes priority over saved settings for the supplied
keys. Existing geometry files without `boundary_meshes` retain shared behavior.
Unknown boundary names and invalid values are rejected. Saved interface meshes
must stay strictly nested for field solving. Generated roughness interfaces
inherit the aperture partition.



## PEC material and fixed vacuum

Use `"PEC": {"kind":"pec"}` in the materials object and assign `"material":"PEC"` to the final exterior region. It is an exact ideal boundary, not a finite conductivity. The ordinary Maxwell `make_solver()` calculates its impedance at any supported beta. Bare PEC has RW=0 and retains indirect space charge; beta=1 makes that contribution zero in this uniform-pipe model. The older `make_solver("pec")` is a separate electrostatic geometry helper.

Finite conducting coatings can be terminated by PEC, with their actual independent interface meshes. Check convergence carefully near resonances; notebook 06 includes a thin PEC-backed coating for which transverse errors remain large at moderate panel counts. PEC is not supported as a finite layer before another material, and conductivity-gradient roughness requires a finite-conductivity substrate.

The innermost vacuum is fixed to conductivity=0 S/m, mu_r=epsilon_r=1 and relaxation_time=0. The built-in air preset uses those same ideal values. Zero conductivity does not mean PEC. Air/Vacuum may terminate finite conductors for beta<1. Finite insulating gaps and the beta=1 open-exterior limit are not implemented.
