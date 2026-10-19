# Python geometry editor and CAD input

ARCS has two entry routes into the same solver-readable geometry: import a
cross-section file, or draw/edit it in a Python desktop application. The editor
uses PySide6. Geometry operations are ordinary Python functions independent
of Qt, so command-line scripts and notebooks can use them too.

## Install and launch

From the extracted source directory, using Python 3.10 or later:

    python -m pip install -e ".[gui,cad]"
    arcs-gui

Alternatively, after installing the dependencies:

    python launch_gui.py
    python launch_gui.py examples/geometries/finite_copper_air.json

Use a Python virtual environment for a separate installation if desired. The
GUI needs a desktop display. No browser, server, or internet connection is
needed while editing. The core impedance code does not import Qt.

## Drawing a cross-section

The default case is a 30 mm radius vacuum aperture with copper filling the
exterior. This is an editable example, not an automatically inferred material.

1. Select **Circle**, **Ellipse**, **Square**, **Rectangle**, **Rounded
   rectangle**, or **Polygon**. Drawing replaces the selected boundary.
2. For a circle, drag from the centre to its radius. For the other dimensioned
   shapes, drag opposite corners. For a polygon, click vertices in perimeter
   order, then press Enter or use Finish polygon.
3. Choose **Dimensions / coordinates** to enter exact values. The Coordinates
   tab accepts one x,y pair per line, with optional x,y header and comments.
   The Native shape / arcs tab accepts the existing ARCS line/arc JSON.
4. **Select / move** selects a visible region and allows its boundary to be
   dragged. The boundary list also selects thin layers that are hard to click.
5. **Crop** keeps the part of a selected boundary inside a dragged rectangle.
   The Drawing operation selector also supports union, subtraction, and
   intersection with newly drawn shapes.
6. Set the beam location numerically or select **Beam** and click inside the
   aperture. Scroll to zoom at the pointer; middle-button drag pans; Fit
   restores the complete geometry view.
7. Save an ARCS JSON case. This contains geometry, units, beam, material
   properties, and mesh settings. Save view exports the actual canvas to PNG.

Undo/redo applies to geometry, materials, units, mesh settings, and beam edits.
Invalid edits are transactional: the previous geometry remains intact.
Changing mm to m, um, or nm converts displayed values while preserving the
physical geometry, beam, and mesh lengths.

## See and update the boundary mesh

The mesh overlay is visible by default. Each physical boundary can have its
own saved mesh settings:

1. Choose **Mesh settings** in the toolbar and click a boundary edge. This tool
   selects without moving geometry. You can also use the physical-boundary
   list, count table or **Edit mesh for** selector on the right.
2. Check the boundary name in **Boundary mesh**. Set maximum panel length,
   curve tolerance, minimum panel count, corner grading and integration order.
   Lengths use the drawing units. Minimum panels is a lower bound; other
   constraints can increase the actual count shown in the table.
3. Click **Generate / update mesh**. Only that boundary's settings are changed.
   They are saved in the ordinary ARCS JSON and used automatically by the solver.
4. Click another boundary to inspect/edit its settings. **Use shared defaults**
   clears only the selected boundary's overrides. Choose **Shared defaults** in
   the target selector to edit settings inherited by unmodified boundaries.

The canvas draws the actual straight panel edges in alternating colours.
The **All boundaries / Selected boundary** selector above it controls display,
independently of the mesh editing target. Save case preserves the overrides;
undo/redo and changes of drawing units preserve their physical meaning.

For very thin curved coatings, independent chord partitions can cross even
when native curves are nested. Such mesh edits are rejected transactionally.
Refine the enclosing boundary or use matching partitions. Generated roughness
interfaces retain the aperture partition and do not have independent overrides.

Meshing controls are at the top of the right panel. **Generate / update mesh**
opens **Mesh only** and activates **Mesh settings**, allowing another boundary
to be clicked immediately. **Inspect mesh** still selects individual panels.
Display controls remain
visible above the drawing. Choose **Geometry +
mesh** to compare it with the original curves. Choose all boundary meshes or
only the inspected boundary. **Show mesh nodes** draws endpoint circles and
is enabled by default; **Show integration points** adds Gauss crosses.

The **Panel detail** inset magnifies a real selected panel, retaining equal x/y
physical scale. Its endpoints stay readable even when the full-view mesh is
too dense to distinguish. **Fit** leaves space beside the geometry for the inset.
Select **Inspect mesh** and click a line panel, or choose **Inspect boundary**
and a one-based **Panel** number below the canvas. Every generated roughness
interface is available in this selector. The selected panel is highlighted;
the details give its endpoint coordinates, physical length and integration
point count. **Zoom to panel** magnifies the main view around that panel.
**Zoom to layer meshes** resolves closely spaced interfaces using their actual
physical separation. **Fit** restores the full cross-section view.

The summary above the canvas shows mesh, panel and node totals for all interfaces.
The table lists per-interface counts, while the selected-boundary information
box shows its panel count and min/max lengths. Zoom around a corner to distinguish short panels.
The same to_panels() endpoints feed the numerical calculation. The shipped
`scripts/capture_gui_mesh.py` checks the displayed arrays against solver meshes
and captures actual windows for a circle, sharp corners with varying coating
thickness, and a 150 nm layer. A display
control changes visibility, not geometry or the solver mesh.

These are boundary line meshes; the Maxwell backend has no volume triangles.
Gauss markers show the geometry/export integration sites. Material-kernel
quadrature separately uses the conductor kernel_order setting (default 12).
Mesh counts do not certify impedance accuracy. Check physical/mesh convergence
for your frequency band.

## Materials and the exterior

The innermost region is always vacuum. Every interface separates the region
on its inner side from the next region. Boundaries must form one aperture and
strictly nested closed loops: touching, intersecting, or disconnected loops
are rejected.

The region after the last boundary extends to infinity. Select its material:

| Physical configuration | Boundaries | Materials, inside to outside |
| --- | --- | --- |
| Infinite copper wall | Aperture only | vacuum → Cu |
| Finite copper wall in air | Aperture, copper outer boundary | vacuum → Cu → air |
| Coating, finite copper wall in air | Aperture, coating outer boundary, copper outer boundary | vacuum → coating → Cu → air |
| Coating on infinite copper | Aperture, coating outer boundary | vacuum → coating → Cu |

For a finite wall, first draw the aperture. Click **+ Draw outer interface** and
draw the wall's outer boundary. The previous exterior becomes a finite layer;
the new exterior defaults to air and can be changed explicitly.

Material properties use SI units: conductivity in S/m, relative permeability,
relative permittivity, and relaxation time in seconds. Cu, Al, SS, and NEG
entries are editable example DC properties, not a material certification.
Air initially means a nonconducting medium with epsilon_r=mu_r=1; this is an
idealization. Smooth nested conducting regions are now matched by the Maxwell solver; an Air/Vacuum exterior is supported for beta<1; apply the chosen beta before solving.

The geometry file can now supply the material for a supported bare-metal
case:

    from arcs import load_geometry
    case = load_geometry("my_bare_copper_case.json")
    solver = case.make_solver("maxwell")
    z_per_m = solver.longitudinal([10., 100., 1000.])

The solver supports nested conducting materials, positive permeability/permittivity and optional Drude conductivity. Smooth coating interfaces are matched. Finite walls followed by ideal Air/Vacuum can be solved for beta<1. The exact beta=1 open-exterior limit and finite insulating gaps remain unsupported.
The editor displays this limitation. Defining geometry does not implement
its electromagnetic model or certify its accuracy.

## File import

Use Open / import, or:

    arcs-import drawing.dxf case.json --exterior Cu
    arcs-import outline.csv case.json --units mm --exterior Cu
    arcs-geometry case.json --preview case.png --export-panels panels.csv

Python API:

    from arcs import import_geometry, load_geometry
    result = import_geometry("drawing.dxf")
    result.geometry.save("case.json")
    print(result.notes)
    case = load_geometry("drawing.dxf")  # automatic DXF dispatch

| Input | Conversion |
| --- | --- |
| ARCS JSON | Exact native shapes, regions, material properties and mesh settings |
| Ordered XY CSV | Polygon with the original corners; units must be supplied |
| Planar DXF | Native circles, full ellipses, lines, circular arcs and polyline bulges; closed splines become polygons |

DXF $INSUNITS determines physical scaling, including conversion from inches
or other supported units to the selected ARCS units. A unitless DXF requires
the user to supply its source units. A conflicting override is rejected.
Line/arc fragments are joined into unambiguous closed loops on their DXF
layer, without adding a missing closing edge. Multiple outlines are ordered
by enclosed area and then checked for strict nesting. Layer names identify
boundaries; ordinary DXF does not specify physical material assignments.
Review every layer material and exterior material after importing.

DXF text/dimension annotations are ignored and reported. Blocks, hatches,
tilted/out-of-plane entities, open splines and partial ellipses are rejected
with a conversion instruction. Export blocks/hatches as simple planar
outlines first. SVG, DWG, STEP/STL, and bitmap tracing are not implemented in
this version. A 3D drawing is never silently projected into 2D.

DXF export retains native circles, ellipses and line/arc boundaries. The ARCS
JSON remains the authoritative file for material properties and beam/mesh
settings; the DXF is a geometry exchange file.

## Cropping precision and topology

Drawing and dimension edits retain native circles, ellipses and arcs. Boolean
operations use Shapely polygon geometry: curved inputs are approximated
using the case's curve tolerance, while original straight segments remain
straight. The native-curve sampling uses a conservative chord error bound.
The result is a polygon; tightening the tolerance and repeating the edit
can change its geometric accuracy. Closed DXF splines use ezdxf flattening
at the supplied tolerance, with an import note recording this approximation.

The curve tolerance controls geometry, not impedance error. Mesh refinement,
quadrature checks, and independent electromagnetic benchmarks are still
needed, especially near corners and closely spaced interfaces.

Boolean operations that create holes, internal material islands, disconnected
apertures, or an empty outline are rejected. Supporting these would require a
general region-adjacency representation and matching field backend. There is
no automatic corner rounding, spline fitting, point sorting, or convex hull.

## Verification

The release checks use offscreen Qt to exercise actual mouse presses,
dragging, polygon clicks, keyboard completion, material selection, units,
and PNG rendering. DXF tests check physical units, native curves, a 150 nm
concentric layer, open outlines, planar restrictions, and polygon corners.
These are geometry/UI checks alongside the existing physics tests.

    QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONPATH=src python -m unittest discover -s tests -v

An actual application-window image can be reproduced headlessly:

    python launch_gui.py examples/geometries/finite_copper_air.json --screenshot editor.png

Primary dependency documentation:

- Qt for Python: https://doc.qt.io/qtforpython-6/
- ezdxf: https://ezdxf.readthedocs.io/
- Shapely: https://shapely.readthedocs.io/

## Finite beta and conductivity-gradient controls

The **Surface roughness** section has an **Enable surface roughness** checkbox.
Check it to enable the model, or uncheck it for a smooth wall, then press
**Apply β / roughness** and save the case. Roughness parameter fields and their
labels are greyed out while unchecked. The Gaussian assumption describes the
distribution of surface heights; it is not a separate physical option.

The right panel stores beam β and optional Gaussian roughness RMS height, slice count and truncation extent. Press Apply β / roughness, then save the case. Profiles retain the previous 32 slices and extent 5; explicit saved settings are preserved. Drawing units apply to RMS height. The supplied aperture is the air-side plane, and the mean surface is `extent_sigma*Rq` outward. The GUI enables a rough bare wall; Python can also build offset rough coating stacks. Numerical sweeps use `examples/run_case.py`.

The blue summary above the canvas always shows **Boundary meshes**, **Panels**, **Nodes**, **Displayed meshes** and **Integration points**, including when the overlay is hidden. Every conducting interface contributes one boundary line mesh; a 32-slice bare rough profile has 33 such meshes including its aperture. For a closed loop, unique nodes equal panels. The per-interface table reports separate panel/node counts; clicking a row selects that mesh for inspection. Use **Zoom to layer meshes** to inspect the actual thin interfaces. These are boundary panels; there is no volume triangle mesh.

An extent above 5 is currently a **mesh-only configuration**. The GUI indicates this limitation, and field solving raises an error. Extending the near-insulating Gaussian tail exposed large numerical discrepancies; the 5-RMS default is reproducible but has an unresolved cutoff error. See [roughness.md](roughness.md).
