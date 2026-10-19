# Boundary panels and integration samples

**Maximum panel length** is the largest allowed straight boundary segment,
in the drawing units. It is independent of the axial pipe length in metres
chosen when running an impedance calculation. Smaller segments resolve
cross-section shape and boundary field variation. A 1 mm default is a starting
choice, not a physical requirement or an accuracy guarantee.

**Curve tolerance** bounds the native curve-to-chord geometry error in the
drawing units. **Minimum panels** supplies a lower bound; length, curvature
and corner constraints can require more. **Corner grading** clusters panels
near genuine corners; 1 gives uniform subdivision. Corners remain exact nodes.
These settings are available for each physical boundary.

## What integration order means

An integral over a panel is a continuous sum. Numerical quadrature estimates
it using a few values of the integrand inside that panel and their weights.
For panel length `ell`, a Gauss-Legendre rule uses

\[
\int_0^\ell g(s)\,ds\;\approx\;
\ell\sum_{k=1}^{q}\bar w_k\,g(\ell t_k),
\qquad \sum_{k=1}^{q}\bar w_k=1.
\]

Here `q` is the number of Gauss points. The locations `t_k` lie inside the
segment and are not uniformly spaced. Their weights are also unequal.
For a 1 mm panel, the four-point rule is:

| Sample | Distance from start [mm] | Fractional weight |
| --- | ---: | ---: |
| 1 | 0.069432 | 0.173927 |
| 2 | 0.330009 | 0.326073 |
| 3 | 0.669991 | 0.326073 |
| 4 | 0.930568 | 0.173927 |

The two endpoint nodes define the segment. The four interior samples are
integration points, shown as crosses in ARCS. They introduce no extra segments
or field unknowns. Four points on each of 189 panels gives 756 integration
points. Eight points gives 1,512 while retaining the same 189 panels and nodes.

![Same single panel, different Gauss sampling](../results/mesh_controls/integration_points.png)

For a smooth polynomial integrand, `q`-point Gauss-Legendre quadrature integrates
degrees through `2q-1` exactly in exact arithmetic. General electromagnetic
kernels can have singular or rapidly varying behavior, requiring different
treatment. Increasing quadrature points and increasing the number of panels
address different numerical errors.

## The current ARCS GUI setting

The control previously named **Integration order** is now **Gauss points / panel**.
Its saved JSON name is still `integration_order`, preserving existing files.
It controls the sampling stored in the geometry mesh, shown on the canvas and
exported to the panel CSV. Use **Show integration points (×)** and **Panel detail**
to view the samples of an individual segment.

In the current Maxwell backend, field assembly builds its own quadrature
from the segment endpoints: bare-wall smooth material kernels default to
`kernel_order=12`; layered kernels use a 16-point rule, with separate adaptive
treatment for self and near interactions. Changing the GUI value from 4 to 8
therefore does **not** change the current Maxwell impedance calculation.
Keep 4 for geometry sampling unless denser display/export sampling is useful.
Assess field accuracy by panel/curve refinement and the separate kernel
settings. The GUI value is not an impedance-accuracy control.
