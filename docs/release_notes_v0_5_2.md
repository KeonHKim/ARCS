# ARCS v0.5.2

The GUI now makes the actual boundary mesh visible immediately. Endpoint
nodes are enabled by default, mesh display controls sit above the drawing,
and generation settings appear at the top of the right sidebar. Generating a
mesh switches to the mesh-only view and enables the inspection tool.

A persistent panel-detail inset shows actual panel coordinates with equal
x/y scale. Users can click a panel, choose its boundary and number, read its
endpoints/length, or zoom the main canvas to that panel. The selector includes
generated roughness interfaces. Layer zoom retains the actual separation;
the full-view Fit reserves space so the inset does not cover the pipe.

Actual Qt captures for a 256-panel circle, polygon corners with nonuniform
coating, and a 150 nm circular coating are in `results/gui/`. The capture script
compares displayed endpoint and integration arrays to solver geometry and
checks that original polygon vertices remain mesh endpoints. Mouse-click,
inspection, visibility, zoom, and generated-interface selection checks are
included in the GUI tests.

All numerical source modules, apart from the package version declaration,
are byte-identical to v0.5.1. The longer notebook retains its original v0.5.1
executed numerical plots with provenance and updated import checks. The
minimal reproducibility notebook is executed with v0.5.2. The 5-RMS roughness
cutoff limitation remains.

## Retained v0.5.1 numerical fix

This fixes a reproducible portability failure in the circular benchmark. ARCS v0.5.0 run with SciPy 1.16.0 produced large mid-frequency errors and spikes in Re(Z), despite good-looking published plots generated with SciPy 1.17.0.

The BEM Green kernels called `scipy.special.kv` directly. A faulty AMOS underflow rescaling in some SciPy builds can return imaginary terms around 0.03 for `K0(680−680j)`, whose true magnitude is about 1.9e−297. These false off-panel couplings corrupt the conductor boundary operator. The independent circular reference used scaled Bessel functions already, so it was unaffected.

Every BEM K evaluation now uses the exact identity `K_n(z) = kve(n,z)*exp(-z)`. This also covers coating/interface and finite-beta kernels. No analytical geometry shortcut, spectral fitting or smoothing is used. The existing numerical quadrature tolerances remain unchanged.

All coupled complex matrix solves explicitly select the general solver, avoiding an independently documented SciPy 1.17.0 automatic-structure bug for complex symmetric matrices. That secondary bug was reproduced with an exactly symmetric material matrix; it is separate from the SciPy 1.16 circular failure.

`examples/02_circular_reproducibility.ipynb` is a minimal fresh 100-frequency circular calculation with environment details, raw signed fields, a comparison plot and numerical error checks. The longer benchmark notebook retains all geometry and transverse examples. Restart the kernel after installation and Run All.

References:
- https://github.com/scipy/xsf/issues/46
- https://github.com/scipy/scipy/issues/23330
- https://github.com/scipy/scipy/pull/24367

The physical model is unchanged in v0.5.2. Earlier GUI screenshots and roughness audits retain their original filenames and provenance; see `release_notes_v0_5_0.md`.
