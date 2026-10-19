"""Build the PEC/material tutorial; execute separately with execute_notebook.py."""
from pathlib import Path
import json
import nbformat

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbformat.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbformat.v4.new_code_cell(text.strip()))

md(r'''
# ARCS 0.6.6: PEC, vacuum and air

PEC is now a selectable material in **Materials → Exterior → PEC**. Use
**View / edit material properties** to inspect the fixed PEC, vacuum and air
settings. Save the case and solve it with the usual Maxwell API/runner.

| Choice | ARCS definition | Field calculation |
|---|---|---|
| Vacuum aperture | σ=0 S/m, εr=μr=1, relaxation time=0 | Required innermost region |
| Air preset | Same ideal parameters as vacuum | Drawing/saving only for wall/exterior regions |
| PEC exterior | Exact tangential-E=0 boundary; no bulk parameters | Bare PEC and conducting coatings backed by PEC |
| Conducting material | Finite positive σ, εr, μr; optional Drude relaxation time | Existing full Maxwell matching |

The air preset does not include actual gas dielectric corrections. An open outer
air region requires dielectric/radiation boundary conditions that are not implemented.
Setting σ=0 does not create PEC: PEC is a separate boundary-condition type.
PEC is supported as the final termination, not as a finite layer before another
material. Conductivity-gradient roughness with PEC is currently rejected.

## How the existing indirect-space-charge calculation works

For a synchronous Fourier component, ARCS uses $k=\omega/(\beta c)$ and
$\kappa=k/\gamma$. In vacuum, the longitudinal spatial Green function is
$G_\kappa=K_0(\kappa|\mathbf r-\mathbf r_s|)/(2\pi)$ for β<1.
The boundary-element single-layer density satisfies

$$S_\kappa\rho=G_\kappa(\mathbf r_b,\mathbf r_s).$$

The regular image Green function inside the pipe is $g_{\rm image}=-S_\kappa\rho$.
It cancels the free-space field at the PEC wall. With ARCS's
$\exp(-ikz+i\omega t)$ convention,

$$\frac{Z_{\parallel,\rm image}}{L}
=-\frac{ikZ_0}{\beta\gamma^2}g_{\rm image}.$$

ARCS differentiates the source and witness kernels analytically for transverse
coefficients. It does not substitute a circular/elliptical analytical result.
The independent circular references below are used only to audit the numerical answer.

The output decomposition is $Z_{\rm wall}=Z_{\rm RW}+Z_{\rm image}$.
For a **bare PEC** wall, $Z_{\rm RW}=0$ exactly. At β=1, the indirect contribution
also vanishes in this longitudinally uniform synchronous model. Direct free-space
space charge and entrance/exit wakes are excluded; this is not the output of a
finite-length PEC pipe in a general 3D wake solver.

The older `case.make_solver("pec")` is a β=1 electrostatic geometry helper, with
no impedance output. The new material option uses `case.make_solver()` (Maxwell)
and supports finite β.

Physics reference: M. Migliorati et al., *Space charge impedance and electromagnetic
fields in elliptical vacuum chambers*, PRAB **21**, 124201 (2018),
[paper](https://cds.cern.ch/record/2650330/files/PhysRevAccelBeams.21.124201.pdf).
The numerical implementation here uses general boundary elements.
''')
code('''
from pathlib import Path
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import arcs
from arcs import PEC, Metal, Wall, Layer, MaxwellSolver, polygon_panels, load_geometry
from arcs.geometry_editor import default_geometry
from arcs.reference import (circular_indirect_space_charge, circular_indirect_dipolar,
                            circular_indirect_quadrupolar, circular_exact, circular_layered_dipolar)
from arcs.plotting import plot_longitudinal
assert arcs.__version__ == "0.6.6", (arcs.__version__, arcs.__file__)
OUTPUT = Path.cwd() / "results" / "pec_materials"
OUTPUT.mkdir(parents=True, exist_ok=True)
print("ARCS:", arcs.__version__, arcs.__file__)
''')
md('''
## 1. Save a PEC case and calculate all components

The example is a 30 mm circular pipe, β=0.7, 1.0 m axial length. It uses 256
boundary panels and 100 frequencies from 10 Hz to 10¹³ Hz. No roughness is enabled.
The intentionally slower beam makes the image impedance visible.
''')
code('''
definition = default_geometry()
definition["name"] = "30 mm PEC circle, beta 0.7"
definition["beta"] = 0.7
definition["regions"][-1]["material"] = "PEC"
definition["mesh"]["minimum_panels"] = 256
case = load_geometry(definition)
case.save(OUTPUT / "pec_circle.json")
solver = case.make_solver(pipe_length_m=1.0)
frequency = np.geomspace(10.0, 1e13, 100)
started = time.perf_counter()
parts = solver.transverse_components(frequency)
response = parts["wall"]
for field in ("longitudinal", "dipolar", "quadrupolar", "constant"):
    np.testing.assert_array_equal(getattr(parts["resistive_wall"], field), 0)
    np.testing.assert_array_equal(getattr(response, field), getattr(parts["indirect_space_charge"], field))
    np.testing.assert_array_equal(getattr(response, field).real, 0)
print("Panel count:", case.report()["boundaries"][0]["panels"])
print("Frequency samples:", frequency.size, "  time [s]:", time.perf_counter()-started)
print("RW is exactly zero; all remaining coefficients are purely imaginary.")
reference_long = circular_indirect_space_charge(frequency, .03, .7)
reference_dip = circular_indirect_dipolar(frequency, .03, .7)
reference_quad = circular_indirect_quadrupolar(frequency, .03, .7)

def resolved_error(actual, reference):
    # All raw samples are retained; this only defines a meaningful error report.
    floor = 1e-8*np.max(abs(reference))
    selected = abs(reference) > floor
    return {"signal_floor": float(floor), "resolved_samples": int(selected.sum()),
            "max_relative_complex_error": float(np.max(abs(actual[selected]/reference[selected]-1))),
            "max_absolute_complex_error": float(np.max(abs(actual-reference)))}

errors = {"longitudinal": resolved_error(response.longitudinal, reference_long),
          "dipolar_xx": resolved_error(response.dipolar[:,0,0], reference_dip),
          "quadrupolar_xx": resolved_error(response.quadrupolar[:,0,0], reference_quad)}
print(json.dumps(errors, indent=2))
np.savetxt(OUTPUT / "pec_spectrum.csv", np.column_stack((frequency,
    response.longitudinal.real, response.longitudinal.imag, reference_long.imag,
    response.dipolar[:,0,0].real, response.dipolar[:,0,0].imag, reference_dip.imag,
    response.quadrupolar[:,0,0].real, response.quadrupolar[:,0,0].imag, reference_quad.imag)),
    delimiter=",", comments="", header="frequency_Hz,Re_long_Ohm,Im_long_Ohm,ref_Im_long_Ohm,Re_dipolar_xx_Ohm_per_m,Im_dipolar_xx_Ohm_per_m,ref_Im_dipolar_xx_Ohm_per_m,Re_quadrupolar_xx_Ohm_per_m,Im_quadrupolar_xx_Ohm_per_m,ref_Im_quadrupolar_xx_Ohm_per_m")
''')
md('''
## 2. Plot the PEC image impedance

Black crosses are independent analytical references. Blue points are BEM values.
Zeros and signs are retained; the imaginary axes use symlog to include zero.
At high frequencies a slower beam's wall field becomes exponentially small.
Low-frequency quadrupolar values can be limited by Cartesian-Hessian cancellation;
the CSV retains them, and the error report states its signal threshold.
''')
code('''
fig = plot_longitudinal(frequency, response.longitudinal, reference=reference_long,
    per_unit_length=False, title="PEC image impedance: 30 mm circle, beta=0.7, L=1.0 m")
fig.axes[0].set_yscale("linear")
fig.axes[1].set_yscale("symlog", linthresh=1e-6)
fig.axes[1].set_ylim(0., 1.3*reference_long.imag.max())
for axis in fig.axes:
    axis.lines[1].set_linestyle("None")
    axis.lines[1].set_marker("x")
    axis.lines[1].set_markevery(5)
fig.savefig(OUTPUT / "pec_longitudinal.png", dpi=180)
plt.close(fig)
display(Image(filename=str(OUTPUT / "pec_longitudinal.png")))

fig, axes = plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
for axis, values, analytical, name in zip(axes,
    (response.dipolar[:,0,0], response.quadrupolar[:,0,0]),
    (reference_dip, reference_quad), ("Dipolar xx", "Quadrupolar xx")):
    axis.plot(frequency, values.imag, "o-", ms=3, label="PEC BEM")
    axis.plot(frequency[::5], analytical.imag[::5], "kx", ms=4, label="Circular analytical")
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=1e-6)
    axis.set(title=name, xlabel="Frequency [Hz]", ylabel="Im(Z transverse) [Ohm/m]")
    axis.grid(True, which="both", alpha=.2)
    axis.legend(fontsize=8)
fig.suptitle("PEC transverse images, beta=0.7, L=1.0 m; real parts are exactly zero")
fig.savefig(OUTPUT / "pec_transverse.png", dpi=180)
plt.close(fig)
display(Image(filename=str(OUTPUT / "pec_transverse.png")))
''')
md('''
## 3. Check the β=1 limit

This is an exact zero in ARCS's output definition, not a very small conductivity
approximation. Compare this to the finite-beta result above. Changing pipe length
scales the uniform-pipe impedance; it does not introduce transition wakes.
''')
code('''
ultra = case.to_dict()
ultra["beta"] = 1.0
ultra["name"] = "30 mm PEC circle, beta 1"
ultra_case = load_geometry(ultra)
ultra_case.save(OUTPUT / "pec_circle_ultrarelativistic.json")
ultra_response = ultra_case.make_solver().transverse(frequency)
for field in ("longitudinal", "dipolar", "quadrupolar", "constant"):
    np.testing.assert_array_equal(getattr(ultra_response, field), 0)
print("All 100 beta=1 longitudinal/transverse samples are exactly zero.")
''')
md('''
## 4. A conducting coating backed by PEC

Finite layers retain their finite conductivity. The final boundary imposes **Ez=0
and Et=0**, and does not set Ht=0: the PEC can carry a surface current. Layer fields
are still eliminated through full Maxwell boundary operators.

This is a deliberately thick **5 mm** test layer, not a representative NEG coating
for the HEB. The aperture and outer coating boundary have independent 128/192-panel
meshes. The circular radial references are independent of those meshes.
''')
code('''
coated = default_geometry()
coated["name"] = "5 mm conducting test layer on PEC"
coated["regions"] = [{"name":"beam","material":"vacuum"},
                     {"name":"coating","material":"NEG"},
                     {"name":"backing","material":"PEC"}]
coated["materials"]["NEG"] = {"conductivity":2e5, "mu_r":1., "epsilon_r":1., "relaxation_time":0.}
coated["interfaces"] = [{"name":"coat_outer","shape":{"type":"circle","radius":35}}]
coated["mesh"].update(minimum_panels=128, max_panel_length=5, curve_tolerance=.1)
coated["boundary_meshes"] = {"coat_outer":{"minimum_panels":192}}
coated_case = load_geometry(coated)
coated_case.save(OUTPUT / "pec_backed_coating.json")
probe = np.array([10., 1e6, 1e9, 1e12])
coated_response = coated_case.make_solver().transverse(probe, component="resistive_wall")
wall = Wall(PEC(), (Layer(Metal(2e5), .005),))
zref = circular_exact(probe, .03, wall, component="resistive_wall")
dref = circular_layered_dipolar(probe, .03, wall, component="resistive_wall")
coated_errors = {"longitudinal": float(np.max(abs(coated_response.longitudinal/zref-1))),
                 "dipolar_xx": float(np.max(abs(coated_response.dipolar[:,0,0]/dref-1)))}
print("Maximum complex relative error at four probes:", coated_errors)
print("The finite coating has nonzero loss; the ideal backing has no bulk loss.")
''')
md('''
## 5. Audit a very thin coating near a THz feature

PEC-backed coatings can have strong mesh sensitivity near resonances. This audit
uses **150 nm**, σ=2×10⁵ S/m, 30 mm aperture radius, β=1, at **1 THz**.
It reports actual longitudinal and dipolar errors at several panel counts, including
an unresolved transverse case. It is not an automatic accuracy certificate.
No analytical substitution, clipping or fitting is used in the solver.
''')
code('''
thin_wall = Wall(PEC(), (Layer(Metal(2e5),150e-9),))
thin_f = np.array([1e12])
thin_zref = circular_exact(thin_f,.03,thin_wall,component="resistive_wall")
thin_dref = circular_layered_dipolar(thin_f,.03,thin_wall,component="resistive_wall")
convergence = []
for count in (64,128,256,384):
    theta = 2*np.pi*np.arange(count)/count
    panels = polygon_panels(.03*np.c_[np.cos(theta),np.sin(theta)],panels_per_edge=1)
    result = MaxwellSolver(panels).transverse(thin_f,thin_wall,component="resistive_wall")
    ze = float(np.max(abs(result.longitudinal/thin_zref-1)))
    de = float(np.max(abs(result.dipolar[:,0,0]/thin_dref-1)))
    convergence.append([count,ze,de])
    print(f"{count} panels/interface: longitudinal {100*ze:.4f}%, dipolar {100*de:.4f}%")
convergence = np.array(convergence)
np.savetxt(OUTPUT/"thin_pec_convergence.csv",convergence,delimiter=",",comments="",
           header="panels_per_interface,longitudinal_relative_complex_error,dipolar_relative_complex_error")
fig,axis = plt.subplots(figsize=(7,4.4),layout="constrained")
axis.loglog(convergence[:,0],100*convergence[:,1],"o-",label="Longitudinal")
axis.loglog(convergence[:,0],100*convergence[:,2],"s-",label="Dipolar xx")
axis.axhline(1.,color="#555555",linestyle="--",label="1% guide")
axis.set(xlabel="Panels per interface",ylabel="Complex relative error [%]",
         title="150 nm coating on PEC at 1 THz: transverse convergence remains slow")
axis.grid(True,which="both",alpha=.2)
axis.legend()
fig.savefig(OUTPUT/"thin_pec_convergence.png",dpi=180)
plt.close(fig)
display(Image(filename=str(OUTPUT/"thin_pec_convergence.png")))
summary = {"version":arcs.__version__,"frequency_samples":100,"frequency_band_Hz":[10.,1e13],
           "pipe_length_m":1.,"beta":.7,"bare_pec_RW_exact_zero":True,
           "beta_one_all_coefficients_exact_zero":True,"indirect_space_charge_errors":errors,
           "thick_coating_probe_errors":coated_errors,"thin_coating_1THz_convergence":convergence.tolist(),
           "thin_coating_transverse_certified":False}
(OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\\n")
''')
md('''
## Run a saved PEC case with one command

From the extracted release folder:

```bash
python examples/run_case.py examples/geometries/pec_circle.json --output results/my_pec
python examples/run_case.py examples/geometries/pec_circle.json --beta 0.7 --output results/my_pec_slow
```

The first example defaults to β=1 and 1.0 m total impedance. The second shows the
finite-beta images. Plotting and unit flags are identical to conducting cases.
''')

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata['language_info'] = {'name':'python','version':'3.12'}
nbformat.write(nb, ROOT/'examples/06_PEC_materials.ipynb')

from sys import path
path.insert(0,str(ROOT/'src'))
from arcs.geometry_editor import default_geometry
from arcs import load_geometry
case = default_geometry()
case['name'] = '30 mm circular PEC pipe'
case['regions'][-1]['material'] = 'PEC'
case['mesh']['minimum_panels'] = 256
load_geometry(case).save(ROOT/'examples/geometries/pec_circle.json')
print(ROOT/'examples/06_PEC_materials.ipynb')
