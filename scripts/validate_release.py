"""Validate current material-selection GUI and installed-wheel release evidence.

All notebooks retain explicitly labelled historical numerical outputs. The
field modules are compared with v0.6.4. This command verifies saved evidence;
it does not rerun the field sweeps or the unit-test suite.
"""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import sys
import nbformat
import numpy as np
from PIL import Image
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arcs
from arcs import Metal, Wall
from arcs.reference import circular_exact, circular_maxwell_dipolar


def read_json(name):
    return json.loads((ROOT / name).read_text())


def notebook_check(name, expected_version):
    nb = nbformat.read(ROOT / name, as_version=4)
    nbformat.validate(nb)
    cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in cells] == list(range(1, len(cells) + 1))
    for cell in cells:
        ast.parse(TransformerManager().transform_cell(cell.source))
    outputs = [output for cell in cells for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    version = nb.metadata.get("numerical_output_provenance", {}).get("executed_version")
    assert version == expected_version, (name, version, expected_version)
    return {"executed_code_cells": len(cells), "errors": 0,
            "embedded_png_plots": sum("image/png" in output.get("data", {}) for output in outputs),
            "executed_version": version, "release_version": arcs.__version__,
            "fresh_for_this_release": version == arcs.__version__}


def spectrum(name, grid):
    data = np.genfromtxt(ROOT / name, delimiter=",", names=True)
    np.testing.assert_array_equal(data[data.dtype.names[0]], grid)
    assert all(np.isfinite(data[field]).all() for field in data.dtype.names)
    return data


def main():
    tests = read_json("results/unit_tests.json")
    install = read_json("results/install_smoke.json")
    assert tests["successful"] and tests["version"] == arcs.__version__
    assert tests["tests_run"] >= 19 and tests["errors"] == tests["failures"] == 0
    final_gui = read_json("results/material_GUI_tests.json")
    assert final_gui["successful"] and final_gui["version"] == arcs.__version__
    retained_full_tests = read_json("results/historical_v0_6_5/unit_tests.json")
    assert retained_full_tests["successful"] and retained_full_tests["tests_run"] == 117
    assert install["version"] == arcs.__version__ and install["wheel_install"]
    assert install["installed_sources_match"] and install["PEC_bare_and_finite_beta_checked"]
    assert install["independent_interface_spectrum_matches_fresh_source"]
    assert install["new_imaginary_sign_checked"] and install["default_pipe_length_m"] == 1.0
    wheel = ROOT / "dist" / f"arcs-{arcs.__version__}-py3-none-any.whl"
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == install["wheel_sha256"]
    baseline = read_json("results/historical_v0_6_4/VALIDATION.json")
    assert baseline["version"] == "0.6.4" and baseline["tests"]["successful"]
    changed_modules = {"src/arcs/__init__.py", "src/arcs/gui.py", "src/arcs/geometry_editor.py", "src/arcs/geometry_io.py"}
    unchanged = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                 for name, digest in baseline["source_sha256"].items() if name not in changed_modules}
    assert all(unchanged.values()) and len(unchanged) >= 15
    retained_names = ["01_circular_benchmark", "02_circular_reproducibility",
                      "03_independent_meshes", "04_pipe_length"]
    retained = {name: notebook_check(f"examples/{name}.ipynb", version)
                for name, version in zip(retained_names, ["0.6.0", "0.6.1", "0.6.1", "0.6.2"])}
    workflow = notebook_check("examples/05_workflow.ipynb", "0.6.4")
    pec_notebook = notebook_check("examples/06_PEC_materials.ipynb", "0.6.4")
    for nb, report_name in [(workflow, "results/05_workflow_execution.json"),
                            (pec_notebook, "results/06_PEC_materials_execution.json")]:
        report = read_json(report_name)
        for field in ("executed_version", "executed_code_cells", "embedded_png_plots", "errors"):
            assert report[field] == nb[field]
        assert nb["executed_code_cells"] >= 6 and nb["embedded_png_plots"] >= 3
        nb["elapsed_seconds"] = report["elapsed_seconds"]
    grid = np.geomspace(10.0, 1e13, 100)
    copper = spectrum("results/ARCS_workflow_tutorial/impedance.csv", grid)
    transverse = spectrum("results/ARCS_workflow_tutorial/transverse.csv", grid)
    z = copper["Re_Z_parallel_Ohm"] + 1j * copper["Im_Z_parallel_Ohm"]
    d = transverse["Re_dipolar_xx_Ohm_per_m"] + 1j * transverse["Im_dipolar_xx_Ohm_per_m"]
    zref = circular_exact(grid, .03, Wall(Metal(5.8e7)))
    dref = circular_maxwell_dipolar(grid, .03, Metal(5.8e7))
    copper_errors = {"longitudinal_max_complex_error_percent": float(100 * np.max(abs(z / zref - 1))),
                     "dipolar_max_complex_error_percent": float(100 * np.max(abs(d / dref - 1)))}
    assert max(copper_errors.values()) < .5
    assert z[np.argmin(abs(grid - 1e12))].imag > 0 and z[np.argmin(abs(grid - 2e12))].imag < 0
    pec = read_json("results/pec_materials/summary.json")
    assert pec["version"] == "0.6.4" and pec["frequency_samples"] == 100
    assert pec["beta_one_all_coefficients_exact_zero"] and pec["bare_pec_RW_exact_zero"]
    image = spectrum("results/pec_materials/pec_spectrum.csv", grid)
    for field in image.dtype.names:
        if field.startswith("Re_"):
            np.testing.assert_array_equal(image[field], 0.0)
    for field, column in [("longitudinal", "long_Ohm"), ("dipolar_xx", "dipolar_xx_Ohm_per_m"),
                           ("quadrupolar_xx", "quadrupolar_xx_Ohm_per_m")]:
        reference = image[f"ref_Im_{column}"]
        selected = abs(reference) >= 1e-8 * np.max(abs(reference))
        error = float(np.max(abs(image[f"Im_{column}"][selected] / reference[selected] - 1)))
        np.testing.assert_allclose(error, pec["indirect_space_charge_errors"][field]["max_relative_complex_error"], rtol=1e-9)
        assert error < .002
    assert not pec["thin_coating_transverse_certified"]
    for name in ("impedance.csv", "transverse.csv"):
        table = spectrum(f"results/pec_runner/{name}", grid)
        for field in table.dtype.names[1:]:
            np.testing.assert_array_equal(table[field], 0.0)
    runner = read_json("results/pec_runner/summary.json")
    assert runner["outer_termination"] == "PEC" and runner["conductivity_S_per_m"] is None
    assert runner["pipe_length_m"] == 1.0 and runner["frequency_samples"] == 100
    gui = read_json("results/gui/pec_material_capture.json")
    boundary_gui = read_json("results/gui/per_boundary_mesh_update.json")
    materials_gui = read_json("results/gui/material_selection_update.json")
    assert materials_gui["version"] == arcs.__version__
    assert materials_gui["predefined_materials"] == ["Vacuum", "PEC", "Air", "Annealed Cu"]
    assert materials_gui["material_ids"] == ["vacuum", "PEC", "air", "Annealed Cu"]
    assert materials_gui["custom_GUI_entry_save_reload_checked"]
    assert materials_gui["old_and_migrated_impedance_arrays_identical"]
    assert gui["version"] == boundary_gui["version"] == arcs.__version__
    assert gui["PEC_visible_in_material_selection"] and gui["saved_case_constructs_PEC_solver"]
    assert boundary_gui["one_boundary_edit_keeps_other_mesh_unchanged"]
    assert boundary_gui["saved_meshes_match_solver"] and boundary_gui["geometry_quadrature_explanation_visible"]
    for report in (gui, boundary_gui, materials_gui):
        with Image.open(ROOT / "results/gui" / report["screenshot"]) as picture:
            picture.verify()
    checks = {"version": arcs.__version__, "completed_at_UTC": datetime.now(timezone.utc).isoformat(),
              "default_pipe_length_m": 1.0, "tests": tests,
              "final_GUI_CAD_tests": final_gui,
              "retained_full_suite_v0_6_5": retained_full_tests,
              "notebook": retained["01_circular_benchmark"], "retained_notebooks": retained,
              "notebook_provenance_note": "All numerical notebook outputs retain their execution provenance. Notebooks 05/06 were executed with v0.6.4; field modules match that version. GUI labels, old/custom case migration and installed-wheel checks are current v0.6.6; the full 117-test v0.6.5 suite is retained separately.",
              "retained_workflow_notebook": workflow, "retained_PEC_notebook": pec_notebook,
              "retained_copper_circular_benchmark": copper_errors, "retained_PEC_benchmark": pec,
              "numerical_modules_identical_to_v0_6_4": unchanged,
              "materials_GUI": materials_gui,
              "PEC_GUI": gui, "per_boundary_GUI": boundary_gui,
              "PEC_runner": runner, "isolated_wheel_install": install,
              "historical_validation": "results/historical_v0_6_4/VALIDATION.json",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "src/arcs").glob("*.py"))},
              "limitations": baseline["limitations"]}
    (ROOT / "VALIDATION.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"version": arcs.__version__, "tests": tests["tests_run"],
                      "retained_notebooks": [workflow, pec_notebook], "copper_errors": copper_errors,
                      "numerical_modules_unchanged": all(unchanged.values()),
                      "materials_GUI": materials_gui,
                      "PEC_signal_errors": pec["indirect_space_charge_errors"],
                      "thin_PEC_coating_certified": False}, indent=2))


if __name__ == "__main__":
    main()
