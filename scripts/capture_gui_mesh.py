"""Capture actual Qt mesh views and record agreement with solver geometry.

Run with Python that has the optional GUI dependencies installed. No impedance
calculation or mesh rescaling is used to create these screenshots.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import arcs
from arcs.gui import GeometryWindow


def main():
    app = QApplication.instance() or QApplication([])
    output = ROOT / "results/gui"
    output.mkdir(parents=True, exist_ok=True)
    report = {"version": arcs.__version__, "renderer": "actual offscreen Qt window", "cases": {}}
    for name, source in (("circle", "bare_copper.json"),
                         ("corners_coating", "nonuniform_coating.json"),
                         ("thin_coating", "coated_circle_150nm.json")):
        window = GeometryWindow(ROOT / "examples/geometries" / source)
        window.resize(1430, 920)
        window.show()
        app.processEvents()
        if name == "circle":
            window.minimum.setValue(256)
        QTest.mouseClick(window.mesh_button, Qt.MouseButton.LeftButton)
        window.quadrature_toggle.setChecked(True)
        window.canvas.fit()
        if name == "thin_coating":
            QTest.mouseClick(window.zoom_mesh_button, Qt.MouseButton.LeftButton)
        else:
            window.inspect_number.setValue(len(window.canvas.mesh.panel_start)//2+1)
        app.processEvents()
        actual = window.document.geometry.material_panels()
        matches = []
        for (_, displayed_name, displayed), (solver_name, solver_mesh) in zip(window._all_mesh_cache, actual):
            assert displayed_name == solver_name
            for field in ("panel_start", "panel_end", "points", "panel_index"):
                np.testing.assert_array_equal(getattr(displayed, field), getattr(solver_mesh, field))
            matches.append({"boundary": displayed_name, "panels": len(displayed.panel_start),
                            "integration_points": displayed.size, "matches_solver_arrays": True})
        if name == "corners_coating":
            corners = window.document.geometry.boundaries[0][1].curves
            for curve in corners:
                vertex = curve.point(0.)
                assert np.any(np.all(window._all_mesh_cache[0][2].panel_start == vertex, axis=1))
        screenshot = output / f"ARCS_v{arcs.__version__}_{name}_mesh.png"
        assert window.grab().save(str(screenshot))
        report["cases"][name] = {"input": source, "screenshot": screenshot.name,
            "counts": dict(window.mesh_counts), "boundaries": matches,
            "nodes_enabled": window.canvas.show_nodes,
            "integration_enabled": window.canvas.show_quadrature,
            "panel_detail_enabled": window.canvas.show_panel_detail,
            "mesh_button_visible": not window.mesh_button.visibleRegion().isEmpty(),
            "inspected_panel": window.canvas.detail_panel+1,
            "panel_info": window.panel_info.text()}
        window.document.saved_data = window.document.data
        window.close()
        app.processEvents()
    path = output / "mesh_visibility.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
