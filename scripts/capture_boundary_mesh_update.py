"""Capture the actual GUI after selecting and editing a coating boundary."""
import json
import os
from pathlib import Path
import sys
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import arcs
from arcs.gui import GeometryWindow

app=QApplication.instance() or QApplication([])
window=GeometryWindow(ROOT/'examples/geometries/independent_meshes.json')
window.show();app.processEvents();window.canvas.fit();app.processEvents()
window.select_tool('mesh')
QTest.mouseClick(window.canvas,Qt.MouseButton.LeftButton,pos=window.canvas.pixel((35,0)).toPoint())
assert window.mesh_target.currentText()=='coat_outer'
assert window.minimum.value()==192
# Apply a real edit to one boundary, then restore 192 for the delivered example.
inner=window._all_mesh_cache[0][2].panel_start.copy()
window.minimum.setValue(224);QTest.mouseClick(window.mesh_button,Qt.MouseButton.LeftButton)
assert len(window._all_mesh_cache[1][2].panel_start)==224
np.testing.assert_array_equal(inner,window._all_mesh_cache[0][2].panel_start)
window.minimum.setValue(192);QTest.mouseClick(window.mesh_button,Qt.MouseButton.LeftButton)
window.mesh_view.setCurrentText('Geometry + mesh')
window.quadrature_toggle.setChecked(True)
window.canvas.fit();app.processEvents()
form=window.mesh_group.layout()
panel_label=form.labelForField(window.panel_length)
gauss_label=form.labelForField(window.integration_order)
assert panel_label.text()=='Maximum panel length'
assert gauss_label.text()=='Gauss points / panel'
for widget in (panel_label,gauss_label,window.panel_length,window.integration_order,window.quadrature_note):
    assert widget.visibleRegion().boundingRect().contains(widget.rect()), 'Mesh control is clipped'
assert panel_label.width()>=panel_label.fontMetrics().horizontalAdvance(panel_label.text())
assert gauss_label.width()>=gauss_label.fontMetrics().horizontalAdvance(gauss_label.text())
output=ROOT/'results/gui';output.mkdir(exist_ok=True)
path=output/f'ARCS_v{arcs.__version__}_per_boundary_mesh.png'
assert window.grab().save(str(path))
case_path=output/'per_boundary_saved_case.json'
window.document.save(case_path)
solver=window.document.geometry.make_solver().backend
for (_,_,p),q in zip(window._all_mesh_cache,(solver.panels,*solver.interfaces)):
    np.testing.assert_array_equal(p.panel_start,q.panel_start)
    np.testing.assert_array_equal(p.points,q.points)
report={'version':arcs.__version__,'renderer':'Actual Qt window, real mouse-click selection',
        'screenshot':path.name,'selected_boundary':window.mesh_target.currentText(),
        'counts':window.mesh_counts,'boundary_panel_counts':{name:len(p.panel_start) for _,name,p in window._all_mesh_cache},
        'one_boundary_edit_keeps_other_mesh_unchanged':True,'saved_meshes_match_solver':True,
        'maximum_panel_length_label_visible_unclipped':True,
        'gauss_points_label_visible_unclipped':True,
        'geometry_quadrature_explanation_visible':True,
        'integration_points_visible_in_panel_detail':window.quadrature_toggle.isChecked(),
        'mesh_controls_visible':all(not w.visibleRegion().isEmpty() for w in
            (window.mesh_target,window.minimum,window.mesh_button,window.reset_mesh_button))}
assert report['mesh_controls_visible']
(output/'per_boundary_mesh_update.json').write_text(json.dumps(report,indent=2)+'\n')
window.document.saved_data=window.document.data;window.close();app.processEvents()
print(json.dumps(report,indent=2))
