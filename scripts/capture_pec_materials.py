"""Exercise the real Qt PEC selection and capture the visible GUI."""
from pathlib import Path
import json
import os
import sys
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from arcs import __version__, PEC, load_geometry
from arcs.gui import GeometryWindow
import numpy as np

app = QApplication.instance() or QApplication([])
window = GeometryWindow()
window.resize(1500,1100)
window.show()
window.document.set_mesh({'minimum_panels':256})
window.refresh()
app.processEvents()
combo = window.regions.cellWidget(1,1)
index = combo.findText('PEC')
assert index >= 0
combo.showPopup()
app.processEvents()
item = combo.model().index(index,0)
combo.view().scrollTo(item)
app.processEvents()
QTest.mouseClick(combo.view().viewport(), Qt.MouseButton.LeftButton,
                pos=combo.view().visualRect(item).center())
app.processEvents()
assert window.document.geometry.regions[-1]['material'] == 'PEC'
assert window.regions.cellWidget(1,1).currentText() == 'PEC'
assert 'RW=0' in window.status.text()
window.mesh_button.click()
window.mesh_view.setCurrentText('Geometry + mesh')
window.quadrature_toggle.setChecked(True)
window.canvas.fit()
app.processEvents()
output = ROOT/'results/gui'
output.mkdir(exist_ok=True)
path = output/f'ARCS_v{__version__}_PEC_material.png'
assert window.grab().save(str(path))
saved = output/'pec_gui_saved_case.json'
window.document.save(saved)
solver = load_geometry(saved).make_solver()
assert isinstance(solver.default_wall, PEC)
np.testing.assert_array_equal(solver.transverse([10.,1e13]).longitudinal, 0)
report = {'version':__version__, 'renderer':'Actual Qt window and dropdown mouse selection',
          'PEC_visible_in_material_selection':True, 'saved_case_constructs_PEC_solver':True,
          'ultrarelativistic_pec_impedance_exact_zero':True,
          'screenshot':path.name, 'counts':window.mesh_counts,
          'materials':window.document.geometry.materials}
(output/'pec_material_capture.json').write_text(json.dumps(report,indent=2)+'\n')
window.document.saved_data = window.document.data
window.close()
app.processEvents()
print(json.dumps(report,indent=2))
