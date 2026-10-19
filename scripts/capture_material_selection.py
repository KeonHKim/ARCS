"""Capture the real material popup and verify old/custom case preservation."""
from pathlib import Path
import copy
import json
import os
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication
import arcs
from arcs import GeometryDocument, load_geometry
from arcs.geometry_editor import BUILTIN_MATERIALS, default_geometry
from arcs.gui import GeometryWindow, MaterialDialog

expected = ['vacuum', 'PEC', 'air', 'Annealed Cu']
display_expected = ['Vacuum', 'PEC', 'Air', 'Annealed Cu']
assert list(BUILTIN_MATERIALS) == expected
assert list(default_geometry()['materials']) == expected
legacy = default_geometry()
legacy['materials']['Cu'] = legacy['materials'].pop('Annealed Cu')
legacy['regions'][-1]['material'] = 'Cu'
old_presets = {name: {'conductivity': sigma, 'mu_r': 1., 'epsilon_r': 1., 'relaxation_time': 0.}
               for name, sigma in [('Al', 3.5e7), ('SS', 1.4e6), ('NEG', 2.e5)]}
legacy['materials'].update(copy.deepcopy(old_presets))
restored = GeometryDocument(legacy)
assert list(restored.data['materials']) == expected
assert restored.data['regions'][-1]['material'] == 'Annealed Cu'
frequencies = [10., 1e6, 1e12]
np.testing.assert_array_equal(load_geometry(legacy).make_solver().longitudinal(frequencies),
                              restored.geometry.make_solver().longitudinal(frequencies))
# A real coating retains its explicitly stored conductivity through migration.
coated = copy.deepcopy(legacy)
coated['interfaces'] = [{'name': 'coat_outer', 'shape': {'type': 'circle', 'radius': 35}}]
coated['regions'] = [{'name': 'beam', 'material': 'vacuum'},
                     {'name': 'coat', 'material': 'NEG'}, {'name': 'backing', 'material': 'Cu'}]
coated_doc = GeometryDocument(coated)
assert coated_doc.data['materials']['NEG'] == old_presets['NEG']
assert 'Al' not in coated_doc.data['materials'] and 'SS' not in coated_doc.data['materials']
np.testing.assert_array_equal(load_geometry(coated).make_solver().longitudinal(frequencies[:2]),
                              coated_doc.geometry.make_solver().longitudinal(frequencies[:2]))
edited = copy.deepcopy(legacy)
edited['materials']['Cu']['conductivity'] = 4.1e7
edited['materials']['SS']['conductivity'] = 1.3e6
edited_doc = GeometryDocument(edited)
assert edited_doc.data['regions'][-1]['material'] == 'Cu'
assert edited_doc.data['materials']['Cu']['conductivity'] == 4.1e7
assert edited_doc.data['materials']['SS']['conductivity'] == 1.3e6
assert 'Al' not in edited_doc.data['materials'] and 'NEG' not in edited_doc.data['materials']

app = QApplication.instance() or QApplication([])
window = GeometryWindow()
window.resize(1500, 1100)
window.show()
window.mesh_button.click()
window.mesh_view.setCurrentText('Geometry + mesh')
window.canvas.fit()
app.processEvents()
combo = window.regions.cellWidget(1, 1)
choices = [combo.itemText(index) for index in range(combo.count())]
assert choices == display_expected + ['Custom…']
assert [combo.itemData(index) for index in range(combo.count()-1)] == expected
assert window.regions.item(0, 1).text() == 'Vacuum · fixed'
assert combo.currentText() == 'Annealed Cu'
output = ROOT / 'results/gui'
output.mkdir(exist_ok=True)
saved = output / 'annealed_copper_gui_case.json'
window.document.save(saved)
solver = load_geometry(saved).make_solver()
assert solver.default_wall.name == 'Annealed Cu' and solver.default_wall.conductivity == 5.8e7
assert solver.pipe_length_m == 1.0
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'coated.json'
    coated_doc.save(path)
    assert GeometryDocument(path).data == coated_doc.data
# Exercise custom entry through the actual GUI action and save/reload it.
original_exec = MaterialDialog.exec
def accept_custom(dialog):
    dialog.name.setText('My coating')
    dialog.values['conductivity'].setText('1234567')
    return 1
try:
    MaterialDialog.exec = accept_custom
    combo.setCurrentText('Custom…')
finally:
    MaterialDialog.exec = original_exec
assert window.document.geometry.materials['My coating']['conductivity'] == 1234567
assert window.document.geometry.regions[-1]['material'] == 'My coating'
window.document.save(output / 'user_material_gui_case.json')
custom = load_geometry(output / 'user_material_gui_case.json')
assert custom.materials['My coating']['conductivity'] == 1234567
window.document = GeometryDocument()
window.refresh(fit=True)
app.processEvents()
combo = window.regions.cellWidget(1, 1)
combo.showPopup()
app.processEvents()
# The headless plugin's virtual screen is smaller than this window. Place the
# actual popup beside its control, with the selected row aligned to the combo.
popup = combo.view().window()
selected = combo.model().index(combo.currentIndex(), 0)
popup.move(combo.mapToGlobal(QPoint(0, -combo.view().visualRect(selected).top())))
app.processEvents()
# Qt's popup is a separate widget. Render both actual widgets at their real
# positions so the saved image includes the material choices on headless Qt.
picture = window.grab()
offset = popup.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))
painter = QPainter(picture)
painter.drawPixmap(offset, popup.grab())
painter.end()
screenshot = output / f'ARCS_v{arcs.__version__}_material_selection.png'
assert picture.save(str(screenshot))
combo.hidePopup()
report = {'version': arcs.__version__, 'renderer': 'Actual Qt window and popup widgets',
          'predefined_materials': display_expected, 'material_ids': expected, 'dropdown_choices': choices,
          'default_exterior': 'Annealed Cu', 'conductivity_S_per_m': 5.8e7,
          'old_Cu_preset_renamed': True, 'unused_legacy_presets_removed': True,
          'active_coating_and_edited_materials_preserved': True,
          'old_and_migrated_impedance_arrays_identical': True,
          'custom_GUI_entry_save_reload_checked': True,
          'screenshot': screenshot.name, 'mesh_counts': window.mesh_counts}
(output / 'material_selection_update.json').write_text(json.dumps(report, indent=2) + '\n')
window.document.saved_data = window.document.data
window.close()
app.processEvents()
print(json.dumps(report, indent=2))
