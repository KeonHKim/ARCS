"""Verify an isolated wheel install, its GUI, and saved boundary settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root',type=Path,required=True,
                    help='Directory containing the isolated wheel-installed arcs package')
args=parser.parse_args()
sys.path.insert(0,str(args.package_root.resolve()))
import numpy as np
import arcs
from arcs import load_geometry, Metal, PEC, Wall
from arcs.reference import circular_exact
from arcs.gui import GeometryWindow
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

installed=Path(arcs.__file__).resolve().parent
assert installed==args.package_root.resolve()/'arcs'
assert arcs.__version__=='0.6.6'
for p in (ROOT/'src/arcs').glob('*.py'):
    assert p.read_bytes()==(installed/p.name).read_bytes(),p.name
case=load_geometry(ROOT/'examples/geometries/independent_meshes.json')
data=np.genfromtxt(ROOT/'results/independent_meshes/spectrum.csv',delimiter=',',names=True)
indices=[0,50,99]
response=case.make_solver().transverse(data['f_Hz'][indices])
np.testing.assert_allclose(response.longitudinal,
    (data['Re_long_Ohm_m']+1j*data['Im_long_Ohm_m'])[indices],rtol=1e-11)
np.testing.assert_allclose(response.dipolar[:,0,0],
    (data['Re_dipolar_Ohm_m2']+1j*data['Im_dipolar_Ohm_m2'])[indices],rtol=1e-11)
bare=load_geometry(ROOT/'examples/geometries/bare_copper.json')
default=bare.make_solver()
assert default.pipe_length_m==1.0 and default.longitudinal_units=='Ohm'
assert default.transverse_coefficient_units=='Ohm/m'
f=np.array([10.,1e6,1e9,1e12,2e12,1e13])
z=bare.make_solver(mesh={'minimum_panels':256}).longitudinal(f)
reference=circular_exact(f,.03,Wall(Metal(5.8e7)))
error=float(np.max(abs(z/reference-1)));assert error<.005
assert z[3].imag>0 and z[4].imag<0
total=bare.make_solver(mesh={'minimum_panels':256},pipe_length_m=4.5)
np.testing.assert_allclose(total.longitudinal(f),4.5*z,rtol=1e-13)
per_response=bare.make_solver(mesh={'minimum_panels':256}).transverse(f[:2])
total_response=total.transverse(f[:2])
for name in ('longitudinal','dipolar','quadrupolar','constant'):
    np.testing.assert_allclose(getattr(total_response,name),4.5*getattr(per_response,name),rtol=1e-13)
assert total_response.pipe_length_m==4.5 and total.longitudinal_units=='Ohm'
pec=load_geometry(ROOT/'examples/geometries/pec_circle.json')
assert isinstance(pec.make_solver().default_wall,PEC)
np.testing.assert_array_equal(pec.make_solver().transverse([10.,1e13]).longitudinal,0)
finite=pec.to_dict();finite['beta']=.7
image=load_geometry(finite).make_solver().transverse([10.,1e6,1e9])
from arcs.reference import circular_indirect_space_charge
np.testing.assert_allclose(image.longitudinal,circular_indirect_space_charge([10.,1e6,1e9],.03,.7),rtol=.001)
app=QApplication.instance() or QApplication([])
window=GeometryWindow(case);window.show();app.processEvents();window.canvas.fit()
form=window.mesh_group.layout()
assert form.labelForField(window.panel_length).text()=='Maximum panel length'
assert form.labelForField(window.integration_order).text()=='Gauss points / panel'
window.select_tool('mesh')
QTest.mouseClick(window.canvas,Qt.MouseButton.LeftButton,pos=window.canvas.pixel((35,0)).toPoint())
assert window.mesh_target.currentText()=='coat_outer'
before=window._all_mesh_cache[0][2].panel_start.copy()
window.minimum.setValue(224);QTest.mouseClick(window.mesh_button,Qt.MouseButton.LeftButton)
assert len(window._all_mesh_cache[1][2].panel_start)==224
np.testing.assert_array_equal(window._all_mesh_cache[0][2].panel_start,before)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'saved.json';window.document.save(path)
    restored=load_geometry(path).make_solver().backend
    assert len(restored.interfaces[0].panel_start)==224
    np.testing.assert_array_equal(restored.interfaces[0].panel_start,window._all_mesh_cache[1][2].panel_start)
window.document.saved_data=window.document.data;window.close();app.processEvents()
wheel=ROOT/'dist/arcs-0.6.6-py3-none-any.whl'
report={'version':arcs.__version__,'wheel_install':True,'PEC_bare_and_finite_beta_checked':True,'installed_package_path':str(installed),
        'installed_sources_match':True,'circle_max_complex_relative_error':error,
        'new_imaginary_sign_checked':True,'independent_interface_spectrum_matches_fresh_source':True,
        'GUI_launched':True,'boundary_click_and_independent_edit_checked':True,
        'saved_case_has_independent_solver_meshes':True,
        'pipe_length_scales_all_impedance_fields':True,'pipe_length_output_units_checked':True,
        'updated_mesh_labels_checked':True,'default_pipe_length_m':default.pipe_length_m,
        'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest()}
(ROOT/'results/install_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
