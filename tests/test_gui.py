"""Actual offscreen Qt mouse/keyboard interaction and rendering checks."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")


@unittest.skipUnless(importlib.util.find_spec("PySide6"),"Optional PySide6 not installed")
class GUIInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from arcs.gui import GeometryWindow
        self.warning=patch("arcs.gui.QMessageBox.warning",return_value=None)
        self.mock_warning=self.warning.start()
        self.window=GeometryWindow()
        self.window.show()
        self.app.processEvents()
        self.window.canvas.fit()
        self.app.processEvents()

    def tearDown(self):
        self.window.document.saved_data=self.window.document.data
        self.window.close()
        self.app.processEvents()
        self.warning.stop()

    def drag(self,tool,start,end):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        canvas=self.window.canvas
        self.window.select_tool(tool)
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=canvas.pixel(start).toPoint())
        QTest.mouseMove(canvas,canvas.pixel(end).toPoint())
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=canvas.pixel(end).toPoint())
        self.app.processEvents()

    def prepare_coating(self, outer_radius=35):
        self.window.document.set_mesh({'minimum_panels':64,'max_panel_length':100,'curve_tolerance':100})
        self.window.document.add_interface({'type':'circle','radius':outer_radius},name='coat_outer',exterior='Annealed Cu')
        self.window.refresh(fit=True)
        self.app.processEvents()

    def test_click_boundaries_edits_independent_meshes_and_saved_solver(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from arcs import load_geometry
        self.prepare_coating()
        shape_before=[self.window.document.shape(i) for i in (0,1)]
        self.window.select_tool('mesh')
        canvas=self.window.canvas
        QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=canvas.pixel((35,0)).toPoint())
        self.assertEqual(self.window.selected,1)
        self.assertEqual(self.window.mesh_target.currentText(),'coat_outer')
        self.window.minimum.setValue(192)
        self.window.integration_order.setValue(5)
        inner_before=self.window._all_mesh_cache[0][2].panel_start.copy()
        QTest.mouseClick(self.window.mesh_button,Qt.MouseButton.LeftButton)
        self.assertEqual(len(self.window._all_mesh_cache[1][2].panel_start),192)
        np.testing.assert_array_equal(self.window._all_mesh_cache[0][2].panel_start,inner_before)
        # The point is slightly outside the aperture: edge picking must beat
        # filled-region selection and open the aperture's mesh settings.
        QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=canvas.pixel((30.02,0)).toPoint())
        self.assertEqual(self.window.mesh_target.currentText(),'aperture')
        self.assertEqual(self.window.minimum.value(),64)
        self.window.minimum.setValue(128)
        QTest.mouseClick(self.window.mesh_button,Qt.MouseButton.LeftButton)
        self.assertEqual([len(p.panel_start) for _,_,p in self.window._all_mesh_cache],[128,192])
        self.window.inspect_boundary.setCurrentIndex(1)
        self.assertEqual(self.window.minimum.value(),192)
        self.assertEqual(self.window.integration_order.value(),5)
        self.assertIn('coat_outer',self.window.mesh_status.text())
        self.assertIn('boundary-specific',self.window.mesh_status.text())
        self.assertEqual([self.window.document.shape(i) for i in (0,1)],shape_before)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'click_saved.json'
            self.window.filename=str(path)
            self.assertTrue(self.window.save_case())
            case=load_geometry(path);solver=case.make_solver().backend
            for (_,_,displayed),actual in zip(self.window._all_mesh_cache,(solver.panels,*solver.interfaces)):
                np.testing.assert_array_equal(displayed.panel_start,actual.panel_start)
                np.testing.assert_array_equal(displayed.points,actual.points)
        self.mock_warning.assert_not_called()

    def test_mesh_defaults_reset_undo_redo_and_list_selection(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        self.prepare_coating()
        self.window.minimum.setValue(128);self.window.apply_mesh()
        self.window.mesh_target.setCurrentText('Shared defaults')
        self.window.minimum.setValue(96);self.window.apply_mesh()
        self.assertEqual([len(p.panel_start) for _,_,p in self.window._all_mesh_cache],[128,96])
        self.window.boundaries.setCurrentRow(1)
        self.assertEqual(self.window.mesh_target.currentText(),'coat_outer')
        self.window.minimum.setValue(192);self.window.apply_mesh()
        QTest.mouseClick(self.window.reset_mesh_button,Qt.MouseButton.LeftButton)
        self.assertEqual(len(self.window._all_mesh_cache[1][2].panel_start),96)
        self.assertFalse(self.window.reset_mesh_button.isEnabled())
        self.window.undo()
        self.assertEqual(self.window.minimum.value(),192)
        self.assertTrue(self.window.reset_mesh_button.isEnabled())
        self.window.redo()
        self.assertEqual(self.window.minimum.value(),96)
        self.assertIn('shared defaults',self.window.mesh_status.text())
        self.mock_warning.assert_not_called()

    def test_click_boundary_mesh_with_overlay_hidden(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        self.prepare_coating()
        self.window.mesh_toggle.setChecked(False)
        self.window.select_tool('mesh')
        QTest.mouseClick(self.window.canvas,Qt.MouseButton.LeftButton,
                         pos=self.window.canvas.pixel((35,0)).toPoint())
        self.assertEqual(self.window.mesh_target.currentText(),'coat_outer')
        self.assertEqual(self.window.selected,1)
        self.mock_warning.assert_not_called()

    def test_crossing_mesh_edit_keeps_last_valid_preview(self):
        self.prepare_coating(30.00015)
        before=self.window.document.data
        meshes=[p.panel_start.copy() for _,_,p in self.window._all_mesh_cache]
        self.window.minimum.setValue(256)
        self.window.apply_mesh()
        self.mock_warning.assert_called_once()
        self.assertIn('cross/touch',self.mock_warning.call_args.args[2])
        self.assertEqual(self.window.document.data,before)
        for expected,(_,_,actual) in zip(meshes,self.window._all_mesh_cache):
            np.testing.assert_array_equal(expected,actual.panel_start)

    def test_mouse_rectangle_crop_undo_redo(self):
        self.drag("rectangle",(-20,-15),(20,15))
        shape=self.window.document.shape(0)
        self.assertEqual(shape["type"],"rectangle")
        self.assertAlmostEqual(shape["width"],40,delta=2/self.window.canvas.scale)
        initial_area=self.window.document.geometry.boundary().area
        self.drag("crop",(-10,-20),(10,20))
        self.assertEqual(self.window.document.shape(0)["type"],"polygon")
        self.assertLess(self.window.document.geometry.boundary().area,initial_area*.6)
        cropped=self.window.document.data
        self.window.undo()
        self.assertEqual(self.window.document.shape(0)["type"],"rectangle")
        self.window.redo()
        self.assertEqual(self.window.document.data,cropped)
        self.mock_warning.assert_not_called()

    def test_polygon_clicks_enter_and_beam(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        canvas=self.window.canvas
        self.window.select_tool("polygon")
        for point in [(-20,-15),(20,-15),(20,15),(-20,15)]:
            QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=canvas.pixel(point).toPoint())
        QTest.keyClick(canvas,Qt.Key.Key_Return)
        self.assertEqual(self.window.document.shape(0)["type"],"polygon")
        self.window.select_tool("beam")
        QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=canvas.pixel((2,3)).toPoint())
        np.testing.assert_allclose(self.window.document.data["beam"],[2,3],atol=1/self.window.canvas.scale)
        self.mock_warning.assert_not_called()

    def test_pec_selection_and_saved_solver(self):
        from arcs import PEC, load_geometry
        combo = self.window.regions.cellWidget(1, 1)
        self.assertGreaterEqual(combo.findText('PEC'), 0)
        combo.setCurrentText('PEC')
        self.app.processEvents()
        self.assertEqual(self.window.document.geometry.regions[-1]['material'], 'PEC')
        self.assertIn('RW=0', self.window.status.text())
        with tempfile.TemporaryDirectory() as d:
            saved = Path(d)/'pec_gui.json'
            self.window.document.save(saved)
            solver = load_geometry(saved).make_solver()
            self.assertIsInstance(solver.default_wall, PEC)
            np.testing.assert_array_equal(solver.transverse([10.,1e13]).longitudinal, 0)

    def test_fixed_material_information(self):
        with patch('arcs.gui.QMessageBox.information',return_value=None) as info:
            self.window.regions.selectRow(0)
            self.window.edit_material()
            self.assertIn('conductivity=0', info.call_args.args[2])
            self.window.regions.cellWidget(1,1).setCurrentText('PEC')
            self.window.regions.selectRow(1)
            self.window.edit_material()
            self.assertIn('tangential E=0', info.call_args.args[2])
            self.window.regions.cellWidget(1,1).setCurrentText('Air')
            self.window.regions.selectRow(1)
            self.window.edit_material()
            self.assertIn('fields are not implemented', info.call_args.args[2])

    def test_material_air_selection_and_units_combo(self):
        combo=self.window.regions.cellWidget(1,1)
        combo.setCurrentText("Air")
        self.assertEqual(self.window.document.geometry.regions[-1]["material"],"air")
        self.assertIn("Air/vacuum outer",self.window.status.text())
        initial=self.window.document.geometry.boundary().area
        self.window.units.setCurrentText("um")
        self.assertAlmostEqual(self.window.document.shape(0)["radius"],30000,places=8)
        self.assertAlmostEqual(self.window.document.geometry.boundary().area,initial,places=14)
        self.mock_warning.assert_not_called()

    def test_drawing_outer_interface_and_native_circle(self):
        self.window.add_button.setChecked(True)
        self.drag("circle",(0,0),(35,0))
        self.assertEqual(len(self.window.document.geometry.boundaries),2)
        self.assertEqual(self.window.document.shape(1)["type"],"circle")
        self.assertEqual([r["material"] for r in self.window.document.geometry.regions],["vacuum","Annealed Cu","air"])
        self.assertFalse(self.window.adding_interface)
        self.mock_warning.assert_not_called()

    def test_beta_roughness_controls_and_generated_meshes(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        self.assertFalse(self.window.rough_toggle.isChecked())
        self.assertTrue(all(not field.isEnabled() for field in self.window.rough_fields))
        self.window.minimum.setValue(32)
        self.window.panel_length.setValue(100)
        self.window.apply_mesh()
        self.window.beam_beta.setValue(.7)
        self.window.rough_toggle.setChecked(True)
        self.assertTrue(all(field.isEnabled() for field in self.window.rough_fields))
        self.window.rough_height.setValue(.0003)
        self.window.rough_slices.setValue(8)
        QTest.mouseClick(self.window.physics_button,Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertEqual(self.window.document.geometry.beta,.7)
        self.assertAlmostEqual(self.window.document.geometry.roughness.rms_height,3e-7)
        self.assertEqual(len(self.window.canvas.meshes),9)
        self.assertEqual(self.window.mesh_counts["meshes"],9)
        self.assertIn("Boundary meshes: 9",self.window.mesh_summary.text())
        self.assertEqual(self.window.mesh_table.rowCount(),9)
        actual=self.window.document.geometry.material_panels()
        for (_,_,displayed),(_,calculated) in zip(self.window.canvas.meshes,actual):
            np.testing.assert_array_equal(displayed.panel_start,calculated.panel_start)
        self.window.rough_toggle.setChecked(False)
        self.assertTrue(all(not field.isEnabled() for field in self.window.rough_fields))
        QTest.mouseClick(self.window.physics_button,Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertIsNone(self.window.document.geometry.roughness)
        self.assertEqual(self.window.document.geometry.beta,.7)
        self.assertEqual(self.window.mesh_counts['meshes'],1)
        with tempfile.TemporaryDirectory() as directory:
            from arcs import load_geometry
            path=Path(directory)/'smooth.json'
            self.window.document.geometry.save(path)
            self.assertIsNone(load_geometry(path).roughness)
        self.window.undo()
        self.assertTrue(self.window.rough_toggle.isChecked())
        self.assertTrue(all(field.isEnabled() for field in self.window.rough_fields))
        self.assertEqual(self.window.mesh_counts['meshes'],9)
        self.window.redo()
        self.assertFalse(self.window.rough_toggle.isChecked())
        self.assertTrue(all(not field.isEnabled() for field in self.window.rough_fields))
        self.assertIsNone(self.window.document.geometry.roughness)
        self.mock_warning.assert_not_called()

    def test_actual_window_render_is_valid_png(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"editor.png"
            self.assertTrue(self.window.grab().save(str(path)))
            with Image.open(path) as picture:
                self.assertGreaterEqual(picture.width,1200)
                self.assertGreaterEqual(picture.height,800)
                picture.verify()
        self.mock_warning.assert_not_called()

    def test_mesh_generation_button_updates_display_and_solver_panels(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        self.window.mesh_toggle.setChecked(False)
        self.window.minimum.setValue(384)
        QTest.mouseClick(self.window.mesh_button,Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(self.window.canvas.show_mesh)
        self.assertTrue(self.window.mesh_toggle.isChecked())
        preview=self.window.canvas.mesh
        self.assertEqual(len(preview.panel_start),384)
        actual=self.window.document.geometry.make_solver().backend.panels
        np.testing.assert_array_equal(preview.panel_start,actual.panel_start)
        np.testing.assert_array_equal(preview.panel_end,actual.panel_end)
        self.assertIn('384 panels',self.window.mesh_info.text())
        self.assertEqual(self.window.mesh_counts["panels"],384)
        self.assertIn('Panels: 384',self.window.mesh_summary.text())
        self.mock_warning.assert_not_called()

    def test_mesh_only_toggle_really_draws_panel_edges(self):
        self.window.mesh_view.setCurrentText('Mesh only')
        self.window.nodes_toggle.setChecked(False)
        self.app.processEvents()
        shown=self.window.canvas.grab().toImage()
        color_count=0
        for x in range(shown.width()):
            for y in range(shown.height()):
                color=shown.pixelColor(x,y)
                if color.red()>150 and 35<color.green()<120 and color.blue()<40:
                    color_count+=1
        self.assertGreater(color_count,100)
        self.window.mesh_toggle.setChecked(False)
        self.app.processEvents()
        hidden=self.window.canvas.grab().toImage()
        self.assertFalse(shown==hidden)
        self.assertTrue(self.window.canvas.mesh_only)
        self.assertIsNone(self.window.canvas.mesh)
        self.assertGreater(self.window.mesh_counts["panels"],0)
        self.assertIn('Displayed meshes: 0',self.window.mesh_summary.text())

    def test_all_interface_meshes_selection_and_quadrature_toggle(self):
        self.window.document.add_interface({'type':'circle','radius':35})
        self.window.refresh()
        self.assertEqual(len(self.window.canvas.meshes),2)
        self.window.select_boundary(1)
        self.window.mesh_scope.setCurrentText('Selected boundary')
        self.window.quadrature_toggle.setChecked(True)
        self.window.nodes_toggle.setChecked(True)
        self.app.processEvents()
        self.assertEqual(len(self.window.canvas.meshes),1)
        self.assertEqual(self.window.mesh_counts["meshes"],2)
        self.assertEqual(self.window.mesh_counts["displayed_meshes"],1)
        self.assertEqual(self.window.canvas.meshes[0][0],1)
        self.assertTrue(self.window.canvas.show_quadrature)
        expected=self.window.document.geometry.to_panels(self.window.document.geometry.boundaries[1][0])
        np.testing.assert_array_equal(self.window.canvas.mesh.points,expected.points)
        self.assertEqual(self.window.canvas.mesh.size,4*len(expected.panel_start))
        self.mock_warning.assert_not_called()

    def test_mesh_regenerates_after_units_and_undo(self):
        self.window.minimum.setValue(384)
        self.window.apply_mesh()
        points=self.window.canvas.mesh.panel_start.copy()
        self.window.units.setCurrentText('um')
        np.testing.assert_allclose(self.window.canvas.mesh.panel_start,points,atol=1e-17)
        self.window.undo()
        np.testing.assert_allclose(self.window.canvas.mesh.panel_start,points,atol=1e-17)
        self.window.undo()
        self.assertLess(len(self.window.canvas.mesh.panel_start),384)

    def test_mesh_controls_and_nodes_are_visible_on_open(self):
        canvas = self.window.canvas
        self.assertTrue(self.window.nodes_toggle.isChecked())
        self.assertTrue(canvas.show_nodes)
        self.assertTrue(canvas.show_panel_detail)
        for control in (self.window.mesh_toggle, self.window.nodes_toggle,
                        self.window.mesh_button, self.window.inspect_number):
            self.assertFalse(control.visibleRegion().isEmpty())
        self.assertLess(self.window.nodes_toggle.y()+self.window.nodes_toggle.height(),canvas.y())
        # The full-view fit leaves the magnified detail beside the pipe.
        rightmost = canvas.pixel((30,0)).x()
        self.assertLess(rightmost,canvas.width()-292)
        with_detail = canvas.grab().toImage()
        self.window.detail_toggle.setChecked(False)
        self.app.processEvents()
        without_detail = canvas.grab().toImage()
        self.assertFalse(with_detail == without_detail)

    def test_click_real_panel_inspects_and_zooms_its_endpoints(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        canvas = self.window.canvas
        mesh = canvas.mesh
        panel = len(mesh.panel_start)//2
        unit = self.window.document.geometry.unit_scale
        start,end = mesh.panel_start[panel]/unit,mesh.panel_end[panel]/unit
        center = (start+end)/2
        self.window.select_tool('inspect')
        QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=canvas.pixel(center).toPoint())
        self.app.processEvents()
        self.assertEqual(self.window.inspect_number.value(),panel+1)
        self.assertEqual(canvas.detail_panel,panel)
        self.assertIs(canvas.detail_mesh,mesh)
        self.assertIn(f'panel {panel+1}',self.window.panel_info.text())
        before = canvas.scale
        QTest.mouseClick(self.window.zoom_panel_button,Qt.MouseButton.LeftButton)
        self.app.processEvents()
        np.testing.assert_allclose(canvas.center,center,atol=1e-14)
        self.assertGreater(canvas.scale,before)
        self.mock_warning.assert_not_called()

    def test_inspection_selects_generated_roughness_interface(self):
        self.window.minimum.setValue(32)
        self.window.panel_length.setValue(100)
        self.window.apply_mesh()
        self.window.document.set_physics(1.,{'model':'conductivity_gradient',
            'rms_height':.0003,'slices':8,'extent_sigma':5.})
        self.window.refresh()
        actual = self.window.document.geometry.material_panels()
        self.window.inspect_boundary.setCurrentIndex(8)
        self.window.mesh_scope.setCurrentText('Selected boundary')
        self.window.inspect_number.setValue(7)
        self.app.processEvents()
        self.assertEqual(len(self.window.canvas.meshes),1)
        self.assertEqual(self.window.canvas.meshes[0][0],8)
        self.assertEqual(self.window.canvas.detail_panel,6)
        np.testing.assert_array_equal(self.window.canvas.detail_mesh.panel_start,actual[8][1].panel_start)
        self.assertIn(actual[8][0],self.window.panel_info.text())
        self.assertEqual(self.window.mesh_counts['meshes'],9)
        self.mock_warning.assert_not_called()


if __name__=="__main__":
    unittest.main()
