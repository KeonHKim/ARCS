"""Geometry invariants important for CAD/drawing accuracy and material intent."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from arcs import GeometryDocument, load_geometry
from arcs.geometry_editor import parse_coordinates, boolean_shape


class GeometryEditorTests(unittest.TestCase):
    def test_independent_mesh_units_defaults_reset_delete_undo_redo(self):
        document=GeometryDocument()
        document.set_mesh({'minimum_panels':64,'max_panel_length':100,'curve_tolerance':100})
        document.add_interface({'type':'circle','radius':35},name='coat_outer',exterior='Annealed Cu')
        document.set_mesh({'minimum_panels':128,'max_panel_length':5},boundary=0)
        document.set_mesh({'minimum_panels':192,'curve_tolerance':.1},boundary='coat_outer')
        before=document.data
        meshes=document.geometry.material_panels()
        document.set_units('um')
        self.assertAlmostEqual(document.data['boundary_meshes']['aperture']['max_panel_length'],5000.)
        self.assertAlmostEqual(document.data['boundary_meshes']['coat_outer']['curve_tolerance'],100.)
        for (_,a),(_,b) in zip(meshes,document.geometry.material_panels()):
            np.testing.assert_allclose(a.panel_start,b.panel_start,atol=2e-17)
        document.undo();self.assertEqual(document.data,before)
        document.set_mesh({'integration_order':6})
        self.assertEqual(document.geometry.mesh_settings('coat_outer').integration_order,6)
        self.assertEqual(document.geometry.mesh_settings().minimum_panels,128)
        document.reset_boundary_mesh('coat_outer')
        self.assertNotIn('coat_outer',document.geometry.boundary_meshes)
        document.undo();self.assertIn('coat_outer',document.geometry.boundary_meshes)
        document.redo();self.assertNotIn('coat_outer',document.geometry.boundary_meshes)
        document.set_mesh({'minimum_panels':192},boundary='coat_outer')
        document.delete_interface(1)
        self.assertNotIn('coat_outer',document.geometry.boundary_meshes)
        document.undo();self.assertIn('coat_outer',document.geometry.boundary_meshes)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'case.json';document.save(path)
            self.assertEqual(GeometryDocument(path).data,document.data)

    def test_invalid_mesh_edit_preserves_geometry_and_history(self):
        document=GeometryDocument()
        document.set_mesh({'minimum_panels':128,'max_panel_length':100,'curve_tolerance':100})
        document.add_interface({'type':'circle','radius':30.00015},name='coat_outer',exterior='Annealed Cu')
        before=document.data;history=len(document._undo)
        with self.assertRaisesRegex(ValueError,'cross/touch'):
            document.set_mesh({'minimum_panels':256},boundary=0)
        self.assertEqual(document.data,before)
        self.assertEqual(len(document._undo),history)
        with self.assertRaisesRegex(ValueError,'existing physical'):
            document.set_mesh({'minimum_panels':256},boundary=99)
        self.assertEqual(document.data,before)

    def test_units_preserve_physical_dimensions_beam_and_mesh(self):
        document = GeometryDocument()
        document.set_beam(2,3)
        initial = document.geometry
        document.set_units("um")
        self.assertAlmostEqual(document.shape(0)["radius"],30000,places=8)
        np.testing.assert_allclose(document.data["beam"],[2000,3000],rtol=2e-15)
        self.assertAlmostEqual(document.data["mesh"]["max_panel_length"],1000,places=9)
        np.testing.assert_allclose(document.geometry.boundary().area,initial.boundary().area,rtol=2e-15)
        np.testing.assert_allclose(document.geometry.beam_position,initial.beam_position)
        self.assertEqual(document.geometry.mesh.max_panel_length,initial.mesh.max_panel_length)
        document.set_units("m")
        document.set_units("mm")
        self.assertAlmostEqual(document.shape(0)["radius"],30,places=12)

    def test_ordered_coordinates_and_sharp_reentrant_vertex(self):
        shape = parse_coordinates("x,y\n-4,-3\n4,-3\n4,0\n0,0\n0,3\n-4,3 # last")
        document = GeometryDocument()
        document.set_shape(0,shape)
        self.assertAlmostEqual(document.geometry.boundary().area/1e-6,36)
        panels = document.geometry.to_panels(mesh={"max_panel_length":1})
        for point in shape["points"]:
            self.assertTrue(np.any(np.linalg.norm(panels.panel_start-np.asarray(point)*.001,axis=1)<1e-14))
        with self.assertRaisesRegex(ValueError,"perimeter|pair"):
            parse_coordinates("0,0,1\n1,1\n2,2")

    @unittest.skipUnless(importlib.util.find_spec("shapely"),"Optional Shapely not installed")
    def test_crop_area_vertices_and_undo_redo(self):
        document = GeometryDocument()
        document.set_shape(0,{"type":"rectangle","width":10,"height":8})
        before = document.data
        document.set_shape(0,{"type":"rectangle","width":6,"height":20,"center":[2,0]},operation="crop")
        self.assertAlmostEqual(document.geometry.boundary().area/1e-6,48)
        corners = sorted(map(tuple,document.shape(0)["points"]))
        self.assertEqual(corners,[(-1.,-4.),(-1.,4.),(5.,-4.),(5.,4.)])
        document.undo()
        self.assertEqual(document.data,before)
        document.redo()
        self.assertAlmostEqual(document.geometry.boundary().area/1e-6,48)

    @unittest.skipUnless(importlib.util.find_spec("shapely"),"Optional Shapely not installed")
    def test_union_and_subtraction_have_independent_known_areas(self):
        square = {"type":"rectangle","width":10,"height":10}
        overlap = {"type":"rectangle","width":10,"height":10,"center":[5,5]}
        document = GeometryDocument()
        document.set_shape(0,square)
        document.set_shape(0,overlap,operation="union")
        self.assertAlmostEqual(document.geometry.boundary().area/1e-6,175)
        document.set_shape(0,square)
        document.set_shape(0,overlap,operation="subtract")
        self.assertAlmostEqual(document.geometry.boundary().area/1e-6,75)
        self.assertIn((0.,0.),map(tuple,document.shape(0)["points"]))

    @unittest.skipUnless(importlib.util.find_spec("shapely"),"Optional Shapely not installed")
    def test_invalid_boolean_and_invalid_nesting_are_transactional(self):
        document = GeometryDocument()
        initial = document.data
        with self.assertRaisesRegex(ValueError,"island/hole"):
            document.set_shape(0,{"type":"circle","radius":5},operation="subtract")
        self.assertEqual(document.data,initial)
        self.assertFalse(document.can_undo)
        with self.assertRaisesRegex(ValueError,"disconnected"):
            document.set_shape(0,{"type":"circle","radius":5,"center":[100,0]},operation="union")
        self.assertEqual(document.data,initial)
        with self.assertRaisesRegex(ValueError,"enclose"):
            document.add_interface({"type":"circle","radius":20})
        self.assertEqual(document.data,initial)

    def test_geometry_crop_chord_error_is_explicit_and_bounded(self):
        from arcs.geometry_editor import outline_points
        points=outline_points({"type":"circle","radius":30},tolerance=.002)
        chord_midpoints=(points+np.roll(points,-1,axis=0))/2
        self.assertLessEqual(np.max(30-np.linalg.norm(chord_midpoints,axis=1)),.002)
        with self.assertRaisesRegex(ValueError,"positive"):
            outline_points({"type":"circle","radius":30},tolerance=0)

    def test_finite_wall_air_round_trip_and_solver_guard(self):
        document=GeometryDocument()
        document.add_interface({"type":"circle","radius":32},name="wall_outer",exterior="air")
        self.assertEqual([r["material"] for r in document.geometry.regions],["vacuum","Annealed Cu","air"])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"case.json"
            document.save(path)
            restored=GeometryDocument(path)
            self.assertEqual(restored.data,document.data)
        self.assertTrue(any("not implemented" in text for text in document.issues()))
        with self.assertRaisesRegex(NotImplementedError,"beta<1"):
            document.geometry.make_solver("maxwell")
        document.delete_interface(1)
        self.assertEqual([r["material"] for r in document.geometry.regions],["vacuum","air"])
        with self.assertRaisesRegex(NotImplementedError,"beta<1"):
            document.geometry.make_solver("maxwell")

    def test_vacuum_is_required_and_properties_checked(self):
        document=GeometryDocument()
        initial=document.data
        with self.assertRaisesRegex(ValueError,"always vacuum"):
            document.set_region(0,"Cu")
        bad=copy.deepcopy(initial)
        bad["regions"][0]["material"]="Cu"
        with self.assertRaisesRegex(ValueError,"innermost"):
            load_geometry(bad)
        bad=copy.deepcopy(initial)
        bad["materials"]["vacuum"]["conductivity"]=1
        with self.assertRaisesRegex(ValueError,"Vacuum must"):
            load_geometry(bad)
        with self.assertRaisesRegex(ValueError,"nonnegative"):
            document.set_region(1,"bad",{"conductivity":-1})
        self.assertEqual(document.data,initial)

    def test_defined_exterior_material_reaches_physics(self):
        document=GeometryDocument()
        document.set_region(1,"custom_copper",{"conductivity":5.8e7})
        mesh={"minimum_panels":64,"max_panel_length":10}
        from arcs import Metal
        solver=document.geometry.make_solver("maxwell",mesh=mesh)
        implicit=solver.longitudinal([10,100])
        explicit=solver.longitudinal([10,100],Metal(5.8e7))
        np.testing.assert_allclose(implicit,explicit,rtol=1e-14)
        self.assertEqual(solver.default_wall.name,"custom_copper")

if __name__=="__main__":
    unittest.main()
