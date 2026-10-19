"""Independent geometry invariants and the complete file-to-solver workflow."""
from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest
import numpy as np
from arcs import load_geometry, Metal
from arcs.geometry_input import main


def case(shape, **extra):
    return load_geometry({"units": "mm", "aperture": shape, **extra})


class GeometryInputChecks(unittest.TestCase):
    def test_geometry_gauss_samples_integrate_without_changing_maxwell_fields(self):
        geometry = case({'type':'rectangle','width':60,'height':40},
            mesh={'max_panel_length':100,'minimum_panels':32,'corner_grading':1})
        q4 = geometry.to_panels(mesh={'integration_order':4})
        q8 = geometry.to_panels(mesh={'integration_order':8})
        np.testing.assert_array_equal(q4.panel_start,q8.panel_start)
        self.assertEqual(q8.size,2*q4.size)
        vector=q4.panel_end[0]-q4.panel_start[0]; length=np.linalg.norm(vector)
        selected=q4.panel_index==0
        fraction=(q4.points[selected]-q4.panel_start[0])@vector/length**2
        # Four Gauss samples integrate a degree-six polynomial exactly.
        self.assertAlmostEqual(np.sum(q4.weights[selected]*fraction**6)/length,1/7,places=14)
        response4=geometry.make_solver(mesh={'integration_order':4}).transverse([1e6],Metal(5.8e7))
        response8=geometry.make_solver(mesh={'integration_order':8}).transverse([1e6],Metal(5.8e7))
        for field in ('longitudinal','dipolar','quadrupolar','constant'):
            np.testing.assert_array_equal(getattr(response4,field),getattr(response8,field))

    def test_independent_interface_meshes_roundtrip_and_solver(self):
        geometry = case({"type":"circle","radius":30},
            interfaces=[{"name":"coat_outer","shape":{"type":"circle","radius":35}}],
            regions=[{"name":"beam","material":"vacuum"},{"name":"coat","material":"NEG"},
                     {"name":"bulk","material":"Cu"}],
            materials={"NEG":{"conductivity":2e5},"Cu":{"conductivity":5.8e7}},
            mesh={"minimum_panels":64,"max_panel_length":100,"curve_tolerance":100},
            boundary_meshes={"aperture":{"minimum_panels":128,"integration_order":3},
                             "coat_outer":{"minimum_panels":192,"integration_order":5}})
        self.assertEqual(len(geometry.to_panels().panel_start),128)
        self.assertEqual(len(geometry.to_panels('coat_outer').panel_start),192)
        self.assertEqual(geometry.to_panels('coat_outer').size,192*5)
        overridden = geometry.to_panels('coat_outer',mesh={'minimum_panels':256})
        self.assertEqual(len(overridden.panel_start),256)
        self.assertEqual(overridden.size,256*5)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'independent.json'
            geometry.save(path)
            restored=load_geometry(path)
            self.assertEqual(restored.to_dict(),geometry.to_dict())
            solver=restored.make_solver().backend
            for displayed, actual in zip(restored.material_panels(),(solver.panels,*solver.interfaces)):
                np.testing.assert_array_equal(displayed[1].panel_start,actual.panel_start)
                np.testing.assert_array_equal(displayed[1].points,actual.points)
        self.assertTrue(geometry.report()['chord_mesh_interfaces_strictly_nested'])

    def test_boundary_mesh_settings_validation_and_inheritance(self):
        shape={"type":"circle","radius":30}
        for overrides in ({"missing":{"minimum_panels":128}},
                          {"aperture":{"minimum_panels":0}},
                          {"aperture":{"minimum_panles":128}},
                          {"aperture":None}):
            with self.assertRaises(ValueError):
                case(shape,boundary_meshes=overrides)
        with self.assertRaisesRegex(ValueError,'boundary_meshes'):
            case(shape,boundary_meshes=[])
        geometry=case(shape,mesh={'max_panel_length':10,'integration_order':6},
                      boundary_meshes={'aperture':{'minimum_panels':128}})
        settings=geometry.mesh_settings()
        self.assertEqual(settings.minimum_panels,128)
        self.assertEqual(settings.integration_order,6)
        self.assertEqual(settings.max_panel_length,.01)
        data=geometry.to_dict();data['mesh']['integration_order']=7
        self.assertEqual(load_geometry(data).mesh_settings().integration_order,7)

    def test_crossing_independent_chords_rejected_before_solving(self):
        geometry=case({"type":"circle","radius":30},
            interfaces=[{"name":"coat_outer","shape":{"type":"circle","radius":30.00015}}],
            regions=[{"name":"beam","material":"vacuum"},{"name":"coat","material":"NEG"},
                     {"name":"bulk","material":"Cu"}],
            materials={"NEG":{"conductivity":2e5},"Cu":{"conductivity":5.8e7}},
            mesh={"minimum_panels":128,"max_panel_length":100,"curve_tolerance":100},
            boundary_meshes={"aperture":{"minimum_panels":256}})
        self.assertFalse(geometry.report()['chord_mesh_interfaces_strictly_nested'])
        with self.assertRaisesRegex(ValueError,'cross/touch'):
            geometry.make_solver()

    def test_explicit_units_and_equivalent_geometry(self):
        millimetres = case({"type": "circle", "radius": 30})
        metres = load_geometry({"units": "m", "aperture": {"type": "circle", "radius": .03}})
        micrometres = load_geometry({"units": "um", "aperture": {"type": "circle", "radius": 30000}})
        np.testing.assert_array_equal(millimetres.to_panels().points, metres.to_panels().points)
        np.testing.assert_allclose(millimetres.to_panels().points, micrometres.to_panels().points, atol=1e-17)
        with self.assertRaisesRegex(ValueError, "units explicitly"):
            load_geometry({"aperture": {"type": "circle", "radius": 30}})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            load_geometry(millimetres.to_dict(), units="m")

    def test_clockwise_csv_closure_and_portable_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"outline.csv").write_text("x,y\n-30,-15\n-30,15\n30,15\n30,-15\n-30,-15\n")
            (root/"pipe.json").write_text(json.dumps({"units": "mm", "aperture": {"type": "csv", "file": "outline.csv"}}))
            geometry = load_geometry(root/"pipe.json")
            self.assertTrue(geometry.boundary().orientation_reversed)
            self.assertAlmostEqual(geometry.boundary().area, .06*.03, places=15)
            geometry.save(root/"portable.json")
            (root/"outline.csv").unlink()
            portable = load_geometry(root/"portable.json")
            np.testing.assert_array_equal(geometry.to_panels().points, portable.to_panels().points)
            with self.assertRaisesRegex(ValueError, "no implied units"):
                load_geometry(root/"absent.csv")

    def test_reentrant_polygon_preserves_vertices_and_area(self):
        vertices = [[-40,-30],[40,-30],[40,0],[0,0],[0,30],[-40,30]]
        geometry = case({"type": "polygon", "points": vertices}, beam=[-20,-10])
        coarse = geometry.to_panels(mesh={"max_panel_length": 10})
        fine = geometry.to_panels(mesh={"max_panel_length": 2})
        for mesh in [coarse, fine]:
            for vertex in np.asarray(vertices)*1e-3:
                self.assertLess(np.min(np.linalg.norm(mesh.panel_start-vertex, axis=1)), 1e-16)
            area = .5*np.sum(mesh.panel_start[:,0]*mesh.panel_end[:,1]-mesh.panel_start[:,1]*mesh.panel_end[:,0])
            self.assertAlmostEqual(area, .0036, places=15)
            np.testing.assert_allclose(mesh.weights @ mesh.normal, [0,0], atol=1e-15)
        self.assertGreater(len(fine.panel_start), len(coarse.panel_start))
        self.assertFalse(geometry.boundary().contains([.02,.015]))
        self.assertFalse(geometry.boundary().contains([0,0]))

    def test_native_circle_chord_error_and_panel_length(self):
        geometry = case({"type": "circle", "radius": 30},
                        mesh={"max_panel_length": 3,"curve_tolerance": .005,"minimum_panels": 16})
        mesh = geometry.to_panels()
        # For circle chords, radial deficit at their midpoint is exact.
        sagitta = .03-np.linalg.norm((mesh.panel_start+mesh.panel_end)/2, axis=1)
        self.assertLessEqual(sagitta.max(), .005e-3*(1+1e-12))
        self.assertLessEqual(np.linalg.norm(mesh.panel_end-mesh.panel_start, axis=1).max(), .003)
        self.assertAlmostEqual(geometry.boundary().area, np.pi*.03**2, places=15)

    def test_rotated_ellipse_bounds(self):
        angle = np.deg2rad(27.)
        geometry = case({"type":"ellipse","semi_axes":[60,30],"center":[5,-4],"rotation_deg":27})
        low, high = geometry.boundary().bounds
        half = np.array([np.hypot(.06*np.cos(angle),.03*np.sin(angle)),
                         np.hypot(.06*np.sin(angle),.03*np.cos(angle))])
        np.testing.assert_allclose(low, [.005,-.004]-half, atol=2e-17)
        np.testing.assert_allclose(high, [.005,-.004]+half, atol=2e-17)
        self.assertAlmostEqual(geometry.boundary().area, np.pi*.06*.03, places=15)

    def test_line_arc_path_area_perimeter_and_exact_containment(self):
        shape={"type":"path","start":[-30,-20],"segments":[
            {"type":"line","to":[30,-20]},
            {"type":"arc","center":[30,0],"to":[30,20],"direction":"ccw"},
            {"type":"line","to":[-30,20]},
            {"type":"arc","center":[-30,0],"to":[-30,-20],"direction":"ccw"}]}
        geometry=case(shape)
        self.assertAlmostEqual(geometry.boundary().area,.06*.04+np.pi*.02**2,places=15)
        self.assertAlmostEqual(geometry.boundary().perimeter,2*.06+2*np.pi*.02,places=15)
        points=np.random.default_rng(18).uniform([-.06,-.03],[.06,.03],size=(200,2))
        expected=np.maximum(abs(points[:,0])-.03,0)**2+points[:,1]**2<.02**2
        actual=np.array([geometry.boundary().contains(point) for point in points])
        np.testing.assert_array_equal(actual,expected)
        for point in [[0,.02],[.05,0],[-.05,0]]:
            self.assertFalse(geometry.boundary().contains(point))

    def test_thin_concentric_and_rotated_ellipse_interfaces(self):
        for inner,outer in [({"type":"circle","radius":30},{"type":"circle","radius":30.00015}),
            ({"type":"ellipse","semi_axes":[60,30],"rotation_deg":31},
             {"type":"ellipse","semi_axes":[60.00015,30.0003],"rotation_deg":31})]:
            geometry=case(inner,interfaces=[{"name":"coating_outer","shape":outer}])
            self.assertEqual(len(geometry.boundaries),2)
            self.assertTrue(geometry.report()["native_interfaces_strictly_nested"])
            with self.assertRaisesRegex(ValueError,"Define material"):
                geometry.make_solver("maxwell")
            self.assertEqual(geometry.make_solver("pec").backend_name,"pec")
        for radius in [30.,29.9]:
            with self.assertRaisesRegex(ValueError,"strictly enclose"):
                case({"type":"circle","radius":30},interfaces=[{"name":"outer","shape":{"type":"circle","radius":radius}}])

    def test_polygon_coating_is_independent_and_nonuniform(self):
        inner={"type":"rectangle","width":60,"height":30}
        outer={"type":"polygon","points":[[-34,-19],[32,-19],[32,18],[-34,18]]}
        geometry=case(inner,interfaces=[{"name":"coating_outer","shape":outer}],
            regions=[{"name":"beam","material":"vacuum"},{"name":"coat","material":"NEG"},{"name":"bulk","material":"Cu"}])
        self.assertTrue(geometry.report()["chord_mesh_interfaces_strictly_nested"])
        low,high=geometry.boundary("coating_outer").bounds
        np.testing.assert_allclose(low,[-.034,-.019])
        np.testing.assert_allclose(high,[.032,.018])
        self.assertNotEqual(abs(low[0])-.03,high[0]-.03)
        with self.assertRaisesRegex(ValueError,"one region"):
            case(inner,interfaces=[{"name":"outer","shape":outer}],regions=[{"name":"beam","material":"vacuum"}])

    def test_native_vs_chord_nesting_are_separate(self):
        # Both circles are valid/nested, but different coarse angular grids
        # produce crossing inscribed polygons. Such a mesh cannot be exported.
        geometry=case({"type":"circle","radius":30},
            interfaces=[{"name":"outer","shape":{"type":"circle","radius":30.2}}],
            mesh={"max_panel_length":2*np.pi*30/10,"curve_tolerance":100,"minimum_panels":4})
        self.assertFalse(geometry.report()["chord_mesh_interfaces_strictly_nested"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"chord meshes"):
                geometry.export_panels(Path(directory)/"bad.csv")

    def test_invalid_units_typos_open_paths_crossings_and_bad_arcs(self):
        inputs=[
            {"type":"polygon","points":[[-1,-1],[1,1],[-1,1],[1,-1]]},
            {"type":"path","start":[0,0],"segments":[{"type":"line","to":[1,0]}]},
            {"type":"path","start":[1,0],"segments":[{"type":"arc","center":[0,0],"to":[0,2]}]},
            {"type":"circle","raduis":30},
            {"type":"rounded_rectangle","width":60,"height":30,"corner_radius":20}]
        for shape in inputs:
            with self.assertRaises(ValueError):
                case(shape)
        with self.assertRaisesRegex(ValueError,"Did you mean 'radius'"):
            case({"type":"circle","raduis":30})
        with self.assertRaisesRegex(ValueError, "Choose backend"):
            case({"type":"rectangle","width":60,"height":30}).make_solver("removed_backend")

    def test_beam_position_reaches_the_solver(self):
        mesh={"max_panel_length":5,"curve_tolerance":.05,"minimum_panels":64}
        original=case({"type":"circle","radius":30},mesh=mesh).make_solver()
        translated=case({"type":"circle","radius":30,"center":[5,-7]},beam=[5,-7],mesh=mesh).make_solver()
        z1=original.longitudinal(10.,Metal(5.8e7))
        z2=translated.longitudinal(10.,Metal(5.8e7))
        self.assertLess(abs(z1/z2-1),5e-12)
        with self.assertRaisesRegex(ValueError,"beam is outside"):
            case({"type":"circle","radius":30,"center":[100,0]}).make_solver()

    def test_command_line_preview_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data={"units":"mm","aperture":{"type":"circle","radius":30}}
            (root/"input.json").write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                status=main([str(root/"input.json"),"--preview",str(root/"mesh.png"),
                             "--export-panels",str(root/"panels.csv"),"--summary",str(root/"report.json")])
            self.assertEqual(status,0)
            self.assertIn("mm input -> metres",stdout.getvalue())
            self.assertGreater((root/"mesh.png").stat().st_size,1000)
            from PIL import Image
            Image.open(root/"mesh.png").verify()
            self.assertTrue((root/"panels.csv").read_text().startswith("boundary,panel,start_x_m"))
            self.assertEqual(json.loads((root/"report.json").read_text())["boundary_count"],1)


if __name__ == "__main__":
    unittest.main()
