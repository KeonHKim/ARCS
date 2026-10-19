"""File conversion tests: units, native curves, closure and unsupported topology."""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from arcs import load_geometry
from arcs.geometry_editor import GeometryDocument
from arcs.geometry_io import import_geometry, export_dxf


@unittest.skipUnless(importlib.util.find_spec("ezdxf"),"Optional ezdxf not installed")
class GeometryIOTests(unittest.TestCase):
    def setUp(self):
        import ezdxf
        self.ezdxf=ezdxf
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)/"drawing.dxf"

    def tearDown(self):
        self.directory.cleanup()

    def test_native_thin_circle_layers_dxf_round_trip(self):
        document=GeometryDocument()
        document.add_interface({"type":"circle","radius":30.00015},name="coating_outer",exterior="Annealed Cu")
        document.set_region(1,"NEG",{"conductivity":2e5})
        export_dxf(document.geometry,self.path)
        result=import_geometry(self.path)
        self.assertEqual([shape.kind for _,shape in result.geometry.boundaries],["circle","circle"])
        np.testing.assert_allclose([shape.area for _,shape in result.geometry.boundaries],
                                  [shape.area for _,shape in document.geometry.boundaries],rtol=2e-15)
        self.assertEqual(result.geometry.units,"mm")
        self.assertEqual(result.geometry.regions[0]["material"],"vacuum")
        self.assertEqual(result.geometry.regions[1]["material"],"unspecified")
        # Same user-facing loader automatically dispatches .dxf.
        self.assertEqual(len(load_geometry(self.path).boundaries),2)

    def test_physical_unit_conversion_and_conflict(self):
        drawing=self.ezdxf.new()
        drawing.header["$INSUNITS"]=1 # inches
        drawing.modelspace().add_circle((0,0),1)
        drawing.saveas(self.path)
        case=import_geometry(self.path).geometry
        self.assertAlmostEqual(case.boundary().definition["radius"],25.4)
        self.assertAlmostEqual(case.boundary().area,math.pi*.0254**2)
        with self.assertRaisesRegex(ValueError,"conflict"):
            import_geometry(self.path,units="mm")
        drawing.header["$INSUNITS"]=0
        drawing.saveas(self.path)
        with self.assertRaisesRegex(ValueError,"no declared units"):
            import_geometry(self.path)
        self.assertEqual(import_geometry(self.path,units="mm").geometry.boundary().definition["radius"],1)

    def test_unordered_lines_arcs_join_into_native_stadium(self):
        drawing=self.ezdxf.new()
        drawing.header["$INSUNITS"]=4
        m=drawing.modelspace()
        m.add_arc((30,0),20,270,90)
        m.add_line((-30,-20),(30,-20))
        m.add_arc((-30,0),20,90,270)
        m.add_line((30,20),(-30,20))
        drawing.saveas(self.path)
        case=import_geometry(self.path).geometry
        self.assertEqual(case.boundary().kind,"path")
        self.assertEqual(sorted(curve.kind for curve in case.boundary().curves),["arc","arc","line","line"])
        self.assertAlmostEqual(case.boundary().area/1e-6,60*40+math.pi*20**2,places=8)
        export_dxf(case,self.path)
        self.assertAlmostEqual(import_geometry(self.path).geometry.boundary().area,case.boundary().area,places=14)

    def test_ellipse_native_axes_rotation_and_export(self):
        document=GeometryDocument()
        document.set_shape(0,{"type":"ellipse","semi_axes":[20,40],"center":[2,3],"rotation_deg":27})
        export_dxf(document.geometry,self.path)
        case=import_geometry(self.path).geometry
        self.assertEqual(case.boundary().kind,"ellipse")
        np.testing.assert_allclose(case.boundary().bounds,document.geometry.boundary().bounds,atol=2e-16)
        self.assertAlmostEqual(case.boundary().area,math.pi*.02*.04,places=14)

    def test_open_branches_crossing_outlines_and_3d_rejected(self):
        drawing=self.ezdxf.new()
        drawing.header["$INSUNITS"]=4
        drawing.modelspace().add_line((0,0),(10,0))
        drawing.saveas(self.path)
        with self.assertRaisesRegex(ValueError,"open"):
            import_geometry(self.path)
        drawing=self.ezdxf.new()
        drawing.header["$INSUNITS"]=4
        drawing.modelspace().add_circle((0,0),30)
        drawing.modelspace().add_circle((20,0),30)
        drawing.saveas(self.path)
        with self.assertRaisesRegex(ValueError,"enclose"):
            import_geometry(self.path)
        drawing=self.ezdxf.new()
        drawing.header["$INSUNITS"]=4
        drawing.modelspace().add_circle((0,0,1),30)
        drawing.saveas(self.path)
        with self.assertRaisesRegex(ValueError,"z=0"):
            import_geometry(self.path)

    def test_annotations_ignored_unknown_geometry_not_silently_omitted(self):
        drawing=self.ezdxf.new()
        drawing.header["$INSUNITS"]=4
        drawing.modelspace().add_circle((0,0),30)
        drawing.modelspace().add_text("Cu",dxfattribs={"height":3})
        drawing.saveas(self.path)
        self.assertTrue(any("annotations" in note for note in import_geometry(self.path).notes))
        drawing.blocks.new("some_geometry")
        drawing.modelspace().add_blockref("some_geometry",(0,0))
        drawing.saveas(self.path)
        with self.assertRaisesRegex(ValueError,"Unsupported DXF entity INSERT"):
            import_geometry(self.path)
        self.path.write_text("This is not a DXF drawing.\n")
        with self.assertRaisesRegex(ValueError,"Could not read DXF"):
            import_geometry(self.path)

    @unittest.skipUnless(importlib.util.find_spec("shapely"),"Optional Shapely not installed")
    def test_round_trip_boolean_polygon_preserves_reentrant_corners(self):
        document=GeometryDocument()
        document.set_shape(0,{"type":"rectangle","width":10,"height":10})
        document.set_shape(0,{"type":"rectangle","width":10,"height":10,"center":[5,5]},operation="subtract")
        export_dxf(document.geometry,self.path)
        case=import_geometry(self.path).geometry
        self.assertAlmostEqual(case.boundary().area/1e-6,75)
        vertices=np.array([curve.point(0.) for curve in case.boundary().curves])
        self.assertTrue(np.any(np.linalg.norm(vertices,axis=1)<1e-14))

    def test_csv_explicit_units_and_no_raster_projection(self):
        path=Path(self.directory.name)/"outline.csv"
        path.write_text("x,y\n-5,-3\n5,-3\n5,3\n-5,3\n")
        with self.assertRaisesRegex(ValueError,"no implied units"):
            import_geometry(path)
        result=import_geometry(path,units="mm")
        self.assertAlmostEqual(result.geometry.boundary().area/1e-6,60)
        result=import_geometry(path,units="mm",exterior="Cu")
        self.assertEqual(result.geometry.regions[-1]["material"],"Cu")
        self.assertEqual(result.geometry.materials["Cu"]["conductivity"],5.8e7)
        with self.assertRaisesRegex(ValueError,"Raster"):
            import_geometry(path.with_suffix(".png"))


if __name__=="__main__":
    unittest.main()
