"""Independent layered modes, profile convergence and corner-preserving meshes."""
from pathlib import Path
import importlib.util
import sys
import unittest
import numpy as np
from arcs import MaxwellSolver, Metal, Wall, Layer, Roughness, polygon_panels, load_geometry, GeometryDocument
from arcs.reference import circular_exact, circular_layered_dipolar, _circular_vacuum_reaction
from arcs.roughness import gradient_layers
from arcs.layered import material_boundaries, offset_boundary
from arcs.pec_panels import PolygonPECSolver


def mesh(n=96):
    theta=np.arange(n)*2*np.pi/n
    return polygon_panels(.03*np.c_[np.cos(theta),np.sin(theta)],panels_per_edge=1)


def supplied_model():
    path=Path(__file__).resolve().parents[1]/"references/SurfaceImpedance/src/surface_impedance/models.py"
    spec=importlib.util.spec_from_file_location("supplied_surface_models",path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


class LayerAndRoughnessChecks(unittest.TestCase):
    def test_extended_tail_rejected_instead_of_unreliable_impedance(self):
        solver = MaxwellSolver(mesh(64), roughness=Roughness(.3e-6, slices=64, extent_sigma=10))
        for frequency in (10., 1e13):
            with self.assertRaisesRegex(NotImplementedError, "near-insulating tails"):
                solver.transverse(frequency, Metal(5.8e7))

    def test_thin_coatings_and_uniform_subdivision(self):
        f=np.array([10.,1e6,1e9,1e11]);metal=Metal(5.8e7)
        solver=MaxwellSolver(mesh())
        for wall in (Wall(metal,(Layer(metal,1e-7),)),
                     Wall(metal,(Layer(Metal(2e5),150e-9),)),
                     Wall(metal,(Layer(Metal(2e5),150e-9),Layer(Metal(1e6),.5e-6)))):
            actual=solver.longitudinal(f,wall)
            expected=circular_exact(f,.03,wall)
            self.assertLess(np.max(abs(actual/expected-1)),.0025)
        coated=Wall(metal,(Layer(Metal(2e5),150e-9),))
        tensor=solver.transverse([10.,1e6,1e9],coated)
        self.assertLess(np.max(abs(tensor.dipolar[:,0,0]/circular_layered_dipolar([10.,1e6,1e9],.03,coated)-1)),.004)
        slow=MaxwellSolver(mesh(),beta=.7)
        wall=Wall(metal,(Layer(Metal(2e5),150e-9),))
        self.assertLess(np.max(abs(slow.longitudinal([1e6,1e9],wall,component='resistive_wall')/
                                 circular_exact([1e6,1e9],.03,wall,.7,component='resistive_wall')-1)),.002)

    def test_gradient_against_independent_cylindrical_layers_and_supplied_ode(self):
        m=Metal(5.8e7);f=np.array([1e9,1e10])
        params={"sigma_metal":m.conductivity,"rq":.3e-6,"mu_r":1.,"xmin_factor":-5.,
                "domain_factor":15.,"step_size":.3e-6/100}
        surface=supplied_model().compute_surface_impedance("rough-single",f,params)
        reference=_circular_vacuum_reaction(f,.03,surface)
        errors=[]
        for slices in (8,16):
            # This supplied reference explicitly starts at -5 Rq. Keep the
            # historical mid-band comparison at that SAME truncation plane.
            rough=Roughness(.3e-6,slices=slices,extent_sigma=5.);solver=MaxwellSolver(mesh(64),roughness=rough)
            actual=solver.longitudinal(f,m)
            depth,media=gradient_layers(Wall(m),rough)
            radial=Wall(m,tuple(Layer(mm,d) for mm,d in zip(media[:-1],np.diff(depth))))
            self.assertLess(np.max(abs(actual/circular_exact(f,.03,radial)-1)),.0006)
            errors.append(abs(actual/reference-1))
        self.assertTrue(np.all(errors[1]<errors[0]))
        self.assertLess(np.max(errors[-1]),.004)

    def test_finite_beta_gradient_and_low_frequency_layers(self):
        metal=Metal(5.8e7);rough=Roughness(.3e-6,slices=8)
        depth,media=gradient_layers(Wall(metal),rough)
        radial=Wall(metal,tuple(Layer(m,d) for m,d in zip(media[:-1],np.diff(depth))))
        solver=MaxwellSolver(mesh(64),beta=.7,roughness=rough)
        actual=solver.transverse([10.,1e8],metal,component='resistive_wall')
        self.assertLess(np.max(abs(actual.longitudinal/circular_exact([10.,1e8],.03,radial,.7,component='resistive_wall')-1)),.005)
        self.assertLess(np.max(abs(actual.dipolar[:,0,0]/circular_layered_dipolar([10.,1e8],.03,radial,.7,component='resistive_wall')-1)),.006)

    def test_independent_noncircular_interfaces_and_rigid_motion(self):
        inner=np.array([[-.04,-.02],[.04,-.02],[.04,.02],[-.04,.02]])
        outer=np.array([[-.043,-.021],[.042,-.021],[.042,.023],[-.043,.023]])
        wall=Wall(Metal(5.8e7),(Layer(Metal(2e5),0.),))
        a=.37;rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        offset=np.array([.005,-.004]);responses=[]
        for transform in (False,True):
            vi=inner@rot.T+offset if transform else inner
            vo=outer@rot.T+offset if transform else outer
            solver=MaxwellSolver(polygon_panels(vi,panels_per_edge=12,grading_power=1.5),
                        beta=.7,interfaces=[polygon_panels(vo,panels_per_edge=16,grading_power=1.5)])
            responses.append(solver.transverse(1e7,wall,offset if transform else (0,0)))
        np.testing.assert_allclose(responses[1].longitudinal,responses[0].longitudinal,rtol=1e-8)
        for name in ("dipolar","quadrupolar"):
            np.testing.assert_allclose(getattr(responses[1],name),
                rot@getattr(responses[0],name)@rot.T,rtol=1e-7,atol=1e-8)

    def test_sharp_reentrant_gradient_geometry_and_units(self):
        vertices=np.array([[-.04,-.03],[.04,-.03],[.04,0],[0,0],[0,.03],[-.04,.03]])
        p=PolygonPECSolver(polygon_panels(vertices,panels_per_edge=4,grading_power=1.5))
        outer=offset_boundary(p,1e-6)
        # The inner reentrant vertex is exact; the outer miter follows both normals.
        self.assertTrue(np.any(np.linalg.norm(p.starts,axis=1)==0))
        self.assertTrue(np.any(np.linalg.norm(outer.starts-np.array([1e-6,1e-6]),axis=1)<1e-15))
        document=GeometryDocument();document.set_physics(.7,{"rms_height":.0003,"slices":8})
        height=document.geometry.roughness.rms_height
        document.set_units("um")
        self.assertAlmostEqual(document.data["roughness"]["rms_height"],.3,places=13)
        self.assertAlmostEqual(document.geometry.roughness.rms_height,height,places=20)
        solver=document.geometry.make_solver(mesh={"minimum_panels":32,"max_panel_length":10000}).backend
        boundaries,media=material_boundaries(solver,Wall(Metal(5.8e7)))
        self.assertEqual(len(boundaries),9)
        self.assertEqual(len(media),9)


if __name__ == "__main__":
    unittest.main()
