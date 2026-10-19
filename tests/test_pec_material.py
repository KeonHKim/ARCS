"""Exact ideal walls, PEC-backed coatings and portable material definitions."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from arcs import PEC, Metal, Wall, Layer, Roughness, MaxwellSolver, load_geometry, polygon_panels
from arcs.geometry_editor import GeometryDocument, default_geometry
from arcs.reference import (circular_exact, circular_layered_dipolar,
                            circular_indirect_space_charge, circular_indirect_dipolar,
                            circular_indirect_quadrupolar)


def circle(n=128, radius=.03):
    theta = 2*np.pi*np.arange(n)/n
    return polygon_panels(radius*np.c_[np.cos(theta), np.sin(theta)], panels_per_edge=1)


class PECMaterialChecks(unittest.TestCase):
    def test_ultrarelativistic_pec_is_exact_zero(self):
        solver = MaxwellSolver(circle(64))
        for wall in (PEC(), Wall(PEC()), Wall(PEC(), (Layer(Metal(2e5), 0.),))):
            parts = solver.transverse_components([10., 1e6, 1e13], wall)
            for response in parts.values():
                for field in ('longitudinal', 'dipolar', 'quadrupolar', 'constant'):
                    np.testing.assert_array_equal(getattr(response, field), 0)
        np.testing.assert_array_equal(MaxwellSolver(circle(64), interfaces=[]).longitudinal(10., PEC()), 0)

    def test_finite_beta_against_independent_circular_formulas(self):
        f = np.array([10., 1e4, 1e6, 1e8, 1e9])
        for beta in (.3, .7, .99):
            solver = MaxwellSolver(circle(), beta=beta)
            parts = solver.transverse_components(f, PEC())
            r = parts['wall']
            np.testing.assert_allclose(r.longitudinal, circular_indirect_space_charge(f, .03, beta), rtol=.001)
            np.testing.assert_allclose(r.dipolar[:, 0, 0], circular_indirect_dipolar(f, .03, beta), rtol=.0015)
            np.testing.assert_allclose(r.quadrupolar[:, 0, 0], circular_indirect_quadrupolar(f, .03, beta),
                                       rtol=.0015, atol=1e-9)
            for field in ('longitudinal', 'dipolar', 'quadrupolar', 'constant'):
                np.testing.assert_array_equal(getattr(parts['resistive_wall'], field), 0)
                np.testing.assert_array_equal(getattr(r, field), getattr(parts['indirect_space_charge'], field))
                np.testing.assert_array_equal(getattr(r, field).real, 0)

    def test_arbitrary_off_axis_pec_uses_existing_image_operators(self):
        panels = polygon_panels([[-.05,-.02],[.05,-.02],[.05,.02],[-.05,.02]],
                                panels_per_edge=12, grading_power=1.5)
        solver = MaxwellSolver(panels, beta=.7)
        point = (.004, -.003)
        f = [10., 1e8]
        actual = solver.transverse(f, PEC(), reference=point)
        image = solver.transverse(f, reference=point, component='indirect_space_charge')
        for field in ('longitudinal', 'dipolar', 'quadrupolar', 'constant'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(image, field))

    def test_pec_backed_coating_against_independent_cylindrical_modes(self):
        f = np.array([10., 1e6, 1e9, 1e12])
        wall = Wall(PEC(), (Layer(Metal(2e5), .005),))
        errors = []
        for n in (64, 128):
            s = MaxwellSolver(circle(n))
            r = s.transverse(f, wall, component='resistive_wall')
            z = circular_exact(f, .03, wall, component='resistive_wall')
            d = circular_layered_dipolar(f, .03, wall, component='resistive_wall')
            errors.append(max(np.max(abs(r.longitudinal/z-1)), np.max(abs(r.dipolar[:, 0, 0]/d-1))))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], .008)
        s = MaxwellSolver(circle(96), beta=.7)
        r = s.transverse(f, wall, component='resistive_wall')
        np.testing.assert_allclose(r.longitudinal, circular_exact(f, .03, wall, .7, component='resistive_wall'), rtol=.01)
        np.testing.assert_allclose(r.dipolar[:, 0, 0], circular_layered_dipolar(f, .03, wall, .7, component='resistive_wall'), rtol=.01)

    def test_independent_interface_mesh_and_pec_backing(self):
        definition = default_geometry()
        definition['regions'] = [{'name':'beam','material':'vacuum'},
                                 {'name':'coat','material':'NEG'}, {'name':'backing','material':'PEC'}]
        definition['materials']['NEG'] = {'conductivity':2e5}
        definition['interfaces'] = [{'name':'coat_outer', 'shape':{'type':'circle','radius':35}}]
        definition['mesh'].update(minimum_panels=64, max_panel_length=5, curve_tolerance=.1)
        definition['boundary_meshes'] = {'coat_outer': {'minimum_panels':96}}
        case = load_geometry(definition)
        solver = case.make_solver(pipe_length_m=None)
        self.assertEqual(len(solver.backend.interfaces[0].panel_start), 96)
        r = solver.transverse([10., 1e9], component='resistive_wall')
        reference = Wall(PEC(), (Layer(Metal(2e5), .005),))
        np.testing.assert_allclose(r.longitudinal, circular_exact([10.,1e9], .03, reference), rtol=.015)

    def test_case_roundtrip_length_and_gui_material_document(self):
        document = GeometryDocument()
        document.set_region(1, 'PEC')
        document.set_physics(beta=.7)
        self.assertEqual(document.issues(), [])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'pec.json'
            document.save(path)
            case = load_geometry(path)
        self.assertEqual(case.materials['PEC'], {'kind':'pec'})
        total = case.make_solver(pipe_length_m=3.)
        per_m = case.make_solver(pipe_length_m=None)
        a, b = total.transverse([10., 1e6]), per_m.transverse([10., 1e6])
        for field in ('longitudinal', 'dipolar', 'quadrupolar', 'constant'):
            np.testing.assert_allclose(getattr(a,field), 3*getattr(b,field), rtol=1e-14)
        self.assertEqual(total.longitudinal_units, 'Ohm')

    def test_pec_is_a_boundary_condition_without_bulk_parameters(self):
        definition = default_geometry()
        for invalid in ({'kind':'pec','conductivity':1e30}, {'conductivity':1e30}, {'kind':'unknown'}):
            data = copy.deepcopy(definition)
            data['materials']['PEC'] = invalid
            with self.assertRaises(ValueError):
                load_geometry(data)
        with self.assertRaisesRegex(ValueError, 'Finite layers'):
            Layer(PEC(), .001)
        data = copy.deepcopy(definition)
        data['regions'][1]['material'] = 'PEC'
        del data['materials']['PEC']
        self.assertIsInstance(load_geometry(data).make_solver().default_wall, PEC)

    def test_unsupported_pec_roughness_and_finite_pec_region_fail_clearly(self):
        s = MaxwellSolver(circle(32), roughness=Roughness(.3e-6, slices=4))
        with self.assertRaisesRegex(NotImplementedError, 'roughness'):
            s.longitudinal(10., PEC())
        data = default_geometry()
        data['interfaces'] = [{'name':'outer', 'shape':{'type':'circle','radius':35}}]
        data['regions'] = [{'name':'beam','material':'vacuum'}, {'name':'layer','material':'PEC'},
                           {'name':'wall','material':'Cu'}]
        with self.assertRaisesRegex(NotImplementedError, 'outer termination'):
            load_geometry(data).make_solver()

    def test_vacuum_and_air_are_not_pec(self):
        for label in ('vacuum', 'air'):
            data = default_geometry()
            data['materials'][label] = {'kind':'pec'}
            with self.assertRaises(ValueError):
                load_geometry(data)
        doc = GeometryDocument()
        doc.set_region(1, 'air')
        self.assertTrue(any('beta<1' in issue for issue in doc.issues()))
        with self.assertRaisesRegex(NotImplementedError, 'beta<1'):
            doc.geometry.make_solver()
        doc.set_physics(.52)
        self.assertFalse(doc.issues())
        self.assertEqual(doc.geometry.make_solver(mesh={'minimum_panels':32,'max_panel_length':10}).backend.beta,.52)

    def test_pec_state_and_diagnostics_do_not_report_fake_skin_depth(self):
        solver = MaxwellSolver(circle(32), beta=.7)
        state = solver.boundary_state(1e6, PEC())
        self.assertEqual(state.component, 'resistive_wall')
        np.testing.assert_array_equal(state.electric_longitudinal, 0)
        self.assertEqual(state.impedance_per_m, 0j)
        info = solver.diagnostics(1e6, PEC())
        self.assertEqual(info['outer_termination'], 'PEC')
        self.assertIsNone(info['attenuation_length_m'])


if __name__ == '__main__':
    unittest.main()
