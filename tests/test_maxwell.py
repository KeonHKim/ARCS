"""Independent cylindrical modes, displaced fields and Maxwell invariants."""
import unittest
import numpy as np
from arcs import MaxwellSolver, Metal, Wall, Layer, polygon_panels, load_geometry
from arcs.constants import C0
from arcs.reference import circular_exact, circular_maxwell_dipolar


def round_mesh(n):
    angle = np.arange(n)*2*np.pi/n
    return polygon_panels(.03*np.c_[np.cos(angle), np.sin(angle)], panels_per_edge=1)


class MaxwellChecks(unittest.TestCase):
    def test_circular_full_band_and_thz_sign_change(self):
        f = np.array([10., 1e3, 1e6, 1e9, 1e11, 1e12, 2e12, 5e12])
        metal = Metal(5.8e7)
        response = MaxwellSolver(round_mesh(256)).transverse(f, metal)
        monopole = circular_exact(f, .03, Wall(metal))
        dipole = circular_maxwell_dipolar(f, .03, metal)
        self.assertLess(np.max(abs(response.longitudinal/monopole-1)), .0015)
        self.assertLess(np.max(abs(response.dipolar[:, 0, 0]/dipole-1)), .0035)
        self.assertGreater(response.longitudinal[-3].imag, 0.)
        self.assertLess(response.longitudinal[-2].imag, 0.)
        self.assertTrue(np.all(response.longitudinal.real > 0))
        np.testing.assert_allclose(response.dipolar[:, 0, 0], response.dipolar[:, 1, 1], rtol=2e-10)
        self.assertLess(np.max(abs(response.dipolar[:, 0, 1]/dipole)), 2e-10)
        self.assertLess(np.max(abs(response.quadrupolar)/abs(dipole[:, None, None])), 2e-10)
        self.assertLess(np.max(abs(response.constant)/abs(dipole[:, None])), 2e-10)

    def test_low_and_high_frequency_mesh_refinement(self):
        f = np.array([10., 2e12]); metal = Metal(5.8e7)
        exact = circular_exact(f, .03, Wall(metal))
        dipole = circular_maxwell_dipolar(f, .03, metal)
        errors = []
        for n in (64, 128, 256):
            r = MaxwellSolver(round_mesh(n)).transverse(f, metal)
            errors.append(np.c_[abs(r.longitudinal/exact-1), abs(r.dipolar[:, 0, 0]/dipole-1)])
        self.assertTrue(np.all(errors[1] < errors[0]))
        self.assertTrue(np.all(errors[2] < errors[1]))

    def test_displaced_fields_validate_analytic_transverse_derivatives(self):
        vertices = [[-.06, -.03], [.06, -.03], [.06, .03], [-.06, .03]]
        solver = MaxwellSolver(polygon_panels(vertices, panels_per_edge=[32, 16, 32, 16], grading_power=2))
        metal = Metal(5.8e7); origin = np.array([.004, -.003]); step = 3e-5
        for f in (100., 2e12):
            tensor = solver.transverse(f, metal, origin)
            factor = C0/(2*np.pi*f)
            np.testing.assert_allclose(tensor.longitudinal, solver.longitudinal(f, metal, origin, origin), rtol=1e-11)
            scale = np.max(abs(tensor.dipolar))
            for i in range(2):
                wi = np.eye(2)[i]*step
                first = (solver.longitudinal(f, metal, origin, origin+wi)
                         - solver.longitudinal(f, metal, origin, origin-wi))/(2*step)
                self.assertLess(abs(tensor.constant[i]-factor*first)/scale, 1e-6)
                for j in range(2):
                    sj = np.eye(2)[j]*step
                    mixed = sum(a*b*solver.longitudinal(f, metal, origin+b*sj, origin+a*wi)
                                for a in (-1, 1) for b in (-1, 1))/(4*step**2)
                    self.assertLess(abs(tensor.dipolar[i, j]-factor*mixed)/scale, 2e-5)
                    if i == j:
                        second = (solver.longitudinal(f, metal, origin, origin+wi)-2*tensor.longitudinal
                                  + solver.longitudinal(f, metal, origin, origin-wi))/step**2
                    else:
                        second = sum(a*b*solver.longitudinal(f, metal, origin, origin+a*wi+b*sj)
                                     for a in (-1, 1) for b in (-1, 1))/(4*step**2)
                    self.assertLess(abs(tensor.quadrupolar[i, j]-factor*second)/scale, 2e-5)
            self.assertLess(abs(np.trace(tensor.quadrupolar))/scale, 1e-12)

    def test_translation_rotation_and_material_dispersion(self):
        vertices = np.array([[-.06, -.03], [.06, -.03], [.06, .03], [-.06, .03]])
        angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        offset = np.array([.003, -.004])
        responses = []
        for v in (vertices, vertices @ rotation.T+offset):
            s = MaxwellSolver(polygon_panels(v, panels_per_edge=[32, 16, 32, 16], grading_power=2))
            responses.append(s.transverse(2e12, Metal(5.8e7), reference=(0., 0.) if len(responses) == 0 else offset))
        for field in ("dipolar", "quadrupolar"):
            expected = rotation @ getattr(responses[0], field) @ rotation.T
            # Graded tiny corner panels in the high-k block matrix amplify
            # coordinate roundoff. Check covariance to 10 ppm, without
            # rotating or symmetrizing the returned physical coefficients.
            np.testing.assert_allclose(getattr(responses[1], field), expected, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(responses[0].longitudinal, responses[1].longitudinal, rtol=1e-6)
        solver = MaxwellSolver(round_mesh(128))
        f = np.array([10., 1e12, 5e12])
        for metal in (Metal(5.8e7, mu_r=1.7, epsilon_r=2.3), Metal(5.8e7, relaxation_time=2e-14)):
            calculated = solver.longitudinal(f, metal)
            self.assertLess(np.max(abs(calculated/circular_exact(f, .03, Wall(metal))-1)), .002)
        drude = Metal(5.8e7, relaxation_time=2e-14)
        d = solver.transverse(f, drude).dipolar[:, 0, 0]
        self.assertLess(np.max(abs(d/circular_maxwell_dipolar(f, .03, drude)-1)), .007)

    def test_matching_residual_default_backend_and_interface_guards(self):
        geometry = load_geometry({"units": "mm", "aperture": {"type": "circle", "radius": 30},
            "mesh": {"minimum_panels": 64, "max_panel_length": 5, "curve_tolerance": .05},
            "regions": [{"name": "beam", "material": "vacuum"}, {"name": "wall", "material": "Cu"}],
            "materials": {"Cu": {"conductivity": 5.8e7}}})
        wrapper = geometry.make_solver()
        self.assertEqual(wrapper.backend_name, "maxwell")
        solver = wrapper.backend; metal = Metal(5.8e7)
        for f in (10., 2e12):
            checks = solver.diagnostics(f, metal)
            self.assertLess(checks["relative_matrix_residual"], 2e-12)
            self.assertTrue(checks["full_vector_maxwell"])
            self.assertFalse(checks["local_surface_impedance_used"])
            state = solver.boundary_state(f, metal)
            self.assertEqual(state.electric_transverse_reaction.shape, (len(solver.pec.lengths), 2))
            np.testing.assert_allclose(state.impedance_per_m, wrapper.longitudinal(f), rtol=2e-13)
        for f in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                solver.longitudinal(f, metal)
        coated = Wall(metal, (Layer(Metal(2e5),150e-9),))
        actual = solver.longitudinal([10.,1e9],coated)
        expected = circular_exact([10.,1e9],.03,coated)
        self.assertLess(np.max(abs(actual/expected-1)),.006)


if __name__ == "__main__":
    unittest.main()
