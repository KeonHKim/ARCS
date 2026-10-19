"""Independent PEC image formulas, rectangle modes, derivatives and convention."""
import unittest
from pathlib import Path
import numpy as np
from scipy.special import k0
from arcs import MaxwellSolver, Metal, Wall, polygon_panels, load_geometry
from arcs.constants import C0, Z0
from arcs.reference import (circular_exact, circular_indirect_space_charge,
                            circular_indirect_dipolar, circular_indirect_quadrupolar)


def circle_panels(n=128):
    theta = np.arange(n)*2*np.pi/n
    return polygon_panels(.03*np.c_[np.cos(theta), np.sin(theta)], panels_per_edge=1)


def rectangle_image(f, beta, a, b, source, witness, modes=512):
    """Independent separated-variable Dirichlet Green function; no BEM kernels."""
    k = 2*np.pi*f/(beta*C0)
    inverse_gamma2 = (1-beta)*(1+beta)
    kappa = k*np.sqrt(inverse_gamma2)
    n = np.arange(1, modes+1)
    alpha = np.sqrt(kappa*kappa+(n*np.pi/(2*a))**2)
    xs, ys = source; xw, yw = witness
    distance = abs(yw-ys)
    g = (np.exp(-alpha*distance)-np.exp(-alpha*(2*b-yw-ys))
         -np.exp(-alpha*(2*b+yw+ys))+np.exp(-alpha*(4*b-distance)))
    g /= 2*alpha*(-np.expm1(-4*alpha*b))
    dirichlet = np.sum(np.sin(n*np.pi*(xs+a)/(2*a))*np.sin(n*np.pi*(xw+a)/(2*a))*g)/a
    free = k0(kappa*np.linalg.norm(np.asarray(witness)-source))/(2*np.pi)
    return 1j*k*Z0*inverse_gamma2/beta*(free-dirichlet)


class SpaceChargeChecks(unittest.TestCase):
    def test_circular_independent_m0_m1_and_quadrupole(self):
        f = np.array([10., 1e4, 1e6, 1e8, 1e9])
        for beta in (.3, .7, .99):
            response = MaxwellSolver(circle_panels(), beta=beta).transverse(
                f, component='indirect_space_charge')
            z = circular_indirect_space_charge(f, .03, beta)
            d = circular_indirect_dipolar(f, .03, beta)
            q = circular_indirect_quadrupolar(f, .03, beta)
            np.testing.assert_allclose(response.longitudinal, z, rtol=.001)
            np.testing.assert_allclose(response.dipolar[:, 0, 0], d, rtol=.0015)
            np.testing.assert_allclose(response.dipolar[:, 1, 1], d, rtol=.0015)
            # Q is O(f² log f) here: the 10 Hz value is below the precision
            # of cancelling large Cartesian Hessian terms. Use an absolute
            # bound there; resolved quadrupoles retain a relative bound.
            np.testing.assert_allclose(response.quadrupolar[:, 0, 0], q, rtol=.0015, atol=1e-9)
            np.testing.assert_allclose(response.quadrupolar[:, 1, 1], q, rtol=.0015, atol=1e-9)
            self.assertTrue(np.all(response.longitudinal.imag > 0))
            np.testing.assert_array_equal(response.longitudinal.real, 0)
            self.assertEqual(response.component, 'indirect_space_charge')

    def test_default_sum_and_material_independence(self):
        solver = MaxwellSolver(circle_panels(96), beta=.7)
        f = [10., 1e6, 1e9]
        parts = solver.transverse_components(f, Metal(5.8e7))
        for field in ('longitudinal', 'dipolar', 'quadrupolar', 'constant'):
            np.testing.assert_array_equal(getattr(parts['wall'], field),
                getattr(parts['resistive_wall'], field)+getattr(parts['indirect_space_charge'], field))
            np.testing.assert_array_equal(getattr(solver.transverse(f, Metal(5.8e7)), field),
                                          getattr(parts['wall'], field))
        a = solver.longitudinal(f, Metal(5.8e7), component='indirect_space_charge')
        b = solver.longitudinal(f, Metal(1e3), component='indirect_space_charge')
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, parts['indirect_space_charge'].longitudinal, rtol=2e-14)
        with self.assertRaisesRegex(ValueError, 'component'):
            solver.longitudinal(10., Metal(5.8e7), component='total_direct')

    def test_beta_one_image_is_exact_zero_and_old_rw_is_conjugated(self):
        solver = MaxwellSolver(circle_panels(64))
        image = solver.transverse([10., 1e13], component='indirect_space_charge')
        for field in ('longitudinal', 'dipolar', 'quadrupolar', 'constant'):
            np.testing.assert_array_equal(getattr(image, field), 0)
        old = Path(__file__).parent/'data/v052_circular_signed_reference.csv'
        if old.exists():
            data = np.genfromtxt(old, delimiter=',', names=True)
            s = MaxwellSolver(circle_panels(256))
            fresh = s.transverse(data['f_Hz'], Metal(5.8e7))
            np.testing.assert_allclose(fresh.longitudinal,
                data['Re_long_Ohm_m']-1j*data['Im_long_Ohm_m'], rtol=3e-11, atol=1e-14)
            np.testing.assert_allclose(fresh.dipolar[:, 0, 0],
                data['Re_dipolar_Ohm_m2']-1j*data['Im_dipolar_Ohm_m2'], rtol=3e-11, atol=1e-13)

    def test_rectangle_independent_modes_and_mesh_convergence(self):
        a, b, beta = .055, .02, .7
        source, witness = (.004, -.006), (-.003, .007)
        f = np.array([10., 1e5, 1e8, 1e9])
        reference = np.array([rectangle_image(x, beta, a, b, source, witness) for x in f])
        errors = []
        for n in (16, 32, 64):
            p = polygon_panels([[-a,-b],[a,-b],[a,b],[-a,b]], panels_per_edge=n, grading_power=1.5)
            z = MaxwellSolver(p, beta=beta).longitudinal(f, source=source, witness=witness,
                                                        component='indirect_space_charge')
            errors.append(max(abs(z/reference-1)))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[-1], .001)

    def test_off_axis_image_derivatives_and_rigid_motion(self):
        vertices = np.array([[-.055,-.02],[.055,-.02],[.055,.02],[-.055,.02]])
        solver = MaxwellSolver(polygon_panels(vertices, panels_per_edge=32, grading_power=1.5), beta=.7)
        point = np.array([.004, -.003]); step = 2e-5; f = 1e8
        r = solver.transverse(f, reference=point, component='indirect_space_charge')
        factor = .7*C0/(2*np.pi*f)
        def z(s, w):
            return solver.longitudinal(f, source=s, witness=w, component='indirect_space_charge')
        for i in range(2):
            wi = np.eye(2)[i]*step
            np.testing.assert_allclose(r.constant[i], factor*(z(point,point+wi)-z(point,point-wi))/(2*step), rtol=1e-5)
            for j in range(2):
                sj = np.eye(2)[j]*step
                mixed = sum(x*y*z(point+y*sj,point+x*wi) for x in (-1,1) for y in (-1,1))/(4*step**2)
                np.testing.assert_allclose(r.dipolar[i,j], factor*mixed, rtol=5e-5)
        k = 1/factor
        np.testing.assert_allclose(np.trace(r.quadrupolar), k*(1-.7**2)*r.longitudinal, rtol=1e-10)
        angle = .37; rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        offset = np.array([.03, -.01])
        moved = MaxwellSolver(polygon_panels(vertices@rotation.T+offset, panels_per_edge=32, grading_power=1.5), beta=.7)
        transformed = moved.transverse(f, reference=point@rotation.T+offset, component='indirect_space_charge')
        for name in ('dipolar', 'quadrupolar'):
            np.testing.assert_allclose(getattr(transformed,name), rotation@getattr(r,name)@rotation.T, rtol=1e-10, atol=1e-9)

    def test_geometry_wrapper_components_and_drude_convention(self):
        case = load_geometry({'units':'mm','beta':.7,'aperture':{'type':'circle','radius':30},
            'regions':[{'name':'beam','material':'vacuum'},{'name':'wall','material':'Cu'}],
            'materials':{'Cu':{'conductivity':5.8e7}}})
        solver = case.make_solver(mesh={'minimum_panels':64,'max_panel_length':5})
        parts = solver.longitudinal_components([10.,1e6])
        np.testing.assert_array_equal(solver.longitudinal([10.,1e6]), parts['wall'])
        self.assertEqual(solver.transverse(1e6,component='indirect_space_charge').component, 'indirect_space_charge')
        metal = Metal(5.8e7, relaxation_time=2e-14)
        self.assertLess(metal.sigma(1e13).imag, 0)
        self.assertGreater(Metal(5.8e7).transverse_wavenumber_squared(1e9).imag, 0)


if __name__ == '__main__':
    unittest.main()
