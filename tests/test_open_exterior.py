"""Independent cylindrical audits for a finite wall in open vacuum/air."""
import unittest
import numpy as np
from arcs import Air,Vacuum,Metal,Wall,Layer,MaxwellSolver,polygon_panels
from arcs.reference import circular_exact,circular_layered_dipolar,circular_maxwell_quadrupolar

class OpenExteriorChecks(unittest.TestCase):
    @staticmethod
    def mesh(n=192):
        t=np.arange(n)*2*np.pi/n
        return polygon_panels(.0313*np.c_[np.cos(t),np.sin(t)],panels_per_edge=1)

    def test_two_conductors_against_independent_cylindrical_fields(self):
        f=np.array([1e4,1e7,1e9]);beta=.52
        wall=Wall(Air(),(Layer(Metal(7.7e5),.0004),Layer(Metal(1e4),.003)))
        actual=MaxwellSolver(self.mesh(),beta=beta).transverse(f,wall,component='resistive_wall')
        np.testing.assert_allclose(actual.longitudinal,circular_exact(f,.0313,wall,beta,component='resistive_wall'),rtol=.006)
        np.testing.assert_allclose(actual.dipolar[:,0,0],circular_layered_dipolar(f,.0313,wall,beta,component='resistive_wall'),rtol=.006)
        np.testing.assert_allclose(actual.quadrupolar[:,0,0],circular_maxwell_quadrupolar(f,.0313,wall,beta,component='resistive_wall'),rtol=.006)
        other=Wall(Vacuum(),wall.layers)
        np.testing.assert_allclose(actual.longitudinal,MaxwellSolver(self.mesh(),beta=beta).longitudinal(f,other,component='resistive_wall'),rtol=1e-12,atol=1e-14)

    def test_no_material_has_zero_wall_field_for_an_off_axis_beam(self):
        mesh=polygon_panels([[-.05,-.02],[.05,-.02],[.05,.02],[-.05,.02]],panels_per_edge=12)
        for beta in [.3,.52,.9]:
            solver=MaxwellSolver(mesh,beta=beta)
            result=solver.transverse([10.,1e8],Air(),reference=(.003,-.001))
            for key in ['longitudinal','dipolar','quadrupolar','constant']:
                np.testing.assert_array_equal(getattr(result,key),0.)

    def test_open_exterior_limit_is_explicit(self):
        with self.assertRaisesRegex(NotImplementedError,'beta<1'):
            MaxwellSolver(self.mesh(32),beta=1.).longitudinal(1e6,Air())
        info=MaxwellSolver(self.mesh(32),beta=.52).diagnostics(1e6,Air())
        self.assertTrue(info['exterior_is_open']);self.assertEqual(info['outer_termination'],'air')
        self.assertTrue(info['air_approximated_as_vacuum'])

if __name__=='__main__':unittest.main()
