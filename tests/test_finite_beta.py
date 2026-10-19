"""Finite-velocity modal references, PDE identities and independent offsets."""
import unittest
import numpy as np
from arcs import MaxwellSolver, Metal, Wall, polygon_panels, load_geometry, GeometryDocument
from arcs.constants import C0
from arcs.reference import circular_exact, circular_maxwell_dipolar, circular_maxwell_quadrupolar


def mesh(n=256):
    angle = np.arange(n)*2*np.pi/n
    return polygon_panels(.03*np.c_[np.cos(angle),np.sin(angle)],panels_per_edge=1)


class FiniteBetaChecks(unittest.TestCase):
    def test_independent_m0_m1_and_quadrupole_modes(self):
        metal = Metal(5.8e7)
        f = np.array([10.,1e4,1e6,1e8,1e9])
        for beta in (.3,.7,.99):
            # Audit RW separately so a large ISC term cannot hide a wall error.
            r = MaxwellSolver(mesh(),beta=beta).transverse(f,metal,component='resistive_wall')
            z = circular_exact(f,.03,Wall(metal),beta,component='resistive_wall')
            d = circular_maxwell_dipolar(f,.03,metal,beta,component='resistive_wall')
            q = circular_maxwell_quadrupolar(f,.03,Wall(metal),beta,component='resistive_wall')
            self.assertLess(np.max(abs(r.longitudinal/z-1)),.003)
            self.assertLess(np.max(abs(r.dipolar[:,0,0]/d-1)),.003)
            np.testing.assert_allclose(r.dipolar[:,0,0],r.dipolar[:,1,1],rtol=1e-9)
            np.testing.assert_allclose(r.quadrupolar[2:,0,0],q[2:],rtol=.003)
            np.testing.assert_allclose(r.quadrupolar[2:,1,1],q[2:],rtol=.003)
            self.assertEqual(r.beam_beta,beta)

    def test_modified_helmholtz_trace_and_panofsky_wenzel(self):
        panels = polygon_panels([[-.06,-.03],[.06,-.03],[.06,.03],[-.06,.03]],
                                panels_per_edge=[32,16,32,16],grading_power=1.5)
        beta=.7;f=3e8;solver=MaxwellSolver(panels,beta=beta)
        metal=Metal(5.8e7);point=np.array([.004,-.003]);step=2e-5
        tensor=solver.transverse(f,metal,point)
        k=2*np.pi*f/(beta*C0);factor=1/k
        np.testing.assert_allclose(np.trace(tensor.quadrupolar),
                                  k*(1-beta*beta)*tensor.longitudinal,rtol=3e-10)
        scale=np.max(abs(tensor.dipolar))
        for i in range(2):
            wi=np.eye(2)[i]*step
            for j in range(2):
                sj=np.eye(2)[j]*step
                mixed=sum(a*b*solver.longitudinal(f,metal,point+b*sj,point+a*wi)
                          for a in (-1,1) for b in (-1,1))/(4*step**2)
                self.assertLess(abs(tensor.dipolar[i,j]-factor*mixed)/scale,2e-5)

    def test_ultrarelativistic_limit_and_material_variation(self):
        p=mesh(128);metal=Metal(5.8e7);f=np.array([1e6,1e9,1e10])
        z1=MaxwellSolver(p).longitudinal(f,metal)
        near=MaxwellSolver(p,beta=1-1e-10).longitudinal(f,metal,component="resistive_wall")
        np.testing.assert_allclose(near,z1,rtol=6e-6)
        for m in (Metal(5.8e7,mu_r=1.7,epsilon_r=2.3),Metal(5.8e7,relaxation_time=2e-14)):
            r=MaxwellSolver(p,beta=.8).transverse([1e6,1e9],m,component='resistive_wall')
            self.assertLess(np.max(abs(r.longitudinal/circular_exact([1e6,1e9],.03,Wall(m),.8,component='resistive_wall')-1)),.002)
            self.assertLess(np.max(abs(r.dipolar[:,0,0]/circular_maxwell_dipolar([1e6,1e9],.03,m,.8,component='resistive_wall')-1)),.003)

    def test_beta_validation_inputs_and_no_mqs_exports(self):
        import arcs
        for beta in (0,-1,1.1,np.inf,np.nan,True):
            with self.assertRaises(ValueError):
                MaxwellSolver(mesh(32),beta=beta)
        self.assertFalse(hasattr(arcs,"MagneticDiffusionSolver"))
        document=GeometryDocument();document.set_physics(.4)
        wrapper=document.geometry.make_solver(mesh={"minimum_panels":64,"max_panel_length":5})
        self.assertEqual(wrapper.backend.beta,.4)
        self.assertEqual(load_geometry(document.geometry.to_dict()).beta,.4)
        with self.assertRaisesRegex(ValueError,"removed"):
            document.geometry.make_solver("diffusion")


if __name__ == "__main__":
    unittest.main()
