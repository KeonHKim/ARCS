"""User-kernel fixtures, analytic resonators and independent Fourier integrals."""
import json
from pathlib import Path
import unittest
import numpy as np
from arcs.wakes import wake_to_impedance, impedance_to_wake, gaussian_bunch_wake
from arcs.constants import C0


class DCTDSTChecks(unittest.TestCase):
    def test_matches_user_wakeSolver_v3_kernel_fixture(self):
        fixture=json.loads((Path(__file__).parent/'data/user_dct_reference.json').read_text())
        for sample in fixture['cases']:
            t=np.arange(len(sample['wake']))*sample['dt_s']
            f,z=wake_to_impedance(t,sample['wake'],plane=sample['plane'])
            expected=np.array(sample['impedance_real'])+1j*np.array(sample['impedance_imag'])
            np.testing.assert_allclose(z,expected,rtol=2e-14,atol=1e-28)
            _,w=impedance_to_wake(f,z,plane=sample['plane'])
            np.testing.assert_allclose(w,sample['wake'],rtol=2e-13,atol=1e-13)

    def test_direct_trapezoidal_integral_and_both_sign_conventions(self):
        t=np.linspace(0,2e-9,257); right=np.exp(-t/3e-10)*(np.cos(2*np.pi*7e9*t)-.2*np.sin(2*np.pi*7e9*t))
        stored=right.copy(); stored[0]*=.5
        f,z=wake_to_impedance(t,stored)
        direct=np.trapezoid(right[None,:]*np.exp(-2j*np.pi*f[:,None]*t),t,axis=1)
        self.assertLess(np.linalg.norm(z-direct)/np.linalg.norm(direct),1e-13)
        fa,za=wake_to_impedance(t,stored,convention='arcs')
        np.testing.assert_array_equal(f,fa); np.testing.assert_array_equal(za,z)
        _,legacy=wake_to_impedance(t,stored,convention="arcs_legacy")
        np.testing.assert_array_equal(legacy,np.conjugate(z))
        ta,w=impedance_to_wake(f,za,convention='arcs')
        np.testing.assert_allclose(ta,t,rtol=2e-14,atol=0)
        np.testing.assert_allclose(w,stored,rtol=2e-12,atol=1e-14)

    def test_analytical_longitudinal_and_transverse_resonators(self):
        f=np.linspace(0,1e12,65537); omega=2*np.pi*f
        wr=2*np.pi*5e9; quality=2.; alpha=wr/(2*quality); wd=np.sqrt(wr**2-alpha**2)
        denominator=wr**2-omega**2+2j*alpha*omega
        z_long=1j*omega*(100*wr/quality)/denominator
        z_trans=1j*1e14*wd/denominator
        for plane,z in (('longitudinal',z_long),('transverse',z_trans)):
            t,w=impedance_to_wake(f,z,plane=plane)
            if plane=='longitudinal':
                expected=100*wr/quality*np.exp(-alpha*t)*(np.cos(wd*t)-alpha/wd*np.sin(wd*t)); expected[0]*=.5
            else:
                expected=1e14*np.exp(-alpha*t)*np.sin(wd*t)
            active=t<=1e-9
            self.assertLess(np.linalg.norm(w[active]-expected[active])/np.linalg.norm(expected[active]),1e-3)

    def test_missing_dc_and_nonuniform_grids_need_explicit_choices(self):
        f=np.geomspace(10,1e6,21); z=(1-1j)*np.sqrt(f)
        with self.assertRaisesRegex(ValueError,'dc_value'): impedance_to_wake(f,z)
        with self.assertRaisesRegex(ValueError,'resample_points'): impedance_to_wake(f,z,dc_value=0)
        t,w=impedance_to_wake(f,z,dc_value=0,resample_points=129,convention='arcs')
        self.assertEqual(len(t),129); self.assertAlmostEqual(t[1],.5e-6)
        self.assertTrue(np.isfinite(w).all())

    def test_padding_preserves_cutoff_half_weight_and_output_sampling(self):
        f=np.linspace(0,1e9,101); z=np.ones(101,dtype=complex)
        _,w0=impedance_to_wake(f,z)
        t,w=impedance_to_wake(f,z,pad=100)
        self.assertAlmostEqual(w[0],w0[0],places=5)
        self.assertAlmostEqual(t[1],.25e-9)
        t2,w2=impedance_to_wake(f,z,tmax=30e-9,samples=501)
        self.assertEqual(len(t2),501); self.assertAlmostEqual(t2[-1],30e-9)
        self.assertTrue(np.isfinite(w2).all())

    def test_gaussian_bunch_transform_matches_direct_signed_fourier_integral(self):
        f=np.linspace(0,1e11,1025); wr=2*np.pi*5e9; omega=2*np.pi*f
        z=1j*omega*100*wr/2/(wr**2-omega**2+1j*omega*wr/2)
        sigma=.01; s,w=gaussian_bunch_wake(f,z,sigma,convention='dct')
        indices=np.array([1010,1018,1024,1030,1040])
        weighted=z*np.exp(-.5*(2*np.pi*f*sigma/C0)**2)
        expected=2*np.trapezoid((weighted[None,:]*np.exp(2j*np.pi*f[None,:]*s[indices,None]/C0)).real,f,axis=1)
        np.testing.assert_allclose(w[indices],expected,rtol=2e-10,atol=1e-5)

    def test_invalid_input_and_transverse_endpoint_guards(self):
        with self.assertRaisesRegex(ValueError,'endpoint'): wake_to_impedance([0,1,2],[1,2,0],plane='transverse')
        with self.assertRaises(ValueError): wake_to_impedance([0,1,2],[1,np.nan,0])
        with self.assertRaises(ValueError): wake_to_impedance([1,2,3],[1,2,0])
        with self.assertRaises(ValueError): impedance_to_wake([0,1,2],[1,1,1],pad=-1)


if __name__=='__main__': unittest.main()
