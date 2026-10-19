"""Two-sided DCT/DST checked against analytic Gaussian Fourier pairs."""
import unittest
import numpy as np
from arcs.wakes import impedance_to_wake, wake_to_impedance, gaussian_bunch_wake
from arcs.constants import C0


class SignedWakeChecks(unittest.TestCase):
    def test_signed_analytic_gaussian_pair_for_both_planes(self):
        tau = 2e-10
        f = np.linspace(0, 8/tau, 2049)
        u = 2*np.pi*f*tau
        fourier = np.sqrt(2*np.pi)*tau*np.exp(-.5*u*u)*(.7-.3j*u)
        for plane, spectrum in (('longitudinal',fourier),('transverse',1j*fourier)):
            t, w = impedance_to_wake(f, spectrum, plane=plane, causal=False)
            expected = np.exp(-.5*(t/tau)**2)*(.7+.3*t/tau)
            np.testing.assert_allclose(w, expected, rtol=2e-11, atol=2e-13)
            ff, zz = wake_to_impedance(t, w, plane=plane, causal=False)
            np.testing.assert_allclose(ff, f, rtol=2e-14, atol=0)
            np.testing.assert_allclose(zz, spectrum, rtol=3e-11, atol=2e-25)

    def test_pure_imaginary_indirect_term_survives_signed_inverse(self):
        f = np.linspace(0, 1e11, 1025); tau = 1e-10
        z = 1j*(2*np.pi*f*tau)*np.exp(-.5*(2*np.pi*f*tau)**2)
        with self.assertRaisesRegex(ValueError, 'causal=False'):
            impedance_to_wake(f, z)
        t, w = impedance_to_wake(f, z, causal=False)
        self.assertGreater(max(abs(w)), 0)
        np.testing.assert_allclose(w, -w[::-1], rtol=1e-13, atol=1e-5)
        self.assertEqual(t[len(t)//2], 0)

    def test_signed_transform_against_direct_integral(self):
        t = np.linspace(-2e-9, 2e-9, 1025)
        w = np.exp(-.5*(t/3e-10)**2)*(1+t/4e-10)
        for plane in ('longitudinal','transverse'):
            f, z = wake_to_impedance(t, w, plane=plane, causal=False)
            index = np.array([0,1,3,9,17,40])
            direct = np.trapezoid(w[None,:]*np.exp(-2j*np.pi*f[index,None]*t), t, axis=1)
            if plane == 'transverse': direct *= 1j
            np.testing.assert_allclose(z[index],direct,rtol=2e-9,atol=1e-24)

    def test_beta_gaussian_bunch_against_direct_integral(self):
        f = np.linspace(0, 1e11, 1025); beta=.7; sigma=.01
        z = 1j*f*np.exp(-(f/2e9)**2)
        s,w = gaussian_bunch_wake(f,z,sigma,beta=beta)
        index = np.array([1010,1018,1024,1030,1040])
        filtered = z*np.exp(-.5*(2*np.pi*f*sigma/(beta*C0))**2)
        expected = 2*np.trapezoid((filtered[None,:]*np.exp(2j*np.pi*f[None,:]*s[index,None]/(beta*C0))).real,f,axis=1)
        np.testing.assert_allclose(w[index],expected,rtol=2e-10,atol=1e3)


if __name__ == '__main__':
    unittest.main()
