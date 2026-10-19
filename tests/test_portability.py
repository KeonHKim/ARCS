"""Regressions for incorrect complex-Bessel and matrix-library branches."""
import unittest
import numpy as np
from arcs.special import decaying_k
from arcs.layered import equilibrated_solve
from arcs import load_geometry, Metal, Wall
from arcs.reference import circular_exact


class PortabilityChecks(unittest.TestCase):
    def test_bessel_underflow_unscaling_against_high_precision_values(self):
        # Independently computed with mpmath at 60 decimal digits. Affected
        # SciPy kv builds can return O(1e-2) imaginary parts here instead.
        argument = np.array([680-680j, 680-1000j])
        expected = (
            [-4.553730032944803e-298+1.8787270101098555e-297j,
             1.901684871999608e-298+1.7134123414795906e-297j],
            [-4.562308745040387e-298+1.8792503633427792e-297j,
             1.8962702041584387e-298+1.7138757583965063e-297j],
        )
        for order in (0, 1):
            np.testing.assert_allclose(decaying_k(order, argument), expected[order],
                                       rtol=2e-13, atol=0.)
        np.testing.assert_array_equal(decaying_k(0, np.array([800-800j])), [0j])

    def test_complex_symmetric_system_uses_correct_general_solve(self):
        matrix = np.array([[182.56985285-64.28859483j, -177.24879835+11.0780499j],
                           [-177.24879835+11.0780499j, 177.24879835-11.0780499j]])
        rhs = np.array([[1., 2.], [3., 4.]])
        result = equilibrated_solve(matrix, rhs)
        np.testing.assert_allclose(matrix @ result, rhs, rtol=2e-14, atol=2e-14)

    def test_circular_real_part_at_previously_corrupted_frequencies(self):
        case = load_geometry({"units": "mm", "aperture": {"type": "circle", "radius": 30},
            "mesh": {"minimum_panels": 256, "max_panel_length": 1., "curve_tolerance": .01},
            "regions": [{"name": "beam", "material": "vacuum"},
                        {"name": "wall", "material": "Cu"}],
            "materials": {"Cu": {"conductivity": 5.8e7}}})
        f = np.geomspace(10., 1e13, 100)[[49, 58, 61, 66, 69, 72, 74]]
        result = case.make_solver().longitudinal(f)
        reference = circular_exact(f, .03, Wall(Metal(5.8e7)))
        np.testing.assert_allclose(result.real, reference.real, rtol=5e-4, atol=0.)
        np.testing.assert_allclose(result.imag, reference.imag, rtol=5e-4, atol=0.)


if __name__ == "__main__":
    unittest.main()
