"""Keep signed reference curves visible when the numerical model stays positive."""
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arcs.plotting import plot_longitudinal


class SignedReferencePlotChecks(unittest.TestCase):
    def test_linear_axes_show_direct_imaginary_component(self):
        f = np.array([10., 100., 1000.]); z = np.array([1+2j, 2-1j, 3+4j])
        figure = plot_longitudinal(f, z, xscale='linear', yscale='linear')
        try:
            for axis in figure.axes:
                self.assertEqual(axis.get_xscale(), 'linear')
                self.assertEqual(axis.get_yscale(), 'linear')
            np.testing.assert_array_equal(figure.axes[1].lines[0].get_ydata(), z.imag)
            self.assertEqual(figure.axes[1].get_ylabel(), 'Im(Z∥/L) [Ω/m]')
        finally:
            plt.close(figure)

    def test_reference_sign_change_sets_scale_and_preserves_data(self):
        f = np.array([1e11, 1e12, 2e12])
        numerical = np.array([.4+.4j, 1.4+1.4j, 2.+2.j])
        reference = np.array([.44+.44j, 2.1+1.2j, 3.7-.86j])
        figure = plot_longitudinal(f, numerical, reference=reference)
        try:
            real, imaginary = figure.axes
            self.assertEqual(real.get_yscale(), 'log')
            self.assertEqual(imaginary.get_yscale(), 'symlog')
            self.assertLess(imaginary.get_ylim()[0], -.86)
            np.testing.assert_array_equal(imaginary.lines[0].get_ydata(), numerical.imag)
            np.testing.assert_array_equal(imaginary.lines[1].get_ydata(), reference.imag)
        finally:
            plt.close(figure)


if __name__ == '__main__':
    unittest.main()
