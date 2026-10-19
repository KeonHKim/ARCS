"""Independent geometry conservation and cylindrical low-frequency checks."""
import unittest
import numpy as np
from arcs import Metal, Wall, polygon_panels, confocal_coating
from arcs.reference import circular_exact
from arcs.constants import MU0


class InterfaceAndLowFrequencyChecks(unittest.TestCase):
    def test_polygon_area_perimeter_and_closed_normals(self):
        vertices = np.array([[-.03, -.015], [.03, -.015], [.03, .015], [-.03, .015]])
        mesh = polygon_panels(vertices)
        self.assertAlmostEqual(mesh.perimeter, .18, places=14)
        area = .5*np.sum(mesh.weights*(mesh.points[:, 0]*mesh.tangent[:, 1]
                                     - mesh.points[:, 1]*mesh.tangent[:, 0]))
        self.assertAlmostEqual(area, .06*.03, places=14)
        np.testing.assert_allclose(mesh.weights @ mesh.normal, [0., 0.], atol=1e-14)
        for vertex in vertices:
            self.assertGreater(np.min(np.linalg.norm(mesh.points-vertex, axis=1)), 0)
        # Exact rectangle faces: sampling must not round or bulge the corners.
        on_edge = np.isclose(abs(mesh.points[:, 0]), .03, atol=1e-14, rtol=0) | np.isclose(
            abs(mesh.points[:, 1]), .015, atol=1e-14, rtol=0)
        self.assertTrue(on_edge.all())

    def test_reentrant_polygon_area(self):
        vertices = np.array([[0., 0.], [3., 0.], [3., 1.], [1., 1.], [1., 2.], [0., 2.]])
        mesh = polygon_panels(vertices)
        area = .5*np.sum(mesh.weights*(mesh.points[:, 0]*mesh.tangent[:, 1]
                                     - mesh.points[:, 1]*mesh.tangent[:, 0]))
        self.assertAlmostEqual(area, 4., places=13)
        self.assertAlmostEqual(mesh.perimeter, 10., places=13)
        np.testing.assert_allclose(mesh.weights @ mesh.normal, [0., 0.], atol=1e-13)

    def test_confocal_interface_geometry_and_normal_thickness(self):
        geometry = confocal_coating(.06, .03, 300e-9)
        a, b = geometry.inner_axes
        outer_a, outer_b = geometry.outer_axes
        self.assertAlmostEqual(a*a-b*b, outer_a**2-outer_b**2, places=15)
        thickness = geometry.normal_thickness(np.array([0., np.pi/2]))
        np.testing.assert_allclose(thickness, [outer_a-a, outer_b-b], rtol=5e-10)
        self.assertGreater(thickness[1], thickness[0])
        theta = np.linspace(0, 2*np.pi, 201)
        point = np.column_stack((a*np.cos(theta), b*np.sin(theta)))
        normal = np.column_stack((np.cos(theta)/a, np.sin(theta)/b))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        outer_point = point+geometry.normal_thickness(theta)[:, None]*normal
        np.testing.assert_allclose((outer_point[:, 0]/outer_a)**2+(outer_point[:, 1]/outer_b)**2,
                                   1., atol=1e-14)

    def test_cylindrical_ultralow_frequency_asymptotic(self):
        f = np.geomspace(1e-10, 1e-6, 21)
        radius, conductivity = .03, 5.8e7
        z = circular_exact(f, radius, Wall(Metal(conductivity, "Cu")))
        q = np.sqrt(1j*2*np.pi*f*MU0*conductivity)
        asymptotic = 1j*f*MU0*(-np.log(q*radius/2)-np.euler_gamma)
        self.assertLess(np.max(abs(z-asymptotic)/abs(z)), 3e-6)
        np.testing.assert_allclose(z.real/(2*np.pi*f*MU0/8), 1., rtol=2e-5)
        self.assertGreater(z.real.min(), 0.)
        self.assertGreater(z.imag.min(), 0.)


if __name__ == "__main__":
    unittest.main()
