"""Independent rectangle series, current conservation and refinement checks."""
import unittest
import numpy as np
from arcs import PolygonPECSolver, polygon_panels
from arcs.plate_reference import (rectangle_surface_integral, rectangle_top_density,
                                 parallel_plate_surface_integral, parallel_plate_density)


def rectangle(a, b, base):
    horizontal = int(np.ceil(base*a/b))
    return polygon_panels([[-a, -b], [a, -b], [a, b], [-a, b]],
                          panels_per_edge=[horizontal, base, horizontal, base])


class PolygonPECChecks(unittest.TestCase):
    def test_finite_rectangles_match_independent_series_and_fields(self):
        b = .03
        for ratio in [1., 2., 4.]:
            a = ratio*b
            solver = PolygonPECSolver(rectangle(a, b, 64))
            reference = rectangle_surface_integral(a, b)
            self.assertLess(abs(solver.surface_integral()/reference-1), 3e-4)
            selected = np.isclose(solver.centres[:, 1], b, atol=1e-14, rtol=0)
            exact = rectangle_top_density(solver.centres[selected, 0], a, b)
            self.assertLess(max(abs(solver.panel_density()[selected]-exact))/max(exact), 8e-4)
            self.assertLess(abs(solver.diagnostics()["image_current_integral"]-1), 4e-7)

    def test_refinement_reduces_physical_reference_error(self):
        a, b = .06, .03
        reference = rectangle_surface_integral(a, b)
        error = []
        for base in [8, 16, 32]:
            result = PolygonPECSolver(rectangle(a, b, base)).surface_integral()
            error.append(abs(result/reference-1))
        self.assertLess(error[-1], error[0]/10)
        self.assertLess(error[1], error[0])

    def test_parallel_plate_limit_and_two_plate_normalization(self):
        b = .03
        target = parallel_plate_surface_integral(b)
        self.assertLess(abs(rectangle_surface_integral(8*b, b)/target-1), 2e-9)
        # Independent numerical integration over BOTH walls checks the factor 2.
        x = np.linspace(-20*b, 20*b, 20001)
        integrated = 2*np.trapezoid(parallel_plate_density(x, b)**2, x)
        self.assertAlmostEqual(integrated/target, 1., places=12)

    def test_off_axis_top_current_matches_green_function_modes(self):
        a, b, source = .06, .03, (.012, .006)
        solver = PolygonPECSolver(rectangle(a, b, 64))
        top = np.isclose(solver.centres[:, 1], b, atol=1e-14, rtol=0)
        computed = solver.lengths[top] @ solver.panel_density(source)[top]
        # Direct separation of variables; integrate sine modes over the top face.
        modes = np.arange(1, 1000, 2)
        k = modes*np.pi/(2*a)
        factor = np.exp(-k*(b-source[1]))*(-np.expm1(-2*k*(source[1]+b)))/(-np.expm1(-4*k*b))
        exact = np.sum(2*np.sin(k*(source[0]+a))*factor/(a*k))
        self.assertLess(abs(computed-exact), 2e-5)

    def test_asymmetric_panel_counts_preserve_geometry(self):
        mesh = rectangle(.12, .03, 16)
        self.assertEqual(len(mesh.panel_start), 160)
        self.assertAlmostEqual(mesh.perimeter, .6, places=14)
        area = .5*np.sum(mesh.weights*(mesh.points[:, 0]*mesh.tangent[:, 1]
                                     - mesh.points[:, 1]*mesh.tangent[:, 0]))
        self.assertAlmostEqual(area, .24*.06, places=14)
        np.testing.assert_allclose(mesh.weights @ mesh.normal, [0., 0.], atol=1e-14)
        with self.assertRaises(ValueError):
            polygon_panels(mesh.vertices, panels_per_edge=[1, 2])

    def test_source_inside_reentrant_polygon_and_boundary_rejection(self):
        vertices = [[0., 0.], [3., 0.], [3., 1.], [1., 1.], [1., 2.], [0., 2.]]
        solver = PolygonPECSolver(polygon_panels(vertices, panels_per_edge=64, grading_power=3.))
        source = (.5, .5)
        self.assertLess(abs(solver.diagnostics(source)["image_current_integral"]-1), 2e-4)
        for point in [(2., 1.5), (1., 1.), (3., .5)]:
            with self.assertRaises(ValueError):
                solver.panel_density(point)


if __name__ == "__main__":
    unittest.main()
