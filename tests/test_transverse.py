"""Saved beam/material inputs reach the Maxwell transverse calculation."""
import unittest
import numpy as np
from arcs import GeometryDocument, Metal


class TransverseChecks(unittest.TestCase):
    def test_geometry_wrapper_uses_beam_and_saved_material(self):
        doc=GeometryDocument(); doc.set_beam(2.,3.)
        solver=doc.geometry.make_solver(mesh={'minimum_panels':64,'max_panel_length':5,'curve_tolerance':.05})
        implicit=solver.transverse([10.,100.])
        explicit=solver.backend.transverse([10.,100.],Metal(5.8e7),reference=(.002,.003))
        np.testing.assert_array_equal(implicit.dipolar,explicit.dipolar)
        self.assertEqual(implicit.dipolar.shape,(2,2,2))
        self.assertEqual(implicit.reference_position_m,(.002,.003))


if __name__=='__main__': unittest.main()
