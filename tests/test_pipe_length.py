"""Uniform-pipe normalization must preserve every complex field component."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arcs import load_geometry
from arcs.plotting import plot_longitudinal, plot_transverse

ROOT = Path(__file__).resolve().parents[1]


def geometry():
    return load_geometry({'units': 'mm', 'aperture': {'type': 'rectangle', 'width': 60, 'height': 40},
        'beam': [5, 3], 'beta': .7,
        'regions': [{'name': 'beam', 'material': 'vacuum'}, {'name': 'wall', 'material': 'Cu'}],
        'materials': {'Cu': {'conductivity': 5.8e7}},
        'mesh': {'minimum_panels': 48, 'max_panel_length': 100,
                 'curve_tolerance': 100, 'corner_grading': 1}})


class PipeLengthChecks(unittest.TestCase):
    def test_all_components_offsets_and_tensors_scale_once(self):
        case = geometry()
        per_length = case.make_solver(pipe_length_m=None)
        f = np.array([10., 1e6, 1e9])
        original = per_length.transverse_components(f)
        original_long = per_length.longitudinal_components(f, witness=(.004, -.002))
        self.assertGreater(np.max(abs(original['wall'].constant)), 0)
        self.assertGreater(np.max(abs(original['wall'].quadrupolar)), 0)
        self.assertGreater(np.max(abs(original['indirect_space_charge'].longitudinal)), 0)
        for length in [1., 4.5]:
            total = case.make_solver(pipe_length_m=length)
            parts = total.transverse_components(f)
            longs = total.longitudinal_components(f, witness=(.004, -.002))
            for component in original:
                for field in ('longitudinal', 'dipolar', 'quadrupolar', 'constant'):
                    np.testing.assert_allclose(getattr(parts[component], field),
                                               length*getattr(original[component], field), rtol=1e-13)
                np.testing.assert_allclose(longs[component], length*original_long[component], rtol=1e-13)
                np.testing.assert_allclose(total.longitudinal(f, component=component),
                                           parts[component].longitudinal, rtol=1e-13)
                np.testing.assert_allclose(total.transverse(f, component=component).dipolar,
                                           parts[component].dipolar, rtol=1e-13)
                self.assertEqual(parts[component].pipe_length_m, length)
                self.assertEqual(parts[component].reference_position_m, original[component].reference_position_m)
                self.assertEqual(parts[component].field_convention, original[component].field_convention)
            np.testing.assert_allclose(parts['wall'].quadrupolar,
                parts['resistive_wall'].quadrupolar+parts['indirect_space_charge'].quadrupolar)
            self.assertEqual(total.longitudinal_units, 'Ohm')
            self.assertEqual(total.transverse_coefficient_units, 'Ohm/m')
            np.testing.assert_array_equal(total.backend.panels.panel_start, per_length.backend.panels.panel_start)
        self.assertIsNone(original['wall'].pipe_length_m)
        self.assertEqual(per_length.longitudinal_units, 'Ohm/m')
        self.assertEqual(per_length.transverse_coefficient_units, 'Ohm/m2')
        converted = original['wall'].for_pipe_length(4.5)
        with self.assertRaisesRegex(ValueError, 'already includes'):
            converted.for_pipe_length(4.5)

    def test_runtime_default_is_one_metre_with_total_units(self):
        case = geometry(); f=np.array([10., 1e6])
        default = case.make_solver()
        self.assertEqual(default.pipe_length_m, 1.0)
        self.assertEqual(default.longitudinal_units, 'Ohm')
        self.assertEqual(default.transverse_coefficient_units, 'Ohm/m')
        response = default.transverse(f)
        per_length = case.make_solver(pipe_length_m=None).transverse(f)
        self.assertEqual(response.pipe_length_m, 1.0)
        for field in ('longitudinal','dipolar','quadrupolar','constant'):
            np.testing.assert_array_equal(getattr(response,field),getattr(per_length,field))
        data=case.to_dict(); data['beta']=1.
        self.assertIsNone(load_geometry(data).make_solver('pec').pipe_length_m)

    def test_invalid_lengths_fail_and_pec_has_no_length_output(self):
        case = geometry()
        for length in [0., -1., float('nan'), float('inf'), True, '2', [2.]]:
            with self.assertRaisesRegex(ValueError, 'pipe_length_m'):
                case.make_solver(pipe_length_m=length)
        data = case.to_dict(); data['beta'] = 1.
        with self.assertRaisesRegex(ValueError, 'PEC geometry helper'):
            load_geometry(data).make_solver('pec', pipe_length_m=2.)

    def test_plot_units_follow_integrated_values_without_rescaling(self):
        f = np.array([10., 1e6])
        response = geometry().make_solver(pipe_length_m=3.).transverse(f)
        long = plot_longitudinal(f, response.longitudinal, per_unit_length=False)
        transverse = plot_transverse(f, response)
        try:
            self.assertEqual(long.axes[1].get_ylabel(), 'Im(Z∥) [Ω]')
            np.testing.assert_array_equal(long.axes[1].lines[0].get_ydata(), response.longitudinal.imag)
            self.assertTrue(all(axis.get_ylabel() == 'Z⊥ [Ω/m]' for axis in transverse.axes))
        finally:
            plt.close(long); plt.close(transverse)

    def test_cli_csv_values_units_and_saved_geometry(self):
        run = runpy.run_path(str(ROOT/'examples/run_case.py'))['main']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'pipe.json'; geometry().save(path)
            for label, length in [('per_length', None), ('total', 3.75), ('default_1m', 1.)]:
                args = [str(path), '--f-max', '1e6', '--points', '2', '--panels', '48',
                        '--output', str(root/label)]
                if length is None:
                    args.append('--per-unit-length')
                elif label != 'default_1m':
                    args.extend(['--pipe-length-m', str(length)])
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
            for filename in ['impedance.csv', 'transverse.csv']:
                first = np.loadtxt(root/'per_length'/filename, delimiter=',', skiprows=1)
                second = np.loadtxt(root/'total'/filename, delimiter=',', skiprows=1)
                np.testing.assert_array_equal(first[:, 0], second[:, 0])
                np.testing.assert_allclose(second[:, 1:], 3.75*first[:, 1:], rtol=1e-13)
                np.testing.assert_array_equal(np.loadtxt(root/'default_1m'/filename,delimiter=',',skiprows=1),first)
            self.assertIn('Re_Z_parallel_Ohm,', (root/'total/impedance.csv').read_text().splitlines()[0])
            self.assertIn('Re_dipolar_xx_Ohm_per_m,', (root/'total/transverse.csv').read_text().splitlines()[0])
            summary = json.loads((root/'total/summary.json').read_text())
            self.assertEqual(summary['pipe_length_m'], 3.75)
            self.assertEqual(summary['normalization'], 'total_uniform_pipe')
            self.assertEqual(summary['transverse_constant_units'], 'Ohm')
            self.assertFalse(summary['entrance_exit_effects_included'])
            self.assertNotIn('pipe_length_m', summary['geometry'])
            default_summary=json.loads((root/'default_1m/summary.json').read_text())
            self.assertEqual(default_summary['pipe_length_m'],1.0)
            self.assertEqual(default_summary['normalization'],'total_uniform_pipe')
            self.assertEqual(default_summary['longitudinal_impedance_units'],'Ohm')
            self.assertEqual(default_summary['dipolar_quadrupolar_units'],'Ohm/m')


if __name__ == '__main__':
    unittest.main()
