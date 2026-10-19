"""Load an ARCS case, solve finite-beta Maxwell impedance, and save raw data/plots.

This tutorial runner does not change the physical model, fit a reference,
clip negative values, or reconstruct a wake from a limited frequency band.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arcs import load_geometry, PEC
from arcs.geometry_input import _save_figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", nargs="?", default=str(ROOT / "examples/geometries/bare_copper.json"))
    parser.add_argument("--units", choices=["m", "mm", "um", "nm"], help="Source units for bare CSV or unitless DXF")
    parser.add_argument("--f-min", type=float, default=10., help="Minimum frequency [Hz]")
    parser.add_argument("--f-max", type=float, default=1e13, help="Maximum frequency [Hz]")
    parser.add_argument("--points", type=int, default=100, help="Number of logarithmically spaced frequencies")
    parser.add_argument("--panels", type=int, default=256, help="Minimum boundary panel count")
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--pipe-length-m", type=float, default=1.0,
                        help="Uniform axial pipe length [m], default 1.0; outputs total impedance.")
    normalization.add_argument("--per-unit-length", dest="pipe_length_m", action="store_const", const=None,
                        default=argparse.SUPPRESS, help="Explicitly return per-metre impedance instead of pipe totals.")
    parser.add_argument("--conductivity", type=float, help="Explicit exterior conductivity override [S/m]")
    parser.add_argument("--beta", type=float, help="Override beta=v/c, 0 < beta <= 1")
    parser.add_argument("--component", choices=['wall','resistive_wall','indirect_space_charge'], default='wall',
                        help="Wall (default) includes RW and indirect space charge; direct SC is excluded")
    parser.add_argument("--roughness-rms-um", type=float, help="Enable gradient roughness with RMS height in micrometres")
    parser.add_argument("--roughness-slices", type=int, default=None, help="BEM material slices per Gaussian transition")
    parser.add_argument("--roughness-extent", type=float, default=None, help="Gaussian truncation extent in RMS heights; new profiles default to 5")
    parser.add_argument("--longitudinal-only", action="store_true", help="Skip transverse tensors")
    parser.add_argument("--output", default="results/my_case", help="Output directory")
    args = parser.parse_args(argv)
    try:
        if not np.isfinite([args.f_min, args.f_max]).all() or not 10 <= args.f_min < args.f_max:
            raise ValueError("Use finite frequencies with 10 Hz <= f-min < f-max. Refine the mesh for your geometry and material.")
        if args.points < 2 or args.panels < 16:
            raise ValueError("Use at least 2 frequencies and 16 minimum panels.")
        case = load_geometry(args.geometry, units=args.units)
        # Apply the override to the case itself so the preview and summary use
        # exactly the same panel construction as the numerical solve.
        definition = case.to_dict()
        definition["mesh"]["minimum_panels"] = args.panels
        if args.beta is not None:
            definition["beta"] = args.beta
        if args.roughness_rms_um is not None:
            definition["roughness"] = {"rms_height":args.roughness_rms_um*1e-6/case.unit_scale,
                "slices":args.roughness_slices if args.roughness_slices is not None else 32,
                "extent_sigma":args.roughness_extent if args.roughness_extent is not None else 5., "model":"conductivity_gradient"}
        if args.roughness_slices is not None:
            if definition.get("roughness") is None:
                raise ValueError("--roughness-slices needs a roughness case or --roughness-rms-um.")
            definition["roughness"]["slices"] = args.roughness_slices
        if args.roughness_extent is not None:
            if definition.get("roughness") is None:
                raise ValueError("--roughness-extent needs a roughness case or --roughness-rms-um.")
            definition["roughness"]["extent_sigma"] = args.roughness_extent
        if args.conductivity is not None:
            label = definition["regions"][-1]["material"]
            properties = definition.setdefault("materials",{}).get(label,{})
            if properties.get("kind") == "pec":
                raise ValueError("PEC has no finite conductivity; choose a conducting exterior to use --conductivity.")
            definition["materials"][label] = {**properties,"conductivity":args.conductivity}
        case = load_geometry(definition)
        wall = None
        solver = case.make_solver("maxwell", pipe_length_m=args.pipe_length_m)
        if wall is None and solver.default_wall is None:
            raise ValueError("Define exterior conductivity in the case's materials, or supply --conductivity in S/m.")
        n_panels = len(solver.backend.panels.panel_start)
        frequencies = np.geomspace(args.f_min, args.f_max, args.points)
        print(f"Case: {case.name}; {n_panels} panels; {len(frequencies)} frequencies.", flush=True)
        print(f"Output: {args.component}, beta={case.beta:g}, exp(+i omega t); direct SC excluded.", flush=True)
        print("Normalization: " + ("per unit pipe length" if args.pipe_length_m is None
              else f"total for {args.pipe_length_m:g} m of uniform pipe") + ".", flush=True)
        impedance = np.empty(len(frequencies), complex)
        dipolar = np.empty((len(frequencies), 2, 2), complex)
        quadrupolar = np.empty_like(dipolar)
        constant = np.empty((len(frequencies), 2), complex)
        for index, frequency in enumerate(frequencies):
            if args.longitudinal_only:
                impedance[index] = solver.longitudinal(float(frequency), wall, component=args.component).item()
            else:
                response = solver.transverse(float(frequency), wall, component=args.component)
                impedance[index] = response.longitudinal.item()
                dipolar[index], quadrupolar[index], constant[index] = response.dipolar, response.quadrupolar, response.constant
            if index == 0 or index == len(frequencies)-1 or (index+1) % max(1, len(frequencies)//5) == 0:
                print(f"  {index+1}/{len(frequencies)} solved: {frequency:g} Hz", flush=True)
        if not np.isfinite(impedance).all():
            raise RuntimeError("The numerical solve produced nonfinite impedance values.")
        if not args.longitudinal_only and not all(np.isfinite(values).all() for values in (dipolar, quadrupolar, constant)):
            raise RuntimeError("The numerical solve produced nonfinite transverse values.")
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)
        long_suffix = "Ohm_per_m" if args.pipe_length_m is None else "Ohm"
        tensor_suffix = "Ohm_per_m2" if args.pipe_length_m is None else "Ohm_per_m"
        np.savetxt(destination / "impedance.csv", np.column_stack((frequencies, impedance.real, impedance.imag)),
                   delimiter=",", comments="",
                   header=f"frequency_Hz,Re_Z_parallel_{long_suffix},Im_Z_parallel_{long_suffix}")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from arcs.plotting import plot_longitudinal, plot_transverse
        figure = plot_longitudinal(frequencies, impedance,
                  per_unit_length=args.pipe_length_m is None,
                  title=f"ARCS · {case.name}\nFull-Maxwell longitudinal impedance · {n_panels} panels")
        _save_figure(figure, destination / "impedance.png", dpi=180)
        plt.close(figure)
        if not args.longitudinal_only:
            from arcs import TransverseImpedance
            response = TransverseImpedance(impedance, dipolar, quadrupolar, constant, case.beam_position, case.beta,
                                          args.component, pipe_length_m=solver.pipe_length_m)
            columns, labels = [frequencies], ["frequency_Hz"]
            for tensor, kind in ((dipolar, "dipolar"), (quadrupolar, "quadrupolar")):
                for i, first in enumerate("xy"):
                    for j, second in enumerate("xy"):
                        z = tensor[:, i, j]
                        columns.extend((z.real, z.imag))
                        labels.extend((f"Re_{kind}_{first}{second}_{tensor_suffix}", f"Im_{kind}_{first}{second}_{tensor_suffix}"))
            for i, name in enumerate("xy"):
                columns.extend((constant[:, i].real, constant[:, i].imag))
                labels.extend((f"Re_constant_{name}_{long_suffix}", f"Im_constant_{name}_{long_suffix}"))
            np.savetxt(destination / "transverse.csv", np.column_stack(columns), delimiter=",", comments="", header=",".join(labels))
            figure = plot_transverse(frequencies, response,
                      title=f"ARCS · {case.name} · {n_panels} panels")
            _save_figure(figure, destination / "transverse.png", dpi=180)
            plt.close(figure)
        preview = case.preview(destination / "geometry.png", show_mesh=True)
        plt.close(preview)
        metal = wall if wall is not None else solver.default_wall
        summary = {"case": case.name, "model": "synchronous full-vector Maxwell boundary elements",
                   "beta": case.beta, "longitudinal_impedance_units": solver.longitudinal_units, "panels": n_panels,
                   "pipe_length_m": solver.pipe_length_m,
                   "normalization": "per_unit_length" if solver.pipe_length_m is None else "total_uniform_pipe",
                   "frequency_samples": len(frequencies), "f_min_Hz": args.f_min, "f_max_Hz": args.f_max,
                   "conductivity_S_per_m": (None if isinstance(getattr(metal,"substrate",metal),PEC) else
                       (metal.substrate.conductivity if hasattr(metal,"substrate") else metal.conductivity)),
                   "outer_termination": "PEC" if isinstance(getattr(metal,"substrate",metal),PEC) else "infinite_conductor",
                   "transverse_calculated": not args.longitudinal_only,
                   "dipolar_quadrupolar_units": solver.transverse_coefficient_units,
                   "transverse_constant_units": solver.constant_units,
                   "entrance_exit_effects_included": False,
                   "transverse_method": "Analytic source/witness derivatives of Maxwell Ez with Panofsky-Wenzel",
                   "beam_position_m": list(case.beam_position), "geometry": case.to_dict(),
                   "full_vector_Maxwell": True, "coating_interfaces": len(case.boundaries)>1, "finite_outer_wall": False,
                   "impedance_component": args.component,
                   "impedance_reference": "PEC pipe" if args.component == 'resistive_wall' else "free space; direct space charge excluded",
                   "field_convention": "exp(-i*k*z+i*omega*t)", "roughness":case.to_dict().get("roughness"),
                   "automatic_accuracy_certification": False,
                   "plot_note": "Plots and CSV use signed Im directly. No fitting or clipping."}
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved longitudinal data/plots, geometry and summary in {destination}.")
        if not args.longitudinal_only:
            print("Saved transverse.csv (full tensors) and transverse.png (x/y dipolar and quadrupolar).")
    except (ValueError, RuntimeError, OSError, NotImplementedError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
