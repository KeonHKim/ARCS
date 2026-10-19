"""Progress and raw-output helpers for the 100-point ARCS notebook sweeps."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import time
import numpy as np
from arcs import TransverseImpedance


_WORKER_SOLVER = None
_WORKER_WALL = None


def _initialize_worker(solver, wall):
    global _WORKER_SOLVER, _WORKER_WALL
    _WORKER_SOLVER, _WORKER_WALL = solver, wall


def _solve_frequency(frequency):
    return _WORKER_SOLVER.transverse(float(frequency), _WORKER_WALL)


def sweep(solver, frequency, wall=None, *, label="ARCS", output=None, workers=None):
    """Solve every requested frequency, with no fitting or interpolation."""
    frequency = np.asarray(frequency, float)
    values = []
    start = time.perf_counter()
    workers = int(os.environ.get("ARCS_SWEEP_WORKERS", "1")) if workers is None else int(workers)
    if workers < 1:
        raise ValueError("Use at least one sweep worker.")
    pool = (ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker,
                               initargs=(solver, wall), mp_context=multiprocessing.get_context("spawn"))
            if workers > 1 else None)
    try:
        calculated = (pool.map(_solve_frequency, frequency) if pool is not None else
                      (solver.transverse(float(f), wall) for f in frequency))
        for index, (f, value) in enumerate(zip(frequency, calculated)):
            for name in ("longitudinal", "dipolar", "quadrupolar", "constant"):
                if not np.all(np.isfinite(getattr(value, name))):
                    raise FloatingPointError(f"{label}: nonfinite {name} at {f:g} Hz")
            values.append(value)
            if index == 0 or index == len(frequency)-1 or (index+1) % 10 == 0:
                print(f"{label}: {index+1}/{len(frequency)} · {f:.5g} Hz · "
                      f"{time.perf_counter()-start:.1f} s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    result = TransverseImpedance(
        np.array([value.longitudinal for value in values]),
        np.array([value.dipolar for value in values]),
        np.array([value.quadrupolar for value in values]),
        np.array([value.constant for value in values]),
        values[0].reference_position_m, values[0].beam_beta,
    )
    if output is not None:
        save_spectrum(output, frequency, result)
    return result


def save_spectrum(path, frequency, result):
    """Export signed real/imaginary longitudinal and all transverse entries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [frequency, result.longitudinal.real, result.longitudinal.imag]
    names = ["f_Hz", "Re_long_Ohm_m", "Im_long_Ohm_m"]
    for tensor, label in ((result.dipolar, "dipolar"), (result.quadrupolar, "quadrupolar")):
        for i, first in enumerate("xy"):
            for j, second in enumerate("xy"):
                columns.extend([tensor[:, i, j].real, tensor[:, i, j].imag])
                names.extend([f"Re_{label}_{first}{second}_Ohm_m2", f"Im_{label}_{first}{second}_Ohm_m2"])
    for i, axis in enumerate("xy"):
        columns.extend([result.constant[:, i].real, result.constant[:, i].imag])
        names.extend([f"Re_constant_{axis}_Ohm_m", f"Im_constant_{axis}_Ohm_m"])
    np.savetxt(path, np.column_stack(columns), delimiter=",", comments="", header=",".join(names))


def sweep_components(solver, frequency, wall=None, *, label="ARCS components", output=None):
    """One field solve per frequency; retain RW and indirect SC separately."""
    frequency = np.asarray(frequency, float)
    values = {name: [] for name in ('wall','resistive_wall','indirect_space_charge')}
    start = time.perf_counter()
    for index, f in enumerate(frequency):
        parts = solver.transverse_components(float(f), wall)
        for name, value in parts.items():
            if not all(np.all(np.isfinite(getattr(value, field)))
                       for field in ('longitudinal','dipolar','quadrupolar','constant')):
                raise FloatingPointError(f"{label}: nonfinite {name} at {f:g} Hz")
            values[name].append(value)
        if index == 0 or index == len(frequency)-1 or (index+1) % 10 == 0:
            print(f"{label}: {index+1}/{len(frequency)} · {f:.5g} Hz · {time.perf_counter()-start:.1f} s", flush=True)
    result = {}
    for name, samples in values.items():
        result[name] = TransverseImpedance(
            *[np.array([getattr(sample, field) for sample in samples])
              for field in ('longitudinal','dipolar','quadrupolar','constant')],
            samples[0].reference_position_m, samples[0].beam_beta, name)
        if output is not None:
            save_spectrum(Path(output).with_name(Path(output).stem+'_'+name+'.csv'), frequency, result[name])
    return result
