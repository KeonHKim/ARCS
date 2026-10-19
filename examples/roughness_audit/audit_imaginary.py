"""Audit the imaginary RW impedance and Gaussian-tail truncation.

The BEM values are the unchanged saved ARCS v0.4.0 benchmark. No BEM run,
reference fitting, smoothing, or rescaling is performed by this script.
The added reference integrates the synchronous circular Maxwell equations
with continuous conductivity, including displacement current and curvature.
Run: python audit_imaginary.py
Requirements: numpy, scipy, matplotlib.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.constants import c as C0, mu_0 as MU0, epsilon_0 as EPS0
from scipy.integrate import solve_ivp
from scipy.special import kve, ndtr

SIGMA = 5.8e7
RQ = 0.3e-6
APERTURE = 0.03
MEAN_RADIUS = APERTURE + 5 * RQ
EXTENTS = np.arange(4.0, 11.0)


def vacuum_reaction(frequency, radius, surface_impedance):
    """Exact beta=1 circular monopole reaction, in ohm/metre."""
    omega = 2 * np.pi * frequency
    zs = np.asarray(surface_impedance)
    return zs / (2 * np.pi * radius) / (1 - (-1j) * omega * EPS0 * zs * radius / 2)


def homogeneous_surface(frequency, radius):
    omega = 2 * np.pi * frequency
    y = SIGMA - (-1j) * omega * EPS0
    q = np.sqrt(-(-1j) * omega * MU0 * SIGMA)
    return q / y * kve(0, q * radius) / kve(1, q * radius)


def continuous_gradient(frequency, extents=EXTENTS, tighter=False):
    """Continuous circular Maxwell reference at fixed mean wall position.

    Fields use exp(-i*k*z + i*omega*t), k=omega/c. In the conductor,
    E_z' = (k**2/y + i*omega*mu0) H_phi,
    H_phi' = y E_z - H_phi/r, y=sigma(r)+i*omega*epsilon0.
    Thus z=-E_z/H_phi obeys z'=y*z*z+z/r-q**2/y.
    For beta=mu_r=epsilon_r=1, q**2=+i*omega*mu0*sigma;
    the algebraically cancelled form avoids subtracting k**2 from itself.
    The exterior starts at +10 Rq, where Gaussian conductivity equals
    the bulk value to double precision; its decaying K0/K1 boundary is exact.
    Each returned inner plane truncates the SAME profile at -extent*Rq.
    """
    omega = 2 * np.pi * frequency
    outer_radius = MEAN_RADIUS + 10 * RQ
    inner_radii = MEAN_RADIUS - np.asarray(extents) * RQ

    def rhs(radius, value):
        sigma = SIGMA * ndtr((radius - MEAN_RADIUS) / RQ)
        y = sigma - (-1j) * omega * EPS0
        q2 = -(-1j) * omega * MU0 * sigma
        return y * value**2 + value / radius - q2 / y

    solution = solve_ivp(
        rhs, (outer_radius, float(inner_radii.min())),
        [homogeneous_surface(frequency, outer_radius)],
        method="DOP853", dense_output=True,
        rtol=1e-12 if tighter else 1e-10,
        atol=1e-15 if tighter else 1e-13,
        max_step=RQ / (16 if tighter else 8),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    surface = solution.sol(inner_radii)[0]
    return vacuum_reaction(frequency, inner_radii, surface)


def verify_homogeneous():
    """Check the radial ODE against independent analytical K0/K1 fields."""
    errors = []
    for frequency in (10.0, 1e7, 1e9, 1e11):
        omega = 2 * np.pi * frequency
        inner, outer = APERTURE, APERTURE + 20 * RQ
        y = SIGMA - (-1j) * omega * EPS0
        q2 = -(-1j) * omega * MU0 * SIGMA
        solution = solve_ivp(
            lambda radius, z: y * z**2 + z / radius - q2 / y,
            (outer, inner), [homogeneous_surface(frequency, outer)],
            method="DOP853", rtol=1e-11, atol=1e-14, max_step=RQ / 8,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        errors.append(abs(solution.y[0, -1] / homogeneous_surface(frequency, inner) - 1))
    maximum = float(max(errors))
    if maximum > 1e-7:
        raise AssertionError(f"Homogeneous circular check failed: {maximum}")
    return maximum


def verify_vacuum_plane():
    """A pure vacuum annulus must preserve the beta=1 beam impedance."""
    errors = []
    for frequency in (10.0, 1e7, 1e9, 1e11):
        omega = 2 * np.pi * frequency
        zs_outer = homogeneous_surface(frequency, MEAN_RADIUS)
        solution = solve_ivp(
            lambda radius, z: (-(-1j) * omega * EPS0) * z**2 + z / radius,
            (MEAN_RADIUS, APERTURE), [zs_outer],
            method="DOP853", rtol=1e-11, atol=1e-14,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        inner = vacuum_reaction(frequency, APERTURE, solution.y[0, -1])
        outer = vacuum_reaction(frequency, MEAN_RADIUS, zs_outer)
        errors.append(abs(inner / outer - 1))
    maximum = float(max(errors))
    if maximum > 1e-7:
        raise AssertionError(f"Vacuum reference-plane invariance failed: {maximum}")
    return maximum


def run(csv_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    saved = np.genfromtxt(csv_path, delimiter=",", names=True)
    frequency = saved["f_Hz"]
    bem = saved["Re_BEM_Ohm_m"] + (-1j) * saved["Im_BEM_Ohm_m"]
    layers32 = saved["Re_circular_layers"] + (-1j) * saved["Im_circular_layers"]
    supplied = saved["Re_supplied_ODE"] + (-1j) * saved["Im_supplied_ODE"]
    smooth = np.array([vacuum_reaction(f, MEAN_RADIUS, homogeneous_surface(f, MEAN_RADIUS))
                       for f in frequency])
    gradient = np.array([continuous_gradient(f) for f in frequency])
    gradient5 = gradient[:, 1]
    gradient8 = gradient[:, 4]
    gradient10 = gradient[:, -1]
    xbem, x5, x8, x10, xsmooth = [z.imag for z in (bem, gradient5, gradient8, gradient10, smooth)]

    # Component errors use Im(Z) denominators, never the total |Z|.
    field_error = 100 * (bem.imag / layers32.imag - 1)
    profile_error = 100 * (layers32.imag / gradient5.imag - 1)
    cutoff_error = 100 * (gradient5.imag / gradient10.imag - 1)
    supplied_error = 100 * (bem.imag / supplied.imag - 1)
    increment_error = 100 * ((xbem - xsmooth) / (x10 - xsmooth) - 1)
    rows = np.column_stack((frequency, bem.real, bem.imag, layers32.imag, supplied.imag,
                           gradient5.real, gradient5.imag, gradient8.imag, gradient10.real,
                           gradient10.imag, smooth.real, smooth.imag, xbem - xsmooth,
                           x10 - xsmooth, field_error, profile_error, cutoff_error,
                           supplied_error, increment_error))
    np.savetxt(output / "imaginary_audit.csv", rows, delimiter=",", comments="", header=
        "f_Hz,Re_BEM,Im_BEM,Im_circular_32,Im_supplied_ODE,Re_continuous_cut5,"
        "Im_continuous_cut5,Im_continuous_cut8,Re_continuous_cut10,Im_continuous_cut10,"
        "Re_smooth_mean_wall,Im_smooth_mean_wall,delta_Im_BEM,delta_Im_cut10,"
        "Im_field_error_percent,Im_profile32_error_percent,Im_cut5_error_percent,"
        "Im_BEM_to_supplied_error_percent,Im_increment_BEM_to_cut10_error_percent")

    # Extend the cutoff study to 10 Hz without claiming new low-frequency BEM runs.
    cutoff_frequencies = np.unique(np.r_[10., 100., 1e3, 1e4, 1e6, frequency])
    cutoff_values = np.array([continuous_gradient(f) for f in cutoff_frequencies])
    np.savetxt(output / "cutoff_convergence.csv",
               np.array([[f, extent, z.real, z.imag,
                          100 * abs(z.imag / values[-1].imag - 1)]
                         for f, values in zip(cutoff_frequencies, cutoff_values)
                         for extent, z in zip(EXTENTS, values)]),
               delimiter=",", comments="", header="f_Hz,extent_Rq,Re_Z_Ohm_m,Im_Z_Ohm_m,Im_error_to_cut10_percent")

    tighter = np.array([continuous_gradient(f, tighter=True) for f in (10., 1e9, 1e11)])
    baseline = np.array([continuous_gradient(f) for f in (10., 1e9, 1e11)])
    integration_error = float(100 * np.max(abs(tighter.imag / baseline.imag - 1)))
    summary = {
        "scope": "Longitudinal imaginary impedance per length; beta=1, circular copper, continuous Gaussian conductivity model",
        "BEM_data": "Unchanged ARCS v0.4.0 saved results: 96 panels/interface, 32 slices, cutoff -5 Rq",
        "mean_wall_radius_m": MEAN_RADIUS, "Rq_m": RQ,
        "mean_wall_held_fixed_for_cutoff_comparison": True,
        "smooth_control": "Copper wall at the same mean radius; beta=1 vacuum-plane invariance checked",
        "new_BEM_runs": False, "reference_fitting": False,
        "max_abs_Im_BEM_to_same32_layers_percent": float(max(abs(field_error))),
        "max_abs_Im_32layers_to_continuous_cut5_percent": float(max(abs(profile_error))),
        "max_abs_Im_BEM_to_supplied_ODE_10MHz_to10GHz_percent": float(max(abs(supplied_error[frequency <= 1e10]))),
        "max_abs_Im_cut5_to_continuous_cut10_percent": float(max(abs(cutoff_error))),
        "max_abs_Im_increment_BEM_to_cut10_percent": float(max(abs(increment_error))),
        "continuous_solver_tolerance_refinement_max_Im_difference_percent": integration_error,
        "homogeneous_K0_K1_check_max_relative_error": verify_homogeneous(),
        "vacuum_plane_invariance_max_relative_error": verify_vacuum_plane(),
        "selected_frequencies": [{"f_Hz": float(frequency[i]),
            "Im_BEM_Ohm_m": float(xbem[i]),
            "Im_continuous_cut5_Ohm_m": float(x5[i]),
            "Im_continuous_cut10_Ohm_m": float(x10[i]),
            "change_cut5_to_cut8_percent_of_cut5": float(100 * (x8[i] / x5[i] - 1)),
            "change_cut5_to_cut10_percent_of_cut5": float(100 * (x10[i] / x5[i] - 1)),
            "Im_BEM_to_supplied_ODE_percent": float(supplied_error[i])}
            for i in range(len(frequency)) if frequency[i] in (1e7, 1e9, 1e10, 1e11)],
        "interpretation": "Close BEM/reference agreement at shared cutoff -5 Rq does not establish Gaussian-tail convergence. This is not experimental validation of geometric roughness.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.1), layout="constrained")
    blue, orange, black, grey = "#1768AC", "#C45B16", "#172B3A", "#929AA2"
    ax = axes[0, 0]
    ax.loglog(frequency, xsmooth, "--", color=grey, label="Smooth wall at the same mean radius")
    ax.loglog(frequency, x5, "-", color=orange, label="Continuous Maxwell gradient: cutoff −5 Rq")
    ax.loglog(frequency, x10, "-", color=black, label="Continuous Maxwell gradient: cutoff −10 Rq")
    ax.loglog(frequency, xbem, "o", color=blue, ms=5, label="Saved ARCS BEM: cutoff −5 Rq")
    ax.set(title="Imaginary longitudinal impedance", ylabel="Im(Z∥/L) [Ω/m]", xlabel="Frequency [Hz]")
    ax.legend(fontsize=7.5)

    ax = axes[0, 1]
    ax.loglog(frequency, xbem - xsmooth, "o--", color=blue, ms=4, label="Saved BEM, cutoff −5 Rq")
    ax.loglog(frequency, x10 - xsmooth, "-", color=black, label="Continuous gradient, cutoff −10 Rq")
    ax.set(title="Reactive increment relative to the smooth mean wall", ylabel="Δ[Im(Z∥/L)] [Ω/m]", xlabel="Frequency [Hz]")
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    ax.semilogx(frequency, field_error, "o-", color=blue, ms=4, label="BEM vs same 32 layers")
    ax.semilogx(frequency, profile_error, "s-", color=orange, ms=4, label="32 layers vs continuous profile, cutoff −5 Rq")
    ax.semilogx(frequency, cutoff_error, "^-", color=black, ms=4, label="Cutoff −5 Rq vs −10 Rq")
    ax.axhline(0, color=grey, lw=.6)
    ax.set(title="Imaginary-part errors have different causes", ylabel="Signed error in Im(Z) [%]", xlabel="Frequency [Hz]")
    ax.legend(fontsize=7.3)

    ax = axes[1, 1]
    for f, color in zip((10., 1e7, 1e9, 1e11), (grey, blue, orange, black)):
        index = int(np.flatnonzero(cutoff_frequencies == f)[0])
        error = 100 * abs(cutoff_values[index].imag / cutoff_values[index, -1].imag - 1)
        error[error == 0] = np.nan  # zero error cannot be displayed on a logarithmic axis
        ax.semilogy(EXTENTS[:-1], error[:-1], "o-", color=color, ms=4, label=f"{f:g} Hz")
    ax.set(title="Air-side cutoff convergence; mean wall fixed", xlabel="Air-side extent [Rq]", ylabel="Absolute Im error relative to −10 Rq [%]", ylim=(1e-9, 100))
    ax.set_xticks(EXTENTS[:-1])
    ax.legend(fontsize=7.5)
    for ax in axes.flat:
        ax.grid(True, which="both", alpha=.18)
    fig.suptitle("ARCS roughness audit: prioritize Im(Z) and the reactive increment\n"
                 "β = 1 · copper · Rq = 0.3 μm · fixed mean radius 30.0015 mm", fontsize=12.5)
    fig.savefig(output / "ARCS_Imaginary_Roughness_Audit.png", dpi=180)
    fig.savefig(output / "ARCS_Imaginary_Roughness_Audit.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--csv", type=Path, default=here / "saved_arcs_spectrum.csv")
    parser.add_argument("--output", type=Path, default=here / "results")
    options = parser.parse_args()
    run(options.csv, options.output)
