ARCS imaginary-part roughness audit
==================================

This standalone audit reuses the saved ARCS v0.4.0 longitudinal BEM results.
It does not change the ARCS solver or run a new boundary-element simulation.
The original spectrum is included as saved_arcs_spectrum.csv.

Install and run:
    python -m pip install numpy scipy matplotlib
    python audit_imaginary.py

Outputs: results/ARCS_Imaginary_Roughness_Audit.png and .pdf, two CSV tables,
and summary.json. Impedances are per unit length in ohm/metre. Positive
plotted values are -Im(Z), with the exp(i*k*z-i*omega*t) convention.

The new independent reference integrates continuous, circular Maxwell fields
for beta=1, mu_r=epsilon_r=1, sigma_bulk=5.8e7 S/m, and Rq=0.3 micrometre.
It retains the synchronous longitudinal wavenumber and displacement current.
For z=-E_z/H_phi, the radial Riccati equation is
    z' = y*z^2 + z/r - q^2/y,
    y = sigma(r)-i*omega*epsilon0,
    q^2 = -i*omega*mu0*sigma(r),
    sigma(r) = sigma_bulk*Phi((r-r_mean)/Rq).
Here q^2 is the algebraically cancelled full-Maxwell expression for this
specific beta=1 material case. It is not a magnetoquasistatic approximation.
The outer boundary is the decaying cylindrical K0/K1 solution at +10 Rq.

For the cutoff comparison, the mean wall radius is held fixed at 30.0015 mm.
Moving the air-side cutoff changes the inner radius to r_mean-extent*Rq.
The exact circular vacuum reaction is evaluated at that radius. The script
checks invariance under a pure vacuum annulus and compares a homogeneous
Maxwell integration with the independent analytical K0/K1 solution. It also
checks the continuous-gradient solution with tighter integration tolerances.

Three distinct errors are reported:
    BEM versus the same 32 circular layers: field/geometry discretization.
    32 circular layers versus a continuous profile, both cut at -5 Rq:
        material-profile discretization.
    Continuous profiles cut at -5 Rq versus -10 Rq, fixed mean wall:
        truncation of the low-conductivity Gaussian tail.
All error denominators use the imaginary component, not total |Z|.
The reactive increment is relative to a smooth wall at the SAME mean radius.

Finding: the old -5 Rq cutoff is not converged in the mid-frequency band.
Increasing the air-side extent changes |Im(Z)| by about 12.0% at 1 GHz,
10.9% at 10 GHz, and 5.26% at 100 GHz, relative to the -5 Rq values.
The close BEM-versus-supplied-reference agreement used the same -5 Rq cutoff
and therefore could not detect this shared truncation issue.

The endpoint conductivity at -5 Rq is about 16.6 S/m. Although tiny relative
to copper, it exceeds omega*epsilon0 (about 0.556 S/m at 10 GHz). Within this
model it is therefore insufficient to judge tail truncation by Gaussian
probability alone; reactive impedance convergence must be checked directly.

The cutoff study includes 10 Hz through 100 GHz. Those additional low-frequency
values come from the continuous circular reference, not new BEM simulations.
They validate the selected conductivity-gradient equations and convergence
checks; they do not validate the homogenized model against a real geometric
rough surface. The source contains no fitting or empirical correction.
