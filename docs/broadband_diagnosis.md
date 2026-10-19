# Broadband Maxwell regression

ARCS v0.5.0 keeps the full vacuum transverse-field reaction restored in v0.3.0. The current β=1 bare-copper circular benchmark spans 100 logarithmically spaced frequencies from 10 Hz to 10 THz and compares independent m=0 and m=1 Maxwell solutions. It includes the THz turnover and the signed imaginary crossing.

The former scalar MQS backend and its diagnostic curves are removed. No backend selection changes the physical model automatically. Finite beta adds the vacuum modified-Helmholtz scale κ=ω/(βγc); it does not remove the synchronous Maxwell coupling.

Run `python examples/benchmark.py`. Signed data and reference errors are in `results/circular_benchmark/`; `restored_response.png` includes a THz close-up. Mesh and material-model assumptions accompany every result. This circular audit is not a guarantee for arbitrary shapes or realistic frequency-dependent metal conductivity.
