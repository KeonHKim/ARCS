"""Module entry point: python -m arcs.geometry_cli input.json --preview mesh.png."""
from .geometry_input import main

if __name__ == "__main__":
    raise SystemExit(main())
