"""Package the current checkout, with SHA-256 manifest and validation checks."""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", "build", ".venv", ".git"}


def include(path):
    parts = path.relative_to(ROOT).parts
    return (path.is_file() and not any(p in SKIP_PARTS or p.endswith(".egg-info") for p in parts)
            and path.suffix not in {".pyc", ".pyo"} and path.name != "manifest.sha256")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Destination ZIP; default is ARCS_v<version>.zip')
    args = parser.parse_args()
    # A release must contain completed numerical/UI/notebook verification.
    validation = json.loads((ROOT / "VALIDATION.json").read_text())
    assert validation["tests"]["successful"]
    assert validation["notebook"]["errors"] == 0
    assert validation["notebook"]["executed_code_cells"] >= 15
    assert validation["notebook"]["embedded_png_plots"] >= 25
    version = validation["version"]
    assert validation["tests"]["version"] == version
    assert validation["isolated_wheel_install"]["version"] == version
    assert validation["materials_GUI"]["version"] == version
    assert all(validation["numerical_modules_identical_to_v0_6_4"].values())
    for name in ("retained_workflow_notebook", "retained_PEC_notebook"):
        assert validation[name]["executed_version"] == "0.6.4"
        assert validation[name]["errors"] == 0
        assert validation[name]["embedded_png_plots"] >= 3
    files = sorted(p for p in ROOT.rglob("*") if include(p))
    manifest = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n"
                       for p in files)
    (ROOT / "manifest.sha256").write_text(manifest)
    destination = args.output.resolve() if args.output else ROOT.parent / f"ARCS_v{version}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Expose the destination only after a complete archive passes CRC and hash
    # verification. A reader must never receive an in-progress ZIP.
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='arcs_package_', suffix='.tmp', delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files + [ROOT / "manifest.sha256"]:
                archive.write(path, arcname=(Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
        with zipfile.ZipFile(staging) as archive:
            assert archive.testzip() is None
            for line in manifest.splitlines():
                expected, relative = line.split("  ", 1)
                assert hashlib.sha256(archive.read(f"{ROOT.name}/{relative}")).hexdigest() == expected
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)
    print(json.dumps({"archive": str(destination), "bytes": destination.stat().st_size,
                      "files": len(files) + 1, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
