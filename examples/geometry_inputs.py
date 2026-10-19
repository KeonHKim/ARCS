"""Preview all supported geometry entry routes without running field physics."""
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arcs import load_geometry
from arcs.geometry_input import _save_figure


def run(output_dir=None):
    output = Path(output_dir or ROOT / "results")
    output.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "examples" / "geometries"
    selected = ["circle", "ellipse", "sharp_L", "rounded_rectangle",
                "line_arc_path", "nonuniform_coating"]
    titles = ["Circle: radius 30 mm", "Ellipse: half axes 60/30 mm",
              "L-shape: sharp reentrant corner", "Rounded rectangle: 8 mm radius",
              "Native line/arc path", "Varying coating thickness\nTwo independent outlines"]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.8), constrained_layout=True)
    for axes, name, title in zip(axes.ravel(), selected, titles):
        load_geometry(folder / (name+".json")).draw(axes)
        axes.set_title(title, fontsize=11)
    fig.suptitle("ARCS geometry input: dimensions, ordered points and independent interfaces\n"
                 "Coordinates shown in mm; line panels and node markers; geometry preview only", fontsize=13)
    _save_figure(fig, output / "geometry_input_examples.png")
    plt.close(fig)
    reports = {}
    for path in sorted(folder.glob("*.json")):
        geometry = load_geometry(path)
        reports[path.stem] = geometry.report()
        geometry.save(output / (path.stem+"_portable_geometry.json"))
        geometry.export_panels(output / (path.stem+"_geometry_panels.csv"))
        print(f"{path.stem}: {len(geometry.boundaries)} boundary/boundaries; "
              + ", ".join(f"{b['name']}={b['panels']} panels" for b in reports[path.stem]["boundaries"]))
    (output / "geometry_input_summary.json").write_text(json.dumps(reports, indent=2)+"\n")
    csv_geometry = load_geometry(folder / "csv_outline.json")
    plt.close(csv_geometry.preview(output / "geometry_csv_preview.png"))
    return reports


if __name__ == "__main__":
    run()
