"""Reproducible editor geometry, native DXF and crop examples; no EM solve."""
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from arcs import GeometryDocument, import_geometry, export_dxf
from arcs.geometry_editor import parse_coordinates


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/"results"
    examples=root/"examples"/"geometries"
    document=GeometryDocument()
    data=document.data
    data["name"]="Finite copper wall with air exterior"
    document.commit(data)
    document.add_interface({"type":"circle","radius":32},name="wall_outer",exterior="air")
    document.save(examples/"finite_copper_air.json")
    export_dxf(document.geometry,examples/"finite_copper_air.dxf")
    restored=import_geometry(examples/"finite_copper_air.dxf")
    error=max(abs(a.area/b.area-1) for (_,a),(_,b) in
              zip(restored.geometry.boundaries,document.geometry.boundaries))
    # A sharp external notch is representable as one connected aperture.
    notch=GeometryDocument()
    notch.set_shape(0,{"type":"rectangle","width":100,"height":60})
    notch.set_shape(0,{"type":"rectangle","width":40,"height":40,"center":[45,25]},
                    operation="subtract")
    data=notch.data
    data["name"]="Rectangular aperture with sharp corner notch"
    notch.commit(data)
    notch.save(examples/"cropped_notch.json")
    export_dxf(notch.geometry,examples/"cropped_notch.dxf")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from arcs.geometry_input import _save_figure
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    document.geometry.draw(axes[0],show_mesh=False)
    notch.geometry.draw(axes[1],show_mesh=True)
    axes[0].set_title("Finite copper wall · air exterior")
    axes[1].set_title("Subtract a rectangle · sharp corners retained")
    fig.suptitle("ARCS geometry input and editing examples · geometry only",fontsize=13)
    _save_figure(fig,output/"geometry_editing_examples.png",dpi=180)
    plt.close(fig)
    summary={"version":"0.4.0","DXF_native_circle_relative_area_round_trip_error":error,
             "finite_wall_material_order":[r["material"] for r in document.geometry.regions],
             "finite_wall_physics_available":False,
             "notch_area_mm2":notch.geometry.boundary().area/1e-6,
             "notch_expected_area_mm2":6000-25*25,
             "DXF_import_notes":list(restored.notes),
             "GUI_geometry_format":"arcs.geometry/1"}
    (output/"geometry_editor_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
