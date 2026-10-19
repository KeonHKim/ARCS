"""Import JSON, ordered XY CSV, and planar DXF into one ARCS geometry format.

DXF circle/ellipse/line/arc/bulge geometry remains native. Closed splines are
approximated with ezdxf flattening at an explicit geometric tolerance. Material
assignments from ordinary DXF are never guessed from layer names.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import copy
import json
import math
import numpy as np
from .geometry_input import load_geometry, UNIT_SCALE, _shape
from .geometry_editor import BUILTIN_MATERIALS, outline_points


@dataclass(frozen=True)
class GeometryImport:
    geometry: object
    notes: tuple


def _import_materials(exterior):
    materials = copy.deepcopy(BUILTIN_MATERIALS)
    if exterior == "Cu":
        # Explicit legacy command-line label; new GUI presets use Annealed Cu.
        materials["Cu"] = copy.deepcopy(BUILTIN_MATERIALS["Annealed Cu"])
    return materials


def _ezdxf():
    try:
        import ezdxf
    except ImportError:
        raise RuntimeError("DXF import/export requires ezdxf. Install ARCS with the cad extra.") from None
    return ezdxf


def _xy(point, factor):
    return [float(point[0])*factor, float(point[1])*factor]


def _planar(entity):
    # Do not silently project a 3D drawing into a different 2D geometry.
    extrusion = getattr(entity.dxf, "extrusion", (0., 0., 1.))
    if not np.allclose(extrusion, (0., 0., 1.), atol=1e-12, rtol=0):
        raise ValueError("DXF must lie in the XY plane with +Z normal; transform tilted entities first.")
    kind = entity.dxftype()
    points = []
    if kind == "LINE":
        points = [entity.dxf.start, entity.dxf.end]
    elif kind in ("CIRCLE", "ARC", "ELLIPSE"):
        points = [entity.dxf.center]
        if kind == "ELLIPSE":
            points.append(entity.dxf.major_axis)
    elif kind == "LWPOLYLINE":
        if abs(entity.dxf.elevation) > 1e-12:
            raise ValueError("DXF entities must lie at z=0; translate the drawing before importing.")
    elif kind == "POLYLINE":
        if not entity.is_2d_polyline:
            raise ValueError("Only 2D POLYLINE entities are supported.")
        points = [v.dxf.location for v in entity.vertices]
        points.append(entity.dxf.elevation)
    elif kind == "SPLINE":
        points = list(entity.control_points)+list(entity.fit_points)
    if any(abs(float(point[2])) > 1e-12 for point in points):
        raise ValueError("DXF entities must lie at z=0; translate the drawing before importing.")


def _segment(entity, factor):
    if entity.dxftype() == "LINE":
        return {"start": _xy(entity.dxf.start, factor),
                "segment": {"type": "line", "to": _xy(entity.dxf.end, factor)}}
    center = _xy(entity.dxf.center, factor)
    radius = float(entity.dxf.radius)*factor
    def endpoint(degrees):
        theta = math.radians(float(degrees))
        return [center[0]+radius*math.cos(theta), center[1]+radius*math.sin(theta)]
    return {"start": endpoint(entity.dxf.start_angle),
            "segment": {"type": "arc", "center": center,
                        "to": endpoint(entity.dxf.end_angle), "direction": "ccw"}}


def _reverse(segment):
    first, part = segment["start"], segment["segment"]
    result = {**part, "to": first}
    if part["type"] == "arc":
        result["direction"] = "cw" if part.get("direction", "ccw") == "ccw" else "ccw"
    return {"start": part["to"], "segment": result}


def _join(segments):
    """Join unordered line/arc pieces without inventing a closing edge."""
    segments = list(segments)
    extent = max(1., max(np.max(np.abs(s["start"])) for s in segments))
    tolerance = 512*np.finfo(float).eps*extent
    def same(a, b):
        return np.linalg.norm(np.asarray(a)-b) <= tolerance
    loops = []
    while segments:
        first = segments.pop(0)
        start, parts = first["start"], [first["segment"]]
        endpoint = parts[-1]["to"]
        while not same(endpoint, start):
            matches = []
            for index, item in enumerate(segments):
                if same(item["start"], endpoint):
                    matches.append((index, item))
                elif same(item["segment"]["to"], endpoint):
                    matches.append((index, _reverse(item)))
            if len(matches) != 1:
                reason = "open" if not matches else "branched/ambiguous"
                raise ValueError(f"DXF outline is {reason}. Join the endpoints into unambiguous closed loops.")
            index, item = matches[0]
            segments.pop(index)
            # A line can adopt the preceding endpoint within roundoff. Arc
            # centres/radii remain native; the geometry validator checks them.
            parts.append(item["segment"])
            endpoint = parts[-1]["to"]
        parts[-1]["to"] = start
        loops.append({"type": "path", "start": start, "segments": parts})
    return loops


def _case_from_outlines(outlines, units, name, exterior, notes):
    if not outlines:
        raise ValueError("The file contains no supported closed cross-section outlines.")
    scale = UNIT_SCALE[units]
    parsed = [(label, shape, _shape(shape, scale, Path.cwd())) for label, shape in outlines]
    parsed.sort(key=lambda item: item[2].area)
    names, interfaces = {"aperture"}, []
    for index, (label, shape, _) in enumerate(parsed[1:], 1):
        label = label if label and label not in names else f"interface_{index}"
        while label in names:
            label += "_outer"
        names.add(label)
        interfaces.append({"name": label, "shape": shape})
    data = {"name": name, "units": units, "aperture": parsed[0][1],
            "interfaces": interfaces, "materials": _import_materials(exterior),
            "regions": [{"name": "beam_region", "material": "vacuum"}]
            +[{"name": f"layer_{index}", "material": "unspecified"}
              for index in range(1, len(parsed))]
            +[{"name": "exterior", "material": exterior}]}
    case = load_geometry(data)  # Requires strict nesting, not merely area order.
    if not case.boundary().contains((0., 0.)):
        try:
            from shapely.geometry import Polygon
            point = Polygon(outline_points(parsed[0][1], units=units)).representative_point()
            data["beam"] = [point.x, point.y]
        except ImportError:
            lo, hi = case.boundary().bounds
            found = None
            for y in np.linspace(lo[1], hi[1], 33)[1:-1]:
                for x in np.linspace(lo[0], hi[0], 33)[1:-1]:
                    if case.boundary().contains((x, y)):
                        found = [x/scale, y/scale]
                        break
                if found:
                    break
            if found is not None:
                data["beam"] = found
        case = load_geometry(data)
        notes.append("Beam position was placed inside the imported aperture; review it before solving.")
    notes.append("Review region materials: DXF layer names describe boundaries, not physical material assignments.")
    return GeometryImport(case, tuple(notes))


def import_geometry(path, *, units=None, target_units="mm", curve_tolerance=.01,
                    exterior="unspecified"):
    """Load a case or convert a CAD cross-section.

    units supplies missing source units; declared DXF units take precedence
    and conflicting overrides are rejected. target_units and tolerance use
    the returned drawing units. No raster images or 3D CAD are projected.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if target_units not in UNIT_SCALE:
        raise ValueError("Choose target units m, mm, um or nm.")
    if not np.isfinite(curve_tolerance) or curve_tolerance <= 0:
        raise ValueError("curve_tolerance must be positive, in target units.")
    if suffix in (".json", ".csv"):
        case = load_geometry(path, units=units)
        if suffix == ".csv" and exterior != "unspecified":
            data = case.to_dict()
            data["regions"][-1]["material"] = exterior
            data["materials"] = _import_materials(exterior)
            case = load_geometry(data)
        return GeometryImport(case, (() if suffix == ".json" else
                             ("CSV contains geometry only; review exterior material and properties.",)))
    if suffix != ".dxf":
        raise ValueError("Import an ARCS JSON case, ordered XY CSV, or planar DXF. Raster images and 3D STEP/STL are not 2D outlines.")
    ezdxf = _ezdxf()
    from ezdxf.units import conversion_factor
    unit_ids = {"m": 6, "mm": 4, "um": 13, "µm": 13, "nm": 12}
    try:
        document = ezdxf.readfile(path)
    except (ezdxf.DXFError, OSError) as error:
        raise ValueError(f"Could not read DXF: {error}") from None
    source_unit = int(document.header.get("$INSUNITS", 0))
    notes = []
    if source_unit == 0:
        if units is None:
            raise ValueError("DXF has no declared units. Supply units='mm' (or m/um/nm); no scale is guessed.")
        if units not in unit_ids:
            raise ValueError("Choose source units m, mm, um or nm.")
        source_unit = unit_ids[units]
        notes.append(f"Unitless DXF: source coordinates explicitly interpreted as {units}.")
    elif units is not None and unit_ids.get(units) != source_unit:
        raise ValueError("The supplied units conflict with the DXF $INSUNITS header.")
    try:
        factor = conversion_factor(source_unit, unit_ids[target_units])
    except Exception:
        raise ValueError("DXF units are unsupported; set a physical $INSUNITS value or export in millimetres.") from None
    outlines, loose = [], {}
    ignored = {}
    for entity in document.modelspace():
        kind, layer = entity.dxftype(), str(entity.dxf.layer)
        if kind in ("TEXT", "MTEXT", "DIMENSION", "POINT", "LEADER", "MLEADER"):
            ignored[kind] = ignored.get(kind, 0)+1
            continue
        if kind not in ("CIRCLE", "ELLIPSE", "LINE", "ARC", "LWPOLYLINE", "POLYLINE", "SPLINE"):
            raise ValueError(f"Unsupported DXF entity {kind} on layer '{layer}'. Explode blocks/hatches into outlines before importing.")
        _planar(entity)
        if kind == "CIRCLE":
            outlines.append((layer, {"type": "circle", "center": _xy(entity.dxf.center, factor),
                                    "radius": float(entity.dxf.radius)*factor}))
        elif kind == "ELLIPSE":
            sweep = float(entity.dxf.end_param-entity.dxf.start_param)
            if abs(sweep-2*np.pi) > 1e-10:
                raise ValueError("Partial DXF ellipses are not supported; export them as a closed spline/polyline.")
            major = np.asarray(entity.dxf.major_axis)[:2]
            a = float(np.linalg.norm(major))*factor
            outlines.append((layer, {"type": "ellipse", "center": _xy(entity.dxf.center, factor),
                                    "semi_axes": [a, a*float(entity.dxf.ratio)],
                                    "rotation_deg": float(np.degrees(np.arctan2(major[1], major[0])))}))
        elif kind in ("LINE", "ARC"):
            loose.setdefault(layer, []).append(_segment(entity, factor))
        elif kind in ("LWPOLYLINE", "POLYLINE"):
            pieces = [_segment(part, factor) for part in entity.virtual_entities()]
            if not pieces:
                raise ValueError("A DXF polyline has no usable segments.")
            if entity.is_closed:
                loops = _join(pieces)
                if len(loops) != 1:
                    raise ValueError("A polyline must contain one closed outline.")
                outlines.append((layer, loops[0]))
            else:
                loose.setdefault(layer, []).extend(pieces)
        else:
            from ezdxf.path import make_path
            if not entity.closed:
                raise ValueError("Open DXF splines are not imported; close/export the outline first.")
            points = [_xy(point, factor) for point in make_path(entity).flattening(
                distance=curve_tolerance/factor)]
            outlines.append((layer, {"type": "polygon", "points": points}))
            notes.append(f"Closed SPLINE on '{layer}' was approximated as a polygon at {curve_tolerance:g} {target_units} flattening tolerance.")
    for label, pieces in loose.items():
        outlines.extend((label, shape) for shape in _join(pieces))
    if ignored:
        notes.append("Ignored drawing annotations: "+", ".join(f"{n} {kind}" for kind, n in ignored.items()))
    notes.append(f"DXF physical units converted to {target_units}; circles, full ellipses and circular arcs remain native.")
    return _case_from_outlines(outlines, target_units, path.stem, exterior, notes)


def export_dxf(geometry, path):
    """Export exact native boundaries to planar DXF; materials remain in JSON."""
    ezdxf = _ezdxf()
    unit_ids = {"m": 6, "mm": 4, "um": 13, "µm": 13, "nm": 12}
    document = ezdxf.new("R2010")
    document.header["$INSUNITS"] = unit_ids[geometry.units]
    space = document.modelspace()
    scale = geometry.unit_scale
    for index, (name, shape) in enumerate(geometry.boundaries):
        layer = "ARCS_"+name
        # DXF layer names exclude these characters.
        for character in '<>/"\\:;?*|=':
            layer = layer.replace(character, "_")
        if layer not in document.layers:
            document.layers.new(layer, dxfattribs={"color": 7 if index == 0 else index+1})
        attributes = {"layer": layer}
        if len(shape.curves) == 1:
            curve = shape.curves[0]
            if curve.kind in ("ellipse", "arc"):
                center, axes, angle, _, sweep = curve.data
                if abs(abs(sweep)-2*np.pi) < 1e-10:
                    if curve.kind == "arc":
                        space.add_circle(np.asarray(center)/scale, axes[0]/scale, dxfattribs=attributes)
                    else:
                        a, b = np.asarray(axes)/scale
                        if b > a:
                            a, b, angle = b, a, angle+np.pi/2
                        space.add_ellipse(np.asarray(center)/scale,
                            major_axis=(a*np.cos(angle), a*np.sin(angle), 0),
                            ratio=b/a, dxfattribs=attributes)
                    continue
        vertices = []
        for curve in shape.curves:
            point = curve.point(0.)/scale
            bulge = 0. if curve.kind == "line" else math.tan(curve.data[-1]/4)
            vertices.append((float(point[0]), float(point[1]), bulge))
        space.add_lwpolyline(vertices, format="xyb", close=True, dxfattribs=attributes)
    document.saveas(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert planar DXF or XY CSV into editable ARCS JSON.")
    parser.add_argument("input")
    parser.add_argument("output", help="Solver-readable .json case")
    parser.add_argument("--units", choices=["m", "mm", "um", "nm"], help="Required for CSV/unitless DXF")
    parser.add_argument("--target-units", default="mm", choices=["m", "mm", "um", "nm"], help="Output units for DXF conversion")
    parser.add_argument("--curve-tolerance", type=float, default=.01, help="Spline flattening tolerance, in target units")
    parser.add_argument("--exterior", default="unspecified", help="User-assigned material label outside the last boundary")
    args = parser.parse_args(argv)
    try:
        result = import_geometry(args.input, units=args.units, target_units=args.target_units,
                                 curve_tolerance=args.curve_tolerance, exterior=args.exterior)
        result.geometry.save(args.output)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"Saved {args.output}: {len(result.geometry.boundaries)} boundary/boundaries, {result.geometry.units}.")
    for note in result.notes:
        print(note)


if __name__ == "__main__":
    main()
