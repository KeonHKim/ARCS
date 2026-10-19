"""UI-independent geometry editing, physical units, booleans and undo/redo.

The editor and file import share the validated ARCS geometry schema. Optional
Shapely is loaded only for boolean geometry operations. No field computation
or analytical impedance reference is used in the editor.
"""
from pathlib import Path
import copy
import json
import math
import re
import numpy as np
from .geometry_input import load_geometry, UNIT_SCALE, _shape, _nested_panels

BUILTIN_MATERIALS = {
    "vacuum": {"conductivity": 0., "mu_r": 1., "epsilon_r": 1., "relaxation_time": 0.},
    "PEC": {"kind": "pec"},
    "air": {"conductivity": 0., "mu_r": 1., "epsilon_r": 1., "relaxation_time": 0.},
    "Annealed Cu": {"conductivity": 5.8e7, "mu_r": 1., "epsilon_r": 1., "relaxation_time": 0.},
}


def _editor_materials(data):
    """Migrate the old GUI catalog while preserving explicit material choices.

    Only the complete former catalog is recognized as automatically supplied.
    Used layers, individual file definitions and changed properties remain
    user-defined materials. An edited copper value is not labelled annealed.
    """
    materials = data.get("materials", {})
    legacy = {name: {"conductivity": sigma, "mu_r": 1., "epsilon_r": 1., "relaxation_time": 0.}
              for name, sigma in (("Al", 3.5e7), ("SS", 1.4e6), ("NEG", 2.e5))}
    if all(name in materials for name in ("vacuum", "air", "Cu", *legacy)):
        used = {region["material"] for region in data["regions"]}
        for name, properties in legacy.items():
            if name not in used and materials[name] == properties:
                materials.pop(name)
    copper = BUILTIN_MATERIALS["Annealed Cu"]
    if materials.get("Cu") == copper and materials.get("Annealed Cu", copper) == copper:
        materials["Annealed Cu"] = materials.pop("Cu")
        for region in data["regions"]:
            if region["material"] == "Cu":
                region["material"] = "Annealed Cu"
    data["materials"] = {**copy.deepcopy(BUILTIN_MATERIALS), **materials}
    return data


def default_geometry():
    return {"format": "arcs.geometry/1", "name": "New ARCS cross-section",
            "units": "mm", "aperture": {"type": "circle", "radius": 30},
            "beam": [0, 0], "interfaces": [],
            "regions": [{"name": "beam_region", "material": "vacuum"},
                        {"name": "exterior", "material": "Annealed Cu"}],
            "materials": copy.deepcopy(BUILTIN_MATERIALS),
            "mesh": {"max_panel_length": 1, "curve_tolerance": .01,
                     "minimum_panels": 128, "corner_grading": 2,
                     "integration_order": 4}}


def parse_coordinates(text):
    """Read ordered XY coordinates, with optional x,y header."""
    rows = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        values = re.split(r"[,;\s]+", line)
        if not rows and [v.lower() for v in values] == ["x", "y"]:
            continue
        if len(values) != 2:
            raise ValueError("Enter one x,y pair per line, in perimeter order.")
        try:
            pair = [float(v) for v in values]
        except ValueError:
            raise ValueError("Coordinates must be numbers; only an optional x,y header is allowed.") from None
        if not np.isfinite(pair).all():
            raise ValueError("Coordinates must be finite.")
        rows.append(pair)
    if len(rows) < 3:
        raise ValueError("A polygon needs at least three ordered coordinate pairs.")
    return {"type": "polygon", "points": rows}


def transform_shape(shape, *, factor=1., translation=(0., 0.), rotation_deg=0.):
    """Uniform scaling about origin, then rotation, then translation."""
    if not np.isfinite([factor, *translation, rotation_deg]).all() or factor <= 0:
        raise ValueError("Use a positive scale and finite transform values.")
    result = copy.deepcopy(shape)
    angle = math.radians(rotation_deg)
    matrix = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
    def point(p):
        return (factor*np.asarray(p) @ matrix.T+translation).tolist()
    kind = result["type"]
    if kind == "polygon":
        result["points"] = [point(p) for p in result["points"]]
    elif kind == "path":
        result["start"] = point(result["start"])
        for segment in result["segments"]:
            segment["to"] = point(segment["to"])
            if "center" in segment:
                segment["center"] = point(segment["center"])
    else:
        result["center"] = point(result.get("center", [0., 0.]))
        for key in ("radius", "width", "height", "corner_radius"):
            if key in result:
                result[key] *= factor
        if "semi_axes" in result:
            result["semi_axes"] = [value*factor for value in result["semi_axes"]]
        if kind != "circle":
            result["rotation_deg"] = result.get("rotation_deg", 0.)+rotation_deg
    return result


def outline_points(shape, tolerance=.01, units="mm"):
    """Sample native outlines with a conservative chord-error bound.

    Lines retain their endpoints. For curves the bound is max|r''|*dt²/8.
    This tolerance is geometric, not an electromagnetic error estimate.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Use a positive curve approximation tolerance.")
    scale = UNIT_SCALE[units]
    native = _shape(shape, scale, Path.cwd())
    points = []
    for curve in native.curves:
        if curve.kind == "line":
            count = 1
        else:
            _, axes, _, _, sweep = curve.data
            count = max(2, math.ceil(abs(sweep)*math.sqrt(max(axes)/(8*tolerance*scale))))
        if len(points)+count > 20000:
            raise ValueError("Curve approximation exceeds 20,000 vertices; increase the geometric tolerance.")
        points.extend(curve.point(np.arange(count)/count)/scale)
    return np.asarray(points)


def boolean_shape(first, second, operation, *, tolerance=.01, units="mm"):
    """Crop/intersect, unite or subtract, preserving one simple aperture."""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise RuntimeError("Boolean tools require Shapely. Install ARCS with the gui extra.") from None
    a = Polygon(outline_points(first, tolerance, units))
    b = Polygon(outline_points(second, tolerance, units))
    if not a.is_valid or not b.is_valid:
        raise ValueError("A boolean input is self-intersecting or unresolved.")
    functions = {"intersect": a.intersection, "crop": a.intersection,
                 "union": a.union, "subtract": a.difference}
    if operation not in functions:
        raise ValueError("Choose union, subtract, intersect or crop.")
    result = functions[operation](b)
    if result.is_empty:
        raise ValueError("The operation would remove the entire boundary.")
    if result.geom_type != "Polygon":
        raise ValueError("The operation produces disconnected apertures. ARCS currently requires one connected outline.")
    if len(result.interiors):
        raise ValueError("The operation produces an internal island/hole. This topology needs a future region-graph solver.")
    return {"type": "polygon", "points": np.asarray(result.exterior.coords)[:-1].tolist()}


class GeometryDocument:
    """Transactional editing: invalid operations leave geometry/history intact."""
    def __init__(self, source=None):
        source = default_geometry() if source is None else source
        geometry = source if hasattr(source, "to_dict") else load_geometry(source)
        self._data = _editor_materials(geometry.to_dict())
        self._geometry = load_geometry(self._data)
        self._undo, self._redo = [], []
        self.saved_data = copy.deepcopy(self._data)

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def geometry(self):
        return self._geometry

    @property
    def dirty(self):
        return self._data != self.saved_data

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def commit(self, data):
        geometry = load_geometry(data)
        normalized = geometry.to_dict()
        if normalized == self._data:
            return
        self._undo.append(copy.deepcopy(self._data))
        self._undo = self._undo[-100:]
        self._redo.clear()
        self._data, self._geometry = normalized, geometry

    def undo(self):
        if self._undo:
            self._redo.append(self._data)
            self._data = self._undo.pop()
            self._geometry = load_geometry(self._data)

    def redo(self):
        if self._redo:
            self._undo.append(self._data)
            self._data = self._redo.pop()
            self._geometry = load_geometry(self._data)

    def shape(self, index):
        return copy.deepcopy(self._data["aperture"] if index == 0
                             else self._data["interfaces"][index-1]["shape"])

    @staticmethod
    def _put_shape(data, index, shape):
        if index == 0:
            data["aperture"] = shape
        else:
            data["interfaces"][index-1]["shape"] = shape

    def set_shape(self, index, shape, *, operation="replace", tolerance=None):
        data = self.data
        if operation != "replace":
            shape = boolean_shape(self.shape(index), shape, operation,
                                  tolerance=tolerance or data["mesh"]["curve_tolerance"],
                                  units=data["units"])
        self._put_shape(data, index, shape)
        self.commit(data)

    def add_interface(self, shape, name=None, exterior="air"):
        data = self.data
        if name is None:
            number = len(data["interfaces"])+1
            name = f"interface_{number}"
            while name in [item["name"] for item in data["interfaces"]]:
                number += 1
                name = f"interface_{number}"
        # The old unbounded region becomes a finite layer; append a new exterior.
        data["regions"][-1]["name"] = f"layer_inside_{name}"
        data["interfaces"].append({"name": name, "shape": shape})
        data["regions"].append({"name": "exterior", "material": exterior})
        self.commit(data)

    def delete_interface(self, index):
        if index == 0:
            raise ValueError("The vacuum aperture cannot be deleted; replace its outline instead.")
        data = self.data
        removed = data["interfaces"].pop(index-1)
        data.get("boundary_meshes", {}).pop(removed["name"], None)
        # Deleting an interface merges regions with the outer region's material.
        data["regions"].pop(index)
        self.commit(data)

    def set_region(self, index, material, properties=None):
        if index == 0 and material != "vacuum":
            raise ValueError("The innermost material is always vacuum.")
        data = self.data
        data["regions"][index]["material"] = material
        if properties is not None:
            data["materials"][material] = properties
        self.commit(data)

    def set_beam(self, x, y):
        data = self.data
        point = [float(x), float(y)]
        if not self.geometry.boundary().contains(np.asarray(point)*self.geometry.unit_scale):
            raise ValueError("Place the beam strictly inside the vacuum aperture.")
        data["beam"] = point
        self.commit(data)

    def set_physics(self, beta=1.0, roughness=None):
        data = self.data
        data["beta"] = beta
        if roughness is None:
            data.pop("roughness", None)
        else:
            data["roughness"] = copy.deepcopy(roughness)
        self.commit(data)

    def transform(self, index, *, translation=(0., 0.), rotation_deg=0.):
        self.set_shape(index, transform_shape(self.shape(index),
                       translation=translation, rotation_deg=rotation_deg))

    def set_units(self, units):
        if units not in UNIT_SCALE:
            raise ValueError("Choose m, mm, um or nm.")
        data = self.data
        factor = UNIT_SCALE[data["units"]]/UNIT_SCALE[units]
        data["aperture"] = transform_shape(data["aperture"], factor=factor)
        for interface in data["interfaces"]:
            interface["shape"] = transform_shape(interface["shape"], factor=factor)
        data["beam"] = (np.asarray(data["beam"])*factor).tolist()
        for key in ("max_panel_length", "curve_tolerance"):
            data["mesh"][key] *= factor
        for settings in data.get("boundary_meshes", {}).values():
            for key in ("max_panel_length", "curve_tolerance"):
                if key in settings:
                    settings[key] *= factor
        if data.get("roughness"):
            data["roughness"]["rms_height"] *= factor
            if data["roughness"].get("interface_heights"):
                data["roughness"]["interface_heights"] = (np.asarray(data["roughness"]["interface_heights"])*factor).tolist()
        data["units"] = units
        self.commit(data)

    def set_mesh(self, values, boundary=None):
        """Update shared defaults or one named/indexed physical boundary.

        Mesh changes are transactional. Independent chord partitions must stay
        nested, especially for very thin curved coatings.
        """
        data = self.data
        if boundary is None:
            data["mesh"].update(values)
        else:
            name = self._boundary_name(boundary)
            data.setdefault("boundary_meshes", {}).setdefault(name, {}).update(values)
        self._check_mesh_nesting(data)
        self.commit(data)

    def reset_boundary_mesh(self, boundary):
        data = self.data
        data.get("boundary_meshes", {}).pop(self._boundary_name(boundary), None)
        self._check_mesh_nesting(data)
        self.commit(data)

    def _boundary_name(self, boundary):
        if isinstance(boundary, (int, np.integer)) and not isinstance(boundary, bool):
            if not 0 <= boundary < len(self.geometry.boundaries):
                raise ValueError("Choose an existing physical boundary.")
            return self.geometry.boundaries[boundary][0]
        self.geometry.boundary(boundary)
        return boundary

    @staticmethod
    def _check_mesh_nesting(data):
        candidate = load_geometry(data)
        meshes = [candidate.to_panels(name) for name, _ in candidate.boundaries]
        if not all(_nested_panels(a, b) for a, b in zip(meshes[:-1], meshes[1:])):
            raise ValueError("The boundary chord meshes cross/touch. Refine the enclosing "
                             "boundary's curve tolerance or maximum panel length, or use "
                             "matching panel partitions for a very thin coating.")

    def save(self, path):
        # Write the shared solver-readable geometry; no separate GUI file format.
        self.geometry.save(path)
        self.saved_data = self.data

    def issues(self):
        issues = []
        case = self.geometry
        if not case.boundary().contains(case.beam_position):
            issues.append("Place the beam inside the vacuum aperture before solving.")
        for region in case.regions[1:]:
            if region["material"] not in case.materials:
                issues.append(f"Define properties for material '{region['material']}'.")
            elif case.materials[region["material"]].get("kind") == "pec" and region is not case.regions[-1]:
                issues.append("PEC is an outer termination; finite layers must be conducting materials.")
        if case.roughness is not None and len(case.boundaries) > 1:
            issues.append("Gradient roughness currently requires a single aperture in the GUI; explicit coating roughness needs a depth-coordinate specification.")
        for region in case.regions[1:]:
            properties=case.materials.get(region["material"], {})
            lossless=properties and properties.get("kind") != "pec" and properties["conductivity"] == 0
            if lossless and region is not case.regions[-1]:
                issues.append("Finite Air/Vacuum gaps are not implemented; use Air/Vacuum only as Exterior.")
            elif lossless:
                if case.beta == 1:
                    issues.append("Open Air/Vacuum exterior requires beta<1; its beta=1 limit is not implemented.")
                if case.roughness is not None:
                    issues.append("Disable conductivity-gradient roughness for an open Air/Vacuum exterior.")
                if any(properties[key] != value for key,value in
                       (("mu_r",1.),("epsilon_r",1.),("relaxation_time",0.))):
                    issues.append("Open Air/Vacuum exterior requires mu_r=epsilon_r=1 and relaxation_time=0.")
            elif properties.get("kind") == "pec" and case.roughness is not None:
                issues.append("Disable conductivity-gradient roughness for PEC; that model needs finite conductivity.")
        return issues
