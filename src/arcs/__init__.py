"""ARCS v0.6.7: Arbitrary Resistive-wall Cross-section Solver for finite-beta beams."""
from .geometry import Contour, circle, ellipse, sampled_smooth_contour
from .materials import Metal, PEC, Vacuum, Air, Layer, Wall
from .reference import circular_exact, circular_indirect_space_charge, circular_indirect_dipolar
from .interfaces import PolygonPanels, polygon_panels, ConfocalCoating, confocal_coating
from .pec_panels import PolygonPECSolver
from .fields import TransverseImpedance
from .roughness import Roughness
from .maxwell import MaxwellSolver, MaxwellBoundaryState
from .geometry_input import GeometryInput, GeometrySolver, MeshSettings, load_geometry
from .geometry_editor import GeometryDocument
from .geometry_io import GeometryImport, import_geometry, export_dxf
from .wakes import wake_to_impedance, impedance_to_wake, gaussian_bunch_wake

__version__ = "0.6.7"
__all__ = ["Contour", "circle", "ellipse", "sampled_smooth_contour",
           "Metal", "PEC", "Vacuum", "Air", "Layer", "Wall", "circular_exact", "circular_indirect_space_charge", "circular_indirect_dipolar",
           "PolygonPanels", "polygon_panels", "ConfocalCoating", "confocal_coating",
           "PolygonPECSolver", "TransverseImpedance", "Roughness",
           "MaxwellSolver", "MaxwellBoundaryState",
           "GeometryInput", "GeometrySolver", "MeshSettings", "load_geometry",
           "GeometryDocument", "GeometryImport", "import_geometry", "export_dxf",
           "wake_to_impedance", "impedance_to_wake", "gaussian_bunch_wake"]
