"""Python desktop cross-section editor (optional PySide6 dependency).

Launch with arcs-gui or python -m arcs.gui. All geometry operations are owned
by GeometryDocument, which is also usable without a GUI.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
try:
    from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QTimer
    from PySide6.QtGui import QAction, QActionGroup, QColor, QPainter, QPainterPath, QPen, QKeySequence
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame,
        QScrollArea, QHBoxLayout, QVBoxLayout, QFormLayout, QGridLayout, QLabel, QPushButton,
        QLineEdit, QComboBox, QListWidget, QGroupBox, QDoubleSpinBox, QSpinBox,
        QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
        QDialogButtonBox, QPlainTextEdit, QTabWidget, QFileDialog,
        QMessageBox, QInputDialog, QToolBar)
except ImportError:
    raise SystemExit('The desktop editor needs PySide6. Run: pip install -e ".[gui,cad]"') from None
from . import __version__
from .geometry_editor import GeometryDocument, BUILTIN_MATERIALS, parse_coordinates, transform_shape
from .geometry_input import _shape, UNIT_SCALE
from .geometry_io import import_geometry, export_dxf

COLORS = {"vacuum": "#eef6ff", "air": "#f4f7fb", "PEC": "#b6beca",
          "Annealed Cu": "#dba47f", "unspecified": "#ded9e9"}
MATERIAL_LABELS = {"vacuum": "Vacuum", "air": "Air"}


def number(value=0, positive=False):
    spin = QDoubleSpinBox()
    spin.setDecimals(9)
    spin.setRange(1e-9 if positive else -1e12, 1e12)
    spin.setValue(float(value))
    spin.setKeyboardTracking(False)
    return spin


class ShapeDialog(QDialog):
    """Exact dimensions, coordinate table or native-path JSON."""
    def __init__(self, shape, units, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Boundary definition · {units}")
        self.resize(530, 500)
        self.units, self.original = units, shape
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Dimensions are in "+units+". Coordinates follow the perimeter order."))
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        primitive = QWidget()
        form = QFormLayout(primitive)
        self.kind = QComboBox()
        self.kind.addItems(["circle", "ellipse", "square", "rectangle", "rounded_rectangle"])
        kind = shape["type"]
        self.kind.setCurrentText(kind if kind in ["circle","ellipse","rectangle","rounded_rectangle"] else "rectangle")
        form.addRow("Shape", self.kind)
        center = shape.get("center", [0, 0])
        axes = shape.get("semi_axes", [30, 20])
        values = {"Center x": center[0], "Center y": center[1],
                  "Radius / semi-axis x": shape.get("radius", axes[0]),
                  "Semi-axis y": axes[1], "Width": shape.get("width", 60),
                  "Height": shape.get("height", 40), "Corner radius": shape.get("corner_radius", 5),
                  "Rotation [degrees]": shape.get("rotation_deg", 0)}
        self.fields = {}
        self.labels = {}
        for name, value in values.items():
            widget = number(value, positive=name not in ("Center x","Center y","Rotation [degrees]"))
            self.fields[name] = widget
            form.addRow(name, widget)
            self.labels[name] = form.labelForField(widget)
        self.tabs.addTab(primitive, "Dimensions")
        self.coordinates = QPlainTextEdit()
        self.coordinates.setPlaceholderText("x, y\n-30, -20\n30, -20\n30, 20\n-30, 20")
        if kind == "polygon":
            self.coordinates.setPlainText("\n".join(f"{x:.12g}, {y:.12g}" for x,y in shape["points"]))
        self.tabs.addTab(self.coordinates, "Coordinates")
        self.raw = QPlainTextEdit(json.dumps(shape, indent=2))
        self.tabs.addTab(self.raw, "Native shape / arcs")
        if kind == "polygon":
            self.tabs.setCurrentIndex(1)
        elif kind == "path":
            self.tabs.setCurrentIndex(2)
        self.kind.currentTextChanged.connect(self.update_fields)
        self.update_fields()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept_checked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.result_shape = None

    def update_fields(self):
        kind = self.kind.currentText()
        visible = {"Center x","Center y"}
        if kind == "circle":
            visible.add("Radius / semi-axis x")
        elif kind == "ellipse":
            visible.update(["Radius / semi-axis x","Semi-axis y","Rotation [degrees]"])
        else:
            visible.update(["Width","Rotation [degrees]"])
            if kind != "square":
                visible.add("Height")
            if kind == "rounded_rectangle":
                visible.add("Corner radius")
        for name, widget in self.fields.items():
            widget.setVisible(name in visible)
            self.labels[name].setVisible(name in visible)

    def accept_checked(self):
        try:
            if self.tabs.currentIndex() == 1:
                shape = parse_coordinates(self.coordinates.toPlainText())
            elif self.tabs.currentIndex() == 2:
                shape = json.loads(self.raw.toPlainText())
            else:
                v = {name: widget.value() for name, widget in self.fields.items()}
                kind = self.kind.currentText()
                shape = {"type": "rectangle" if kind == "square" else kind,
                         "center": [v["Center x"], v["Center y"]]}
                if kind == "circle":
                    shape["radius"] = v["Radius / semi-axis x"]
                elif kind == "ellipse":
                    shape["semi_axes"] = [v["Radius / semi-axis x"], v["Semi-axis y"]]
                    shape["rotation_deg"] = v["Rotation [degrees]"]
                else:
                    shape.update(width=v["Width"], height=v["Width"] if kind == "square" else v["Height"],
                                 rotation_deg=v["Rotation [degrees]"])
                    if kind == "rounded_rectangle":
                        shape["corner_radius"] = v["Corner radius"]
            _shape(shape, UNIT_SCALE[self.units], Path.cwd())
        except (ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Check the boundary", str(error))
            return
        self.result_shape = shape
        self.accept()


class MaterialDialog(QDialog):
    def __init__(self, name="custom", properties=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Material properties")
        self.resize(430, 270)
        form = QFormLayout(self)
        self.name = QLineEdit(name)
        form.addRow("Material name", self.name)
        properties = properties or {"conductivity": 1e6, "mu_r": 1., "epsilon_r": 1., "relaxation_time": 0.}
        self.values = {}
        for label, key in [("Conductivity [S/m]","conductivity"),("Relative permeability","mu_r"),
                           ("Relative permittivity","epsilon_r"),("Relaxation time [s]","relaxation_time")]:
            # Scientific notation allows small relaxation times without rounding.
            widget = QLineEdit(f"{properties.get(key, 0):.12g}")
            self.values[key] = widget
            form.addRow(label, widget)
        form.addRow(QLabel("Air uses an ideal nonconducting εr=μr=1 model.\nProperties are saved in SI units."))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def definition(self):
        return self.name.text().strip(), {key: float(widget.text()) for key, widget in self.values.items()}


class GeometryCanvas(QWidget):
    shapeCreated = Signal(object, str)
    boundarySelected = Signal(int)
    moveRequested = Signal(int, float, float)
    beamRequested = Signal(float, float)
    pointerMoved = Signal(float, float)
    panelSelected = Signal(int, int)
    boundaryMeshSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(540, 500)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.document = None
        self.selected = 0
        self.tool = "select"
        self.center = np.array([0., 0.])
        self.scale = 7.
        self.anchor = self.current = None
        self.polygon = []
        self.pan_anchor = None
        self.mesh = None
        self.meshes = []
        self.show_mesh = False
        self.mesh_only = False
        self.show_nodes = True
        self.show_quadrature = False
        self.show_panel_detail = True
        self.detail_mesh = None
        self.detail_name = ""
        self.detail_boundary = 0
        self.detail_panel = 0
        self.move_shape = None

    def world(self, point):
        return self.center+np.array([(point.x()-self.width()/2)/self.scale,
                                   -(point.y()-self.height()/2)/self.scale])

    def pixel(self, point):
        delta = (np.asarray(point)-self.center)*self.scale
        return QPointF(self.width()/2+delta[0], self.height()/2-delta[1])

    def fit(self):
        if self.document is None:
            return
        scale = self.document.geometry.unit_scale
        bounds = [shape.bounds for _, shape in self.document.geometry.boundaries]
        lo = np.min([b[0] for b in bounds], axis=0)/scale
        hi = np.max([b[1] for b in bounds], axis=0)/scale
        # Reserve space beside the full cross-section for the panel detail;
        # fitting must not put that inset on top of the geometry being shown.
        width = max(.4*self.width(), self.width()-304) if self.show_mesh and self.show_panel_detail else self.width()
        self.scale = .78*min(width/max(hi[0]-lo[0], 1e-12),
                             self.height()/max(hi[1]-lo[1], 1e-12))
        self.center = (lo+hi)/2+np.array([(self.width()-width)/(2*self.scale), 0.])
        self.update()

    def set_tool(self, tool):
        self.tool = tool
        self.cancel()
        self.setCursor(Qt.CursorShape.ArrowCursor if tool == "select" else
                       Qt.CursorShape.PointingHandCursor if tool in ("inspect", "mesh") else Qt.CursorShape.CrossCursor)

    def cancel(self):
        self.anchor = self.current = self.move_shape = None
        self.polygon = []
        self.update()

    def _points(self, shape):
        native = _shape(shape, UNIT_SCALE[self.document.geometry.units], Path.cwd())
        result = []
        for curve in native.curves:
            count = 1 if curve.kind == "line" else max(16, math.ceil(abs(curve.data[-1])*80))
            result.extend(curve.point(np.arange(count)/count)/self.document.geometry.unit_scale)
        return np.asarray(result)

    def _path(self, points, closed=True):
        path = QPainterPath()
        if len(points):
            path.moveTo(self.pixel(points[0]))
            for point in points[1:]:
                path.lineTo(self.pixel(point))
            if closed:
                path.closeSubpath()
        return path

    def _draw_shape(self, painter, shape, color, selected=False, draft=False):
        path = self._path(self._points(shape))
        pen = QPen(QColor("#176b98" if selected or draft else "#53657a"))
        pen.setWidthF(2.4 if selected else 1.4)
        if draft:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor(color))
        painter.drawPath(path)
        if selected and shape["type"] == "polygon" and len(shape["points"]) <= 32:
            painter.setBrush(QColor("#176b98"))
            for point in shape["points"]:
                painter.drawEllipse(self.pixel(point), 3., 3.)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f6f8fb"))
        if self.document is None:
            return
        # Grid is in physical drawing units; vertical coordinates increase upward.
        approximate = 80/self.scale
        base = 10**math.floor(math.log10(max(approximate, 1e-15)))
        step = next((value*base for value in (1,2,5,10) if value*base >= approximate), 10*base)
        lo = self.world(QPointF(0, self.height()))
        hi = self.world(QPointF(self.width(), 0))
        painter.setPen(QPen(QColor("#dfe6ed"), 1))
        for x in np.arange(math.ceil(lo[0]/step)*step, hi[0], step):
            p = self.pixel((x, 0))
            painter.drawLine(QPointF(p.x(), 0), QPointF(p.x(), self.height()))
        for y in np.arange(math.ceil(lo[1]/step)*step, hi[1], step):
            p = self.pixel((0, y))
            painter.drawLine(QPointF(0, p.y()), QPointF(self.width(), p.y()))
        data = self.document.data
        if not (self.show_mesh and self.mesh_only):
            for index in range(len(self.document.geometry.boundaries)-1, -1, -1):
                shape = self.move_shape if index == self.selected and self.move_shape else self.document.shape(index)
                color = COLORS.get(data["regions"][index]["material"], "#cdbde0")
                self._draw_shape(painter, shape, color, index == self.selected)
        if self.show_mesh:
            palette = [("#0067b1", "#d45500"), ("#007f64", "#8554a5")]
            scale = self.document.geometry.unit_scale
            for boundary_index, name, mesh in self.meshes:
                colors = palette[boundary_index % len(palette)]
                starts, ends = mesh.panel_start/scale, mesh.panel_end/scale
                painter.setBrush(Qt.BrushStyle.NoBrush)
                # Draw actual straight panel edges; native curves stay separate.
                for panel_index, (start, end) in enumerate(zip(starts, ends)):
                    painter.setPen(QPen(QColor(colors[panel_index % 2]),
                                        2.6 if boundary_index == self.selected else 2.0))
                    painter.drawLine(self.pixel(start), self.pixel(end))
                if self.show_quadrature:
                    painter.setPen(QPen(QColor("#b32183"), 1.3))
                    for point in mesh.points/scale:
                        pixel = self.pixel(point)
                        painter.drawLine(pixel+QPointF(-2,-2), pixel+QPointF(2,2))
                        painter.drawLine(pixel+QPointF(-2,2), pixel+QPointF(2,-2))
                if self.show_nodes:
                    painter.setPen(QPen(QColor("#153c54"), 1.2))
                    painter.setBrush(QColor("#ffffff"))
                    for point in starts:
                        painter.drawEllipse(self.pixel(point), 2.1, 2.1)
                if boundary_index == self.detail_boundary and self.detail_mesh is mesh:
                    panel = self.detail_panel
                    painter.setPen(QPen(QColor("#be184b"), 4.2))
                    painter.drawLine(self.pixel(starts[panel]), self.pixel(ends[panel]))
                    painter.setBrush(QColor("#ffffff"))
                    for point in (starts[panel], ends[panel]):
                        painter.drawEllipse(self.pixel(point), 4., 4.)
        if self.anchor is not None and self.current is not None and self.tool != "select":
            shape = self.drag_shape()
            if shape:
                self._draw_shape(painter, shape, "#80d7edf4", draft=True)
        if self.polygon:
            points = self.polygon+([self.current.tolist()] if self.current is not None else [])
            painter.setPen(QPen(QColor("#176b98"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self._path(points, closed=False))
        origin = self.pixel((0.,0.))
        painter.setPen(QPen(QColor("#66768b"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(origin.x(),0), QPointF(origin.x(),self.height()))
        painter.drawLine(QPointF(0,origin.y()), QPointF(self.width(),origin.y()))
        beam = self.pixel(data["beam"])
        painter.setPen(QPen(QColor("#ca3651"), 2))
        painter.setBrush(QColor("#ca3651"))
        painter.drawEllipse(beam, 4., 4.)
        painter.drawLine(beam+QPointF(-9,0), beam+QPointF(9,0))
        painter.drawLine(beam+QPointF(0,-9), beam+QPointF(0,9))
        painter.setPen(QColor("#243e55"))
        painter.drawText(beam+QPointF(10,-10), "Beam")
        # A compact physical scale bar avoids crowding the drawing.
        length = step*self.scale
        y = self.height()-24
        painter.setPen(QPen(QColor("#365068"), 2))
        painter.drawLine(QPointF(24,y), QPointF(24+length,y))
        painter.drawLine(QPointF(24,y-4), QPointF(24,y+4))
        painter.drawLine(QPointF(24+length,y-4), QPointF(24+length,y+4))
        painter.drawText(QPointF(24,y-9), f"{step:g} {data['units']}")
        painter.drawText(QPointF(self.width()-110, self.height()-20), "x →     y ↑")
        if self.show_mesh and self.show_panel_detail and self.detail_mesh is not None:
            self._draw_panel_detail(painter)

    def _draw_panel_detail(self, painter):
        """Magnify one real solver panel with an undistorted physical scale."""
        painter.save()
        box = QRectF(self.width()-292, 16, 276, 198)
        painter.setPen(QPen(QColor("#8ba9bb"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(QColor("#104a6a"))
        painter.drawText(box.adjusted(12, 8, -12, -8), Qt.AlignmentFlag.AlignTop,
                         "PANEL DETAIL · same physical mesh")
        mesh, panel = self.detail_mesh, self.detail_panel
        title = f"{self.detail_name} · panel {panel+1:,} / {len(mesh.panel_start):,}"
        title = painter.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, 252)
        painter.drawText(QPointF(box.left()+12, box.top()+40), title)
        plot = box.adjusted(14, 52, -14, -30)
        unit = self.document.geometry.unit_scale
        start, end = mesh.panel_start[panel]/unit, mesh.panel_end[panel]/unit
        vector = end-start
        length = np.linalg.norm(vector)
        center = (start+end)/2
        span = np.maximum(abs(vector), .5*length)
        zoom = .62*min(plot.width()/span[0], plot.height()/span[1])

        def pixel(point):
            delta = (point-center)*zoom
            return QPointF(plot.center().x()+delta[0], plot.center().y()-delta[1])

        painter.save()
        painter.setClipRect(plot)
        for index in ((panel-1) % len(mesh.panel_start), panel,
                      (panel+1) % len(mesh.panel_start)):
            color = "#be184b" if index == panel else "#267bac"
            painter.setPen(QPen(QColor(color), 3 if index == panel else 2))
            first, last = mesh.panel_start[index]/unit, mesh.panel_end[index]/unit
            painter.drawLine(pixel(first), pixel(last))
        if self.show_quadrature:
            painter.setPen(QPen(QColor("#b32183"), 1.7))
            for point in mesh.points[mesh.panel_index == panel]/unit:
                p = pixel(point)
                painter.drawLine(p+QPointF(-3,-3), p+QPointF(3,3))
                painter.drawLine(p+QPointF(-3,3), p+QPointF(3,-3))
        # Endpoints remain visible in the detail even if the full-view node
        # overlay is disabled or a dense mesh merges at screen resolution.
        painter.setPen(QPen(QColor("#153c54"), 1.5))
        painter.setBrush(QColor("#ffffff"))
        for label, point in (("start", start), ("end", end)):
            p = pixel(point)
            painter.drawEllipse(p, 4.5, 4.5)
            painter.drawText(p+QPointF(7,-7), label)
        painter.restore()
        painter.setPen(QColor("#104a6a"))
        painter.drawText(QPointF(box.left()+12, box.bottom()-10),
                         f"Length: {length:.6g} {self.document.geometry.units} · equal x/y scale")
        painter.restore()

    def hit_panel(self, point):
        """Find the closest displayed line segment within eight screen pixels."""
        best, distance = None, 8/self.scale
        unit = self.document.geometry.unit_scale
        for index, _, mesh in self.meshes:
            starts, vectors = mesh.panel_start/unit, (mesh.panel_end-mesh.panel_start)/unit
            t = np.clip(np.sum((point-starts)*vectors, axis=1)/np.sum(vectors*vectors, axis=1), 0, 1)
            distances = np.linalg.norm(starts+t[:,None]*vectors-point, axis=1)
            panel = int(np.argmin(distances))
            if distances[panel] < distance:
                best, distance = (index, panel), distances[panel]
        return best

    def drag_shape(self):
        if self.anchor is None or self.current is None:
            return None
        first, second = self.anchor, self.current
        delta = second-first
        if self.tool == "circle":
            radius = float(np.linalg.norm(delta))
            return None if radius == 0 else {"type":"circle","center":first.tolist(),"radius":radius}
        if self.tool == "square":
            edge = float(np.max(abs(delta)))
            second = first+np.where(delta < 0, -edge, edge)
        width, height = abs(second-first)
        if width == 0 or height == 0:
            return None
        center = ((first+second)/2).tolist()
        if self.tool == "ellipse":
            return {"type":"ellipse","center":center,"semi_axes":[width/2,height/2]}
        shape = {"type":"rounded_rectangle" if self.tool == "rounded_rectangle" else "rectangle",
                 "center":center,"width":float(width),"height":float(height)}
        if self.tool == "rounded_rectangle":
            shape["corner_radius"] = float(min(width,height)*.15)
        return shape

    def hit(self, point):
        outline = self.hit_boundary(point)
        if outline is not None and outline < len(self.document.geometry.boundaries):
            return outline
        case = self.document.geometry
        candidates = [index for index, (_, shape) in enumerate(case.boundaries)
                      if shape.contains(point*case.unit_scale)]
        # The innermost containing boundary is the visible filled region.
        if candidates:
            return candidates[0]
        return self.selected

    def hit_boundary(self, point):
        """Pick a visible boundary edge before considering filled regions."""
        panel = self.hit_panel(point)
        if panel is not None:
            return panel[0]
        if self.show_mesh and self.mesh_only:
            return None
        best, distance = None, 8/self.scale
        for index, (_, shape) in enumerate(self.document.geometry.boundaries):
            starts = self._points(shape.definition)
            vectors = np.roll(starts, -1, axis=0)-starts
            t = np.clip(np.sum((point-starts)*vectors, axis=1)/np.sum(vectors*vectors, axis=1), 0, 1)
            closest = float(np.min(np.linalg.norm(starts+t[:, None]*vectors-point, axis=1)))
            if closest < distance:
                best, distance = index, closest
        return best

    def mousePressEvent(self, event):
        self.setFocus()
        point = self.world(event.position())
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_anchor = (event.position(), self.center.copy())
            return
        if event.button() == Qt.MouseButton.RightButton and self.tool == "polygon":
            self.finish_polygon()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.tool == "mesh":
            boundary = self.hit_boundary(point)
            if boundary is not None:
                self.boundaryMeshSelected.emit(boundary)
        elif self.tool == "inspect":
            hit = self.hit_panel(point)
            if hit is not None:
                self.panelSelected.emit(*hit)
        elif self.tool == "beam":
            self.beamRequested.emit(*point)
        elif self.tool == "polygon":
            self.polygon.append(point.tolist())
            self.current = point
        else:
            self.anchor = self.current = point
            if self.tool == "select":
                self.selected = self.hit(point)
                self.boundarySelected.emit(self.selected)
        self.update()

    def mouseMoveEvent(self, event):
        point = self.world(event.position())
        self.pointerMoved.emit(*point)
        if self.pan_anchor is not None:
            pixel, center = self.pan_anchor
            delta = event.position()-pixel
            self.center = center+np.array([-delta.x(),delta.y()])/self.scale
        self.current = point
        if self.anchor is not None and self.tool == "select":
            self.move_shape = transform_shape(self.document.shape(self.selected),
                                              translation=point-self.anchor)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_anchor = None
            return
        if event.button() != Qt.MouseButton.LeftButton or self.anchor is None:
            return
        self.current = self.world(event.position())
        if self.tool == "select":
            delta = self.current-self.anchor
            if np.linalg.norm(delta)*self.scale > 2:
                self.moveRequested.emit(self.selected, *delta)
        else:
            shape = self.drag_shape()
            if shape:
                self.shapeCreated.emit(shape, "intersect" if self.tool == "crop" else "draw")
        self.anchor = self.current = self.move_shape = None
        self.update()

    def finish_polygon(self):
        points = []
        for point in self.polygon:
            if not points or not np.array_equal(point, points[-1]):
                points.append(point)
        if len(points) >= 3:
            self.shapeCreated.emit({"type":"polygon","points":points}, "draw")
        self.cancel()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.tool == "polygon":
            self.finish_polygon()
        elif event.key() == Qt.Key.Key_Backspace and self.tool == "polygon" and self.polygon:
            self.polygon.pop()
            self.update()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        before = self.world(event.position())
        factor = 1.15**(event.angleDelta().y()/120)
        self.scale = min(1e12, max(1e-12, self.scale*factor))
        self.center += before-self.world(event.position())
        self.update()


class GeometryWindow(QMainWindow):
    def __init__(self, source=None):
        super().__init__()
        self.document = GeometryDocument(source)
        self.filename = None
        self.selected = 0
        self.edit_shared_mesh = False
        self.inspected_mesh_index = 0
        self.adding_interface = False
        self.setWindowTitle("ARCS · Geometry editor")
        self.resize(1430, 920)
        self.setStyleSheet("""
            QMainWindow, QWidget {font-family: Sans Serif; font-size: 12px; color: #23384d;}
            QMainWindow {background: #f4f7fa;}
            QToolBar {background: #e7eef4; padding: 7px; spacing: 6px; border: 0;}
            QToolButton {padding: 6px 9px; border-radius: 4px;}
            QToolButton:checked {background: #bddced; color: #104a6a;}
            QGroupBox {font-weight: 600; margin-top: 12px; padding-top: 15px; border: 1px solid #d5dfe8; border-radius: 6px;}
            QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 4px;}
            QPushButton {padding: 7px; border: 1px solid #c4d2dd; border-radius: 4px; background: #f6fafc;}
            QPushButton:checked {background: #c6e4f3;}
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit {padding: 4px; border: 1px solid #bfcedb; border-radius: 3px; background: white;}
            QListWidget, QTableWidget {border: 1px solid #cdd9e3; background: white;}
        """)
        self._build()
        self.refresh(fit=True)

    def _action(self, bar, label, callback, shortcut=None):
        action = QAction(label, self)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        bar.addAction(action)
        return action

    def _build(self):
        files = self.addToolBar("Files")
        self._action(files, "New", self.new_case, QKeySequence.StandardKey.New)
        self._action(files, "Open / import", self.open_case, QKeySequence.StandardKey.Open)
        self._action(files, "Save case", self.save_case, QKeySequence.StandardKey.Save)
        self._action(files, "Export DXF", self.save_dxf)
        self._action(files, "Export mesh", self.save_mesh)
        self._action(files, "Save view", self.save_view)
        files.addSeparator()
        self.undo_action = self._action(files, "Undo", self.undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._action(files, "Redo", self.redo, "Ctrl+Shift+Z")
        self._action(files, "Fit", lambda: self.canvas.fit(), "F")
        self.addToolBarBreak()
        tools = self.addToolBar("Drawing tools")
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions = {}
        for key, title in [("select","Select / move"),("mesh","Mesh settings"),("inspect","Inspect mesh"),("circle","Circle"),("ellipse","Ellipse"),
                           ("square","Square"),("rectangle","Rectangle"),
                           ("rounded_rectangle","Rounded rectangle"),("polygon","Polygon"),
                           ("crop","Crop"),("beam","Beam")]:
            action = QAction(title,self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, key=key: self.select_tool(key))
            self.tool_group.addAction(action)
            tools.addAction(action)
            self.tool_actions[key] = action
        self.tool_actions["select"].setChecked(True)
        self._action(tools, "Finish polygon", lambda: self.canvas.finish_polygon())
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(12,12,12,12)
        outer.setSpacing(12)
        self.setCentralWidget(central)
        left = QWidget()
        left.setFixedWidth(235)
        l = QVBoxLayout(left)
        l.setContentsMargins(0,0,0,0)
        title = QLabel("ARCS")
        title.setStyleSheet("font-size: 30px; font-weight: 700; color: #165777;")
        l.addWidget(title)
        l.addWidget(QLabel(f"Cross-section editor · v{__version__}"))
        l.addWidget(QLabel("Physical boundaries"))
        self.boundaries = QListWidget()
        self.boundaries.setMinimumHeight(170)
        self.boundaries.currentRowChanged.connect(self.select_boundary)
        l.addWidget(self.boundaries)
        self.add_button = QPushButton("+ Draw outer interface")
        self.add_button.setCheckable(True)
        self.add_button.toggled.connect(self.set_add_interface)
        l.addWidget(self.add_button)
        edit = QPushButton("Dimensions / coordinates")
        edit.clicked.connect(self.edit_boundary)
        l.addWidget(edit)
        delete = QPushButton("Remove selected interface")
        delete.clicked.connect(self.delete_boundary)
        l.addWidget(delete)
        composition = QGroupBox("Drawing operation")
        cl = QVBoxLayout(composition)
        self.operation = QComboBox()
        self.operation.addItems(["Replace selected","Union with selected","Subtract from selected","Intersect with selected"])
        cl.addWidget(self.operation)
        tip = QLabel("Crop: drag a rectangle to keep.\nPolygon: click vertices, then Enter.\nCircle: drag from its centre.\nOther shapes: drag opposite corners.")
        tip.setWordWrap(True)
        cl.addWidget(tip)
        l.addWidget(composition)
        self.mesh_toggle = QCheckBox("Show boundary mesh")
        self.mesh_toggle.setChecked(True)
        self.mesh_toggle.toggled.connect(self.update_mesh)
        self.mesh_view = QComboBox()
        self.mesh_view.addItems(["Geometry + mesh", "Mesh only"])
        self.mesh_view.currentIndexChanged.connect(self.update_mesh)
        self.mesh_scope = QComboBox()
        self.mesh_scope.addItems(["All boundaries", "Selected boundary"])
        self.mesh_scope.currentIndexChanged.connect(self.update_mesh)
        self.nodes_toggle = QCheckBox("Show mesh nodes")
        self.nodes_toggle.setChecked(True)
        self.nodes_toggle.toggled.connect(self.update_mesh)
        self.quadrature_toggle = QCheckBox("Show integration points (×)")
        self.quadrature_toggle.toggled.connect(self.update_mesh)
        self.detail_toggle = QCheckBox("Panel detail")
        self.detail_toggle.setChecked(True)
        self.detail_toggle.toggled.connect(self.update_mesh)
        self.zoom_mesh_button = QPushButton("Zoom to layer meshes")
        self.zoom_mesh_button.clicked.connect(self.zoom_layer_meshes)
        self.mesh_table = QTableWidget(0, 3)
        self.mesh_table.setHorizontalHeaderLabels(["Boundary", "Panels", "Nodes"])
        self.mesh_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            self.mesh_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.mesh_table.verticalHeader().hide()
        self.mesh_table.setMaximumHeight(165)
        self.mesh_table.setMinimumHeight(80)
        self.mesh_table.setToolTip("Actual generated boundary meshes. A closed line mesh has one unique endpoint node per panel.")
        self.mesh_table.cellClicked.connect(lambda row, column: self.choose_inspection_boundary(row))
        l.addWidget(self.mesh_table)
        self.mesh_info = QLabel()
        self.mesh_info.setWordWrap(True)
        self.mesh_info.setStyleSheet("font-size: 11px;")
        l.addWidget(self.mesh_info)
        self.boundary_info = QLabel()
        self.boundary_info.setWordWrap(True)
        l.addWidget(self.boundary_info)
        l.addStretch()
        l.addWidget(QLabel("Scroll: zoom at pointer\nMiddle drag: pan\nEsc: cancel drawing"))
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setFixedWidth(255)
        outer.addWidget(left_scroll)
        drawing = QWidget()
        dl = QVBoxLayout(drawing)
        dl.setContentsMargins(0,0,0,0)
        header = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("Geometry name")
        self.name.editingFinished.connect(self.rename_case)
        header.addWidget(self.name)
        self.units = QComboBox()
        self.units.addItems(["mm","m","um","nm"])
        self.units.currentTextChanged.connect(self.change_units)
        header.addWidget(self.units)
        dl.addLayout(header)
        self.mesh_summary = QLabel()
        self.mesh_summary.setObjectName("mesh_summary")
        self.mesh_summary.setWordWrap(True)
        self.mesh_summary.setStyleSheet("font-size: 15px; font-weight: 600; color: #104a6a; "
                                       "background: #e5f1f8; padding: 9px; border-radius: 5px;")
        self.mesh_summary.setToolTip("Counts include every generated interface, even when its overlay is hidden or only one boundary is displayed.")
        dl.addWidget(self.mesh_summary)
        mesh_controls = QHBoxLayout()
        mesh_controls.addWidget(self.mesh_toggle)
        mesh_controls.addWidget(self.mesh_view)
        mesh_controls.addWidget(self.mesh_scope)
        dl.addLayout(mesh_controls)
        mesh_markers = QHBoxLayout()
        for widget in (self.nodes_toggle, self.quadrature_toggle, self.detail_toggle):
            mesh_markers.addWidget(widget)
        mesh_markers.addStretch()
        dl.addLayout(mesh_markers)
        self.canvas = GeometryCanvas()
        self.canvas.shapeCreated.connect(self.apply_drawn_shape)
        self.canvas.boundarySelected.connect(self.select_boundary)
        self.canvas.boundaryMeshSelected.connect(self.edit_mesh_boundary)
        self.canvas.moveRequested.connect(self.move_boundary)
        self.canvas.beamRequested.connect(self.place_beam)
        self.canvas.pointerMoved.connect(lambda x,y: self.pointer.setText(f"x {x:.6g}   y {y:.6g} {self.document.geometry.units}"))
        self.canvas.panelSelected.connect(self.inspect_panel)
        dl.addWidget(self.canvas,1)
        inspect = QHBoxLayout()
        inspect.addWidget(QLabel("Inspect boundary"))
        self.inspect_boundary = QComboBox()
        self.inspect_boundary.setMinimumWidth(100)
        self.inspect_boundary.currentIndexChanged.connect(self.choose_inspection_boundary)
        inspect.addWidget(self.inspect_boundary, 1)
        inspect.addWidget(QLabel("Panel"))
        self.inspect_number = QSpinBox()
        self.inspect_number.setRange(1, 1)
        self.inspect_number.valueChanged.connect(self.update_panel_inspection)
        inspect.addWidget(self.inspect_number)
        self.zoom_panel_button = QPushButton("Zoom to panel")
        self.zoom_panel_button.clicked.connect(self.zoom_to_panel)
        inspect.addWidget(self.zoom_panel_button)
        inspect.addWidget(self.zoom_mesh_button)
        dl.addLayout(inspect)
        self.panel_info = QLabel()
        self.panel_info.setWordWrap(True)
        self.panel_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        dl.addWidget(self.panel_info)
        self.mesh_legend = QLabel("Boundary mesh: alternating line panels · ○ nodes · × integration points")
        self.mesh_legend.setWordWrap(True)
        self.mesh_legend.setStyleSheet("color: #165777; font-size: 11px;")
        dl.addWidget(self.mesh_legend)
        self.hint = QLabel("Select a drawing tool or import a planar DXF / coordinate file.")
        self.hint.setWordWrap(True)
        dl.addWidget(self.hint)
        outer.addWidget(drawing,1)
        right = QWidget()
        right.setFixedWidth(360)
        r = QVBoxLayout(right)
        r.setContentsMargins(0,0,0,0)
        mesh = QGroupBox("Boundary mesh")
        self.mesh_group = mesh
        mf = QFormLayout(mesh)
        self.mesh_target = QComboBox()
        self.mesh_target.currentIndexChanged.connect(self.choose_mesh_target)
        self.mesh_target.setToolTip("Click a boundary with Mesh settings, select it in the boundary list, "
                                   "or choose it here. Shared defaults affect settings without boundary overrides.")
        mf.addRow("Edit mesh for", self.mesh_target)
        self.mesh_status = QLabel()
        self.mesh_status.setWordWrap(True)
        self.mesh_status.setStyleSheet("font-size: 11px; color: #165777;")
        mf.addRow(self.mesh_status)
        self.panel_length, self.curve_tol = number(1,True),number(.01,True)
        self.minimum = QSpinBox()
        self.minimum.setRange(1,20000)
        self.grading = number(2,True)
        self.grading.setMinimum(1)
        self.integration_order = QSpinBox()
        self.integration_order.setRange(1,64)
        mf.addRow("Maximum panel length",self.panel_length)
        mf.addRow("Curve tolerance",self.curve_tol)
        mf.addRow("Minimum panels",self.minimum)
        mf.addRow("Corner grading",self.grading)
        mf.addRow("Gauss points / panel",self.integration_order)
        self.minimum.setToolTip("Minimum number of panels for this boundary. Length, curve and corner "
                                "constraints can increase the actual count shown in the table.")
        self.integration_order.setToolTip("Integration order = weighted Gauss sample points inside each panel. "
                                         "4 means four samples, not four panels. This setting controls geometry "
                                         "display/export; Maxwell kernel integration uses separate settings.")
        self.quadrature_note = QLabel("Weighted integration samples inside each panel. "
                                      "Display/export setting; Maxwell field integrals use separate quadrature.")
        self.quadrature_note.setWordWrap(True)
        self.quadrature_note.setStyleSheet("font-size: 11px; color: #165777;")
        self.quadrature_note.setMinimumHeight(4*self.quadrature_note.fontMetrics().lineSpacing())
        mf.addRow(self.quadrature_note)
        self.panel_length.setToolTip("Maximum boundary panel length in the selected drawing units. "
                                     "Axial pipe length is set when running the impedance calculation.")
        self.curve_tol.setToolTip("Native curve-to-chord tolerance in the selected drawing units; not an impedance error target.")
        self.mesh_button = QPushButton("Generate / update mesh")
        self.mesh_button.clicked.connect(self.apply_mesh)
        mf.addRow(self.mesh_button)
        self.reset_mesh_button = QPushButton("Use shared defaults")
        self.reset_mesh_button.clicked.connect(self.reset_mesh)
        self.reset_mesh_button.setToolTip("Remove only the selected boundary's saved overrides and regenerate it.")
        mf.addRow(self.reset_mesh_button)
        self.rough_mesh_note = QLabel("Generated roughness interfaces inherit the aperture's panel partition.")
        self.rough_mesh_note.setWordWrap(True)
        self.rough_mesh_note.setStyleSheet("font-size: 11px;")
        mf.addRow(self.rough_mesh_note)
        r.addWidget(mesh)
        materials = QGroupBox("Materials · inside to outside")
        ml = QVBoxLayout(materials)
        self.regions = QTableWidget(0,2)
        self.regions.setHorizontalHeaderLabels(["Region","Material"])
        self.regions.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.regions.verticalHeader().hide()
        self.regions.setMaximumHeight(220)
        ml.addWidget(self.regions)
        properties = QPushButton("View / edit material properties")
        properties.clicked.connect(self.edit_material)
        ml.addWidget(properties)
        note = QLabel("Vacuum aperture: σ=0, εr=μr=1.\nPEC: exact ideal wall; choose Exterior.\nAnnealed Cu preset: σ=5.8×10⁷ S/m.\nAir preset: σ=0, εr=μr=1.\nOpen Air/Vacuum exterior requires β<1.\nOther materials: Custom…")
        note.setWordWrap(True)
        ml.addWidget(note)
        r.addWidget(materials)
        beam = QGroupBox("Beam position")
        bf = QFormLayout(beam)
        self.beam_x, self.beam_y = number(), number()
        bf.addRow("x",self.beam_x)
        bf.addRow("y",self.beam_y)
        button = QPushButton("Apply beam position")
        button.clicked.connect(lambda: self.place_beam(self.beam_x.value(),self.beam_y.value()))
        bf.addRow(button)
        self.beam_beta = number(1, True)
        self.beam_beta.setRange(1e-6,1.)
        self.beam_beta.setDecimals(10)
        bf.addRow("β = v/c", self.beam_beta)
        r.addWidget(beam)
        rough = QGroupBox("Surface roughness")
        self.rough_group = rough
        rough.setStyleSheet("QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled {background: #edf0f2; color: #8793a0;} QLabel:disabled {color: #8793a0;}")
        rf = QFormLayout(rough)
        self.rough_toggle = QCheckBox("Enable surface roughness")
        self.rough_toggle.setToolTip("Include surface roughness using a Gaussian distribution of surface heights. Click Apply β / roughness to commit this setting.")
        rf.addRow(self.rough_toggle)
        self.rough_height = number(.0003,True)
        self.rough_height.setDecimals(10)
        self.rough_height.setToolTip("RMS height in drawing units. The mean wall lies extent × RMS outward from the aperture reference plane.")
        rf.addRow("RMS roughness", self.rough_height)
        self.rough_slices = QSpinBox()
        self.rough_slices.setRange(4,256)
        self.rough_slices.setValue(32)
        rf.addRow("Profile slices",self.rough_slices)
        self.rough_extent = number(5., True)
        self.rough_extent.setRange(4., 12.)
        self.rough_extent.setDecimals(2)
        self.rough_extent.setToolTip("Gaussian truncation extent in RMS heights. Extents above 5 can be meshed but are currently rejected by the BEM field solver because of near-insulating-tail conditioning. Changing extent moves the mean wall relative to the input aperture.")
        rf.addRow("Extent [RMS heights]", self.rough_extent)
        self.rough_fields = (self.rough_height, self.rough_slices, self.rough_extent)
        self.rough_labels = tuple(rf.labelForField(field) for field in self.rough_fields)
        self.rough_toggle.toggled.connect(self.update_roughness_controls)
        self.update_roughness_controls()
        self.physics_button = QPushButton("Apply β / roughness")
        self.physics_button.clicked.connect(self.apply_physics)
        rf.addRow(self.physics_button)
        note = QLabel("Model: Gaussian surface-height distribution.\nUnchecked: smooth wall after Apply.\nAperture = air-side reference plane.\nMean wall = aperture + extent × RMS.\nBEM extent above 5: unsupported.\n5-RMS profile has a cutoff error.\nDepth profile; no longitudinal bumps.")
        note.setWordWrap(True)
        rf.addRow(note)
        r.addWidget(rough)
        state = QGroupBox("Geometry / solver status")
        sl = QVBoxLayout(state)
        self.status = QLabel()
        self.status.setWordWrap(True)
        sl.addWidget(self.status)
        r.addWidget(state)
        r.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right)
        scroll.setFixedWidth(380)
        outer.addWidget(scroll)
        self.pointer = QLabel()
        self.statusBar().addPermanentWidget(self.pointer)
        self.statusBar().showMessage("Ready · save an ARCS JSON case to use the same geometry in Python.")

    def refresh(self, fit=False):
        data = self.document.data
        self.selected = min(self.selected,len(self.document.geometry.boundaries)-1)
        self.name.setText(data["name"])
        self.units.blockSignals(True)
        self.units.setCurrentText(data["units"])
        self.units.blockSignals(False)
        self.boundaries.blockSignals(True)
        self.boundaries.clear()
        for index, (name, shape) in enumerate(self.document.geometry.boundaries):
            material = data['regions'][index]['material']
            self.boundaries.addItem(f"{name}\n{shape.kind} · inside: {MATERIAL_LABELS.get(material, material)}")
        self.boundaries.setCurrentRow(self.selected)
        self.boundaries.blockSignals(False)
        self.regions.setRowCount(len(data["regions"]))
        for index, region in enumerate(data["regions"]):
            label = "Vacuum aperture" if index == 0 else ("Exterior (∞)" if index == len(data["regions"])-1 else f"Layer {index}")
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.regions.setItem(index,0,item)
            if index == 0:
                item = QTableWidgetItem("Vacuum · fixed")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.regions.setItem(index,1,item)
            else:
                combo = QComboBox()
                labels = list(data["materials"])
                if region["material"] not in labels:
                    labels.append(region["material"])
                for material in labels:
                    combo.addItem(MATERIAL_LABELS.get(material, material), material)
                combo.addItem("Custom…", "Custom…")
                for position, material in enumerate(labels):
                    if data["materials"].get(material, {}).get("kind") == "pec":
                        combo.setItemData(position, "Exact PEC boundary: tangential E=0. Use as Exterior; no conductivity parameter.", Qt.ItemDataRole.ToolTipRole)
                    elif material == "air":
                        combo.setItemData(position, "Ideal air preset (σ=0, εr=μr=1). Open Exterior is supported for β<1; finite insulating gaps are not implemented.", Qt.ItemDataRole.ToolTipRole)
                    elif material == "Annealed Cu":
                        combo.setItemData(position, "Annealed Cu preset: conductivity=5.8e7 S/m, εr=μr=1, relaxation time=0. Inspect/edit properties for your material.", Qt.ItemDataRole.ToolTipRole)
                    elif material not in BUILTIN_MATERIALS:
                        combo.setItemData(position, "User-defined material from this case; inspect/edit its saved properties.", Qt.ItemDataRole.ToolTipRole)
                combo.setCurrentIndex(combo.findData(region["material"]))
                combo.currentIndexChanged.connect(lambda selected,index=index,widget=combo:
                    self.change_material(index,widget.itemData(selected)) if selected >= 0 else None)
                self.regions.setCellWidget(index,1,combo)
        for widget,value in [(self.beam_x,data["beam"][0]),(self.beam_y,data["beam"][1])]:
            widget.setValue(value)
        self.load_mesh_controls()
        self.beam_beta.setValue(data.get("beta",1.))
        values = data.get("roughness") or {}
        self.rough_toggle.setChecked(bool(values))
        self.rough_height.setValue(values.get("rms_height", .3e-6/self.document.geometry.unit_scale))
        self.rough_slices.setValue(values.get("slices",32))
        self.rough_extent.setValue(values.get("extent_sigma",5.))
        shape = self.document.geometry.boundaries[self.selected][1]
        self.boundary_info.setText(f"Area: {shape.area/self.document.geometry.unit_scale**2:.7g} {data['units']}²\n"
                                  f"Perimeter: {shape.perimeter/self.document.geometry.unit_scale:.7g} {data['units']}")
        issues = self.document.issues()
        if self.document.geometry.roughness is not None and self.document.geometry.roughness.extent_sigma > 5:
            issues.append("Extended roughness mesh is visible, but the field solver rejects extent > 5 because near-insulating tails are unstable. The 5-RMS profile is truncated; see the cutoff audit.")
        if self.document.geometry.materials.get(data["regions"][-1]["material"], {}).get("kind") == "pec":
            physics = ("PEC wall: RW=0; indirect space charge retained.\nAt β=1 this uniform-pipe impedance is zero."
                       if len(data["regions"]) == 2 else
                       "Conducting coating(s) on exact PEC.\nCoating losses are retained; PEC backing is lossless.")
        else:
            physics = "Wall impedance: RW + indirect space charge."
        physics += "\nDirect space charge excluded.\nPlots and CSV use Im(Z) directly.\nAccuracy still requires mesh and physics checks."
        self.status.setText("Valid closed, nested geometry.\n\n"+("\n\n".join(issues) if issues else physics))
        self.canvas.document = self.document
        self.canvas.selected = self.selected
        self.canvas.move_shape = None
        self.update_mesh()
        if fit:
            self.canvas.fit()
        self.undo_action.setEnabled(self.document.can_undo)
        self.redo_action.setEnabled(self.document.can_redo)
        self.setWindowTitle(("● " if self.document.dirty else "")+"ARCS · "+data["name"])

    def run_edit(self, operation, message="", fit=False):
        try:
            operation()
        except (ValueError, RuntimeError, OSError, NotImplementedError) as error:
            QMessageBox.warning(self,"Geometry unchanged",str(error))
            return False
        self.refresh(fit)
        if message:
            self.statusBar().showMessage(message,10000)
        return True

    def select_boundary(self,index):
        if 0 <= index < len(self.document.geometry.boundaries):
            self.selected = index
            self.inspected_mesh_index = index
            self.edit_shared_mesh = False
            self.refresh()

    def load_mesh_controls(self):
        case = self.document.geometry
        self.mesh_target.blockSignals(True)
        self.mesh_target.clear()
        for name, _ in case.boundaries:
            self.mesh_target.addItem(name, name)
        self.mesh_target.addItem("Shared defaults", None)
        self.mesh_target.setCurrentIndex(len(case.boundaries) if self.edit_shared_mesh else self.selected)
        self.mesh_target.blockSignals(False)
        name = None if self.edit_shared_mesh else case.boundaries[self.selected][0]
        settings = case.mesh if name is None else case.mesh_settings(name)
        values = settings.input_values(case.unit_scale)
        for widget, key in ((self.panel_length, "max_panel_length"), (self.curve_tol, "curve_tolerance"),
                            (self.minimum, "minimum_panels"), (self.grading, "corner_grading"),
                            (self.integration_order, "integration_order")):
            widget.setValue(values[key])
        custom = bool(name is not None and case.boundary_meshes.get(name))
        self.reset_mesh_button.setEnabled(custom)
        self.mesh_status.setText("Shared defaults · boundaries without overrides" if name is None else
                                 f"{name} · {'boundary-specific settings' if custom else 'shared defaults'}\n"
                                 f"Lengths in {case.units}. Click Generate to apply.")
        self.rough_mesh_note.setVisible(case.roughness is not None)

    def choose_mesh_target(self, index):
        if index < 0:
            return
        if self.mesh_target.itemData(index) is None:
            self.edit_shared_mesh = True
            self.load_mesh_controls()
        else:
            self.select_boundary(index)

    def edit_mesh_boundary(self, index):
        if index < len(self.document.geometry.boundaries):
            self.select_boundary(index)
        else:
            self.select_boundary(0)
            self.choose_inspection_boundary(index)
            self.statusBar().showMessage("Generated roughness interfaces inherit the aperture mesh; editing aperture settings.", 10000)
        self.minimum.setFocus(Qt.FocusReason.OtherFocusReason)

    def select_tool(self,tool):
        self.tool_actions[tool].setChecked(True)
        self.canvas.set_tool(tool)
        self.hint.setText("Draw the next outer boundary, enclosing all existing boundaries." if self.adding_interface else
            {"crop":"Drag a rectangle: keep the selected outline inside it.",
             "circle":"Drag from the circle centre to its radius.",
             "polygon":"Click ordered vertices. Enter/right click finishes; Backspace removes a vertex.",
             "beam":"Click inside the vacuum aperture to place the beam.",
             "inspect":"Click a visible mesh panel, or choose its boundary and panel number below the drawing.",
             "mesh":"Click a boundary edge to edit its own mesh settings in the right panel. No geometry is moved.",
             "select":"Click an outline/region to select it. Drag to move it."}.get(tool,"Drag opposite corners. Edit exact dimensions afterwards."))

    def set_add_interface(self,checked):
        self.adding_interface = checked
        self.select_tool(self.canvas.tool)

    def apply_drawn_shape(self,shape,mode):
        if self.adding_interface and mode != "intersect":
            operation = lambda: self.document.add_interface(shape)
        else:
            kind = "intersect" if mode == "intersect" else ["replace","union","subtract","intersect"][self.operation.currentIndex()]
            operation = lambda: self.document.set_shape(self.selected,shape,operation=kind)
        if self.run_edit(operation,"Boundary updated."):
            if self.adding_interface and mode != "intersect":
                self.selected = len(self.document.geometry.boundaries)-1
                self.add_button.setChecked(False)
                self.refresh()
            self.select_tool("select")

    def edit_boundary(self):
        dialog = ShapeDialog(self.document.shape(self.selected),self.document.geometry.units,self)
        if dialog.exec():
            self.apply_drawn_shape(dialog.result_shape,"draw")

    def delete_boundary(self):
        self.run_edit(lambda: self.document.delete_interface(self.selected),"Interface removed.")

    def move_boundary(self,index,dx,dy):
        self.run_edit(lambda: self.document.transform(index,translation=(dx,dy)),"Boundary moved.")

    def place_beam(self,x,y):
        self.run_edit(lambda: self.document.set_beam(x,y),"Beam position updated.")

    def change_material(self,index,name):
        if name == "Custom…":
            self.edit_material(index,custom=True)
        else:
            self.run_edit(lambda: self.document.set_region(index,name),"Material assigned.")

    def edit_material(self,index=None,custom=False):
        # clicked(bool) is not a region index.
        if isinstance(index,bool) or index is None:
            index = self.regions.currentRow() if self.regions.currentRow() >= 0 else 1
        index = min(index,len(self.document.geometry.regions)-1)
        region = self.document.data["regions"][index]
        name = region["material"]
        if index == 0:
            QMessageBox.information(self,"Vacuum properties","The aperture is fixed vacuum: conductivity=0 S/m, εr=μr=1 and relaxation time=0. Direct space charge is excluded from ARCS impedance output.")
            self.refresh()
            return
        if not custom and self.document.data["materials"].get(name, {}).get("kind") == "pec":
            QMessageBox.information(self,"PEC properties","PEC is an exact boundary condition: tangential E=0. It has no finite conductivity, permittivity, permeability or relaxation-time settings. Choose it as Exterior. Finite conducting coatings may precede it; conductivity-gradient roughness is unsupported with PEC.")
            self.refresh()
            return
        if name == "vacuum" and not custom:
            QMessageBox.information(self,"Vacuum properties","Vacuum is fixed: conductivity=0 and εr=μr=1. It is always the aperture material. Open Vacuum as Exterior is supported for β<1; finite insulating gaps are not implemented.")
            self.refresh()
            return
        if name == "air" and not custom:
            values = self.document.data["materials"][name]
            stored = (f"This case: conductivity={values['conductivity']:g} S/m, "
                      f"εr={values['epsilon_r']:g}, μr={values['mu_r']:g}, "
                      f"relaxation time={values['relaxation_time']:g} s.\n\n")
            QMessageBox.information(self,"Air preset",stored+"The built-in Air preset uses conductivity=0 S/m, εr=μr=1 and relaxation time=0, exactly like ideal Vacuum. Choose it as Exterior with β<1 and smooth conducting layers. The exact β=1 open-exterior limit, finite insulating gaps and other dielectric exterior values are not implemented.")
            self.refresh()
            return
        properties = self.document.data["materials"].get(name)
        if properties is not None and properties.get("kind") == "pec":
            properties = None
        dialog = MaterialDialog("custom" if custom or name in ("air","unspecified") else name,
                                properties,self)
        if dialog.exec():
            try:
                label,properties = dialog.definition()
            except ValueError:
                QMessageBox.warning(self,"Material unchanged","Enter finite numeric properties.")
                self.refresh()
                return
            self.run_edit(lambda: self.document.set_region(index,label,properties),"Material properties saved.")
        else:
            self.refresh()

    def change_units(self,units):
        self.run_edit(lambda: self.document.set_units(units),"Display units changed; physical dimensions preserved.",fit=True)

    def rename_case(self):
        data = self.document.data
        data["name"] = self.name.text()
        self.run_edit(lambda: self.document.commit(data))

    def update_roughness_controls(self, checked=None):
        enabled = self.rough_toggle.isChecked()
        for widget in self.rough_fields + self.rough_labels:
            widget.setEnabled(enabled)

    def apply_physics(self):
        roughness = None
        if self.rough_toggle.isChecked():
            roughness = {"model":"conductivity_gradient", "rms_height":self.rough_height.value(),
                         "slices":self.rough_slices.value(), "extent_sigma":self.rough_extent.value()}
        self.run_edit(lambda: self.document.set_physics(self.beam_beta.value(),roughness),
                      "Beam velocity and conductivity-gradient settings updated.")

    def apply_mesh(self):
        values = {"max_panel_length":self.panel_length.value(),"curve_tolerance":self.curve_tol.value(),
                  "minimum_panels":self.minimum.value(),"corner_grading":self.grading.value(),
                  "integration_order":self.integration_order.value()}
        boundary = None if self.edit_shared_mesh else self.document.geometry.boundaries[self.selected][0]
        target = "Shared defaults" if boundary is None else boundary
        if self.run_edit(lambda: self.document.set_mesh(values, boundary), f"{target}: mesh settings saved and regenerated."):
            self.mesh_toggle.setChecked(True)
            self.mesh_view.setCurrentText("Mesh only")
            self.select_tool("mesh")
            self.update_mesh()

    def reset_mesh(self):
        if not self.edit_shared_mesh:
            self.run_edit(lambda: self.document.reset_boundary_mesh(self.selected),
                          "Selected boundary now uses shared mesh defaults.")

    def update_mesh(self):
        self.canvas.show_mesh = self.mesh_toggle.isChecked()
        self.canvas.mesh_only = self.mesh_view.currentIndex() == 1
        self.canvas.show_nodes = self.nodes_toggle.isChecked()
        self.canvas.show_quadrature = self.quadrature_toggle.isChecked()
        self.canvas.show_panel_detail = self.detail_toggle.isChecked()
        self.canvas.detail_mesh = None
        self.canvas.mesh = None
        self.canvas.meshes = []
        self.mesh_legend.setVisible(self.canvas.show_mesh)
        try:
            case = self.document.geometry
            # Generate/cache ALL interfaces independently of display controls.
            # Hiding the overlay or selecting one outline preserves totals.
            if getattr(self, "_mesh_geometry", None) is not case:
                actual = case.material_panels() if case.roughness is not None else [
                    (name, case.to_panels(name)) for name, _ in case.boundaries]
                self._all_mesh_cache = [(i, name, mesh) for i, (name, mesh) in enumerate(actual)]
                self._mesh_geometry = case
            meshes = self._all_mesh_cache
            self.inspected_mesh_index = min(self.inspected_mesh_index, len(meshes)-1)
            self.inspect_boundary.blockSignals(True)
            self.inspect_boundary.clear()
            self.inspect_boundary.addItems([name for _, name, _ in meshes])
            self.inspect_boundary.setCurrentIndex(self.inspected_mesh_index)
            self.inspect_boundary.blockSignals(False)
            panels = sum(len(mesh.panel_start) for _, _, mesh in meshes)
            nodes = panels  # each interface is a separate closed line mesh
            integration = sum(mesh.size for _, _, mesh in meshes)
            visible = meshes if self.mesh_scope.currentIndex() == 0 else [
                item for item in meshes if item[0] == self.inspected_mesh_index]
            if self.canvas.show_mesh:
                self.canvas.meshes = visible
                self.canvas.mesh = next((mesh for index, _, mesh in visible if index == self.inspected_mesh_index), None)
            self.mesh_counts = {"meshes": len(meshes), "panels": panels, "nodes": nodes,
                                "integration_points": integration,
                                "displayed_meshes": len(self.canvas.meshes)}
            self.mesh_summary.setText(f"Boundary meshes: {len(meshes):,}   ·   Panels: {panels:,}   ·   Nodes: {nodes:,}\n"
                                      f"Displayed meshes: {len(self.canvas.meshes):,}   ·   Integration points: {integration:,}")
            self.mesh_table.setRowCount(len(meshes))
            for row, (_, name, mesh) in enumerate(meshes):
                for column, value in enumerate((name, f"{len(mesh.panel_start):,}", f"{len(mesh.panel_start):,}")):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.mesh_table.setItem(row, column, item)
            selected_mesh = next((mesh for index, _, mesh in meshes if index == self.selected), meshes[0][2])
            lengths = np.linalg.norm(selected_mesh.panel_end-selected_mesh.panel_start, axis=1)/case.unit_scale
            self.mesh_info.setText(f"Selected boundary: {len(lengths):,} panels\n"
                                   f"Panel length {lengths.min():.4g}–{lengths.max():.4g} {case.units}\n"
                                   + ("Overlay hidden; mesh counts remain visible." if not self.canvas.show_mesh
                                      else "Alternating edges show the actual line mesh."))
            self.zoom_mesh_button.setEnabled(len(meshes) > 1)
            self.inspect_number.setEnabled(True)
            self.zoom_panel_button.setEnabled(True)
            self.inspect_number.blockSignals(True)
            self.inspect_number.setMaximum(len(meshes[self.inspected_mesh_index][2].panel_start))
            self.inspect_number.blockSignals(False)
            self.update_panel_inspection()
        except (ValueError, RuntimeError, NotImplementedError) as error:
            self.mesh_counts = None
            self.mesh_summary.setText("Mesh construction failed: "+str(error))
            self.mesh_table.setRowCount(0)
            self.mesh_info.setText(str(error))
            self.zoom_mesh_button.setEnabled(False)
            self.inspect_number.setEnabled(False)
            self.zoom_panel_button.setEnabled(False)
            self.panel_info.setText("No generated panel is available to inspect.")
            self.statusBar().showMessage(str(error))
        self.canvas.update()

    def choose_inspection_boundary(self, index):
        if index < 0:
            return
        self.edit_shared_mesh = False
        self.selected = index if index < len(self.document.geometry.boundaries) else 0
        self.canvas.selected = self.selected
        self.boundaries.blockSignals(True)
        self.boundaries.setCurrentRow(self.selected)
        self.boundaries.blockSignals(False)
        self.load_mesh_controls()
        case = self.document.geometry
        shape = case.boundaries[self.selected][1]
        self.boundary_info.setText(f"Area: {shape.area/case.unit_scale**2:.7g} {case.units}²\n"
                                  f"Perimeter: {shape.perimeter/case.unit_scale:.7g} {case.units}")
        self.inspected_mesh_index = index
        self.inspect_number.setValue(1)
        self.mesh_toggle.setChecked(True)
        self.update_mesh()

    def inspect_panel(self, boundary, panel):
        self.choose_inspection_boundary(boundary)
        self.inspect_number.setValue(panel+1)

    def update_panel_inspection(self):
        meshes = getattr(self, "_all_mesh_cache", [])
        if not meshes or self.inspected_mesh_index >= len(meshes):
            return
        index, name, mesh = meshes[self.inspected_mesh_index]
        panel = min(self.inspect_number.value()-1, len(mesh.panel_start)-1)
        self.canvas.detail_mesh = mesh
        self.canvas.detail_boundary, self.canvas.detail_panel = index, panel
        self.canvas.detail_name = name
        unit, units = self.document.geometry.unit_scale, self.document.geometry.units
        start, end = mesh.panel_start[panel]/unit, mesh.panel_end[panel]/unit
        length = np.linalg.norm(end-start)
        count = np.count_nonzero(mesh.panel_index == panel)
        self.panel_info.setText(f"{name} · panel {panel+1:,} / {len(mesh.panel_start):,} · "
                               f"length {length:.7g} {units} · {count} integration points\n"
                               f"Start ({start[0]:.8g}, {start[1]:.8g}) → "
                               f"End ({end[0]:.8g}, {end[1]:.8g}) {units}")
        self.canvas.update()

    def zoom_to_panel(self):
        mesh = self.canvas.detail_mesh
        if mesh is None:
            return
        panel, unit = self.canvas.detail_panel, self.document.geometry.unit_scale
        start, end = mesh.panel_start[panel]/unit, mesh.panel_end[panel]/unit
        self.canvas.center = (start+end)/2
        span = max(4*np.linalg.norm(end-start), np.finfo(float).tiny)
        self.canvas.scale = .7*min(self.canvas.width(), self.canvas.height())/span
        self.mesh_toggle.setChecked(True)
        self.canvas.update()

    def zoom_layer_meshes(self):
        """Show actual thin interfaces near the aperture's rightmost point."""
        meshes = getattr(self, "_all_mesh_cache", [])
        if len(meshes) < 2:
            return
        first = meshes[0][2].panel_start
        point = first[np.argmax(first[:, 0])]
        last = meshes[-1][2].panel_start
        outer = last[np.argmin(np.linalg.norm(last-point, axis=1))]
        unit = self.document.geometry.unit_scale
        span = max(5*np.linalg.norm(outer-point)/unit, 1e-9)
        self.canvas.center = (point+outer)/(2*unit)
        self.canvas.scale = .8*min(self.canvas.width(), self.canvas.height())/span
        self.mesh_scope.setCurrentIndex(0)
        self.mesh_toggle.setChecked(True)
        self.canvas.update()

    def undo(self):
        self.document.undo()
        self.refresh()

    def redo(self):
        self.document.redo()
        self.refresh()

    def _discard(self):
        if not self.document.dirty:
            return True
        answer = QMessageBox.question(self,"Unsaved geometry","Save changes before closing this case?",
            QMessageBox.StandardButton.Save|QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_case()
        return True

    def new_case(self):
        if self._discard():
            self.document = GeometryDocument()
            self.filename = None
            self.selected = 0
            self.refresh(fit=True)

    def open_case(self):
        path,_ = QFileDialog.getOpenFileName(self,"Open / import cross-section","",
            "2D geometry (*.json *.csv *.dxf);;ARCS cases (*.json);;Ordered coordinates (*.csv);;DXF drawings (*.dxf)")
        if not path or not self._discard():
            return
        units = None
        if Path(path).suffix.lower() == ".csv":
            units,ok = QInputDialog.getItem(self,"Coordinate units","CSV coordinates are in:",["mm","m","um","nm"],0,False)
            if not ok:
                return
        try:
            try:
                result = import_geometry(path,units=units,target_units=self.document.geometry.units,
                                         curve_tolerance=self.curve_tol.value())
            except ValueError as error:
                if "no declared units" not in str(error):
                    raise
                units,ok = QInputDialog.getItem(self,"DXF source units","Unitless DXF coordinates are in:",["mm","m","um","nm"],0,False)
                if not ok:
                    return
                result = import_geometry(path,units=units,target_units=self.document.geometry.units,
                                         curve_tolerance=self.curve_tol.value())
            self.document = GeometryDocument(result.geometry)
            self.filename = path if Path(path).suffix.lower() == ".json" else None
            self.selected = 0
            self.refresh(fit=True)
            if result.notes:
                QMessageBox.information(self,"Import summary","\n\n".join(result.notes))
        except (ValueError,RuntimeError,OSError) as error:
            QMessageBox.warning(self,"Import failed",str(error))

    def save_case(self):
        path = self.filename
        if not path:
            path,_ = QFileDialog.getSaveFileName(self,"Save editable ARCS geometry","cross_section.json","ARCS geometry (*.json)")
        if not path:
            return False
        if not Path(path).suffix:
            path += ".json"
        if self.run_edit(lambda: self.document.save(path),"Geometry and material properties saved."):
            self.filename = path
            return True
        return False

    def save_dxf(self):
        path,_ = QFileDialog.getSaveFileName(self,"Export planar DXF","cross_section.dxf","DXF (*.dxf)")
        if path:
            self.run_edit(lambda: export_dxf(self.document.geometry,path),"DXF geometry exported. Material properties remain in the ARCS JSON.")

    def save_mesh(self):
        path,_ = QFileDialog.getSaveFileName(self,"Export boundary panels","boundary_panels.csv","CSV (*.csv)")
        if path:
            self.run_edit(lambda: self.document.geometry.export_panels(path),"Boundary panels exported in metres.")

    def save_view(self):
        path,_ = QFileDialog.getSaveFileName(self,"Save geometry view","cross_section.png","PNG (*.png)")
        if path and not self.canvas.grab().save(path):
            QMessageBox.warning(self,"Could not save view","The image could not be written.")

    def closeEvent(self,event):
        if self._discard():
            event.accept()
        else:
            event.ignore()


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARCS Python desktop cross-section editor.")
    parser.add_argument("input",nargs="?",help="Optional ARCS JSON case")
    parser.add_argument("--screenshot",help="Render the actual editor window to PNG and exit")
    args = parser.parse_args(argv)
    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    application = QApplication.instance() or QApplication(sys.argv[:1])
    window = GeometryWindow(args.input)
    window.show()
    def capture():
        window.canvas.fit()
        application.processEvents()
        if not window.grab().save(args.screenshot):
            print("Could not save the editor screenshot.",file=sys.stderr)
            application.exit(1)
        else:
            application.quit()
    if args.screenshot:
        QTimer.singleShot(100,capture)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
