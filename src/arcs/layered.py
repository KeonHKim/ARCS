"""Nonlocal full-Maxwell matching of nested conducting material regions.

Each homogeneous annulus is represented by scalar single-layer potentials
for Ez and Hz on both interfaces. Normal derivatives use the region Green representation (inner normals
point into the aperture); tangential derivatives use those single-layer
potentials. Matching at the outer interface eliminates its two field
traces, recursively producing the exterior admittance at the vacuum boundary.
The vacuum solve retains all four reaction fields. No reference impedance is
used in assembly. The exterior is an infinite conductor or an exact PEC
termination (Ez=Et=0); no volume kernel is constructed inside PEC.
"""
import numpy as np
from .special import decaying_k as kv
from scipy.integrate import quad_vec
from scipy.linalg import solve
from numpy.polynomial.legendre import leggauss
from .constants import C0, Z0, MU0
from .interfaces import polygon_panels
from .pec_panels import PolygonPECSolver
from .materials import PEC


def equilibrated_solve(matrix, rhs):
    rows = np.max(abs(matrix), axis=1)
    scaled = matrix/rows[:, None]
    columns = np.max(abs(scaled), axis=0)
    return solve(scaled/columns[None, :], rhs/rows[:, None],
                 check_finite=False, assume_a="gen")/columns[:, None]


def offset_boundary(p, distance):
    """Parallel straight-panel offset with exact miters at polygon corners.

    No corner is rounded. Offsets that cross, collapse or invert are rejected.
    The input panel partition is retained, including its corner grading.
    """
    n = p.normal
    previous = np.roll(n, 1, axis=0)
    denominator = 1+np.sum(previous*n, axis=1)
    if np.any(denominator < 1e-12):
        raise ValueError("Offset is undefined at a reversing boundary segment.")
    vertices = p.starts+distance*(previous+n)/denominator[:, None]
    panels = polygon_panels(vertices, panels_per_edge=1, grading_power=1.)
    from .geometry_input import _nested_panels
    if distance > 0 and not _nested_panels(p.panels, panels):
        raise ValueError("Material offset crosses/touches the aperture; reduce thickness or change geometry.")
    return PolygonPECSolver(panels)


def kernel(target, source, q, selfjump=None):
    """Finite straight-panel Yukawa value, target gradient and double layer.

    Closely spaced interfaces use a hyperbolic integration coordinate. This
    resolves their normal separation independently of the panel length. It
    evaluates the finite-panel Green function, not a local wall model.
    """
    nodes, gauss_weights = leggauss(16)
    vectors = source.ends-source.starts
    positions = source.starts[:, None, :] + (nodes[None, :, None]+1)/2*vectors[:, None, :]
    displacement = target.centres[:, None, None, :]-positions[None, :, :, :]
    radius = np.linalg.norm(displacement, axis=-1)
    weights = source.lengths[:, None]*gauss_weights/2
    single = np.sum(kv(0, q*radius)*weights[None, :, :], axis=2)/(2*np.pi)
    local = target.centres[:, None, :]-source.starts[None, :, :]
    projection = np.einsum("ijk,jk->ij", local, source.tangent)
    height = np.einsum("ijk,jk->ij", local, source.normal)
    double = (np.sum(q*kv(1, q*radius)/radius*weights[None, :, :], axis=2)
              * height/(2*np.pi))
    near = ((abs(height) > 1e-14*source.scale)
            & (abs(height) < .3*source.lengths[None, :])
            & (projection > -.2*source.lengths[None, :])
            & (projection < 1.2*source.lengths[None, :]))
    if selfjump is not None:
        np.fill_diagonal(near, False)
    pairs = np.argwhere(near)
    if len(pairs):
        rows, columns = pairs.T
        h = abs(height[rows, columns])
        along = projection[rows, columns]
        length = source.lengths[columns]
        lower = np.arcsinh(-along/h)
        upper = np.arcsinh((length-along)/h)

        def integrand(t):
            coordinate = lower+(upper-lower)*t
            r = h*np.cosh(coordinate)
            return np.vstack((kv(0, q*r)*r, q*h*kv(1, q*r)))*(upper-lower)[None, :]

        integrated = quad_vec(integrand, 0., 1., epsabs=1e-12, epsrel=1e-10)[0]/(2*np.pi)
        single[rows, columns] = integrated[0]
        double[rows, columns] = np.sign(height[rows, columns])*integrated[1]
    if selfjump is not None:
        upper = np.minimum(source.lengths/2, 40/q.real)
        self_integral = upper*quad_vec(lambda t: kv(0, q*upper*t), 0., 1.,
                                       epsabs=1e-12, epsrel=1e-11)[0]/np.pi
        np.fill_diagonal(single, self_integral)
        np.fill_diagonal(double, 0.)
    # The tangential source-panel integral is an exact endpoint difference.
    first = np.linalg.norm(local, axis=-1)
    last = np.linalg.norm(local-vectors[None, :, :], axis=-1)
    along = (kv(0, q*first)-kv(0, q*last))/(2*np.pi)
    gradient = (along[:, :, None]*source.tangent[None, :, :]
                - double[:, :, None]*source.normal[None, :, :])
    normal = np.einsum("ijk,ik->ij", gradient, target.normal)
    tangent = np.einsum("ijk,ik->ij", gradient, target.tangent)
    if selfjump is not None:
        np.fill_diagonal(normal, selfjump)
        np.fill_diagonal(tangent, 0.)
    return single, normal, tangent, double


def _material_scales(f, metal, beta):
    k = 2*np.pi*f/(beta*C0)
    y = metal.admittivity(f)
    q2 = metal.transverse_wavenumber_squared(f, beta)
    return k, y, q2, np.sqrt(q2)


def _field_map(normal, tangent, k, y, q2, mu_r, beta):
    # (u,v) -> (Et/(k² Z0 I), Ht/I), with the global outward normal.
    return np.block([[tangent/q2, -beta*mu_r*normal/q2],
                     [-1j*k*Z0*y*normal/q2, k*k*tangent/q2]])


def adm_outer(p, f, metal, beta):
    """Infinite conductor's nonlocal two-field exterior admittance."""
    k, y, q2, q = _material_scales(f, metal, beta)
    single, normal, tangent, double = kernel(p, p, q, -.5)
    from .materials import Vacuum
    if isinstance(metal, Vacuum):
        # Direct exterior single-layer representation: derivative of the
        # same trace. Avoid a discrete Calderon product, whose cancellation
        # error is amplified by 1/kappa^2 in a lossless exterior.
        normal_map = equilibrated_solve(single.T, normal.T).T
    else:
        normal_map = -equilibrated_solve(single, .5*np.eye(len(single))-double)
    tangent_map = equilibrated_solve(single.T, tangent.T).T
    return _field_map(normal_map, tangent_map, k, y, q2, metal.mu_r, beta)


def trace_tangent(p):
    """Differentiate a boundary trace using its nonlocal Laplace interpolant.

    Tangential differentiation is geometrical, independent of material q.
    The derivative of a prescribed zero PEC trace must be exactly zero.
    A short-range material Green kernel with constant source densities
    cannot represent this derivative reliably when its range is far below
    panel length. The normal material maps still use full Maxwell kernels.
    """
    if not hasattr(p, '_trace_tangent'):
        displacement = p.centres[:, None, :]-p.starts[None, :, :]
        projection = np.einsum('ijk,jk->ij', displacement, p.tangent)
        height = np.einsum('ijk,jk->ij', displacement, p.normal)
        a, b = p.lengths-projection, -projection
        along = np.log((a*a+height*height)/(b*b+height*height))/(4*np.pi)
        across = np.sign(height)*(np.arctan2(a, abs(height))-np.arctan2(b, abs(height)))/(2*np.pi)
        gradient = along[..., None]*p.tangent-across[..., None]*p.normal
        tangent = np.einsum('ijk,ik->ij', gradient, p.tangent)
        np.fill_diagonal(tangent, 0.)
        p._trace_tangent = equilibrated_solve(p.single_layer.T, tangent.T).T
    return p._trace_tangent


def adm_layer(inner, outer, f, metal, beta, exterior, *, pec_backed=False):
    """Eliminate one annulus by matching both tangential fields at its exterior."""
    k, y, q2, q = _material_scales(f, metal, beta)
    singles, tangents, doubles = [], [], []
    for i, target in enumerate((inner, outer)):
        single_row, tangent_row, double_row = [], [], []
        for j, source in enumerate((inner, outer)):
            jump = (-.5 if i == 0 else .5) if i == j else None
            single, _, tangent, double = kernel(target, source, q, jump)
            single_row.append(single)
            tangent_row.append(tangent)
            # Domain normals: -global at inner; +global at outer.
            double_row.append(double*(-1 if j == 0 else 1))
        singles.append(single_row)
        tangents.append(tangent_row)
        doubles.append(double_row)
    single = np.block(singles)
    ni, no = len(inner.lengths), len(outer.lengths)
    # Green identity: S*g_domain=(1/2+D_domain)*u. Restore global normals.
    signs = np.r_[-np.ones(ni), np.ones(no)]
    normal = signs[:, None]*equilibrated_solve(single, .5*np.eye(len(single))+np.block(doubles))
    if pec_backed or isinstance(exterior, PEC):
        # A trace's tangential derivative depends only on that boundary's
        # trace, not on values prescribed on a different interface.
        tangent = np.block([[trace_tangent(inner), np.zeros((ni, no))],
                            [np.zeros((no, ni)), trace_tangent(outer)]])
    else:
        tangent = equilibrated_solve(single.T, np.block(tangents).T).T

    def field(i, j):
        rows = slice(0, ni) if i == 0 else slice(ni, ni+no)
        columns = slice(0, ni) if j == 0 else slice(ni, ni+no)
        return _field_map(normal[rows, columns], tangent[rows, columns],
                          k, y, q2, metal.mu_r, beta)

    fii, fio = field(0, 0), field(0, 1)
    foi, foo = field(1, 0), field(1, 1)
    if isinstance(exterior, PEC):
        # Exact PEC conditions on the outer wall: Ez=0 and Et=0.
        # The coating's Hz trace is retained; Ht is a surface current and
        # must not be forced to zero. No exterior volume Green operator.
        constraint = np.vstack((np.c_[np.eye(no), np.zeros((no, no))], foo[:no]))
        drive = np.vstack((np.zeros((no, 2*ni)), foi[:no]))
        return fii-fio@equilibrated_solve(constraint, drive)
    return fii-fio@equilibrated_solve(foo-exterior, foi)


def material_boundaries(solver, wall):
    """Construct/cache actual interfaces, independently of frequency."""
    key = (wall, solver.roughness)
    if getattr(solver, "_layer_geometry_key", None) == key:
        return solver._layer_geometry
    inner = solver.pec
    if solver.roughness is not None:
        if solver.interfaces is not None:
            raise NotImplementedError("Use offset Wall layers for conductivity-gradient roughness; explicit rough interfaces need a depth-coordinate specification.")
        from .roughness import gradient_layers
        depths, materials = gradient_layers(wall, solver.roughness)
        boundaries = [inner]+[offset_boundary(inner, depth) for depth in depths[1:]]
    elif solver.interfaces is not None:
        if len(solver.interfaces) != len(wall.layers):
            raise ValueError("Provide one outer interface per finite layer.")
        boundaries = [inner]+[PolygonPECSolver(p) for p in solver.interfaces]
        materials = [layer.metal for layer in wall.layers]+[wall.substrate]
    else:
        layers = [layer for layer in wall.layers if layer.thickness > 0]
        depths = np.r_[0., np.cumsum([layer.thickness for layer in layers])]
        boundaries = [inner]+[offset_boundary(inner, depth) for depth in depths[1:]]
        materials = [layer.metal for layer in layers]+[wall.substrate]
    from .geometry_input import _nested_panels
    if not all(_nested_panels(a.panels, b.panels) for a,b in zip(boundaries[:-1],boundaries[1:])):
        raise ValueError("The material interface meshes must be strictly nested.")
    solver._layer_geometry_key = key
    solver._layer_geometry = (boundaries, materials)
    return boundaries, materials


def wall_admittance(solver, f, wall):
    if solver.roughness is not None and solver.roughness.extent_sigma > 5:
        raise NotImplementedError(
            "BEM roughness with extent_sigma > 5 is currently unsupported: "
            "near-insulating tails make the two-field layer elimination unreliable. "
            "extent_sigma=5 reproduces the truncated profile, not the converged full Gaussian. "
            "Geometry meshes remain available; see docs/roughness.md for the cutoff audit.")
    boundaries, materials = material_boundaries(solver, wall)
    admittance = materials[-1] if isinstance(materials[-1], PEC) else adm_outer(
        boundaries[-1], float(f), materials[-1], solver.beta)
    for index in range(len(boundaries)-2, -1, -1):
        admittance = adm_layer(boundaries[index], boundaries[index+1], float(f),
                               materials[index], solver.beta, admittance,
                               pec_backed=isinstance(materials[-1], PEC))
    return admittance, len(boundaries)
