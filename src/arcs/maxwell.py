"""Synchronous full-Maxwell boundary elements for 0 < beta <= 1.

The vacuum longitudinal electric and magnetic fields and both Cartesian
components of its transverse reaction field are retained. The conductor
uses a nonlocal modified-Helmholtz Dirichlet-to-Neumann operator. No local
surface-impedance approximation or cross-section reference is used here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve
from scipy.special import xlogy
from .special import decaying_k as kv
from .constants import C0, MU0, Z0
from .materials import Metal, PEC, Vacuum, Air, Wall, frequency_array
from .kernels import PanelKernels
from .fields import TransverseImpedance


@dataclass(frozen=True)
class MaxwellBoundaryState:
    """Physical reaction-field traces per ampere of beam current.

    electric_longitudinal and electric_transverse_reaction are in Ohm/m;
    magnetic_longitudinal is in 1/m. The transverse electric trace excludes
    the PEC beam field. Its normal component is added to the PEC drive in
    the matching equations. Axes of the transverse array are (panel, x/y).
    """
    electric_longitudinal: np.ndarray
    magnetic_longitudinal: np.ndarray
    electric_transverse_reaction: np.ndarray
    impedance_per_m: complex
    relative_residual: float
    component: str = "resistive_wall"
    field_convention: str = "exp(-i*k*z+i*omega*t)"


class MaxwellSolver(PanelKernels):
    """Full synchronous Maxwell wall impedance relative to free space.

    Four constant single-layer densities per vacuum panel describe Wx, Wy,
    u=(Ez-Ez_PEC)/(-i*k*Z0*I), v=Hz/(-i*k*I). The vacuum operator is Laplace
    at beta=1 and modified Helmholtz with kappa=omega/(beta*gamma*c) below 1.
    Tangential E and H are matched to nonlocal material field operators.
    Metal displacement current, permeability and Drude conductivity are kept.
    PEC material is an exact ideal termination: bare PEC has RW=0, and
    finite conducting layers can be backed by tangential-E=0 conditions.

    Default output is resistive wall plus PEC indirect space charge. The
    singular free-space direct space-charge field is excluded. Its regular
    PEC image field uses this same boundary mesh, including analytic source
    and witness derivatives. component='resistive_wall' returns the previous
    physical contribution with the new exp(+i*omega*t) field convention.
    No analytical cross-section reference enters this class.
    """

    def __init__(self, panels, kernel_order=12, logarithm_reference=None, beta=1.0,
                 roughness=None, interfaces=None):
        self.beta = validate_beta(beta)
        if roughness is not None:
            from .roughness import Roughness
            if not isinstance(roughness,Roughness):
                raise TypeError("Supply Roughness(...) or None for the wall-depth model.")
        self.roughness = roughness
        self.interfaces = tuple(interfaces) if interfaces is not None else None
        super().__init__(panels, kernel_order, logarithm_reference)
        p = self.pec
        displacement = p.centres[:, None, :]-p.starts[None, :, :]
        gradient = self._single_layer_gradient(displacement)
        self._normal = np.einsum("ijk,ik->ij", gradient, p.normal)
        self._tangential = np.einsum("ijk,ik->ij", gradient, p.tangent)
        # Interior normal jump of the logarithmic single-layer potential.
        np.fill_diagonal(self._normal, .5)
        np.fill_diagonal(self._tangential, 0.)
        nx, ny = p.normal.T
        tx, ty = p.tangent.T
        normal, tangent = self._normal, self._tangential
        self._div_x = nx[:, None]*normal+tx[:, None]*tangent
        self._div_y = ny[:, None]*normal+ty[:, None]*tangent
        self._curl_x = tx[:, None]*normal-nx[:, None]*tangent
        self._curl_y = ty[:, None]*normal-ny[:, None]*tangent

    def _wall(self, wall):
        if isinstance(wall, (Metal, PEC, Vacuum)):
            wall = Wall(wall)
        if not isinstance(wall, Wall):
            raise TypeError("Supply Metal, PEC, Vacuum, Air or a Wall ordered from vacuum outward.")
        if isinstance(wall.substrate, Vacuum):
            if self.beta == 1:
                raise NotImplementedError("Open Air/Vacuum exterior requires beta<1; the beta=1 exterior limit is not implemented.")
            if self.roughness is not None:
                raise NotImplementedError("Open Air/Vacuum exterior with conductivity-gradient roughness is not validated; use explicit smooth conducting layers.")
        if isinstance(wall.substrate, PEC):
            if self.roughness is not None:
                raise NotImplementedError("Conductivity-gradient roughness needs finite conductivity; disable it for PEC.")
            if self.interfaces is not None and len(self.interfaces) != len(wall.layers):
                raise ValueError("Provide one outer interface per finite layer before the PEC termination.")
        return wall

    @staticmethod
    def _bare_pec(wall):
        return isinstance(wall.substrate, PEC) and not any(layer.thickness > 0 for layer in wall.layers)

    def _bare_free(self, wall):
        return (isinstance(wall.substrate, Vacuum) and not self.interfaces
                and not any(layer.thickness > 0 for layer in wall.layers))

    def _vacuum_operators(self, kappa):
        """Yukawa single-layer traces, with stable analytic Laplace splitting."""
        if kappa == 0:
            return self.pec.single_layer, self._normal, self._tangential
        radius = self._distance
        argument = kappa*radius
        if kappa*self.pec.scale <= 1:
            single = self.pec.single_layer + np.sum(
                (kv(0, argument)+np.log(radius/self.pec.logarithm_reference))
                * self._weights[None, :, :], axis=-1)/(2*np.pi)
            correction = np.sum(((1-argument*kv(1, argument))/radius**2)[..., None]
                * self._displacement*self._weights[None, :, :, None], axis=2)/(2*np.pi)
            normal = self._normal + np.einsum("ijk,ik->ij", correction, self.pec.normal)
            tangent = self._tangential + np.einsum("ijk,ik->ij", correction, self.pec.tangent)
        else:
            single, _ = self._conductor_operators(complex(kappa))
            gradient = np.sum((-kappa*kv(1, argument)/radius)[..., None]
                * self._displacement*self._weights[None, :, :, None], axis=2)/(2*np.pi)
            normal = np.einsum("ijk,ik->ij", gradient, self.pec.normal)
            tangent = np.einsum("ijk,ik->ij", gradient, self.pec.tangent)
            np.fill_diagonal(normal, .5)
            np.fill_diagonal(tangent, 0.)
        return single, normal, tangent

    def _source_basis(self, source, kappa, single, derivatives=False):
        point = self.pec._source(source)
        if kappa == 0:
            if derivatives:
                drive, derivative, _ = self.pec.source_derivatives(point)
                return np.column_stack((drive, derivative))
            return self.pec.panel_density(point)[:, None]
        delta = self.pec.centres-np.asarray(point)
        radius = np.linalg.norm(delta, axis=1)
        drive = kv(0, kappa*radius)/(2*np.pi)
        if derivatives:
            derivative = (kappa*kv(1, kappa*radius)/radius)[:, None]*delta/(2*np.pi)
            drive = np.column_stack((drive, derivative))
        else:
            drive = drive[:, None]
        return solve(single, drive, check_finite=False, assume_a="gen")

    def _scales(self, f):
        k = 2*np.pi*float(f)/(self.beta*C0)
        return k, k*np.sqrt((1-self.beta)*(1+self.beta))

    def _single_layer_gradient(self, displacement):
        """Exact target gradient, with no jump, for straight source panels."""
        p = self.pec
        projection = np.einsum("...jk,jk->...j", displacement, p.tangent)
        height = np.einsum("...jk,jk->...j", displacement, p.normal)
        a, b = p.lengths-projection, -projection
        along = np.log((a*a+height*height)/(b*b+height*height))/(4*np.pi)
        across = (np.sign(height)*(np.arctan2(a, abs(height))
                                  - np.arctan2(b, abs(height)))/(2*np.pi))
        return along[..., None]*p.tangent-across[..., None]*p.normal

    def _laplace_witness_operators(self, witness):
        """Analytic single-layer value, gradient and Hessian inside vacuum."""
        point = np.asarray(self.pec._source(witness))
        p = self.pec
        displacement = point-p.starts
        projection = np.einsum("jk,jk->j", displacement, p.tangent)
        height = abs(np.einsum("jk,jk->j", displacement, p.normal))

        def primitive(a):
            return .5*xlogy(a, a*a+height*height)-a+height*np.arctan2(a, height)

        value = -(primitive(p.lengths-projection)-primitive(-projection)
                  - p.lengths*np.log(p.logarithm_reference))/(2*np.pi)
        gradient = self._single_layer_gradient(displacement)
        end_displacement = point-p.ends
        endpoint = (end_displacement/np.sum(end_displacement**2, axis=1)[:, None]
                    - displacement/np.sum(displacement**2, axis=1)[:, None])/(2*np.pi)
        along = np.einsum("jk,jk->j", p.tangent, endpoint)
        across = np.einsum("jk,jk->j", p.normal, endpoint)
        tt = p.tangent[:, :, None]*p.tangent[:, None, :]
        nn = p.normal[:, :, None]*p.normal[:, None, :]
        tn = p.tangent[:, :, None]*p.normal[:, None, :]
        hessian = along[:, None, None]*(tt-nn)+across[:, None, None]*(tn+tn.transpose(0, 2, 1))
        return value, gradient, hessian

    def _witness_operators(self, witness, kappa=0.):
        if kappa == 0:
            return self._laplace_witness_operators(witness)
        point = np.asarray(self.pec._source(witness))
        delta = point-self._positions
        radius = np.linalg.norm(delta, axis=-1)
        argument = kappa*radius
        k0, k1 = kv(0, argument), kv(1, argument)
        g = -kappa*k1/radius
        h = ((kappa*kappa*k0+2*kappa*k1/radius)[..., None, None]
             * delta[..., :, None]*delta[..., None, :]/radius[..., None, None]**2
             - (kappa*k1/radius)[..., None, None]*np.eye(2))/(2*np.pi)
        if kappa*self.pec.scale <= 1:
            value, gradient, hessian = self._laplace_witness_operators(witness)
            value += np.sum((k0+np.log(radius/self.pec.logarithm_reference))
                            *self._weights, axis=1)/(2*np.pi)
            gradient += np.sum(((1-argument*k1)/radius**2)[..., None]*delta
                               *self._weights[..., None], axis=1)/(2*np.pi)
            lap_h = (2*delta[..., :, None]*delta[..., None, :]/radius[..., None, None]**4
                     - np.eye(2)/radius[..., None, None]**2)/(2*np.pi)
            hessian += np.sum((h-lap_h)*self._weights[..., None, None], axis=1)
        else:
            value = np.sum(k0*self._weights, axis=1)/(2*np.pi)
            gradient = np.sum(g[..., None]*delta*self._weights[..., None], axis=1)/(2*np.pi)
            hessian = np.sum(h*self._weights[..., None, None], axis=1)
        return value, gradient, hessian

    def _solve_fields(self, f, wall, source=(0., 0.), derivatives=False):
        """Solve div/curl and both material matching equations at this beta."""
        k, kappa = self._scales(f)
        s, n, t = self._vacuum_operators(kappa)
        basis = self._source_basis(source, kappa, s, derivatives)
        nx, ny = self.pec.normal.T
        tx, ty = self.pec.tangent.T
        zero = np.zeros_like(s)
        if wall.layers or self.roughness is not None or self.interfaces is not None:
            from .layered import wall_admittance
            admittance, interface_count = wall_admittance(self, f, wall)
            count = len(s)
            eu = admittance[:count, :count] @ s
            ev = admittance[:count, count:] @ s
            hu = admittance[count:, :count] @ s
            hv = admittance[count:, count:] @ s
        else:
            metal = wall.substrate
            y = complex(metal.admittivity(f))
            q2 = complex(metal.transverse_wavenumber_squared(f, self.beta))
            q = np.sqrt(q2)
            single_metal, double_metal = self._conductor_operators(q)
            conductor = -solve(single_metal, .5*self._identity-double_metal,
                               check_finite=False, assume_a="gen") @ s
            eu, ev = t/q2, -self.beta*metal.mu_r*conductor/q2
            hu, hv = -1j*k*Z0*y*conductor/q2, k*k*t/q2
            interface_count = 1
        matrix = np.block([
            [nx[:, None]*n+tx[:, None]*t, ny[:, None]*n+ty[:, None]*t, -s, zero],
            [tx[:, None]*n-nx[:, None]*t, ty[:, None]*n-ny[:, None]*t, zero, self.beta*s],
            [tx[:, None]*s, ty[:, None]*s, -eu, -ev],
            [k*k*nx[:, None]*s, k*k*ny[:, None]*s, -n-self.beta*hu, -self.beta*hv],
        ])
        count = len(s)
        rhs = np.zeros((4*count, basis.shape[1]), complex)
        rhs[3*count:] = -self.beta*basis
        rows = np.max(abs(matrix), axis=1)
        scaled = matrix/rows[:, None]
        columns = np.max(abs(scaled), axis=0)
        density = solve(scaled/columns[None, :], rhs/rows[:, None],
                        check_finite=False, assume_a="gen")/columns[:, None]
        rhs_norm = np.linalg.norm(rhs)
        residual = np.linalg.norm(matrix @ density-rhs)/rhs_norm if rhs_norm else 0.
        return density, k, kappa, s, float(residual), interface_count, basis

    @staticmethod
    def _positive(f):
        frequency = frequency_array(f)
        if np.any(frequency <= 0):
            raise ValueError("MaxwellSolver requires positive frequencies in Hz; supply DC limits separately.")
        return frequency

    def boundary_state(self, f, wall, source=(0., 0.), witness=None):
        """Expose RW-only reaction traces per ampere, at one frequency.

        These are the matching unknowns relative to PEC, rather than the
        complete wall field relative to free space. See state.component.
        """
        frequency = self._positive(f)
        if frequency.ndim != 0:
            raise ValueError("boundary_state requires one positive frequency in Hz.")
        wall = self._wall(wall)
        if self._bare_pec(wall) and not self.interfaces:
            self.pec._source(source)
            self.pec._source(source if witness is None else witness)
            count = len(self.pec.lengths)
            return MaxwellBoundaryState(np.zeros(count, complex), np.zeros(count, complex),
                                        np.zeros((count, 2), complex), 0j, 0.)
        if self._bare_free(wall):
            k,kappa=self._scales(frequency)
            single,normal,tangent=self._vacuum_operators(kappa)
            basis=self._source_basis(source,kappa,single)[:,0]
            electric=-(1-self.beta)*(1+self.beta)/self.beta*basis
            derivative=(normal@electric)[:,None]*self.pec.normal+(tangent@electric)[:,None]*self.pec.tangent
            value,_,_=self._witness_operators(source if witness is None else witness,kappa)
            return MaxwellBoundaryState(-1j*k*Z0*(single@electric),np.zeros(len(single),complex),
                                        k*k*Z0/kappa**2*derivative,
                                        complex(1j*k*Z0*(value@electric)),0.)
        density, k, kappa, single, residual, _, _ = self._solve_fields(frequency, wall, source)
        value, _, _ = self._witness_operators(source if witness is None else witness, kappa)
        count = len(single)
        traces = single @ density[:, 0].reshape(4, count).T
        return MaxwellBoundaryState(-1j*k*Z0*traces[:, 2], -1j*k*traces[:, 3],
                    k*k*Z0*traces[:, :2],
                    complex(1j*k*Z0*(value @ density[2*count:3*count, 0])), residual)

    def _longitudinal_terms(self, f, wall, source, witness, include_rw):
        frequency = self._positive(f)
        source = self.pec._source(source)
        witness = self.pec._source(source if witness is None else witness)
        bare_free=False
        if include_rw:
            wall = self._wall(wall)
            bare_free=self._bare_free(wall)
            if self._bare_pec(wall) and not self.interfaces:
                include_rw = False
            if bare_free: include_rw=False
        rw = np.zeros(frequency.shape, complex)
        isc = np.zeros_like(rw)
        image_factor = (1-self.beta)*(1+self.beta)/self.beta
        for index in np.ndindex(frequency.shape):
            if include_rw:
                density, k, kappa, single, _, _, basis = self._solve_fields(frequency[index], wall, source)
            elif image_factor:
                k, kappa = self._scales(frequency[index])
                single, _, _ = self._vacuum_operators(kappa)
                basis = self._source_basis(source, kappa, single)
            else:
                continue
            value, _, _ = self._witness_operators(witness, kappa)
            if include_rw:
                count = len(single)
                rw[index] = 1j*k*Z0*(value @ density[2*count:3*count, 0])
            if image_factor:
                # phi_image=-S*rho. The free-space singularity is never evaluated.
                isc[index] = 1j*k*Z0*image_factor*(value @ basis[:, 0])
        if bare_free:
            # A fictitious interface between identical vacuum regions has
            # exactly no wall field. In the PEC-reference decomposition the
            # material correction cancels the image contribution.
            rw=-isc
        return {"resistive_wall": rw, "indirect_space_charge": isc, "wall": rw+isc}

    def longitudinal(self, f, wall=None, source=(0., 0.), witness=None, *, component="wall"):
        """Wall impedance in Ohm/m: RW + indirect SC, excluding direct SC.

        component is wall (default), resistive_wall or indirect_space_charge.
        Materials are unnecessary for the PEC indirect-space-charge component.
        All complex values use exp(-i*k*z+i*omega*t); plots display Im directly.
        """
        component = validate_component(component)
        return self._longitudinal_terms(f, wall, source, witness,
                                       component != "indirect_space_charge")[component]

    def longitudinal_components(self, f, wall, source=(0., 0.), witness=None):
        """Return a dict of wall, resistive_wall and indirect_space_charge arrays.

        At most one material-field solve per frequency supplies all terms.
        Bare PEC uses only the image boundary problem, with exact RW=0.
        """
        return self._longitudinal_terms(f, wall, source, witness, True)

    def _transverse_terms(self, f, wall, reference, include_rw):
        """Analytic dipolar and quadrupolar derivatives of RW and PEC images.

        Panofsky-Wenzel: Zperp=(beta*c/omega)*grad_witness Zparallel.
        No finite differences enter these derivatives. At finite beta the
        quadrupolar trace equals (kappa**2/k)*Zparallel, not zero.
        """
        frequency = self._positive(f)
        bare_free=False
        if include_rw:
            wall = self._wall(wall)
            bare_free=self._bare_free(wall)
            if self._bare_pec(wall) and not self.interfaces:
                include_rw = False
            if bare_free: include_rw=False
        point = self.pec._source(reference)
        def arrays():
            return [np.zeros(frequency.shape, complex),
                    np.zeros(frequency.shape+(2, 2), complex),
                    np.zeros(frequency.shape+(2, 2), complex),
                    np.zeros(frequency.shape+(2,), complex)]
        rw, isc = arrays(), arrays()
        image_factor = (1-self.beta)*(1+self.beta)/self.beta
        count = len(self.pec.lengths)
        for index in np.ndindex(frequency.shape):
            if include_rw:
                density, k, kappa, _, _, _, basis = self._solve_fields(frequency[index], wall, point, True)
            elif image_factor:
                k, kappa = self._scales(frequency[index])
                single, _, _ = self._vacuum_operators(kappa)
                basis = self._source_basis(point, kappa, single, True)
            else:
                continue
            value, gradient, hessian = self._witness_operators(point, kappa)
            for output, electric, factor in ((rw, density[2*count:3*count] if include_rw else None, 1.),
                                             (isc, basis, image_factor)):
                if electric is None or factor == 0:
                    continue
                output[0][index] = 1j*k*Z0*factor*(value @ electric[:, 0])
                output[1][index] = 1j*Z0*factor*(gradient.T @ electric[:, 1:])
                output[2][index] = 1j*Z0*factor*np.einsum("nij,n->ij", hessian, electric[:, 0])
                output[3][index] = 1j*Z0*factor*(gradient.T @ electric[:, 0])
        if bare_free: rw=[-value for value in isc]
        terms = {"resistive_wall": rw, "indirect_space_charge": isc,
                 "wall": [a+b for a,b in zip(rw, isc)]}
        return {name: TransverseImpedance(*values, point, self.beta, name)
                for name, values in terms.items()}

    def transverse(self, f, wall=None, reference=(0., 0.), *, component="wall"):
        """Wall longitudinal, dipolar/quadrupolar tensors and constant kick.

        Default includes indirect space charge, evaluated on the boundary mesh.
        All derivatives are analytic; no finite source displacement is fitted.
        """
        component = validate_component(component)
        return self._transverse_terms(f, wall, reference,
                                     component != "indirect_space_charge")[component]

    def transverse_components(self, f, wall, reference=(0., 0.)):
        """Return three TransverseImpedance objects; bare PEC needs only images."""
        return self._transverse_terms(f, wall, reference, True)

    def diagnostics(self, f, wall, source=(0., 0.)):
        """Matching residual and retained physics; refine to assess accuracy."""
        f = float(self._positive(f))
        wall = self._wall(wall)
        bare_pec = self._bare_pec(wall) and not self.interfaces
        bare_free=self._bare_free(wall)
        if bare_pec or bare_free:
            self.pec._source(source)
            k, kappa = self._scales(f)
            residual, interfaces = None, 1
        else:
            _, k, kappa, _, residual, interfaces, _ = self._solve_fields(f, wall, source)
        if isinstance(wall.substrate, PEC):
            attenuation = None
        else:
            q = np.sqrt(wall.substrate.transverse_wavenumber_squared(f, self.beta))
            attenuation = float(1/q.real)
        return {"boundary_count": interfaces, "panels": len(self.pec.lengths),
                "independent_vacuum_unknowns_after_elimination":
                    (len(self.pec.lengths) if self.beta < 1 else 0) if bare_pec or bare_free else 4*len(self.pec.lengths),
                "kernel_order": self.kernel_order,
                "relative_matrix_residual": residual,
                "attenuation_length_m": attenuation,
                "outer_termination": ("PEC" if isinstance(wall.substrate, PEC) else
                                      "air" if isinstance(wall.substrate, Air) else
                                      "vacuum" if isinstance(wall.substrate, Vacuum) else "infinite_conductor"),
                "vacuum_k_times_extent": float(k*self.pec.scale),
                "vacuum_kappa_times_extent": float(kappa*self.pec.scale),
                "full_vector_maxwell": True, "beam_beta": self.beta,
                "coating_interfaces": bool(wall.layers or self.interfaces),
                "roughness_model": "conductivity_gradient" if self.roughness is not None else None,
                "finite_outer_wall": bool(wall.layers and isinstance(wall.substrate, Vacuum)),
                "exterior_is_open": isinstance(wall.substrate, Vacuum),
                "air_approximated_as_vacuum": isinstance(wall.substrate, Air),
                "impedance_reference": "free space (direct space charge excluded)",
                "indirect_space_charge_included": True,
                "space_charge_method": "PEC image single-layer BEM on aperture mesh",
                "field_convention": "exp(-i*k*z+i*omega*t)",
                "local_surface_impedance_used": False,
                "accuracy_certified_by_residual_alone": False}


def validate_beta(beta):
    if isinstance(beta, (bool, np.bool_)) or not np.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("Beam beta=v/c must be finite with 0 < beta <= 1.")
    return float(beta)


def validate_component(component):
    if component not in ("wall", "resistive_wall", "indirect_space_charge"):
        raise ValueError("component must be wall, resistive_wall or indirect_space_charge.")
    return component
