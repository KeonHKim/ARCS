"""Independent circular benchmarks; these functions are never called by BEM."""
import numpy as np
from scipy.special import ive, kve
from .constants import C0, EPS0, MU0, Z0
from .materials import PEC, frequency_array


def _circular_vacuum_reaction(f, radius, zs, beta=1.0):
    """Exact vacuum monopole reaction relative to the same-beta PEC pipe."""
    from .maxwell import validate_beta
    beta = validate_beta(beta)
    f = frequency_array(f)
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    zs = np.broadcast_to(np.asarray(zs, complex), f.shape)
    kappa = 2*np.pi*f/(beta*C0)*np.sqrt((1-beta)*(1+beta))
    x = kappa*radius
    i0 = ive(0, x)
    ratio = np.full(f.shape, radius/2)
    np.divide(ive(1,x), kappa*i0, out=ratio, where=kappa != 0)
    return (zs*np.exp(-2*x)/(2*np.pi*radius*i0**2)
            /(1+1j*2*np.pi*f*EPS0*zs*ratio))


def _synchronous_constants(f, material, beta=1.0):
    from .materials import Vacuum
    if isinstance(material, Vacuum) and beta == 1:
        raise NotImplementedError("Open Air/Vacuum exterior reference requires beta<1.")
    omega = 2*np.pi*f
    y = material.admittivity(f)
    kappa2 = (omega/(beta*C0))**2*(1-beta)*(1+beta)
    q = np.sqrt(kappa2+(omega/C0)**2*(1-material.mu_r*material.epsilon_r)
                +1j*omega*MU0*material.mu_r*material.sigma(f))
    return q, q/y


def circular_cylindrical_surface(f, radius, wall, beta=1.0):
    """Exact m=0 cylindrical layers at the specified beam beta, semi-infinite last layer.

    Uses scaled modified Bessel functions and inward reflection recursion.
    Coating thicknesses are radial. Includes material displacement current and
    synchronous longitudinal wavenumber.
    """
    from .maxwell import validate_beta
    beta = validate_beta(beta)
    f = frequency_array(f)
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    shape = f.shape
    freq = f.ravel()
    answer = np.zeros(freq.shape, complex)
    mask = freq > 0
    fp = freq[mask]
    if not fp.size:
        return answer.reshape(shape)
    interfaces = radius + np.r_[0., np.cumsum([x.thickness for x in wall.layers])]
    outer = interfaces[-1]
    if isinstance(wall.substrate, PEC):
        z = np.zeros(fp.shape, complex)
    else:
        q, zc = _synchronous_constants(fp, wall.substrate, beta)
        z = zc*kve(0, q*outer)/kve(1, q*outer)
    for i in range(len(wall.layers)-1, -1, -1):
        layer = wall.layers[i]
        if layer.thickness == 0:
            continue
        q, zc = _synchronous_constants(fp, layer.metal, beta)
        inner, outer = interfaces[i], interfaces[i+1]
        qi, qo = q*inner, q*outer
        k0i, k0o = kve(0, qi), kve(0, qo)
        i0i, i0o = ive(0, qi), ive(0, qo)
        k1i = kve(1, qi)/k0i
        k1o = kve(1, qo)/k0o
        i1i = ive(1, qi)/i0i
        i1o = ive(1, qo)/i0o
        reflection = (z/zc*k1o-1)/(1+z/zc*i1o)
        propagation = (i0i/i0o)*(k0o/k0i)*np.exp(-q*layer.thickness-(q*layer.thickness).real)
        ratio = reflection*propagation
        z = zc*(1+ratio)/(k1i-ratio*i1i)
    answer[mask] = z
    return answer.reshape(shape)


def _component(rw, isc, component):
    from .maxwell import validate_component
    component = validate_component(component)
    return {"resistive_wall": rw, "indirect_space_charge": isc, "wall": rw+isc}[component]


def circular_indirect_space_charge(f, radius, beta=1.0):
    """Centred circular PEC image impedance per length, Ohm/m.

    Mounet et al., HB2021 MOP10 Eq. (1), in the new ARCS convention.
    No finite beam radius is introduced; the free-space field is excluded.
    This analytical function is for validation and never enters the BEM solve.
    """
    from .maxwell import validate_beta
    beta = validate_beta(beta)
    f = frequency_array(f)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be finite and positive.")
    result = np.zeros(f.shape, complex)
    if beta == 1:
        return result
    active = f > 0
    k = 2*np.pi*f[active]/(beta*C0)
    inverse_gamma2 = (1-beta)*(1+beta)
    x = k*np.sqrt(inverse_gamma2)*radius
    ratio = kve(0, x)/ive(0, x)*np.exp(-2*x)
    result[active] = 1j*k*Z0*inverse_gamma2/beta*ratio/(2*np.pi)
    return result


def circular_indirect_dipolar(f, radius, beta=1.0):
    """Centred circular PEC dipolar image impedance per length, Ohm/m²."""
    from .maxwell import validate_beta
    beta = validate_beta(beta)
    f = frequency_array(f)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be finite and positive.")
    result = np.zeros(f.shape, complex)
    if beta == 1:
        return result
    inverse_gamma2 = (1-beta)*(1+beta)
    active = f > 0
    k = 2*np.pi*f[active]/(beta*C0)
    x = k*np.sqrt(inverse_gamma2)*radius
    # x² K1(x)/I1(x) tends to 2, avoiding a 0*infinity DC product.
    factor = x*x*kve(1, x)/ive(1, x)*np.exp(-2*x)
    result[active] = 1j*Z0*inverse_gamma2/(4*np.pi*beta*radius**2)*factor
    result[~active] = 1j*Z0*inverse_gamma2/(2*np.pi*beta*radius**2)
    return result


def circular_indirect_quadrupolar(f, radius, beta=1.0):
    """Each centred circular quadrupolar diagonal, Ohm/m²."""
    k = 2*np.pi*frequency_array(f)/(beta*C0)
    return .5*k*(1-beta)*(1+beta)*circular_indirect_space_charge(f, radius, beta)


def circular_exact(f, radius, wall, beta=1.0, *, component="wall"):
    """Independent cylindrical m=0 wall impedance, RW + indirect SC, Ohm/m."""
    rw = _circular_vacuum_reaction(f, radius,
                circular_cylindrical_surface(f, radius, wall, beta), beta)
    return _component(rw, circular_indirect_space_charge(f, radius, beta), component)


def circular_maxwell_dipolar(f, radius, metal, beta=1.0, *, component="wall"):
    """Independent full-Maxwell bare circular m=1 field matching, Ohm/m².

    Both longitudinal fields and both tangential matching equations are
    solved analytically, with finite-beta vacuum I1 and exterior K1 modes.
    Displacement current, mu_r, epsilon_r and Drude conductivity are retained.
    This function never enters the BEM field assembly.
    """
    from .maxwell import validate_beta
    beta = validate_beta(beta)
    f = frequency_array(f)
    if radius <= 0 or np.any(f <= 0):
        raise ValueError("Use a positive radius and positive frequencies.")
    if isinstance(metal, PEC):
        return _component(np.zeros(f.shape, complex), circular_indirect_dipolar(f, radius, beta), component)
    k = 2*np.pi*f/(beta*C0)
    kappa = k*np.sqrt((1-beta)*(1+beta))
    x = kappa*radius
    y = metal.admittivity(f)
    q, _ = _synchronous_constants(f, metal, beta)
    q2 = q*q
    f1 = q*kve(0,q*radius)/kve(1,q*radius)+1/radius
    small = x < 1e-3
    h = (1+x*x/4-x**4/96)/radius
    j = .5+x*x/24-x**4/256
    factor = 1-x*x/4+7*x**4/192
    if np.any(~small):
        xx = x[~small]
        hh = kappa[~small]*ive(0,xx)/ive(1,xx)-1/radius
        h = np.array(h,copy=True); j = np.array(j,copy=True); factor = np.array(factor,copy=True)
        h[~small] = hh
        j[~small] = (hh*hh-1/radius**2)/kappa[~small]**2
        factor[~small] = xx*xx*np.exp(-2*xx)/(4*ive(1,xx)**2)
    rho = kappa*kappa/q2
    denominator = (k*k/(beta*(h+metal.mu_r*f1*rho)) *
        (j-h*h/(k*k)+beta*beta*h*metal.mu_r*f1/q2+2/(radius*radius*q2)
         -kappa*kappa/(radius*radius*q2*q2)) -1j*k*Z0*y*f1/q2)
    rw = -1j*Z0/(np.pi*radius**3)*factor/denominator
    return _component(rw, circular_indirect_dipolar(f, radius, beta), component)


def circular_maxwell_quadrupolar(f, radius, wall, beta=1.0, *, component="wall"):
    """Centred circle: equal diagonal entries, zero off-diagonal entries."""
    from .maxwell import validate_beta
    beta = validate_beta(beta)
    f = frequency_array(f)
    k = 2*np.pi*f/(beta*C0)
    return .5*k*(1-beta)*(1+beta)*circular_exact(f,radius,wall,beta,component=component)


def _mode_layer_admittance(f, inner, outer, metal, beta, exterior, mode=1):
    """Independent cylindrical I_m/K_m annulus, no boundary discretization."""
    k = 2*np.pi*f/(beta*C0)
    y = metal.admittivity(f)
    q, _ = _synchronous_constants(f, metal, beta)
    qi, qo = q*inner, q*outer
    regular_ratio = ive(mode, qi)/ive(mode, qo)*np.exp(qi.real-qo.real)
    decaying_ratio = kve(mode, qo)/kve(mode, qi)*np.exp(qi-qo)
    regular_inner = q*ive(mode-1, qi)/ive(mode, qi)-mode/inner
    regular_outer = q*ive(mode-1, qo)/ive(mode, qo)-mode/outer
    decaying_inner = -q*kve(mode-1, qi)/kve(mode, qi)-mode/inner
    decaying_outer = -q*kve(mode-1, qo)/kve(mode, qo)-mode/outer
    values = np.array([[1., regular_ratio], [decaying_ratio, 1.]])
    derivatives = np.array([[decaying_inner, regular_inner*regular_ratio],
                            [decaying_outer*decaying_ratio, regular_outer]])
    normal = np.linalg.solve(values.T, derivatives.T).T
    q2 = q*q

    def field(i, j):
        radius = inner if i == 0 else outer
        tangent = mode/radius if i == j else 0.
        return np.array([[-tangent/q2, -beta*metal.mu_r*normal[i,j]/q2],
                         [-1j*k*Z0*y*normal[i,j]/q2, k*k*tangent/q2]])

    if isinstance(exterior, PEC):
        constraint = np.vstack(([1., 0.], field(1, 1)[0]))
        drive = np.vstack(([0., 0.], field(1, 0)[0]))
        return field(0, 0)-field(0, 1)@np.linalg.solve(constraint, drive)
    return field(0,0)-field(0,1)@np.linalg.solve(field(1,1)-exterior,field(1,0))


def circular_layered_dipolar(f, radius, wall, beta=1.0, *, component="wall"):
    """Independent full circular m=1 multilayer solution, Ohm/m².

    Each layer uses regular I1 and decaying K1 radial fields. Their exact
    analytic derivatives match Et and Ht across each circular interface.
    The finite-beta vacuum reaction and source/witness normalization are
    retained. This validates coated/graded BEM dipoles independently.
    """
    from .maxwell import validate_beta
    beta = validate_beta(beta)
    frequency = frequency_array(f)
    if radius <= 0 or np.any(frequency <= 0):
        raise ValueError("Use a positive radius and positive frequencies.")
    layers = [layer for layer in wall.layers if layer.thickness > 0]
    if isinstance(wall.substrate, PEC) and not layers:
        return _component(np.zeros(frequency.shape, complex),
                          circular_indirect_dipolar(frequency, radius, beta), component)
    radii = radius+np.r_[0.,np.cumsum([layer.thickness for layer in layers])]
    result = np.empty(frequency.shape,complex)
    for index in np.ndindex(frequency.shape):
        freq = float(frequency[index])
        k = 2*np.pi*freq/(beta*C0)
        kappa = k*np.sqrt((1-beta)*(1+beta))
        r = radii[-1]
        if isinstance(wall.substrate, PEC):
            admittance = wall.substrate
        else:
            q, _ = _synchronous_constants(freq,wall.substrate,beta)
            q2 = q*q; y = wall.substrate.admittivity(freq)
            normal = -q*kve(0,q*r)/kve(1,q*r)-1/r
            admittance = np.array([[-1/(r*q2),-beta*wall.substrate.mu_r*normal/q2],
                                  [-1j*k*Z0*y*normal/q2,k*k/(r*q2)]])
        for i in range(len(layers)-1,-1,-1):
            admittance = _mode_layer_admittance(freq,radii[i],radii[i+1],
                                                layers[i].metal,beta,admittance)
        x = kappa*radius
        if x < 1e-3:
            h = (1+x*x/4-x**4/96)/radius
            j = .5+x*x/24-x**4/256
            factor = 1-x*x/4+7*x**4/192
        else:
            h = kappa*ive(0,x)/ive(1,x)-1/radius
            j = (h*h-1/radius**2)/(kappa*kappa)
            factor = x*x*np.exp(-2*x)/(4*ive(1,x)**2)
        a,b,c,d = admittance.ravel()
        denominator = (k*k*j-h*h+beta*k*k*h*b-beta*h*c-k*k*a/radius+d/radius
                       +kappa*kappa*(d*a-c*b))
        result[index] = (-1j*Z0/(np.pi*radius**3)*factor
                         *(beta*h+kappa*kappa*b)/denominator)
    return _component(result, circular_indirect_dipolar(f, radius, beta), component)
