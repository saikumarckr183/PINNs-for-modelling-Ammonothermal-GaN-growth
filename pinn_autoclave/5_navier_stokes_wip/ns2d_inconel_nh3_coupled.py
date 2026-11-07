r"""
Inconel (solid conduction) + NH3 (Boussinesq Navier-Stokes) model.

This couples two materials on one shared axisymmetric meridional domain:

    Inconel (solid):
        steady conduction, temperature only,
        T = T_outer(z) = T0_K + T1*z hard-enforced on the true outer wall
        (r=R_OUT, z=0, z=H_OUT).

    NH3 (fluid):
        steady axisymmetric Boussinesq Navier-Stokes,
        unknowns [u_r, u_z, p, T].

At the shared Inconel/NH3 wall (cylindrical sidewall, CryS-MAS lower
polyline, top face):

    - no-slip (u_r=u_z=0)               hard, via NH3 post-processing
    - temperature continuity T_INC=T_NH3 hard, via field composition
      inside each residual: T_NH3(x) = T_INC(x) + wall(x)*correction(x)
    - flux continuity k_inc*dT_inc/dn = k_nh3*dT_nh3/dn   soft residual

Axis regularity (r=0) is structural for both materials: the network
input is rho=(r/R)^2, and u_r is separately multiplied by r.

Pressure has no boundary condition anywhere (closed cavity, no
inlet/outlet), so it is only defined up to an additive constant; one
point value is pinned via a DataResidual, following
``navier_stokes_2d_NS2d-C.py``'s ``PointValueDataResidual`` pattern.

Geometry and Inconel material properties are copied unchanged from
``2D heat Inconel + Ammonia.py``. NH3 fluid properties are the
Schimmel et al. 2023 values, with a constant (not temperature
dependent) thermal conductivity for this first version.

Escaping the trivial u=0 solution
----------------------------------
u=0, T=T_outer(z) is an exact solution of the interior equations (it
satisfies continuity, momentum reduces to ordinary hydrostatic pressure,
energy is trivially satisfied since T is linear in z) and every boundary
condition.

Rayleigh-number continuation is *not* included in this first version
(see the module-level ``RAYLEIGH_NUMBER`` printed at import time -- it
is very large here, i.e. genuinely supercritical) and should be added
if the two safeguards above are not enough on their own.

"""

import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scimba_jax.domains.domain_mapping import DomainMapping
from scimba_jax.domains.meshless_domains.domains_2d import (
    Segment2D,
    Square2D,
)
from scimba_jax.nonlinear_approximation.approximation_spaces.approximation_spaces import (
    ApproximationSpace,
)
from scimba_jax.nonlinear_approximation.integration.monte_carlo import (
    DomainSampler,
    TensorizedSampler,
)
from scimba_jax.nonlinear_approximation.model_class.funcparam_vectorial import (
    ParamScalarFunction,
    ParamVecFunction,
)
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import (
    BoundaryResidual,
    DataResidual,
    InteriorResidual,
)

jax.config.update("jax_enable_x64", True)

SAVE_DIR = Path("output_inconel_nh3_boussinesq")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Geometry (unchanged from "2D heat Inconel + Ammonia.py")

R_OUT = 0.035
H_OUT = 0.370

R_NH3 = 0.012
Z_NH3_TOP = 0.335

# CryS-MAS lower NH3/Inconel boundary polyline.
NH3_LOWER_R = jnp.array(
    [0.0000, 0.0031, 0.0060, 0.0085, 0.0104, 0.0116, 0.0120],
    dtype=jnp.float64,
)

NH3_LOWER_Z = jnp.array(
    [0.0230, 0.0234, 0.0246, 0.0265, 0.0290, 0.0319, 0.0350],
    dtype=jnp.float64,
)

Z_NH3_MIN = float(NH3_LOWER_Z[0])
Z_NH3_MAX_LOWER = float(NH3_LOWER_Z[-1])


# Inconel material and outer thermal boundary

def k_inc_value(temperature_K):
    """Inconel conductivity [W/(m K)], same correlation as the conduction case."""
    return 6.9995 + 0.0151 * temperature_K


T0_K = 611.15
T1 = -300.0


def T_outer(x):
    """Prescribed external Inconel temperature profile [K]: T0_K + T1*z."""
    x = jnp.asarray(x, dtype=jnp.float64)
    z = x[..., 1:2]
    return T0_K + z * T1


# NH3 fluid properties (Schimmel et al. 2023 values, constant conductivity)

RHO_NH3 = 233.950
MU_NH3 = 3.673e-5
CP_NH3 = 4216.5
K_NH3 = 70.63350679  # constant for this first version (not temperature-dependent)
ALPHA_NH3 = 2.631e-3

NU_NH3 = MU_NH3 / RHO_NH3
KAPPA_NH3 = K_NH3 / (RHO_NH3 * CP_NH3)

GRAVITY_Z = -9.81


# Reference scales

# Inconel-side scales, identical convention to the conduction script.
DELTA_T_REF = abs(T1) * H_OUT  # wall-to-wall temperature drop [K]
L_REF = R_OUT

T_K_MIN_FOR_SCALING = 495.0
T_K_MAX_FOR_SCALING = 615.0
T_K_SCALE_GRID = jnp.linspace(T_K_MIN_FOR_SCALING, T_K_MAX_FOR_SCALING, 2001)
K_INC_SCALE_VALUES = k_inc_value(T_K_SCALE_GRID)

if bool(jnp.any(K_INC_SCALE_VALUES <= 0.0)):
    raise ValueError("Inconel conductivity is not positive in the scaling range.")

K_INC_MAX_REF = float(jnp.max(K_INC_SCALE_VALUES))
K_REF = max(K_INC_MAX_REF, K_NH3)  # Inconel dominates; NH3 is ~100x smaller
FLUX_REF = K_REF * DELTA_T_REF / L_REF
PDE_R_WEIGHTED_REF = FLUX_REF

# NH3-side scales: buoyancy velocity/pressure scales, and per-equation

L_REF_NH3 = Z_NH3_TOP - Z_NH3_MIN  # cavity height [m]

U_REF = (ALPHA_NH3 * abs(GRAVITY_Z) * DELTA_T_REF * L_REF_NH3) ** 0.5
P_REF = RHO_NH3 * U_REF**2

PRANDTL_NUMBER = NU_NH3 / KAPPA_NH3
RAYLEIGH_NUMBER = (
    ALPHA_NH3 * abs(GRAVITY_Z) * DELTA_T_REF * L_REF_NH3**3
) / (NU_NH3 * KAPPA_NH3)

MOMENTUM_REF = U_REF**2 / L_REF_NH3
CONTINUITY_REF = U_REF / L_REF_NH3
ENERGY_REF = U_REF * DELTA_T_REF / L_REF_NH3

RADIAL_MOMENTUM_REF = MOMENTUM_REF * R_NH3**2
AXIAL_MOMENTUM_REF = MOMENTUM_REF * R_NH3
CONTINUITY_R_REF = CONTINUITY_REF * R_NH3
ENERGY_R_REF = ENERGY_REF * R_NH3

# Buoyancy reference temperature: outer-law value at the cavity mid-height.
T_BOUSSINESQ_REF = T0_K + T1 * 0.5 * (Z_NH3_MIN + Z_NH3_TOP)

# Symmetry-breaking seed amplitude (physical m/s), see nh3_post_processing.
SEED_AMPLITUDE = 0.5 * U_REF

print("\n" + "=" * 72)
print("REFERENCE SCALES")
print("=" * 72)
print(f"DELTA_T_REF   = {DELTA_T_REF:.4f} K")
print(f"L_REF_NH3     = {L_REF_NH3:.6f} m")
print(f"U_REF         = {U_REF:.6e} m/s")
print(f"P_REF         = {P_REF:.6e} Pa")
print(f"NU_NH3        = {NU_NH3:.6e} m^2/s")
print(f"KAPPA_NH3     = {KAPPA_NH3:.6e} m^2/s")
print(f"Prandtl (Pr)  = {PRANDTL_NUMBER:.6f}")
print(f"Rayleigh (Ra) = {RAYLEIGH_NUMBER:.6e}  (supercritical -- expect a hard problem)")
print(f"K_REF (flux)  = {K_REF:.4f} W/(m K)  (Inconel-side dominates)")
print(
    "Per-equation residual references: "
    f"radial={RADIAL_MOMENTUM_REF:.4e}, axial={AXIAL_MOMENTUM_REF:.4e}, "
    f"continuity={CONTINUITY_R_REF:.4e}, energy={ENERGY_R_REF:.4e}"
)


# Training settings

N_COLLOC = {
    "inc_bottom": 1000,
    "inc_side": 4000,
    "inc_top": 1500,
    "inc_lower": 3500,
    "nh3_lower": 2500,
    "nh3_upper": 5000,
}

N_EPOCHS = 1000
HIDDEN_INCONEL = [16, 16, 16]
HIDDEN_NH3 = [16, 16, 16, 16]


# Reference coordinates and mapped main domain

XI_NH3 = R_NH3 / R_OUT
ETA_NH3_MIN = Z_NH3_MIN / H_OUT
ETA_NH3_MAX_LOWER = Z_NH3_MAX_LOWER / H_OUT
ETA_NH3_TOP = Z_NH3_TOP / H_OUT
XI_NH3_LOWER = NH3_LOWER_R / R_OUT
ETA_NH3_LOWER = NH3_LOWER_Z / H_OUT

main_mapping = DomainMapping.rectangle(
    origin=jnp.array([0.0, 0.0]),
    x_dir=jnp.array([R_OUT, 0.0]),
    y_dir=jnp.array([0.0, H_OUT]),
)

main_domain = Square2D(
    [(0.0, 1.0), (0.0, 1.0)],
    is_main_domain=True,
    label_str="main",
)

main_domain.set_mapping(
    main_mapping,
    bounds_postmap=[(0.0, R_OUT), (0.0, H_OUT)],
)

# Volumetric sampling subdomains
inc_bottom = Square2D(
    [(0.0, 1.0), (0.0, ETA_NH3_MIN)],
    is_main_domain=False,
    label_str="inc_bottom",
)

inc_side = Square2D(
    [(XI_NH3, 1.0), (ETA_NH3_MIN, ETA_NH3_TOP)],
    is_main_domain=False,
    label_str="inc_side",
)

inc_top = Square2D(
    [(0.0, 1.0), (ETA_NH3_TOP, 1.0)],
    is_main_domain=False,
    label_str="inc_top",
)

inc_lower = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MIN, ETA_NH3_MAX_LOWER)],
    is_main_domain=False,
    label_str="inc_lower",
)

nh3_lower = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MIN, ETA_NH3_MAX_LOWER)],
    is_main_domain=False,
    label_str="nh3_lower",
)

nh3_upper = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MAX_LOWER, ETA_NH3_TOP)],
    is_main_domain=False,
    label_str="nh3_upper",
)

for domain in (inc_bottom, inc_side, inc_top, inc_lower, nh3_lower, nh3_upper):
    main_domain.add_subdomain(domain)


# Interface boundary segments: flux continuity ONLY (soft). Temperature
# continuity and no-slip are hard-enforced structurally, so no Segment2D
# objects are needed for them.

domain_if_side_q = Segment2D(
    pt1=(XI_NH3, ETA_NH3_MAX_LOWER),
    pt2=(XI_NH3, ETA_NH3_TOP),
    label_str="if_side_q",
)

domain_if_top_q = Segment2D(
    pt1=(0.0, ETA_NH3_TOP),
    pt2=(XI_NH3, ETA_NH3_TOP),
    label_str="if_top_q",
)

N_LOWER_INTERFACE_SEGMENTS = len(NH3_LOWER_R) - 1
lower_flux_domains = []

for i in range(N_LOWER_INTERFACE_SEGMENTS):
    p1 = (float(XI_NH3_LOWER[i]), float(ETA_NH3_LOWER[i]))
    p2 = (float(XI_NH3_LOWER[i + 1]), float(ETA_NH3_LOWER[i + 1]))

    lower_flux_domains.append(
        Segment2D(pt1=p1, pt2=p2, label_str="if_lower_q", label_idx=i + 1)
    )

for boundary in (domain_if_side_q, domain_if_top_q, *lower_flux_domains):
    main_domain.add_bc_domain(boundary)

BOUNDARY_GROUPS = {
    "interface_flux": [
        domain_if_side_q.get_label(),
        domain_if_top_q.get_label(),
        *[domain.get_label() for domain in lower_flux_domains],
    ],
}

main_domain.set_boundaries_dict(BOUNDARY_GROUPS)

N_BC_COLLOC = {
    "interface_flux": {label: 300 for label in BOUNDARY_GROUPS["interface_flux"]},
}

INTERFACE_Q_DOMAINS = {
    domain_if_side_q.get_label(): domain_if_side_q,
    domain_if_top_q.get_label(): domain_if_top_q,
}
for domain in lower_flux_domains:
    INTERFACE_Q_DOMAINS[domain.get_label()] = domain


# Geometry predicates in PHYSICAL coordinates

def z_bottom_ammonia(r):
    r = jnp.asarray(r, dtype=jnp.float64)
    r_clipped = jnp.clip(r, NH3_LOWER_R[0], NH3_LOWER_R[-1])
    return jnp.interp(r_clipped, NH3_LOWER_R, NH3_LOWER_Z)


def in_nh3(r, z):
    return (r >= 0.0) & (r <= R_NH3) & (z >= z_bottom_ammonia(r)) & (z <= Z_NH3_TOP)


def in_lower_ammonia(r, z):
    return (
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom_ammonia(r))
        & (z <= Z_NH3_MAX_LOWER)
    )


# Neural fields: coordinate scaling and hard-constraint post-processing

def scaled_coordinates(x):
    """rho=(r/R_OUT)^2, eta=z/H_OUT -- structural axis regularity for both
    materials."""
    x = jnp.asarray(x, dtype=jnp.float64)
    r_scaled = x[..., 0:1] / R_OUT
    z_scaled = x[..., 1:2] / H_OUT
    return jnp.concatenate([r_scaled**2, z_scaled], axis=-1)


def outer_distance_factor(x):
    """Zero on the true outer Inconel wall (r=R_OUT, z=0, z=H_OUT)."""
    r = x[..., 0:1]
    z = x[..., 1:2]
    d_side = 1.0 - (r / R_OUT) ** 2
    d_bottom = z / H_OUT
    d_top = (H_OUT - z) / H_OUT
    return d_side * d_bottom * d_top


def inconel_post_processing(theta_inc, x):
    return T_outer(x) + DELTA_T_REF * outer_distance_factor(x) * theta_inc


def nh3_interface_distance_factor(x):
    """Zero on the shared Inconel/NH3 wall (bottom polyline, side, top);
    positive inside the NH3 cavity."""
    x = jnp.asarray(x, dtype=jnp.float64)
    r = x[..., 0:1]
    z = x[..., 1:2]
    z_bottom = z_bottom_ammonia(r)
    d_bottom = (z - z_bottom) / H_OUT
    d_top = (Z_NH3_TOP - z) / H_OUT
    d_side = (R_NH3 - r) / R_OUT
    return d_bottom * d_top * d_side


def nh3_post_processing(output, x):
    """
    Hard-enforces, everywhere:
        - u_r = 0 at r=0 (structural r factor) and at the Inconel wall
          (wall factor);
        - u_z = 0 at the Inconel wall (wall factor); u_z is NOT forced to
          zero on the axis (axial flow through the centerline is physical);
        - a symmetry-breaking seed, scaled by the buoyancy velocity U_REF,
          to avoid the u=0 trivial solution.

    Temperature continuity with Inconel is NOT applied here (this function
    has no access to the Inconel field). It is applied downstream, inside
    each residual, by adding the Inconel field to the correction returned
    here: T_NH3(x) = T_INC(x) + wall(x)*DELTA_T_REF*t_raw. This function
    only returns the wall(x)*DELTA_T_REF*t_raw part.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    r = x[..., 0:1]
    z = x[..., 1:2]

    wall = jnp.maximum(nh3_interface_distance_factor(x), 0.0)
    r_scaled = r / R_NH3
    z_scaled = (z - Z_NH3_MIN) / L_REF_NH3

    ur_raw = output[..., 0:1]
    uz_raw = output[..., 1:2]
    p_raw = output[..., 2:3]
    t_raw = output[..., 3:4]

    # Single convective cell, smooth and nonzero in the interior; wall(x)
    # makes it vanish at the true boundary without the seed needing to.
    seed_ur_shape = jnp.sin(jnp.pi * z_scaled) * r_scaled
    seed_uz_shape = jnp.cos(jnp.pi * z_scaled)

    ur = r_scaled * wall * (U_REF * ur_raw + SEED_AMPLITUDE * seed_ur_shape)
    uz = wall * (U_REF * uz_raw + SEED_AMPLITUDE * seed_uz_shape)
    pressure = P_REF * p_raw
    temperature_correction = wall * DELTA_T_REF * t_raw

    return jnp.concatenate([ur, uz, pressure, temperature_correction], axis=-1)


# Conductivity and Inconel conduction operator

def k_inc(T):
    return ParamScalarFunction(
        T.dims,
        lambda space, x: k_inc_value(T(space, x)),
        f_type=T.f_type,
    )


def axisymmetric_conduction(T, conductivity, space, x):
    dT_dr = T.partial_derivative("x", 0)
    dT_dz = T.partial_derivative("x", 1)
    q_r = conductivity(T) * dT_dr
    q_z = conductivity(T) * dT_dz
    r = x[..., 0:1]
    raw_residual = (
        r * q_r.partial_derivative("x", 0)(space, x)
        + q_r(space, x)
        + r * q_z.partial_derivative("x", 1)(space, x)
    )
    return raw_residual / PDE_R_WEIGHTED_REF


class InconelResidual(InteriorResidual):
    def __init__(self, domain, lower_box=False):
        super().__init__(domain=domain, size=1, model_type="x")
        self.lower_box = lower_box

    def construct_residual(self, *vars):
        T_inc, _ = vars

        def residual(space, x):
            value = axisymmetric_conduction(T_inc, k_inc, space, x)

            if not self.lower_box:
                return value

            r = x[..., 0:1]
            z = x[..., 1:2]
            return jnp.where(in_lower_ammonia(r, z), 0.0, value)

        return ParamScalarFunction(T_inc.dims, residual, f_type=T_inc.f_type)


# NH3 axisymmetric Boussinesq residual

def r_coord(x):
    return x[0]


def r2_coord(x):
    return x[0] ** 2


class NH3BoussinesqResidual(InteriorResidual):
    r"""
    Steady axisymmetric Boussinesq Navier-Stokes for NH3, r-weighted.
    Physical (SI) fields throughout; each of the four combined r-weighted
    residuals is divided by its own reference scale (RADIAL_MOMENTUM_REF,
    AXIAL_MOMENTUM_REF, CONTINUITY_R_REF, ENERGY_R_REF).

    Temperature continuity with Inconel is hard-enforced here by using
        temperature = T_inc_field + nh3_field.component(3)

    nh3_lower/nh3_upper are rectangular sampling envelopes that include
    some points outside the true (polyline-bounded) NH3 region, so each
    equation is masked by in_nh3(r, z), same convention as the conduction
    script's AmmoniaResidual.
    """

    def __init__(self, domain):
        super().__init__(domain=domain, size=4, model_type="x", f_rhs=None)

    def construct_residual(self, *vars):
        t_inc_field, nh3_field = vars

        ur = nh3_field.component(0)
        uz = nh3_field.component(1)
        pressure = nh3_field.component(2)
        temperature = t_inc_field + nh3_field.component(3)

        ur_r = ur.partial_derivative("x", 0)
        ur_z = ur.partial_derivative("x", 1)
        uz_r = uz.partial_derivative("x", 0)
        uz_z = uz.partial_derivative("x", 1)
        ur_rr = ur_r.partial_derivative("x", 0)
        ur_zz = ur_z.partial_derivative("x", 1)
        uz_rr = uz_r.partial_derivative("x", 0)
        uz_zz = uz_z.partial_derivative("x", 1)
        t_r = temperature.partial_derivative("x", 0)
        t_z = temperature.partial_derivative("x", 1)
        t_rr = t_r.partial_derivative("x", 0)
        t_zz = t_z.partial_derivative("x", 1)
        p_r = pressure.partial_derivative("x", 0)
        p_z = pressure.partial_derivative("x", 1)

        radial_momentum_base = (
            ur * ur_r
            + uz * ur_z
            + p_r / RHO_NH3
            - NU_NH3 * (ur_rr + ur_zz)
        )
        radial_momentum = (
            radial_momentum_base * r2_coord
            - (ur_r * r_coord) * NU_NH3
            + NU_NH3 * ur
        ) / RADIAL_MOMENTUM_REF

        axial_momentum_base = (
            ur * uz_r
            + uz * uz_z
            + p_z / RHO_NH3
            - ALPHA_NH3 * (temperature - T_BOUSSINESQ_REF) * GRAVITY_Z
            - NU_NH3 * (uz_rr + uz_zz)
        )
        axial_momentum = (
            axial_momentum_base * r_coord - NU_NH3 * uz_r
        ) / AXIAL_MOMENTUM_REF

        continuity = (
            ur_r * r_coord + ur + uz_z * r_coord
        ) / CONTINUITY_R_REF

        energy_base = ur * t_r + uz * t_z - KAPPA_NH3 * (t_rr + t_zz)
        energy = (energy_base * r_coord - KAPPA_NH3 * t_r) / ENERGY_R_REF

        def masked(expr):
            def residual(space, x):
                r = x[..., 0:1]
                z = x[..., 1:2]
                return jnp.where(in_nh3(r, z), expr(space, x), 0.0)

            return ParamScalarFunction({"x": 2}, residual, f_type="x")

        return ParamVecFunction.cat(
            [
                masked(radial_momentum),
                masked(axial_momentum),
                masked(continuity),
                masked(energy),
            ]
        )


# Interface flux residual (soft; only remaining interface condition)

class InterfaceFluxResidual(BoundaryResidual):
    """k_inc(T_inc)*dT_inc/dn - K_NH3*dT_nh3/dn = 0 on the shared wall.

    NH3-side flux is purely conductive: u=0 there by the hard no-slip
    constraint, so the convective contribution vanishes identically.
    """

    def __init__(self):
        super().__init__(domain=INTERFACE_Q_DOMAINS, size=1, model_type="x")

    def construct_residual(self, *vars):
        t_inc_field, nh3_field = vars
        temperature_nh3 = t_inc_field + nh3_field.component(3)

        d_inc_dr = t_inc_field.partial_derivative("x", 0)
        d_inc_dz = t_inc_field.partial_derivative("x", 1)
        d_nh3_dr = temperature_nh3.partial_derivative("x", 0)
        d_nh3_dz = temperature_nh3.partial_derivative("x", 1)
        k_inc_field = k_inc(t_inc_field)

        def residual(space, x, n):
            n_r = n[..., 0:1]
            n_z = n[..., 1:2]
            grad_inc_normal = n_r * d_inc_dr(space, x) + n_z * d_inc_dz(space, x)
            grad_nh3_normal = n_r * d_nh3_dr(space, x) + n_z * d_nh3_dz(space, x)
            flux_jump = (
                k_inc_field(space, x) * grad_inc_normal - K_NH3 * grad_nh3_normal
            )
            return flux_jump / FLUX_REF

        return ParamScalarFunction({"x": 2}, residual, f_type="x_n")


# Pressure gauge: one point-value data residual, since the cavity has no
# pressure boundary condition anywhere (same pattern as
# navier_stokes_2d_NS2d-C.py's PointValueDataResidual).

PRESSURE_PIN_POINT = jnp.array(
    [[0.0, 0.5 * (Z_NH3_MIN + Z_NH3_TOP)]], dtype=jnp.float64
)


class PressurePinResidual(DataResidual):
    def __init__(self):
        x_data = PRESSURE_PIN_POINT
        y_data = jnp.zeros((1, 1), dtype=jnp.float64)
        super().__init__(1, "x", (x_data, y_data))

    def construct_residual(self, *vars):
        _, nh3_field = vars
        # Normalized like every other residual in this file, so its weight
        # can be chosen on the same footing as the rest (not on raw Pa^2).
        return nh3_field.component(2) / P_REF


# Physical model

class InconelNH3BoussinesqModel(AbstractPhysicalModel):
    def __init__(self):
        super().__init__(main_domain=main_domain)

        self.physical_residuals = {
            "inc_bottom": InconelResidual(inc_bottom),
            "inc_side": InconelResidual(inc_side),
            "inc_top": InconelResidual(inc_top),
            "inc_lower": InconelResidual(inc_lower, lower_box=True),
            "nh3_lower": NH3BoussinesqResidual(nh3_lower),
            "nh3_upper": NH3BoussinesqResidual(nh3_upper),
            "interface_flux": InterfaceFluxResidual(),
        }

        self.data_residuals = {"pressure_pin": PressurePinResidual()}


model = InconelNH3BoussinesqModel()

# Sampler

domain_sampler = DomainSampler(main_domain, main_domain_fully_covered=True)

sampler = TensorizedSampler(
    [domain_sampler],
    model_type="x",
    bc=True,
    ic=False,
    data_samplers=model.data_residuals,
)

# Approximation space

def build_mlp(in_size, out_size, hidden_sizes, key):
    try:
        return MLP(
            in_size=in_size,
            out_size=out_size,
            hidden_sizes=hidden_sizes,
            key=key,
            activation=jnp.tanh,
        )
    except TypeError:
        return MLP(in_size=in_size, out_size=out_size, hidden_sizes=hidden_sizes, key=key)


def build_space(seed=42):
    key = jax.random.PRNGKey(seed)
    key_inc, key_nh3 = jax.random.split(key, 2)

    nn_inc = build_mlp(2, 1, HIDDEN_INCONEL, key_inc)
    nn_nh3 = build_mlp(2, 4, HIDDEN_NH3, key_nh3)

    return ApproximationSpace(
        {"x": 2},
        [(nn_inc, "scalar", None), (nn_nh3, "vec", 4)],
        model_type="x",
        pre_processing=scaled_coordinates,
        post_processing=[inconel_post_processing, nh3_post_processing],
    )


# Loss weights

W_COMMON = {
    "inc_bottom": (150.0,),
    "inc_side": (350.0,),
    "inc_top": (200.0,),
    "inc_lower": (350.0,),
    # NH3 interior: [radial, axial, continuity, energy], momentum favored
    # over continuity (same reasoning as "Navier stokes 2d testcase.py").
    "nh3_lower": (5.0, 5.0, 1.0, 1.0),
    "nh3_upper": (5.0, 5.0, 1.0, 1.0),
    "interface_flux": (600.0,),
    "pressure_pin": (1.0,),
}


# Evaluation helpers (space-aware; combine both materials' fields)

def evaluate_temperature_inc(space, x):
    x = jnp.asarray(x, dtype=jnp.float64)
    temperature = space.create_variables()[0]
    if x.ndim == 2:
        return jax.vmap(lambda x_one: temperature(space, x_one))(x)
    return temperature(space, x)


def evaluate_nh3_field(space, x):
    """Return [u_r, u_z, p, T] at physical points, T already combined with
    the Inconel field (T_NH3 = T_INC + correction)."""
    x = jnp.asarray(x, dtype=jnp.float64)

    def one_point(x_one):
        t_inc_field, nh3_field = space.create_variables()
        raw = nh3_field(space, x_one)
        t_inc = t_inc_field(space, x_one)
        return jnp.concatenate([raw[0:3], t_inc + raw[3:4]], axis=-1)

    if x.ndim == 2:
        return jax.vmap(one_point)(x)
    return one_point(x)

# Geometry masks and plotting helpers

def z_bottom_ammonia_numpy(r):
    r = np.asarray(r, dtype=np.float64)
    r_clipped = np.clip(r, float(NH3_LOWER_R[0]), float(NH3_LOWER_R[-1]))
    return np.interp(
        r_clipped,
        np.asarray(NH3_LOWER_R, dtype=np.float64),
        np.asarray(NH3_LOWER_Z, dtype=np.float64),
    )


def in_nh3_numpy(r, z):
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return (r >= 0.0) & (r <= R_NH3) & (z >= z_bottom_ammonia_numpy(r)) & (z <= Z_NH3_TOP)


def draw_geometry(ax):
    lower_r = np.asarray(NH3_LOWER_R, dtype=np.float64)
    lower_z = np.asarray(NH3_LOWER_Z, dtype=np.float64)

    ax.plot([0.0, 0.0], [0.0, H_OUT * 1e3], "k", lw=1.2)
    ax.plot([R_OUT * 1e3, R_OUT * 1e3], [0.0, H_OUT * 1e3], "k", lw=1.2)
    ax.plot([0.0, R_OUT * 1e3], [0.0, 0.0], "k", lw=1.2)
    ax.plot([0.0, R_OUT * 1e3], [H_OUT * 1e3, H_OUT * 1e3], "k", lw=1.2)
    ax.plot([R_NH3 * 1e3, R_NH3 * 1e3], [Z_NH3_MAX_LOWER * 1e3, Z_NH3_TOP * 1e3], "k", lw=1.4)
    ax.plot([0.0, R_NH3 * 1e3], [Z_NH3_TOP * 1e3, Z_NH3_TOP * 1e3], "k", lw=1.4)
    ax.plot(lower_r * 1e3, lower_z * 1e3, "k", lw=1.4)

    ax.set_aspect("equal")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")


def evaluate_grid(space, n_r=161, n_z=501):
    r = np.linspace(0.0, R_OUT, n_r)
    z = np.linspace(0.0, H_OUT, n_z)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    points = np.column_stack([rr.ravel(), zz.ravel()])

    nh3_mask = in_nh3_numpy(points[:, 0], points[:, 1])
    inc_mask = ~nh3_mask

    temperature = np.empty(points.shape[0], dtype=np.float64)
    ur = np.full(points.shape[0], np.nan, dtype=np.float64)
    uz = np.full(points.shape[0], np.nan, dtype=np.float64)
    pressure = np.full(points.shape[0], np.nan, dtype=np.float64)

    if np.any(inc_mask):
        temperature[inc_mask] = np.asarray(
            evaluate_temperature_inc(space, jnp.asarray(points[inc_mask]))
        ).reshape(-1)

    if np.any(nh3_mask):
        nh3_values = np.asarray(evaluate_nh3_field(space, jnp.asarray(points[nh3_mask])))
        ur[nh3_mask] = nh3_values[:, 0]
        uz[nh3_mask] = nh3_values[:, 1]
        pressure[nh3_mask] = nh3_values[:, 2]
        temperature[nh3_mask] = nh3_values[:, 3]

    shape = rr.shape
    return {
        "rr": rr,
        "zz": zz,
        "nh3_mask": nh3_mask.reshape(shape),
        "temperature": temperature.reshape(shape),
        "ur": ur.reshape(shape),
        "uz": uz.reshape(shape),
        "pressure": pressure.reshape(shape),
    }


def symmetric_limits(values):
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return (-1.0, 1.0)
    maximum = np.max(np.abs(valid))
    if maximum <= 0.0:
        maximum = 1.0
    return (-maximum, maximum)


def save_figure(figure, filename):
    filepath = SAVE_DIR / filename
    figure.savefig(filepath, dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot:\n{filepath.resolve()}")


def plot_temperature(space):
    field = evaluate_grid(space)
    rr, zz, temperature = field["rr"], field["zz"], field["temperature"]

    figure, ax = plt.subplots(figsize=(6.0, 10.0), constrained_layout=True)
    contour = ax.contourf(rr * 1e3, zz * 1e3, temperature, levels=60, cmap="inferno")
    figure.colorbar(contour, ax=ax).set_label("Temperature [K]")
    draw_geometry(ax)
    ax.set_title("Inconel + NH3 (Boussinesq) temperature field")
    save_figure(figure, "01_temperature.png")


def plot_velocity(space):
    field = evaluate_grid(space)
    rr, zz, ur, uz = field["rr"], field["zz"], field["ur"], field["uz"]
    speed = np.sqrt(ur**2 + uz**2)

    figure, ax = plt.subplots(figsize=(6.0, 10.0), constrained_layout=True)
    contour = ax.contourf(rr * 1e3, zz * 1e3, speed, levels=60, cmap="viridis")
    figure.colorbar(contour, ax=ax).set_label(r"$|\mathbf{u}|$ [m/s]")
    draw_geometry(ax)

    stride_r, stride_z = 8, 15
    r_q = rr[::stride_z, ::stride_r]
    z_q = zz[::stride_z, ::stride_r]
    ur_q = ur[::stride_z, ::stride_r]
    uz_q = uz[::stride_z, ::stride_r]
    valid = np.isfinite(ur_q) & np.isfinite(uz_q)
    ax.quiver(
        r_q[valid] * 1e3, z_q[valid] * 1e3, ur_q[valid], uz_q[valid],
        color="white", width=0.0025,
    )

    ax.set_title("NH3 velocity magnitude and meridional vectors")
    save_figure(figure, "02_velocity.png")

    print("\n" + "=" * 72)
    print("DIRECT VELOCITY MAGNITUDE CHECK (trivial-solution diagnostic)")
    print("=" * 72)
    finite_speed = speed[np.isfinite(speed)]
    max_speed = float(np.max(finite_speed)) if finite_speed.size else 0.0
    print(f"max|u| = {max_speed:.6e} m/s  (buoyancy scale U_REF = {U_REF:.3e} m/s)")
    if max_speed < 1.0e-6:
        print(">>> CONCLUSION: velocity field is essentially ZERO. Trivial solution.")
    else:
        print(">>> CONCLUSION: velocity field is nonzero.")


def plot_pressure(space):
    field = evaluate_grid(space)
    rr, zz, pressure = field["rr"], field["zz"], field["pressure"]
    pressure_display = pressure - np.nanmean(pressure)
    p_min, p_max = symmetric_limits(pressure_display)

    figure, ax = plt.subplots(figsize=(6.0, 10.0), constrained_layout=True)
    contour = ax.contourf(
        rr * 1e3, zz * 1e3, pressure_display, levels=60, cmap="coolwarm",
        vmin=p_min, vmax=p_max,
    )
    figure.colorbar(contour, ax=ax).set_label("Pressure minus grid mean [Pa]")
    draw_geometry(ax)
    ax.set_title("NH3 pressure field")
    save_figure(figure, "03_pressure.png")


def plot_divergence(space):
    field = evaluate_grid(space)
    rr, zz, ur, uz = field["rr"], field["zz"], field["ur"], field["uz"]

    r_values = rr[0, :]
    z_values = zz[:, 0]
    dr = float(r_values[1] - r_values[0])
    dz = float(z_values[1] - z_values[0])

    ur_filled = np.nan_to_num(ur, nan=0.0)
    uz_filled = np.nan_to_num(uz, nan=0.0)
    dur_dr = np.gradient(ur_filled, dr, axis=1)
    duz_dz = np.gradient(uz_filled, dz, axis=0)
    r_safe = np.maximum(rr, 1.0e-8)
    divergence = dur_dr + ur_filled / r_safe + duz_dz
    divergence[~field["nh3_mask"]] = np.nan
    divergence[:, 0] = np.nan

    d_min, d_max = symmetric_limits(divergence)

    figure, ax = plt.subplots(figsize=(6.0, 10.0), constrained_layout=True)
    contour = ax.contourf(
        rr * 1e3, zz * 1e3, divergence, levels=60, cmap="RdBu_r", vmin=d_min, vmax=d_max,
    )
    figure.colorbar(contour, ax=ax).set_label(r"$\nabla_{\mathrm{axi}} \cdot u$ [1/s]")
    draw_geometry(ax)
    ax.set_title("Finite-difference divergence diagnostic")
    save_figure(figure, "04_divergence.png")


def plot_loss_history(pinn):
    figure, ax = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    try:
        pinn.losses.plot(
            ax,
            loss_groups=["inc_bottom", "inc_side", "inc_top", "inc_lower",
                         "nh3_lower", "nh3_upper", "interface_flux", "pressure_pin"],
        )
    except Exception as error:  # noqa: BLE001
        print(f"\nLoss-group plot failed ({error}); using default plot.")
        pinn.losses.plot(ax)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Inconel + NH3 Boussinesq PINN loss history")
    ax.grid(True, which="both", alpha=0.30)
    save_figure(figure, "05_loss_history.png")


def plot_temperature_vs_fvm(space, excel_path, sheet_name="Sheet1"):
    """Temperature-only comparison against the conduction-case FVM data
    (no velocity reference is available)."""
    excel_path = Path(excel_path).resolve()
    if not excel_path.exists():
        print(f"\nFVM file not found, skipping comparison:\n{excel_path}")
        return

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    required = {"x", "y", "Temperature"}
    if not required.issubset(df.columns):
        print(f"\nFVM file missing columns {required}; skipping comparison.")
        return

    df = df.dropna(subset=["x", "y", "Temperature"]).copy()
    r_fvm = df["x"].to_numpy(dtype=np.float64)
    z_fvm = df["y"].to_numpy(dtype=np.float64)
    T_fvm = df["Temperature"].to_numpy(dtype=np.float64)

    if {"T0", "T1"}.issubset(df.columns):
        keep = np.isclose(df["T0"].to_numpy(dtype=np.float64), T0_K) & np.isclose(
            df["T1"].to_numpy(dtype=np.float64), T1
        )
        r_fvm, z_fvm, T_fvm = r_fvm[keep], z_fvm[keep], T_fvm[keep]

    nh3_mask = in_nh3_numpy(r_fvm, z_fvm)
    points = np.column_stack([r_fvm, z_fvm])
    T_pinn = np.empty(len(T_fvm), dtype=np.float64)

    if np.any(~nh3_mask):
        T_pinn[~nh3_mask] = np.asarray(
            evaluate_temperature_inc(space, jnp.asarray(points[~nh3_mask]))
        ).reshape(-1)
    if np.any(nh3_mask):
        T_pinn[nh3_mask] = np.asarray(
            evaluate_nh3_field(space, jnp.asarray(points[nh3_mask]))
        )[:, 3]

    error = T_pinn - T_fvm
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    max_abs = float(np.max(np.abs(error)))

    print("\n" + "=" * 72)
    print("TEMPERATURE VS. CONDUCTION-CASE FVM DATA (velocity has no reference)")
    print("=" * 72)
    print(f"RMSE [K]: {rmse:.6e}   MAE [K]: {mae:.6e}   Max |error| [K]: {max_abs:.6e}")
    print(
        "Note: this is a *reference* comparison only -- the conduction FVM data was "
        "generated without convection, so agreement is not expected to be exact "
        "inside the NH3 cavity."
    )


# Run

if __name__ == "__main__":
    space = build_space(seed=42)

    print("\nNumber of neural models:", len(space.models))

    pinn = Projector(
        model,
        space,
        sampler,
        weights=W_COMMON,
        one_loss_by_equation=True,
    )

    key = jax.random.PRNGKey(42)
    key, samples = sampler.sample(key, N_COLLOC, N_BC_COLLOC)

    print("\n" + "=" * 72)
    print("SAMPLE KEYS")
    print("=" * 72)
    print(samples.keys())

    print("\n" + "=" * 72)
    print("INITIAL LOSSES")
    print("=" * 72)
    print(pinn.evaluate_losses(space, samples))

    start = timeit.default_timer()
    key, pinn = pinn.project(key, space, N_EPOCHS, N_COLLOC, N_BC_COLLOC, n_dl=1)
    elapsed = timeit.default_timer() - start

    print("\n" + "=" * 72)
    print("BEST LOSS")
    print("=" * 72)
    print(pinn.best_loss)
    print(f"\nTraining time: {elapsed:.2f} s")

    plot_loss_history(pinn)
    plot_temperature(pinn.space)
    plot_velocity(pinn.space)
    plot_pressure(pinn.space)
    plot_divergence(pinn.space)

    PROJECT_DIR = Path(__file__).resolve().parents[1] / "data"
    FVM_FILE = PROJECT_DIR / "complete_data_inc_nh3.xlsx"
    plot_temperature_vs_fvm(pinn.space, FVM_FILE)

    print("\nAll plots saved in:\n" + str(SAVE_DIR.resolve()))
