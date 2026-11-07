import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm

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
)
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import (
    BoundaryResidual,
    InteriorResidual,
)

jax.config.update("jax_enable_x64", True)

SAVE_DIR = Path("output_inc_nh3")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Physical geometry and material properties

# Outer autoclave geometry
R_OUT = 0.035
H_OUT = 0.370


# NH3 cavity geometry
R_NH3 = 0.012
Z_NH3_TOP = 0.335


# CryS-MAS lower NH3/Inconel boundary polyline.
NH3_LOWER_R = jnp.array(
    [
        0.0000,
        0.0031,
        0.0060,
        0.0085,
        0.0104,
        0.0116,
        0.0120,
    ],
    dtype=jnp.float64,
)


NH3_LOWER_Z = jnp.array(
    [
        0.0230,
        0.0234,
        0.0246,
        0.0265,
        0.0290,
        0.0319,
        0.0350,
    ],
    dtype=jnp.float64,
)


# Lowest NH3 point, located on the symmetry axis.
Z_NH3_MIN = float(NH3_LOWER_Z[0])


# Height at which the lower polyline reaches the NH3
Z_NH3_MAX_LOWER = float(NH3_LOWER_Z[-1])

# Temperature-dependent thermal-conductivity correlations

CONDUCTIVITY_TEMPERATURE_UNIT = "K"


def conductivity_temperature(
    temperature_K,
):
    if CONDUCTIVITY_TEMPERATURE_UNIT == "K":
        return temperature_K

    if CONDUCTIVITY_TEMPERATURE_UNIT == "C":
        return temperature_K - 273.15

    raise ValueError("CONDUCTIVITY_TEMPERATURE_UNIT must be either 'K' or 'C'.")


def k_inc_value(
    temperature_K,
):
    temperature_input = conductivity_temperature(
        temperature_K,
    )

    return 6.9995 + 0.0151 * temperature_input


def k_nh3_value(
    temperature_K,
):
    temperature_input = conductivity_temperature(
        temperature_K,
    )

    # Ammonia thermal conductivity, taken from the NH3 material block of
    # Autoclave.mat (the material database used for the FV reference run):
    #
    #   name: NH3
    #   thermal conductivity: 0.3039497375-0.00019097*T+0.0000001887*T*T
    #
    # ~0.256 W/m/K over 520-595 K, essentially constant.
    #
    # The cubic previously used here,
    #   70.63350679-0.19209961*T+0.00020083*T^2-0.00000007*T^3,
    # is the conductivity of TC_shell_Al2O3 (the alumina thermocouple
    # shell) from the same file -- ~55x too high for ammonia.
    return (
        0.3039497375
        - 0.00019097 * temperature_input
        + 0.0000001887 * temperature_input**2
    )


# Fixed external temperature boundary condition
T0_K = 611.15
T1 = -300.0


T0_C = T0_K - 273.15
T_REF = T0_K


def T_outer(x):
    """
    Prescribed external Inconel temperature profile [K]:

        T_outer(z) = T0_K + T1 * z.

    Imposed on the bottom (z = 0), top (z = H_OUT) AND outer sidewall
    (r = R_OUT), as in the FV reference 2D_Heat_data.csv, which reproduces it
    on r = R_OUT to 5e-4 K.
    """
    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    z = x[..., 1:2]

    return T0_K + z * T1


# Geometry predicates in PHYSICAL coordinates
def z_bottom_ammonia(r):
    r = jnp.asarray(
        r,
        dtype=jnp.float64,
    )

    r_clipped = jnp.clip(
        r,
        NH3_LOWER_R[0],
        NH3_LOWER_R[-1],
    )

    return jnp.interp(
        r_clipped,
        NH3_LOWER_R,
        NH3_LOWER_Z,
    )


def in_lower_ammonia(r, z):
    return (
        (r >= 0.0) & (r <= R_NH3) & (z >= z_bottom_ammonia(r)) & (z <= Z_NH3_MAX_LOWER)
    )


def in_nh3(r, z):
    return (r >= 0.0) & (r <= R_NH3) & (z >= z_bottom_ammonia(r)) & (z <= Z_NH3_TOP)


# Material registry
#
# Adding a material is adding one entry. Everything downstream -- the
# Kirchhoff potential and its inverse, the conductance-weighted axial
# coordinate, the interior residual, the interface ansatz -- is derived from
# this table, so none of it has to be touched when the NH3 cavity is split
# into further subdomains with their own conductivities.
#
# ``region`` is a predicate on (r, z); ``None`` marks the matrix material
# that fills whatever no other entry claims. Entries are tested in order,
# first match wins, so a new inclusion goes BEFORE the material it sits in.
MATERIAL_SPECS = [
    {
        "name": "nh3",
        # Autoclave.mat, NH3
        "k": [0.3039497375, -0.00019097, 0.0000001887],
        "region": lambda r, z: in_nh3(r, z),
    },
    {
        "name": "inconel",
        # Autoclave.mat, Autoclave (Inconel 718)
        "k": [6.9995, 0.0151],
        "region": None,
    },
]

MATRIX_MATERIAL = "inconel"

# Temperature window the inverse Kirchhoff table has to cover. Generous on
# both sides of the Dirichlet data so no iterate can leave it.
T_TABLE_MIN = 300.0
T_TABLE_MAX = 900.0
_T_TABLE = jnp.linspace(T_TABLE_MIN, T_TABLE_MAX, 8001)


def conductivity_poly(coefficients, temperature_K):
    """k(T) = sum_i c_i T^i."""
    total = jnp.zeros_like(temperature_K)

    for power, coefficient in enumerate(coefficients):
        total = total + coefficient * temperature_K**power

    return total


def kirchhoff_poly(coefficients, temperature_K):
    """K(T) = integral of k, i.e. sum_i c_i T^(i+1) / (i+1)."""
    total = jnp.zeros_like(temperature_K)

    for power, coefficient in enumerate(coefficients):
        total = total + coefficient * temperature_K ** (power + 1) / (power + 1)

    return total


class Material:
    """One material: its conductivity, its Kirchhoff potential, its inverse.

    The Kirchhoff potential K(T) = integral_0^T k is strictly increasing
    wherever k > 0, so it always inverts. The inverse is tabulated once and
    polished with two Newton steps, which is exact to round-off and makes no
    assumption about the degree of k -- a cubic inverts as readily as the
    linear Inconel law.
    """

    def __init__(self, name, k_coefficients, region):
        self.name = name
        self.k_coefficients = list(k_coefficients)
        self.region = region

        self._k_table = kirchhoff_poly(self.k_coefficients, _T_TABLE)

        if bool(jnp.any(jnp.diff(self._k_table) <= 0.0)):
            raise ValueError(
                f"K(T) is not increasing for material '{name}' over "
                f"[{T_TABLE_MIN}, {T_TABLE_MAX}] K: k must stay positive."
            )

    def k(self, temperature_K):
        return conductivity_poly(self.k_coefficients, temperature_K)

    def kirchhoff(self, temperature_K):
        return kirchhoff_poly(self.k_coefficients, temperature_K)

    def kirchhoff_inverse(self, potential):
        """T such that K(T) = potential: table lookup plus Newton polish."""
        temperature = jnp.interp(potential, self._k_table, _T_TABLE)

        for _ in range(2):
            temperature = temperature - (
                self.kirchhoff(temperature) - potential
            ) / self.k(temperature)

        return temperature


MATERIALS = {
    spec["name"]: Material(
        spec["name"],
        spec["k"],
        spec["region"],
    )
    for spec in MATERIAL_SPECS
}

MATRIX = MATERIALS[MATRIX_MATERIAL]


def material_indicator(name, r, z):
    """Whether ``(r, z)`` belongs to material ``name`` (first match wins)."""
    claimed = jnp.zeros_like(jnp.asarray(r), dtype=bool)

    for spec in MATERIAL_SPECS:
        if spec["region"] is None:
            here = jnp.ones_like(claimed)
        else:
            here = spec["region"](r, z)

        mine = here & (~claimed)

        if spec["name"] == name:
            return mine

        claimed = claimed | here

    raise KeyError(name)


# Leading-order profile
#
# All three outer faces carry the same Dirichlet datum T_outer(z), and the
# sidewall pins the Inconel to it: the FV solution departs from T_outer by
# ~0.02 K everywhere except within a few millimetres of the two cavity ends
# (+1.8 K at the floor, -2.6 K at the roof, both on the axis). T_outer is
# therefore the leading behaviour, in both materials.
#
# This replaces the conductance-weighted coordinate S(z) = int dz / C(z) used
# while the sidewall was adiabatic. That coordinate is the exact solution of
# a 1-D column whose total flow Q is constant along z; with a Dirichlet
# sidewall heat crosses r = R_OUT, Q varies by ~15 % along the height, and
# the column model is simply the wrong leading order. The networks now see
# plain z / H_OUT.
def axial_coordinate(z):
    """Axial coordinate handed to the NETWORKS."""
    return z / H_OUT


_K_BOTTOM = MATRIX.kirchhoff(T0_K)
_K_TOP = MATRIX.kirchhoff(T0_K + T1 * H_OUT)


def T_axial_profile(z):
    """The leading-order temperature profile T_outer(z)."""
    return T0_K + T1 * z


def matrix_potential_base(z):
    """Matrix Kirchhoff potential of the leading profile, K_matrix(T_outer(z)).

    Equal to K_matrix(T_outer) on every outer face, so a correction multiplied
    by a factor vanishing on those faces leaves all three Dirichlet
    conditions exact. It is NOT affine in z (K is quadratic in T for the
    Inconel law): Delta K = k'(T) T1^2 is small and left to the correction.
    """
    return MATRIX.kirchhoff(T_axial_profile(z))


# Scale of the axial heat flow [W]: a uniform Inconel column at the mid-range
# temperature under the imposed gradient.
FLUX_BALANCE_REF = (
    jnp.pi * R_OUT**2 * MATRIX.k(T0_K + 0.5 * T1 * H_OUT) * abs(T1)
)


def T_base(x):
    """The 1-D profile as a function of the full coordinate."""
    x = jnp.asarray(x, dtype=jnp.float64)

    return T_axial_profile(x[..., 1:2])


# Training settings
# Half the points and twice the epochs of the adiabatic setup. ENG was still
# descending steeply at 200 epochs, and its cost per epoch scales with the
# number of points, so the same wall time buys more steps. Measured against
# FV (T0 = 611.15 K, T1 = -300 K/m), same wall time: 17 500 points x 200
# epochs -> 0.18 K RMS; 8 750 x 400 -> 0.078 K; 4 375 x 800 -> 0.092 K.
N_COLLOC = {
    # Inconel interior sampling regions
    "inc_bottom": 500,
    "inc_side": 2000,
    "inc_top": 750,
    "inc_lower": 1750,
    # NH3 interior sampling regions
    "nh3_upper": 2500,
    "nh3_lower": 1250,
    # Integral constraint: each sample is a whole cross section.
    "flux_balance": 40,
}


N_EPOCHS = 400

LEARNING_RATE = 1.0e-4

HIDDEN = [24, 24, 24]

# Width of the two interface-trace networks (functions on a curve).
HIDDEN_TRACE = [16, 16]

# Nondimensional reference scales
DELTA_T_REF = abs(T1) * H_OUT


# Interface temperature residual scale [K].
INTERFACE_T_REF = 5

# Fixed conductivity reference for residual normalization

T_K_MIN_FOR_SCALING = 495.0
T_K_MAX_FOR_SCALING = 615.0

T_K_SCALE_GRID = jnp.linspace(
    T_K_MIN_FOR_SCALING,
    T_K_MAX_FOR_SCALING,
    2001,
)

K_INC_SCALE_VALUES = k_inc_value(
    T_K_SCALE_GRID,
)

K_NH3_SCALE_VALUES = k_nh3_value(
    T_K_SCALE_GRID,
)

# Fail early if the correlation/unit choice produces an
# unphysical conductivity in the anticipated range.
if bool(jnp.any(K_INC_SCALE_VALUES <= 0.0)):
    raise ValueError()

if bool(jnp.any(K_NH3_SCALE_VALUES <= 0.0)):
    raise ValueError()

K_INC_MIN_REF = float(jnp.min(K_INC_SCALE_VALUES))

K_INC_MAX_REF = float(jnp.max(K_INC_SCALE_VALUES))

K_NH3_MIN_REF = float(jnp.min(K_NH3_SCALE_VALUES))

K_NH3_MAX_REF = float(jnp.max(K_NH3_SCALE_VALUES))

# One fixed conservative global scale for all variable-k PDE
# and flux residuals.
K_REF = max(
    K_INC_MAX_REF,
    K_NH3_MAX_REF,
)

# Characteristic length [m].
L_REF = R_OUT


# Reference scale for the original non-r-weighted PDE operator:
PDE_REF = K_REF * DELTA_T_REF / L_REF**2


# Reference heat-flux scale:
FLUX_REF = K_REF * DELTA_T_REF / L_REF


# Reference scale for the NH3/Inconel interface flux residual.
#
# NOT FLUX_REF. The condition is k_inc dT_inc/dn = k_nh3 dT_nh3/dn, and both
# sides equal the TRUE flux crossing the cavity wall, which is set by the
# poorly conducting side: ~0.26 * 111 / 0.035 ~ 8e2 W/m2, against a FLUX_REF
# of ~5e4 W/m2 built from k_inc. Dividing by FLUX_REF makes the condition 60x
# too lenient: a residual reading 5e-3 is a 270 W/m2 flux jump, three times
# the ~88 W/m2 that physically crosses the cavity. The mis-balanced heat then
# shifts how much the Inconel annulus carries, which is what put a smooth
# few-K wave in z into the solution while every loss term still looked small.
INTERFACE_FLUX_REF = K_NH3_MAX_REF * DELTA_T_REF / L_REF


# Reference scale for the r-weighted Laplacian of the potential.
#
# In Kirchhoff variables there is only ONE scale: grad K is a heat flux, so
# the operator scales like (K_bottom - K_top) / L_REF whatever the material.
# The per-material normalisation the temperature form needed -- and which was
# a real bug when a single global K_REF left the NH3 residual ~64x too loose
# -- simply does not arise here.
PDE_POTENTIAL_REF = (_K_BOTTOM - _K_TOP) / L_REF

# ... except that the potential of a material spans K_m(T_bottom) - K_m(T_top),
# which for the NH3 is ~60x smaller than for the Inconel. Normalising the NH3
# residual by the Inconel span left it ~60x too loose (~3600x in the squared
# loss): trained NH3 residuals read 1e-4 against 1e-3 for the Inconel while the
# NH3 field carried errors of several kelvin. Each material's residual is
# therefore normalised by its own span.
NH3_PDE_POTENTIAL_REF = (
    MATERIALS["nh3"].kirchhoff(T0_K) - MATERIALS["nh3"].kirchhoff(T0_K + T1 * H_OUT)
) / L_REF


EPS_AXIS = 1.0e-7

T_INTERNAL_CENTER = T0_K + 0.5 * T1 * H_OUT

T_INTERNAL_RANGE = -T1 * H_OUT


# Dimensionless multiplier of the raw NH3 neural-network output.
#
# 0.1, not 1.0. internal_reference_profile is algebraically identical to
# T_outer(z), so this scales the DEVIATION from the correct leading profile,
# whose true size is ~2 K -- 1.0 allowed +-111 K of slack. That slack is not
# free: inside the interface band the field is blended as
# chi*G + (1-chi)*B, and d(chi)/dr picks up a radial component wherever
# Gamma is not perpendicular to the axis (the cavity floor meets r = 0 at a
# 7.4 degree tilt). The resulting axis-symmetry violation is proportional to
# (G - B), so an oversized background scale shows up directly as dT/dr != 0
# on the symmetry axis.
NH3_CORRECTION_SCALE = 0.1

# Reference coordinates
XI_NH3 = R_NH3 / R_OUT


ETA_NH3_MIN = Z_NH3_MIN / H_OUT

ETA_NH3_MAX_LOWER = Z_NH3_MAX_LOWER / H_OUT

ETA_NH3_TOP = Z_NH3_TOP / H_OUT


# Lower NH3/Inconel polyline in normalized reference coordinates.
XI_NH3_LOWER = NH3_LOWER_R / R_OUT

ETA_NH3_LOWER = NH3_LOWER_Z / H_OUT

# Main mapped autoclave domain
main_mapping = DomainMapping.rectangle(
    origin=jnp.array([0.0, 0.0]),
    x_dir=jnp.array([R_OUT, 0.0]),
    y_dir=jnp.array([0.0, H_OUT]),
)


main_domain = Square2D(
    [
        (0.0, 1.0),
        (0.0, 1.0),
    ],
    is_main_domain=True,
    label_str="main",
)


main_domain.set_mapping(
    main_mapping,
    bounds_postmap=[
        (0.0, R_OUT),
        (0.0, H_OUT),
    ],
)

# Volumetric sampling subdomains
inc_bottom = Square2D(
    [
        (0.0, 1.0),
        (0.0, ETA_NH3_MIN),
    ],
    is_main_domain=False,
    label_str="inc_bottom",
)

# Inconel outside the NH3 sidewall:
inc_side = Square2D(
    [
        (XI_NH3, 1.0),
        (
            ETA_NH3_MIN,
            ETA_NH3_TOP,
        ),
    ],
    is_main_domain=False,
    label_str="inc_side",
)

# Inconel above NH3:
inc_top = Square2D(
    [
        (0.0, 1.0),
        (
            ETA_NH3_TOP,
            1.0,
        ),
    ],
    is_main_domain=False,
    label_str="inc_top",
)


# Lower-envelope rectangle:
inc_lower = Square2D(
    [
        (0.0, XI_NH3),
        (
            ETA_NH3_MIN,
            ETA_NH3_MAX_LOWER,
        ),
    ],
    is_main_domain=False,
    label_str="inc_lower",
)

# NH3 sampling subdomains
nh3_lower = Square2D(
    [
        (0.0, XI_NH3),
        (
            ETA_NH3_MIN,
            ETA_NH3_MAX_LOWER,
        ),
    ],
    is_main_domain=False,
    label_str="nh3_lower",
)


# Upper NH3 cavity:
nh3_upper = Square2D(
    [
        (0.0, XI_NH3),
        (
            ETA_NH3_MAX_LOWER,
            ETA_NH3_TOP,
        ),
    ],
    is_main_domain=False,
    label_str="nh3_upper",
)


flux_balance = Square2D(
    [
        (0.0, 1.0),
        (0.0, 1.0),
    ],
    is_main_domain=False,
    label_str="flux_balance",
)


# Attach all volumetric sampling regions to main_domain.
for domain in (
    inc_bottom,
    inc_side,
    inc_top,
    inc_lower,
    nh3_lower,
    nh3_upper,
    flux_balance,
):
    main_domain.add_subdomain(
        domain,
    )


# Boundary segments
# External autoclave boundaries

domain_outer_side = Segment2D(
    pt1=(
        1.0,
        0.0,
    ),
    pt2=(
        1.0,
        1.0,
    ),
    label_str="outer_side",
)

domain_outer_bottom = Segment2D(
    pt1=(
        0.0,
        0.0,
    ),
    pt2=(
        1.0,
        0.0,
    ),
    label_str="outer_bottom",
)

domain_outer_top = Segment2D(
    pt1=(
        0.0,
        1.0,
    ),
    pt2=(
        1.0,
        1.0,
    ),
    label_str="outer_top",
)


# Symmetry-axis material segments
domain_axis_inc_bottom = Segment2D(
    pt1=(
        0.0,
        0.0,
    ),
    pt2=(
        0.0,
        ETA_NH3_MIN,
    ),
    label_str="axis_inc_bottom",
)

domain_axis_nh3 = Segment2D(
    pt1=(
        0.0,
        ETA_NH3_MIN,
    ),
    pt2=(
        0.0,
        ETA_NH3_TOP,
    ),
    label_str="axis_nh3",
)

domain_axis_inc_top = Segment2D(
    pt1=(
        0.0,
        ETA_NH3_TOP,
    ),
    pt2=(
        0.0,
        1.0,
    ),
    label_str="axis_inc_top",
)

# NH3/Inconel interface: cylindrical sidewall
domain_if_side_T = Segment2D(
    pt1=(
        XI_NH3,
        ETA_NH3_MAX_LOWER,
    ),
    pt2=(
        XI_NH3,
        ETA_NH3_TOP,
    ),
    label_str="if_side_T",
)

domain_if_side_q = Segment2D(
    pt1=(
        XI_NH3,
        ETA_NH3_MAX_LOWER,
    ),
    pt2=(
        XI_NH3,
        ETA_NH3_TOP,
    ),
    label_str="if_side_q",
)

# NH3/Inconel interface: upper horizontal face

domain_if_top_T = Segment2D(
    pt1=(
        0.0,
        ETA_NH3_TOP,
    ),
    pt2=(
        XI_NH3,
        ETA_NH3_TOP,
    ),
    label_str="if_top_T",
)

domain_if_top_q = Segment2D(
    pt1=(
        0.0,
        ETA_NH3_TOP,
    ),
    pt2=(
        XI_NH3,
        ETA_NH3_TOP,
    ),
    label_str="if_top_q",
)


# Lower NH3/Inconel interface
N_LOWER_INTERFACE_SEGMENTS = len(NH3_LOWER_R) - 1


lower_temperature_domains = []
lower_flux_domains = []


for i in range(N_LOWER_INTERFACE_SEGMENTS):
    p1 = (
        float(XI_NH3_LOWER[i]),
        float(ETA_NH3_LOWER[i]),
    )

    p2 = (
        float(XI_NH3_LOWER[i + 1]),
        float(ETA_NH3_LOWER[i + 1]),
    )

    # Temperature-continuity segment.
    lower_temperature_domains.append(
        Segment2D(
            pt1=p1,
            pt2=p2,
            label_str="if_lower_T",
            label_idx=i + 1,
        )
    )

    # Flux-continuity segment.
    lower_flux_domains.append(
        Segment2D(
            pt1=p1,
            pt2=p2,
            label_str="if_lower_q",
            label_idx=i + 1,
        )
    )


# Register physical boundary domains
for boundary in (
    # External Inconel boundaries
    domain_outer_side,
    domain_outer_bottom,
    domain_outer_top,
    # Symmetry-axis segments
    domain_axis_inc_bottom,
    domain_axis_nh3,
    domain_axis_inc_top,
    # Vertical NH3/Inconel sidewall interface
    domain_if_side_T,
    domain_if_side_q,
    # Upper horizontal NH3/Inconel interface
    domain_if_top_T,
    domain_if_top_q,
    # Six lower piecewise-linear NH3/Inconel segments
    *lower_temperature_domains,
    *lower_flux_domains,
):
    main_domain.add_bc_domain(
        boundary,
    )


# Boundary groups

BOUNDARY_GROUPS = {
    "outer_side": [
        domain_outer_side.get_label(),
    ],
    # Axis regularity.
    "axis_inc": [
        domain_axis_inc_bottom.get_label(),
        domain_axis_inc_top.get_label(),
    ],
    "axis_nh3": [
        domain_axis_nh3.get_label(),
    ],
    # NH3/Inconel temperature continuity.
    "interface_temperature": [
        domain_if_side_T.get_label(),
        domain_if_top_T.get_label(),
        *[domain.get_label() for domain in lower_temperature_domains],
    ],
    # NH3/Inconel conductive-flux continuity.
    "interface_flux": [
        domain_if_side_q.get_label(),
        domain_if_top_q.get_label(),
        *[domain.get_label() for domain in lower_flux_domains],
    ],
}


main_domain.set_boundaries_dict(
    BOUNDARY_GROUPS,
)


# Boundary collocation counts
N_BC_COLLOC = {
    group_name: {boundary_label: 300 for boundary_label in boundary_labels}
    for group_name, boundary_labels in BOUNDARY_GROUPS.items()
}


# Axis samples.
for boundary_label in N_BC_COLLOC["axis_inc"]:
    N_BC_COLLOC["axis_inc"][boundary_label] = 300


for boundary_label in N_BC_COLLOC["axis_nh3"]:
    N_BC_COLLOC["axis_nh3"][boundary_label] = 300


# Interface samples.
for boundary_label in N_BC_COLLOC["interface_temperature"]:
    N_BC_COLLOC["interface_temperature"][boundary_label] = 750


for boundary_label in N_BC_COLLOC["interface_flux"]:
    N_BC_COLLOC["interface_flux"][boundary_label] = 750


# Neural temperature trial functions
def scaled_coordinates(x):
    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    r_scaled = x[..., 0:1] / R_OUT
    z_scaled = axial_coordinate(x[..., 1:2])

    # Quadratic in r/R_OUT, so any network of rho satisfies dT/dr = 0 on the
    # symmetry axis. The smoothstep used while the sidewall was adiabatic
    # (which also flattened rho at r = R_OUT) has no role any more: the
    # sidewall is Dirichlet, carried by outer_distance_factor.
    rho = r_scaled**2

    eta = z_scaled

    return jnp.concatenate(
        [
            rho,
            eta,
        ],
        axis=-1,
    )


# The NH3 network gets its own scaling.
#
# scaled_coordinates normalizes on the FULL autoclave, so the NH3
# network only ever sees rho = (r / R_OUT)**2 in [0, 0.118]: 12% of its
# radial input range, squeezed against 0 where tanh is close to linear.
# That is the direction in which flux continuity is enforced on the
# vertical NH3/Inconel sidewall, so the NH3 field is nearly blind
# exactly where the interface condition is hardest.
#
# Normalizing on the cavity instead spreads both inputs over [0, 1].
# rho stays quadratic in r so that dT/dr = 0 on the symmetry axis
# remains automatic (no axis residual is imposed anywhere).
def nh3_scaled_coordinates(x):
    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    r_scaled = x[..., 0:1] / R_NH3

    sigma = axial_coordinate(x[..., 1:2])
    sigma_min = axial_coordinate(jnp.asarray(Z_NH3_MIN))
    sigma_max = axial_coordinate(jnp.asarray(Z_NH3_TOP))

    z_scaled = (sigma - sigma_min) / (sigma_max - sigma_min)

    rho = r_scaled**2
    eta = z_scaled

    return jnp.concatenate(
        [
            rho,
            eta,
        ],
        axis=-1,
    )


ENVELOPE_LENGTH = R_OUT


def outer_distance_factor(
    x,
):
    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    r = x[..., 0:1]
    z = x[..., 1:2]

    # The FV reference imposes T = T_outer(z) on the bottom, top AND outer
    # sidewall, so the factor vanishes on all three, and peaks at 1.
    #
    # The sidewall factor is 1 - (r / R_OUT)^2 rather than (R_OUT - r) / R_OUT:
    # it must be EVEN in r, otherwise its r-derivative at r = 0 would break the
    # axis condition that nothing else imposes.
    #
    # The axial factors saturate over ENVELOPE_LENGTH = R_OUT instead of being
    # the parabola z (H_OUT - z) / H_OUT^2. With a Dirichlet sidewall,
    # perturbations decay over ~R_OUT, and the largest ones sit at the cavity
    # ends, only ~35 mm from the top and bottom faces. The parabola is 0.31 there
    # (3.5 mm above the cavity roof) and starved the correction exactly where it
    # was needed: +1.6 K in the cap with a 0.5 K budget. Measured on two seeds
    # at 400 epochs: 0.078 / 0.078 K RMS against FV with the parabola,
    # 0.071 / 0.060 K with the saturating factors.
    d_bottom = jnp.tanh(z / ENVELOPE_LENGTH)

    d_top = jnp.tanh((H_OUT - z) / ENVELOPE_LENGTH)

    d_side = 1.0 - (r / R_OUT) ** 2

    return d_bottom * d_top * d_side


def nh3_interface_distance_factor(
    x,
):
    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    r = x[..., 0:1]
    z = x[..., 1:2]

    z_bottom = z_bottom_ammonia(
        r,
    )

    # Dimensionless distances across the NH3 cavity.
    d_bottom = (z - z_bottom) / H_OUT

    d_top = (Z_NH3_TOP - z) / H_OUT

    d_side = (R_NH3 - r) / R_OUT

    return d_bottom * d_top * d_side


# NH3/Inconel interface curve, as an ordered list of segments
#
# Gamma runs from the axis at the cavity floor, along the lower polyline, up
# the cylindrical sidewall and back along the top face to the axis. Its two
# endpoints sit on r = 0, which is the symmetry axis and not a material
# interface, so Gamma is an open curve.
_IF_POINTS = [
    (float(NH3_LOWER_R[i]), float(NH3_LOWER_Z[i])) for i in range(len(NH3_LOWER_R))
] + [
    (R_NH3, Z_NH3_TOP),
    (0.0, Z_NH3_TOP),
]

_IF_P0 = jnp.array(_IF_POINTS[:-1], dtype=jnp.float64)
_IF_P1 = jnp.array(_IF_POINTS[1:], dtype=jnp.float64)

# Half-width of the band in which the interface ansatz is active [m].
#
# Must be smaller than the distance from Gamma to every outer boundary
# (0.023 m to r = R_OUT and to z = 0, 0.035 m to z = H_OUT) so that the outer
# Dirichlet hard constraints are untouched, and smaller than the
# cavity's own half-width (0.012 m) so the band does not meet itself.
INTERFACE_BAND = 0.005

# Guard so that d(sqrt)/dx stays finite for a point sitting exactly on Gamma.
_IF_EPS2 = 1.0e-24


def interface_projection(x):
    """Closest point of Gamma to ``x``, and the distance to it.

    Returns ``(distance, projected_point)``. The projection is what makes the
    construction work: a function of the projected point is constant along
    the normal, so its normal derivative vanishes and the flux trace below is
    the ONLY thing that sets dT/dn on either side.
    """
    d = _IF_P1 - _IF_P0
    len2 = jnp.sum(d * d, axis=-1)

    t = jnp.sum((x - _IF_P0) * d, axis=-1) / len2
    t = jnp.clip(t, 0.0, 1.0)

    proj = _IF_P0 + t[:, None] * d
    diff = x - proj
    dist2 = jnp.sum(diff * diff, axis=-1)

    nearest = jnp.argmin(dist2)

    return (
        jnp.sqrt(dist2[nearest] + _IF_EPS2),
        proj[nearest],
    )


def interface_phi(x):
    """Signed distance to Gamma, positive inside the NH3 cavity."""
    distance, projected = interface_projection(x)

    sign = jnp.where(
        in_nh3(x[0], x[1]),
        1.0,
        -1.0,
    )

    return sign * distance, projected


def interface_cutoff(phi):
    """Band cutoff: 1 on Gamma, 0 beyond INTERFACE_BAND, flat at both ends.

    ``chi'(0) = 0`` is essential. The normal derivative of chi * G would
    otherwise contribute chi' * G at the interface and destroy the flux
    identity that the whole construction is built on.
    """
    q = phi / INTERFACE_BAND

    return jnp.where(
        jnp.abs(q) < 1.0,
        (1.0 - q * q) ** 2,
        0.0,
    )


def interface_trace_features(x):
    """Network input for the two interface traces: the projected point.

    Both traces are functions of the projection alone, hence constant along
    normals to Gamma.
    """
    x = jnp.asarray(x, dtype=jnp.float64)

    _, projected = interface_phi(x)

    # QUADRATIC in the projected radius, for the same reason the field
    # pre-processings are: along the cavity's top and bottom faces the
    # projection is (r, const), so a trace that were linear in r_gamma would
    # have a nonzero radial derivative on the symmetry axis and would break
    # the axis condition that nothing else imposes. Squaring makes the traces
    # even in r, so dU/dr = dW/dr = 0 at r = 0 by construction.
    return jnp.concatenate(
        [
            (projected[0:1] / R_NH3) ** 2,
            projected[1:2] / H_OUT,
        ],
        axis=-1,
    )


# Scale [K] of the NH3 field's departure from its background T_outer(z). The
# true departure is ~0.02 K in the cavity bulk and up to ~2.6 K at its two
# ends; keeping it small matters because any mismatch between the base and
# the background is multiplied by chi'' ~ 8/delta^2 inside the band.
NH3_TRACE_CORR = 2.0

# The same budgets expressed as potentials: a correction of X kelvin is a
# correction of roughly X * k(T) in K. Stated this way the numbers stay
# interpretable in kelvin as materials are added.
NH3_TRACE_CORR_K = NH3_TRACE_CORR * 0.26


# Scale of the interface temperature trace around the linear profile [K].
# Correction budgets [K].
#
# All three are deviations from the leading profile T_outer(z). With the
# Dirichlet sidewall that profile is ~0.02 K from the FV solution in the
# bulk, and at most ~2.6 K from it, at the two cavity ends.
# outer_distance_factor peaks at 1, so INC_CORRECTION_SCALE is the scale of
# the correction in kelvin.
# 0.5 K, not 3.0: the strong form is nearly blind to smooth error, so slack
# in this budget is spent on error almost for free. Measured at 200 epochs:
# Inconel bulk error 0.068 -> 0.032 K RMS when going from 3.0 to 0.5.
INC_CORRECTION_SCALE = 0.5

INC_CORRECTION_SCALE_K = INC_CORRECTION_SCALE * 15.4

INTERFACE_U_SCALE = 3.0


def identity_post_processing(raw, x):
    """All four models are combined by hand in evaluate_temperature."""
    return raw


def internal_reference_profile(x):
    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    z_scaled = x[..., 1:2] / H_OUT

    return T_INTERNAL_CENTER + T_INTERNAL_RANGE * (0.5 - z_scaled)


# Trial functions with BOTH interface conditions hard-constrained
#
# Let phi be the signed distance to Gamma (positive in NH3), P(x) the closest
# point of Gamma, chi a band cutoff with chi(0) = 1 and chi'(0) = 0, and let
#
#     U(P)  be a shared interface TEMPERATURE trace   [K]
#     W(P)  be a shared interface NORMAL FLUX trace   [W/m2]
#
# both functions of the projection alone. Define, per material m,
#
#     G_m(x) = U + phi * W / k_m(U)
#     T_m(x) = chi * G_m + (1 - chi) * B_m(x)
#
# with B_m the old background ansatz, which already carries the outer
# Dirichlet hard constraints.
#
# On Gamma, phi = 0 and chi = 1, so T_inc = T_nh3 = U: temperature
# continuity is exact. For the normal derivative, chi'(0) = 0 kills the
# cutoff term, dU/dn = 0 because U is a function of the projection, and
# d(phi^2 ...)/dn = 0, leaving dT_m/dn = W / k_m(U), hence
#
#     k_inc dT_inc/dn = W = k_nh3 dT_nh3/dn
#
# so flux continuity is exact too, with k evaluated at the shared interface
# temperature -- which is exactly the physical condition. Both interface
# residuals can therefore be deleted from the loss.
#
# Away from the band chi = 0 and each field falls back to B_m, so the outer
# boundary conditions are untouched (INTERFACE_BAND is smaller than the
# distance from Gamma to every outer face).


def background_inconel(raw, x):
    """Inconel background POTENTIAL: Dirichlet on all three outer faces, hard.

    K is monotone in T, so pinning K to K(T_outer) on a Dirichlet face pins T
    to T_outer exactly.
    """
    return (
        matrix_potential_base(x[..., 1:2])
        + INC_CORRECTION_SCALE_K * outer_distance_factor(x) * raw
    )


def _potential_at_point(space, x, material):
    """Kirchhoff potential of one material at ONE point ``x`` of shape (2,).

    Working in K rather than T buys three things at once, and every one of
    them is material-agnostic:

      * the interior equation div(k grad T) = 0 becomes the LINEAR, constant
        coefficient Laplace equation ``Delta K = 0`` -- identical for every
        material, so the residual and its normalisation stop depending on
        which material a point is in;
      * flux continuity ``k dT/dn`` continuous becomes ``dK/dn`` continuous,
        which is linear in the network outputs;
      * temperature continuity is K_m^{-1}(K_m) matching, which the shared
        trace U delivers by construction.

    The interface ansatz is correspondingly simpler than in T: the flux
    trace enters as ``+ phi * W`` with no division by k(U).
    """
    variables = space.create_variables()

    raw_inc = variables[0](space, x)
    raw_nh3 = variables[1](space, x)
    raw_u = variables[2](space, x)
    raw_w = variables[3](space, x)

    phi, projected = interface_phi(x)
    phi = jnp.reshape(phi, (1,))
    chi = interface_cutoff(phi)

    z_gamma = projected[1:2]
    u_trace = T_axial_profile(z_gamma) + INTERFACE_U_SCALE * raw_u
    w_trace = INTERFACE_FLUX_REF * raw_w

    if material == "nh3":
        # Same blend as the Inconel: interface ansatz in the band, a smooth
        # background beyond it. The background used to be K_nh3(U(P(x))),
        # i.e. the interface trace extended along normals. That extension is
        # discontinuous on the cavity's medial axis, where P(x) jumps from the
        # floor (or roof) to the sidewall: on r = 0 at z = 35.5 mm and
        # z = 323.5 mm, U jumps by ~3.4 K, which a smooth correction network
        # cannot cancel. With the Dirichlet sidewall, T_outer(z) is within
        # ~0.02 K of the solution in the cavity bulk, so it is the natural
        # background and has no such jump. (Measured: NH3 error near the
        # cavity ends 0.84 -> 0.35 K RMS, max 5.1 -> 1.7 K, at 200 epochs.)
        near_interface = MATERIALS["nh3"].kirchhoff(u_trace) + phi * w_trace
        background = (
            MATERIALS["nh3"].kirchhoff(T_axial_profile(x[..., 1:2]))
            + NH3_TRACE_CORR_K * raw_nh3
        )

        return chi * near_interface + (1.0 - chi) * background

    near_interface = MATRIX.kirchhoff(u_trace) + phi * w_trace

    return chi * near_interface + (1.0 - chi) * background_inconel(raw_inc, x)


def _temperature_at_point(space, x, material):
    """Physical temperature: invert the material's Kirchhoff potential."""
    name = "nh3" if material == "nh3" else MATRIX_MATERIAL

    return MATERIALS[name].kirchhoff_inverse(_potential_at_point(space, x, material))


def _evaluate_pointwise(space, x, material, fn):
    x = jnp.asarray(x, dtype=jnp.float64)

    if x.ndim == 2:
        return jax.vmap(lambda x_one: fn(space, x_one, material))(x)

    return fn(space, x, material)


def evaluate_temperature(space, material, x):
    return _evaluate_pointwise(space, x, material, _temperature_at_point)


def evaluate_potential(space, material, x):
    return _evaluate_pointwise(space, x, material, _potential_at_point)


def T_inc_eval(space, x):
    return evaluate_temperature(space, "inc", x)


def T_nh3_eval(space, x):
    return evaluate_temperature(space, "nh3", x)


def K_inc_eval(space, x):
    return evaluate_potential(space, "inc", x)


def K_nh3_eval(space, x):
    return evaluate_potential(space, "nh3", x)


K_INC = ParamScalarFunction(
    {"x": 2},
    K_inc_eval,
    f_type="x",
)


K_NH3 = ParamScalarFunction(
    {"x": 2},
    K_nh3_eval,
    f_type="x",
)


T_INC = ParamScalarFunction(
    {"x": 2},
    T_inc_eval,
    f_type="x",
)


T_NH3 = ParamScalarFunction(
    {"x": 2},
    T_nh3_eval,
    f_type="x",
)


def k_inc(
    T,
):
    return ParamScalarFunction(
        T.dims,
        lambda space, x: k_inc_value(
            T(
                space,
                x,
            )
        ),
        f_type=T.f_type,
    )


def k_nh3(
    T,
):
    return ParamScalarFunction(
        T.dims,
        lambda space, x: k_nh3_value(
            T(
                space,
                x,
            )
        ),
        f_type=T.f_type,
    )


# r-weighted axisymmetric steady-conduction operator
def axisymmetric_potential_laplacian(K, space, x, reference):
    """r-weighted axisymmetric Laplacian of a Kirchhoff potential.

        div(k grad T) = div(grad K) = Laplace(K)

    so the interior equation is  d_r(r d_r K) + r d_zz K = 0,  in which no
    conductivity appears at all. One operator for every material; only the
    ``reference`` it is normalised by is per material (the span of that
    material's potential).
    """
    dK_dr = K.partial_derivative("x", 0)
    dK_dz = K.partial_derivative("x", 1)

    r = x[..., 0:1]

    raw_residual = (
        r * dK_dr.partial_derivative("x", 0)(space, x)
        + dK_dr(space, x)
        + r * dK_dz.partial_derivative("x", 1)(space, x)
    )

    return raw_residual / reference


# Interior residuals
class InconelResidual(InteriorResidual):
    def __init__(
        self,
        domain,
        lower_box=False,
    ):
        super().__init__(
            domain=domain,
            size=1,
            model_type="x",
        )

        self.lower_box = lower_box

    def construct_residual(self, *vars):
        # NOT unpacked from *vars. The space now holds four raw models
        # (two fields plus the two interface traces) with identity
        # post-processing, so vars[0] is the bare network output. The
        # physical temperature is assembled in T_inc_eval, which is what
        # T_INC wraps.
        K_field = K_INC

        def residual(space, x):
            value = axisymmetric_potential_laplacian(
                K_field,
                space,
                x,
                PDE_POTENTIAL_REF,
            )

            if not self.lower_box:
                return value

            r = x[..., 0:1]
            z = x[..., 1:2]

            return jnp.where(
                in_lower_ammonia(
                    r,
                    z,
                ),
                0.0,
                value,
            )

        return ParamScalarFunction(
            K_field.dims,
            residual,
            f_type=K_field.f_type,
        )


class AmmoniaResidual(InteriorResidual):
    def __init__(
        self,
        domain,
        lower_box=False,
    ):
        super().__init__(
            domain=domain,
            size=1,
            model_type="x",
        )

        self.lower_box = lower_box

    def construct_residual(self, *vars):
        # See InconelResidual: the assembled field, not vars[1].
        K_field = K_NH3

        def residual(space, x):
            value = axisymmetric_potential_laplacian(
                K_field,
                space,
                x,
                NH3_PDE_POTENTIAL_REF,
            )

            r = x[..., 0:1]
            z = x[..., 1:2]

            # The same full-domain mask is appropriate for the
            # lower and upper NH3 sampling envelopes.
            return jnp.where(
                in_nh3(
                    r,
                    z,
                ),
                value,
                0.0,
            )

        return ParamScalarFunction(
            K_field.dims,
            residual,
            f_type=K_field.f_type,
        )


# Slab energy balance (integral / weak-form constraint)
#
# Q(z) = integral of -k dT/dz 2 pi r dr is the axial heat flow through the
# cross section at height z. While the sidewall was adiabatic, Q had to be
# independent of z. With T prescribed on r = R_OUT it is NOT: k varies by
# ~10 % between the two ends, so Q does too, and the difference crosses the
# sidewall (the DG reference measures Q from 18.0 W at the bottom to 15.2 W
# at the top of the cavity). Imposing Q(z) = Q(z_ref) would now be wrong.
#
# What holds instead is the energy balance of the slab between z_ref and z:
#
#     Q(z) - Q(z_ref) = integral_{z_ref}^{z} 2 pi R_OUT k dT/dr|_{R_OUT} dz'
#
# (heat entering through the wall, outward normal +r). This is the PDE
# integrated against the slab indicator, i.e. still the crudest weak form,
# now with the sidewall term that the adiabatic case let drop. In Kirchhoff
# variables k grad T = grad K, so both sides are plain derivatives of the
# assembled potentials.
#
# Toggle. Measured at 200 epochs against FV: switching it on took the error
# from 0.53 to 0.35 K RMS (NH3 0.96 -> 0.59 K) for +0.04 K in the Inconel.
USE_FLUX_BALANCE = True

_N_QUAD_PANEL = 16

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_N_QUAD_PANEL)


def _panel(a, b):
    """Map the Gauss-Legendre rule from [-1, 1] onto [a, b]."""
    return (
        0.5 * (b - a) * _GL_NODES + 0.5 * (a + b),
        0.5 * (b - a) * _GL_WEIGHTS,
    )


# Two panels, split at the cavity wall: the integrand has a kink there
# whenever the cavity is present, and a single rule across it would not
# integrate accurately.
_Q_R_INNER, _Q_W_INNER = _panel(0.0, R_NH3)
_Q_R_OUTER, _Q_W_OUTER = _panel(R_NH3, R_OUT)

_Q_R = jnp.array(np.concatenate([_Q_R_INNER, _Q_R_OUTER]), dtype=jnp.float64)
_Q_W = jnp.array(np.concatenate([_Q_W_INNER, _Q_W_OUTER]), dtype=jnp.float64)

# Reference height every other cross-section is compared against. Using a
# DIFFERENCE avoids needing to know the true Q, which keeps this constraint
# free of any reference data.
Z_FLUX_REFERENCE = 0.5 * H_OUT

# Gauss-Legendre rule on [-1, 1] for the sidewall integral, mapped per sample
# onto [Z_FLUX_REFERENCE, z]. The wall flux is smooth (the cavity is 23 mm
# away), but it changes sign near both cavity ends, hence more nodes than a
# single cross-section panel.
_N_QUAD_SIDE = 24

_GL_SIDE_NODES, _GL_SIDE_WEIGHTS = np.polynomial.legendre.leggauss(_N_QUAD_SIDE)
_GL_SIDE_NODES = jnp.array(_GL_SIDE_NODES, dtype=jnp.float64)
_GL_SIDE_WEIGHTS = jnp.array(_GL_SIDE_WEIGHTS, dtype=jnp.float64)


def _cross_section_flow(dK_inc_dz, dK_nh3_dz, space, z):
    """Q(z) = integral of -dK/dz 2 pi r dr over the cross section."""
    points = jnp.stack(
        [_Q_R, jnp.full_like(_Q_R, z)],
        axis=-1,
    )

    def one_point(x_one):
        gradient = jnp.where(
            in_nh3(x_one[0], x_one[1]),
            dK_nh3_dz(space, x_one),
            dK_inc_dz(space, x_one),
        )

        return -gradient[0]

    integrand = jax.vmap(one_point)(points)

    return jnp.sum(_Q_W * integrand * 2.0 * jnp.pi * _Q_R)


def _sidewall_inflow(dK_inc_dr, space, z_a, z_b):
    """Heat entering through r = R_OUT between z_a and z_b (oriented)."""
    z_nodes = 0.5 * (z_b - z_a) * _GL_SIDE_NODES + 0.5 * (z_a + z_b)
    weights = 0.5 * (z_b - z_a) * _GL_SIDE_WEIGHTS

    points = jnp.stack(
        [jnp.full_like(z_nodes, R_OUT), z_nodes],
        axis=-1,
    )

    wall_flux = jax.vmap(lambda x_one: dK_inc_dr(space, x_one)[0])(points)

    return jnp.sum(weights * wall_flux) * 2.0 * jnp.pi * R_OUT


class FluxBalanceResidual(InteriorResidual):
    """Q(z) - Q(z_ref) - sidewall inflow = 0, normalised by the axial flow scale."""

    def __init__(self, domain):
        super().__init__(
            domain=domain,
            size=1,
            model_type="x",
        )

    def construct_residual(self, *vars):
        dK_inc_dr = K_INC.partial_derivative("x", 0)
        dK_inc_dz = K_INC.partial_derivative("x", 1)
        dK_nh3_dz = K_NH3.partial_derivative("x", 1)

        def residual(space, x):
            z = x[..., 1]

            here = _cross_section_flow(dK_inc_dz, dK_nh3_dz, space, z)
            there = _cross_section_flow(
                dK_inc_dz, dK_nh3_dz, space, Z_FLUX_REFERENCE
            )
            side = _sidewall_inflow(dK_inc_dr, space, Z_FLUX_REFERENCE, z)

            return jnp.reshape(
                (here - there - side) / FLUX_BALANCE_REF,
                (1,),
            )

        return ParamScalarFunction(
            {"x": 2},
            residual,
            f_type="x",
        )


# Boundary residual domain dictionaries
OUTER_DOMAINS = {
    domain_outer_side.get_label(): domain_outer_side,
    domain_outer_bottom.get_label(): domain_outer_bottom,
    domain_outer_top.get_label(): domain_outer_top,
}


AXIS_INC_DOMAINS = {
    domain_axis_inc_bottom.get_label(): domain_axis_inc_bottom,
    domain_axis_inc_top.get_label(): domain_axis_inc_top,
}


AXIS_NH3_DOMAINS = {
    domain_axis_nh3.get_label(): domain_axis_nh3,
}


INTERFACE_T_DOMAINS = {
    domain_if_side_T.get_label(): domain_if_side_T,
    domain_if_top_T.get_label(): domain_if_top_T,
}


INTERFACE_Q_DOMAINS = {
    domain_if_side_q.get_label(): domain_if_side_q,
    domain_if_top_q.get_label(): domain_if_top_q,
}


for domain in lower_temperature_domains:
    INTERFACE_T_DOMAINS[domain.get_label()] = domain


for domain in lower_flux_domains:
    INTERFACE_Q_DOMAINS[domain.get_label()] = domain


# Boundary residuals


class InterfaceTemperatureResidual(BoundaryResidual):
    """
    Enforce perfect thermal-contact temperature continuity on all
    physical NH3/Inconel interfaces:

        T_INC - T_NH3 = 0.
    """

    def __init__(self):
        super().__init__(
            domain=INTERFACE_T_DOMAINS,
            size=1,
            model_type="x",
        )

    def construct_residual(
        self,
        *vars,
    ):
        def residual(
            space,
            x,
            n,
        ):
            return (
                T_INC(
                    space,
                    x,
                )
                - T_NH3(
                    space,
                    x,
                )
            ) / INTERFACE_T_REF

        return ParamScalarFunction(
            {"x": 2},
            residual,
            f_type="x_n",
        )


class InterfaceFluxResidual(
    BoundaryResidual,
):
    def __init__(
        self,
    ):
        super().__init__(
            domain=INTERFACE_Q_DOMAINS,
            size=1,
            model_type="x",
        )

    def construct_residual(
        self,
        *vars,
    ):
        dT_inc_dr = T_INC.partial_derivative(
            "x",
            0,
        )

        dT_inc_dz = T_INC.partial_derivative(
            "x",
            1,
        )

        dT_nh3_dr = T_NH3.partial_derivative(
            "x",
            0,
        )

        dT_nh3_dz = T_NH3.partial_derivative(
            "x",
            1,
        )

        k_inc_field = k_inc(
            T_INC,
        )

        k_nh3_field = k_nh3(
            T_NH3,
        )

        def residual(
            space,
            x,
            n,
        ):
            """
            Evaluate:

                [k_inc(T_inc) * dT_inc/dn
                 - k_nh3(T_nh3) * dT_nh3/dn]
                / INTERFACE_FLUX_REF.
            """

            n_r = n[..., 0:1]

            n_z = n[..., 1:2]

            dT_inc_dn = n_r * dT_inc_dr(
                space,
                x,
            ) + n_z * dT_inc_dz(
                space,
                x,
            )

            dT_nh3_dn = n_r * dT_nh3_dr(
                space,
                x,
            ) + n_z * dT_nh3_dz(
                space,
                x,
            )

            k_inc_local = k_inc_field(
                space,
                x,
            )

            k_nh3_local = k_nh3_field(
                space,
                x,
            )

            flux_jump = k_inc_local * dT_inc_dn - k_nh3_local * dT_nh3_dn

            return flux_jump / INTERFACE_FLUX_REF

        return ParamScalarFunction(
            {"x": 2},
            residual,
            f_type="x_n",
        )


# Physical model
class AutoclaveConductionModel(AbstractPhysicalModel):
    def __init__(self):
        super().__init__(
            main_domain=main_domain,
        )

        self.physical_residuals = {
            # Inconel interior PDE residuals
            "inc_bottom": InconelResidual(
                inc_bottom,
            ),
            "inc_side": InconelResidual(
                inc_side,
            ),
            "inc_top": InconelResidual(
                inc_top,
            ),
            "inc_lower": InconelResidual(
                inc_lower,
                lower_box=True,
            ),
            # NH3 interior PDE residuals
            "nh3_upper": AmmoniaResidual(
                nh3_upper,
            ),
            "nh3_lower": AmmoniaResidual(
                nh3_lower,
                lower_box=True,
            ),
            # The NH3/Inconel temperature- and flux-continuity conditions
            # used to be soft residuals here. They are now exact by
            # construction (see the trial functions above), so they carry no
            # loss term at all.
        }

        if USE_FLUX_BALANCE:
            self.physical_residuals["flux_balance"] = FluxBalanceResidual(
                flux_balance,
            )

        self.data_residuals = {}


model = AutoclaveConductionModel()


# Collocation sampler
domain_sampler = DomainSampler(
    main_domain,
    main_domain_fully_covered=True,
)


sampler = TensorizedSampler(
    [domain_sampler],
    model_type="x",
    bc=True,
    ic=False,
)


# Neural-network approximation space
def build_mlp(
    in_size,
    out_size,
    hidden_sizes,
    key,
):
    try:
        return MLP(
            in_size=in_size,
            out_size=out_size,
            hidden_sizes=hidden_sizes,
            key=key,
            activation=jnp.tanh,
        )

    except TypeError:
        # Compatibility fallback for ScimBa/JAX versions where MLP
        # does not expose the activation keyword argument.
        return MLP(
            in_size=in_size,
            out_size=out_size,
            hidden_sizes=hidden_sizes,
            key=key,
        )


def build_space(seed=42):
    key = jax.random.PRNGKey(
        seed,
    )

    key_inc, key_nh3, key_u, key_w = jax.random.split(
        key,
        4,
    )

    nn_inc = build_mlp(
        in_size=2,
        out_size=1,
        hidden_sizes=HIDDEN,
        key=key_inc,
    )

    nn_nh3 = build_mlp(
        in_size=2,
        out_size=1,
        hidden_sizes=HIDDEN,
        key=key_nh3,
    )

    # The two interface traces live on a curve, so they are 1-D objects and
    # need far less capacity than the two fields.
    nn_u = build_mlp(
        in_size=2,
        out_size=1,
        hidden_sizes=HIDDEN_TRACE,
        key=key_u,
    )

    nn_w = build_mlp(
        in_size=2,
        out_size=1,
        hidden_sizes=HIDDEN_TRACE,
        key=key_w,
    )

    return ApproximationSpace(
        {"x": 2},
        [
            (nn_inc, "scalar", None),
            (nn_nh3, "scalar", None),
            (nn_u, "scalar", None),
            (nn_w, "scalar", None),
        ],
        model_type="x",
        pre_processing=[
            scaled_coordinates,
            nh3_scaled_coordinates,
            interface_trace_features,
            interface_trace_features,
        ],
        # All four raw outputs are combined by hand in
        # _temperature_at_point, which is the only place that can see more
        # than one model at a time.
        post_processing=[
            identity_post_processing,
            identity_post_processing,
            identity_post_processing,
            identity_post_processing,
        ],
    )


# Loss weights

W_COMMON = {
    "inc_bottom": (150.0,),
    "inc_side": (350.0,),
    "inc_top": (200.0,),
    "inc_lower": (350.0,),
    "nh3_lower": (250.0,),
    "nh3_upper": (450.0,),
}

if USE_FLUX_BALANCE:
    W_COMMON["flux_balance"] = (100.0,)


# Plotting and post-training diagnostics


def read_fvm_table(path, sheet_name="Sheet1"):
    """FV nodes as a DataFrame, from an Excel workbook or a CSV file.

    2D_Heat_data.csv stacks several (T0, T1) boundary cases; only the one this
    PINN solves is kept, so every consumer below sees a single case, as it
    did with the one-case Excel workbook.
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    if {"T0", "T1"}.issubset(df.columns):
        df = df[np.isclose(df["T0"], T0_K) & np.isclose(df["T1"], T1)].copy()

        if len(df) == 0:
            raise ValueError(f"No FV case with T0={T0_K}, T1={T1} in {path}")

    return df


def z_bottom_ammonia_numpy(r):
    r = np.asarray(
        r,
        dtype=np.float64,
    )

    r_clipped = np.clip(
        r,
        float(NH3_LOWER_R[0]),
        float(NH3_LOWER_R[-1]),
    )

    return np.interp(
        r_clipped,
        np.asarray(
            NH3_LOWER_R,
            dtype=np.float64,
        ),
        np.asarray(
            NH3_LOWER_Z,
            dtype=np.float64,
        ),
    )


def in_ammonia_numpy(r, z):
    r = np.asarray(
        r,
        dtype=np.float64,
    )

    z = np.asarray(
        z,
        dtype=np.float64,
    )

    z_bottom = z_bottom_ammonia_numpy(
        r,
    )

    return (r >= 0.0) & (r <= R_NH3) & (z >= z_bottom) & (z <= Z_NH3_TOP)


def material_masks_numpy(r, z):
    nh3_mask = in_ammonia_numpy(
        r,
        z,
    )

    inc_mask = ~nh3_mask

    return (
        inc_mask,
        nh3_mask,
    )


# Final neural-field evaluation helper
def evaluate_field(space, points, material):
    x = jnp.asarray(
        points,
        dtype=jnp.float64,
    )

    if material == "inc":
        field = T_INC

    elif material == "nh3":
        field = T_NH3

    else:
        raise ValueError(f"Unknown material '{material}'. Use 'inc' or 'nh3'.")

    return np.asarray(
        field(
            space,
            x,
        )
    ).reshape(
        -1,
    )


# ── Improved plot_solution: higher resolution + isotherm lines + optional
#    FVM-error overlay ─────────────────────────────────────────────────────
# Replaces your existing plot_solution. Same signature and outputs
# temperature_field.png as before, PLUS a second file
# temperature_field_isotherms.png with labeled contour lines (much better
# for spotting kinks at the material interface), and optionally a third
# file temperature_field_vs_fvm_error.png if you pass fvm_excel_path.


def plot_solution(
    space,
    n_r=400,
    n_z=1200,
    fvm_excel_path=None,
    fvm_sheet_name="Sheet1",
):
    r = np.linspace(0.0, R_OUT, n_r)
    z = np.linspace(0.0, H_OUT, n_z)

    rr, zz = np.meshgrid(r, z, indexing="xy")

    points = np.column_stack([rr.ravel(), zz.ravel()])

    inc_mask, nh3_mask = material_masks_numpy(points[:, 0], points[:, 1])

    temperature = np.empty(points.shape[0], dtype=np.float64)

    if np.any(inc_mask):
        temperature[inc_mask] = evaluate_field(space, points[inc_mask], "inc")

    if np.any(nh3_mask):
        temperature[nh3_mask] = evaluate_field(space, points[nh3_mask], "nh3")

    temperature = temperature.reshape(rr.shape)

    def draw_geometry(ax):
        ax.plot(
            np.full(2, R_NH3 * 1.0e3),
            np.array([Z_NH3_MAX_LOWER, Z_NH3_TOP], dtype=np.float64) * 1.0e3,
            "k",
            lw=1.2,
        )
        ax.plot(
            np.array([0.0, R_NH3], dtype=np.float64) * 1.0e3,
            np.full(2, Z_NH3_TOP * 1.0e3),
            "k",
            lw=1.2,
        )
        ax.plot(
            np.asarray(NH3_LOWER_R, dtype=np.float64) * 1.0e3,
            np.asarray(NH3_LOWER_Z, dtype=np.float64) * 1.0e3,
            "k",
            lw=1.4,
        )
        ax.plot(
            np.full(2, R_OUT * 1.0e3),
            np.array([0.0, H_OUT], dtype=np.float64) * 1.0e3,
            "k",
            lw=1.2,
        )
        ax.plot(
            np.array([0.0, R_OUT], dtype=np.float64) * 1.0e3,
            np.zeros(2, dtype=np.float64),
            "k",
            lw=1.2,
        )
        ax.plot(
            np.array([0.0, R_OUT], dtype=np.float64) * 1.0e3,
            np.full(2, H_OUT * 1.0e3),
            "k",
            lw=1.2,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("z [mm]")

    # ── Plot 1: original filled contour (kept, higher resolution now) ──
    figure, ax = plt.subplots(figsize=(5.8, 10.0))

    contour = ax.contourf(
        rr * 1.0e3,
        zz * 1.0e3,
        temperature,
        levels=100,
        cmap="turbo",
    )
    colorbar = figure.colorbar(contour, ax=ax)
    colorbar.set_label("Temperature [K]")

    draw_geometry(ax)
    ax.set_title(
        f"Axisymmetric Inconel--NH3 conduction\nT0 = {T0_K:.1f} K, T1 = {T1:.1f} K/m"
    )

    figure.tight_layout()
    figure_path = SAVE_DIR / "temperature_field.png"
    figure.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print("Saved PINN temperature plot in:")
    print(figure_path.resolve())

    # ── Plot 2: filled contour + labeled isotherm lines ──
    # Isotherms make interface kinks/discontinuities visible directly,
    # since a real flux-continuity problem shows as a bent contour line
    # right at the black interface boundary.
    figure2, ax2 = plt.subplots(figsize=(5.8, 10.0))

    contour2 = ax2.contourf(
        rr * 1.0e3,
        zz * 1.0e3,
        temperature,
        levels=100,
        cmap="turbo",
    )
    colorbar2 = figure2.colorbar(contour2, ax=ax2)
    colorbar2.set_label("Temperature [K]")

    line_levels = np.linspace(temperature.min(), temperature.max(), 20)
    lines = ax2.contour(
        rr * 1.0e3,
        zz * 1.0e3,
        temperature,
        levels=line_levels,
        colors="white",
        linewidths=0.6,
    )
    ax2.clabel(lines, inline=True, fontsize=6, fmt="%.0f")

    draw_geometry(ax2)
    ax2.set_title(
        "Axisymmetric Inconel--NH3 conduction\n"
        "Isotherms every "
        f"{(line_levels[1] - line_levels[0]):.2f} K"
    )

    figure2.tight_layout()
    figure2_path = SAVE_DIR / "temperature_field_isotherms.png"
    figure2.savefig(figure2_path, dpi=250, bbox_inches="tight")
    plt.close(figure2)

    print("Saved isotherm plot in:")
    print(figure2_path.resolve())

    # ── Plot 3 (optional): dense PINN-vs-FVM error field, if FVM data given ──
    # This is the plot that actually answers "how accurate is the PINN",
    # since it plots ERROR directly rather than absolute temperature (whose
    # color scale hides sub-Kelvin discrepancies).
    if fvm_excel_path is not None:
        df_fvm = read_fvm_table(fvm_excel_path, sheet_name=fvm_sheet_name)
        r_fvm = df_fvm["x"].to_numpy(dtype=np.float64)
        z_fvm = df_fvm["y"].to_numpy(dtype=np.float64)
        T_fvm = df_fvm["Temperature"].to_numpy(dtype=np.float64)

        fvm_points = np.column_stack([r_fvm, z_fvm])
        inc_mask_fvm, nh3_mask_fvm = material_masks_numpy(r_fvm, z_fvm)

        T_pinn_fvm = np.empty(len(T_fvm), dtype=np.float64)
        if np.any(inc_mask_fvm):
            T_pinn_fvm[inc_mask_fvm] = evaluate_field(
                space,
                fvm_points[inc_mask_fvm],
                "inc",
            )
        if np.any(nh3_mask_fvm):
            T_pinn_fvm[nh3_mask_fvm] = evaluate_field(
                space,
                fvm_points[nh3_mask_fvm],
                "nh3",
            )

        error_fvm = T_pinn_fvm - T_fvm
        limit = np.max(np.abs(error_fvm))

        figure3, ax3 = plt.subplots(figsize=(5.8, 10.0))

        scatter = ax3.scatter(
            r_fvm * 1.0e3,
            z_fvm * 1.0e3,
            c=error_fvm,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            s=14,
            marker="s",
            linewidths=0.0,
        )
        colorbar3 = figure3.colorbar(scatter, ax=ax3)
        colorbar3.set_label("PINN - FVM [K]")

        draw_geometry(ax3)
        ax3.set_title(
            "PINN error vs. FVM reference\n"
            f"RMSE = {np.sqrt(np.mean(error_fvm**2)):.3f} K, "
            f"MAE = {np.mean(np.abs(error_fvm)):.3f} K"
        )

        figure3.tight_layout()
        figure3_path = SAVE_DIR / "temperature_field_vs_fvm_error.png"
        figure3.savefig(figure3_path, dpi=250, bbox_inches="tight")
        plt.close(figure3)

        print("Saved PINN-vs-FVM error field in:")
        print(figure3_path.resolve())


def make_axisymmetric_residual(
    temperature_function,
    conductivity_function,
    pde_ref,
):
    def residual_on_points(
        space,
        x,
    ):
        x = jnp.asarray(
            x,
            dtype=jnp.float64,
        )

        def temperature_at_one_point(
            x_one,
        ):
            return jnp.ravel(
                temperature_function(
                    space,
                    x_one,
                )
            )[0]

        def radial_flux_at_one_point(
            x_one,
        ):
            temperature = temperature_at_one_point(
                x_one,
            )

            dT_dr = jax.grad(
                temperature_at_one_point,
            )(x_one)[0]

            conductivity = conductivity_function(
                temperature,
            )

            return conductivity * dT_dr

        def axial_flux_at_one_point(
            x_one,
        ):
            temperature = temperature_at_one_point(
                x_one,
            )

            dT_dz = jax.grad(
                temperature_at_one_point,
            )(x_one)[1]

            conductivity = conductivity_function(
                temperature,
            )

            return conductivity * dT_dz

        radial_flux_gradient = jax.grad(
            radial_flux_at_one_point,
        )

        axial_flux_gradient = jax.grad(
            axial_flux_at_one_point,
        )

        def residual_at_one_point(
            x_one,
        ):
            r = x_one[0]

            q_r = radial_flux_at_one_point(
                x_one,
            )

            dq_r_dr = radial_flux_gradient(x_one)[0]

            dq_z_dz = axial_flux_gradient(x_one)[1]

            raw_residual = r * dq_r_dr + q_r + r * dq_z_dz

            return raw_residual / pde_ref

        return jax.vmap(
            residual_at_one_point,
        )(x)

    return residual_on_points


# The residual MAPS are still drawn from the temperature form, which stays
# valid and is the more readable diagnostic; only the trained loss moved to
# the potential. Each normalises by its material's potential span, as the loss.
inc_conduction_residual = make_axisymmetric_residual(
    T_inc_eval,
    k_inc_value,
    PDE_POTENTIAL_REF,
)

nh3_conduction_residual = make_axisymmetric_residual(
    T_nh3_eval,
    k_nh3_value,
    NH3_PDE_POTENTIAL_REF,
)


def apply_interior_material_mask(
    residual_name,
    r_grid,
    z_grid,
    values,
):
    """
    Return PDE residual values inside the correct physical material
    region and NaN outside that region.
    """
    r_np = np.asarray(
        r_grid,
        dtype=np.float64,
    )

    z_np = np.asarray(
        z_grid,
        dtype=np.float64,
    )

    values_np = np.asarray(
        values,
        dtype=np.float64,
    )

    if residual_name == "inc_bottom":
        mask = (r_np >= 0.0) & (r_np <= R_OUT) & (z_np >= 0.0) & (z_np <= Z_NH3_MIN)

    elif residual_name == "inc_lower":
        z_polyline = z_bottom_ammonia_numpy(
            r_np,
        )

        mask = (
            (r_np >= 0.0)
            & (r_np <= R_OUT)
            & (z_np >= Z_NH3_MIN)
            & (z_np <= Z_NH3_MAX_LOWER)
            & ((r_np > R_NH3) | (z_np < z_polyline))
        )

    elif residual_name == "inc_side":
        mask = (
            (r_np >= R_NH3)
            & (r_np <= R_OUT)
            & (z_np >= Z_NH3_MIN)
            & (z_np <= Z_NH3_TOP)
        )

    elif residual_name == "inc_top":
        mask = (r_np >= 0.0) & (r_np <= R_OUT) & (z_np >= Z_NH3_TOP) & (z_np <= H_OUT)

    elif residual_name == "nh3_lower":
        z_polyline = z_bottom_ammonia_numpy(
            r_np,
        )

        mask = (
            (r_np >= 0.0)
            & (r_np <= R_NH3)
            & (z_np >= z_polyline)
            & (z_np <= Z_NH3_MAX_LOWER)
        )

    elif residual_name == "nh3_upper":
        mask = (
            (r_np >= 0.0)
            & (r_np <= R_NH3)
            & (z_np >= Z_NH3_MAX_LOWER)
            & (z_np <= Z_NH3_TOP)
        )

    else:
        raise ValueError(f"Unknown interior residual name: {residual_name}")

    return np.where(
        mask,
        values_np,
        np.nan,
    )


def plot_all_interior_residuals(
    space,
    n_r=151,
    n_z=251,
):
    residual_regions = (
        (
            "inc_bottom",
            "Inconel: bottom",
            inc_conduction_residual,
            0.0,
            R_OUT,
            0.0,
            Z_NH3_MIN,
        ),
        (
            "inc_lower",
            "Inconel: lower polyline region",
            inc_conduction_residual,
            0.0,
            R_OUT,
            Z_NH3_MIN,
            Z_NH3_MAX_LOWER,
        ),
        (
            "inc_side",
            "Inconel: side region",
            inc_conduction_residual,
            R_NH3,
            R_OUT,
            Z_NH3_MIN,
            Z_NH3_TOP,
        ),
        (
            "inc_top",
            "Inconel: top",
            inc_conduction_residual,
            0.0,
            R_OUT,
            Z_NH3_TOP,
            H_OUT,
        ),
        (
            "nh3_lower",
            "NH3: lower polyline region",
            nh3_conduction_residual,
            0.0,
            R_NH3,
            Z_NH3_MIN,
            Z_NH3_MAX_LOWER,
        ),
        (
            "nh3_upper",
            "NH3: upper cavity region",
            nh3_conduction_residual,
            0.0,
            R_NH3,
            Z_NH3_MAX_LOWER,
            Z_NH3_TOP,
        ),
    )

    plot_data = []

    for (
        residual_name,
        title,
        residual_function,
        r_min,
        r_max,
        z_min,
        z_max,
    ) in residual_regions:
        r_plot_min = max(
            r_min,
            10.0 * EPS_AXIS,
        )

        r_values = jnp.linspace(
            r_plot_min,
            r_max,
            n_r,
        )

        z_values = jnp.linspace(
            z_min,
            z_max,
            n_z,
        )

        r_grid, z_grid = jnp.meshgrid(
            r_values,
            z_values,
            indexing="xy",
        )

        x_eval = jnp.column_stack(
            [
                r_grid.reshape(
                    -1,
                ),
                z_grid.reshape(
                    -1,
                ),
            ]
        )

        residual_values = residual_function(
            space,
            x_eval,
        )

        residual_values = jnp.asarray(
            residual_values,
        ).reshape(
            n_z,
            n_r,
        )

        masked_values = apply_interior_material_mask(
            residual_name,
            r_grid,
            z_grid,
            residual_values,
        )

        absolute_values = np.abs(
            masked_values,
        )

        positive_values = absolute_values[
            np.isfinite(
                absolute_values,
            )
            & (absolute_values > 0.0)
        ]

        if positive_values.size == 0:
            raise ValueError(f"No positive residual values remain for {residual_name}.")

        print(
            f"{residual_name}: "
            f"min |R| = {np.min(positive_values):.3e}, "
            f"max |R| = {np.max(positive_values):.3e}, "
            f"mean |R| = {np.mean(positive_values):.3e}"
        )

        plot_data.append(
            (
                residual_name,
                title,
                absolute_values,
                float(r_plot_min),
                float(r_max),
                float(z_min),
                float(z_max),
            )
        )

    all_positive_values = np.concatenate(
        [
            values[np.isfinite(values) & (values > 0.0)]
            for (
                _,
                _,
                values,
                _,
                _,
                _,
                _,
            ) in plot_data
        ]
    )

    common_vmin = np.min(
        all_positive_values,
    )

    common_vmax = np.max(
        all_positive_values,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 10.0),
        constrained_layout=True,
    )

    image = None

    for ax, (
        residual_name,
        title,
        absolute_values,
        r_min,
        r_max,
        z_min,
        z_max,
    ) in zip(
        axes.ravel(),
        plot_data,
    ):
        plot_values = np.where(
            absolute_values > 0.0,
            absolute_values,
            np.nan,
        )

        image = ax.pcolormesh(
            np.linspace(
                r_min * 1.0e3,
                r_max * 1.0e3,
                n_r,
            ),
            np.linspace(
                z_min * 1.0e3,
                z_max * 1.0e3,
                n_z,
            ),
            plot_values,
            shading="auto",
            cmap="magma",
            norm=LogNorm(
                vmin=common_vmin,
                vmax=common_vmax,
            ),
        )

        ax.set_title(
            f"{title}\n"
            r"$|\mathcal{R}_{\mathrm{PDE}}|$"
        )

        ax.set_xlabel(
            "r [mm]",
        )

        ax.set_ylabel(
            "z [mm]",
        )

    colorbar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.88,
    )

    colorbar.set_label(
        r"Absolute scaled PDE residual "
        r"$|\mathcal{R}_{\mathrm{PDE}}|$"
    )

    output_path = SAVE_DIR / "all_interior_residual_maps.png"

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure,
    )

    print("Saved interior residual maps in:")

    print(
        output_path.resolve(),
    )


# One-dimensional FVM versus PINN line comparison
# One-dimensional FVM/PINN line comparison
def plot_fvm_vs_pinn(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_inconel_nh3_line_comparison.png",
):
    """
    Compare FVM temperatures with the Inconel + NH3 PINN prediction.

    Required Excel columns:
    - x            radial coordinate r [m]
    - y            axial coordinate z [m]
    - Temperature  FVM temperature [K]

    The Excel input must contain one vertical line:
    all x values must have the same radial coordinate.
    """

    excel_path = Path(excel_path).resolve()

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (SAVE_DIR / output_name).resolve()

    comparison_csv = (
        SAVE_DIR / "fvm_vs_pinn_inconel_nh3_line_comparison.csv"
    ).resolve()

    if not excel_path.exists():
        raise FileNotFoundError(f"1D FVM line Excel file does not exist:\n{excel_path}")

    df = read_fvm_table(
        excel_path,
        sheet_name=sheet_name,
    )

    required_columns = {
        "x",
        "y",
        "Temperature",
    }

    missing_columns = required_columns.difference(
        df.columns,
    )

    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[
            "x",
            "y",
            "Temperature",
        ],
    ).copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after removing NaN values.")

    r_fvm = df["x"].to_numpy(
        dtype=np.float64,
    )

    z_fvm = df["y"].to_numpy(
        dtype=np.float64,
    )

    T_fvm = df["Temperature"].to_numpy(
        dtype=np.float64,
    )

    # This must be a line at one radius, not the full 2D data.
    if not np.allclose(
        r_fvm,
        r_fvm[0],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        unique_r = np.unique(
            r_fvm,
        )

        raise ValueError(
            "plot_fvm_vs_pinn() expects one vertical FVM line "
            "with a constant x/r coordinate.\n"
            f"Found {len(unique_r)} radial values.\n"
            "Extract one line from the 2D workbook first."
        )

    fvm_points = np.column_stack(
        [
            r_fvm,
            z_fvm,
        ]
    )

    # TWO MATERIALS ONLY.
    inc_mask, nh3_mask = material_masks_numpy(
        r_fvm,
        z_fvm,
    )

    assigned_mask = inc_mask | nh3_mask

    if not np.all(assigned_mask):
        raise ValueError(
            f"{np.count_nonzero(~assigned_mask)} FVM point(s) "
            "are outside both Inconel and NH3 masks."
        )

    T_pinn = np.empty(
        len(T_fvm),
        dtype=np.float64,
    )

    if np.any(inc_mask):
        T_pinn[inc_mask] = evaluate_field(
            space,
            fvm_points[inc_mask],
            "inc",
        )

    if np.any(nh3_mask):
        T_pinn[nh3_mask] = evaluate_field(
            space,
            fvm_points[nh3_mask],
            "nh3",
        )

    error_K = T_pinn - T_fvm

    abs_error_K = np.abs(
        error_K,
    )

    rmse_K = np.sqrt(
        np.mean(
            error_K**2,
        )
    )

    mae_K = np.mean(
        abs_error_K,
    )

    max_abs_error_K = np.max(
        abs_error_K,
    )

    material = np.where(
        nh3_mask,
        "NH3",
        "Inconel",
    )

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "pinn_minus_fvm_K": error_K,
            "absolute_error_K": abs_error_K,
            "material": material,
        }
    )

    comparison_df.to_csv(
        str(comparison_csv),
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM VS PINN LINE COMPARISON: INCONEL + NH3")
    print("=" * 72)

    print(f"Excel file:\n{excel_path}")

    print(f"Number of FVM points: {len(T_fvm)}")

    print(f"Probe radius [m]: {r_fvm[0]:.8f}")

    print(f"Axial-coordinate range [m]: {z_fvm.min():.8f} to {z_fvm.max():.8f}")

    print(f"NH3 points: {np.count_nonzero(nh3_mask)}")

    print(f"Inconel points: {np.count_nonzero(inc_mask)}")

    print(f"RMSE [K]: {rmse_K:.8e}")

    print(f"MAE [K]: {mae_K:.8e}")

    print(f"Maximum absolute error [K]: {max_abs_error_K:.8e}")

    order = np.argsort(
        z_fvm,
    )

    z_sorted = z_fvm[order]
    T_fvm_sorted = T_fvm[order]
    T_pinn_sorted = T_pinn[order]
    error_sorted = error_K[order]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                3,
                1,
            ],
        },
    )

    axes[0].plot(
        z_sorted * 1.0e3,
        T_fvm_sorted,
        "o",
        ms=4.0,
        color="black",
        label="FVM",
    )

    axes[0].plot(
        z_sorted * 1.0e3,
        T_pinn_sorted,
        "-",
        lw=2.0,
        color="tab:red",
        label="PINN",
    )

    axes[0].set_ylabel(
        "Temperature [K]",
    )

    axes[0].set_title(f"FVM vs PINN temperature: r = {r_fvm[0] * 1.0e3:.3f} mm")

    axes[0].grid(
        True,
        alpha=0.30,
    )

    axes[0].legend()

    axes[0].text(
        0.02,
        0.03,
        (
            f"RMSE = {rmse_K:.3e} K\n"
            f"MAE = {mae_K:.3e} K\n"
            f"Max |error| = "
            f"{max_abs_error_K:.3e} K"
        ),
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.90,
        },
    )

    # Inconel/NH3 interface lines for this vertical probe.
    r_probe = float(r_fvm[0])

    if 0.0 <= r_probe <= R_NH3:
        z_nh3_bottom = float(
            z_bottom_ammonia_numpy(
                np.array(
                    [r_probe],
                    dtype=np.float64,
                )
            )[0]
        )

        for z_interface in (
            z_nh3_bottom,
            Z_NH3_TOP,
        ):
            axes[0].axvline(
                z_interface * 1.0e3,
                color="0.35",
                lw=1.0,
                ls="--",
                alpha=0.80,
            )

            axes[1].axvline(
                z_interface * 1.0e3,
                color="0.35",
                lw=1.0,
                ls="--",
                alpha=0.80,
            )

    axes[1].axhline(
        0.0,
        color="black",
        lw=1.0,
    )

    axes[1].plot(
        z_sorted * 1.0e3,
        error_sorted,
        "-o",
        ms=3.0,
        lw=1.2,
        color="tab:blue",
    )

    axes[1].set_xlabel(
        "z [mm]",
    )

    axes[1].set_ylabel(
        "PINN - FVM [K]",
    )

    axes[1].grid(
        True,
        alpha=0.30,
    )

    fig.tight_layout()

    fig.savefig(
        str(figure_path),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig,
    )

    print(f"Saved FVM/PINN line plot in:\n{figure_path}")

    print(f"Saved FVM/PINN line data in:\n{comparison_csv}")

    return comparison_df


def plot_fvm_vs_pinn_relative_error(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_inconel_nh3_line_relative_error.png",
):
    """
    Plot the relative error (PINN - FVM) / FVM [%] along a single
    vertical FVM line.

    Required Excel columns:
    - x            radial coordinate r [m]
    - y            axial coordinate z [m]
    - Temperature  FVM temperature [K]

    The Excel input must contain one vertical line:
    all x values must have the same radial coordinate.
    """

    excel_path = Path(excel_path).resolve()

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (SAVE_DIR / output_name).resolve()

    comparison_csv = (
        SAVE_DIR / "fvm_vs_pinn_inconel_nh3_line_relative_error.csv"
    ).resolve()

    if not excel_path.exists():
        raise FileNotFoundError(f"1D FVM line Excel file does not exist:\n{excel_path}")

    df = read_fvm_table(
        excel_path,
        sheet_name=sheet_name,
    )

    required_columns = {
        "x",
        "y",
        "Temperature",
    }

    missing_columns = required_columns.difference(
        df.columns,
    )

    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[
            "x",
            "y",
            "Temperature",
        ],
    ).copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after removing NaN values.")

    r_fvm = df["x"].to_numpy(
        dtype=np.float64,
    )

    z_fvm = df["y"].to_numpy(
        dtype=np.float64,
    )

    T_fvm = df["Temperature"].to_numpy(
        dtype=np.float64,
    )

    # This must be a line at one radius, not the full 2D data.
    if not np.allclose(
        r_fvm,
        r_fvm[0],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        unique_r = np.unique(
            r_fvm,
        )

        raise ValueError(
            "plot_fvm_vs_pinn_relative_error() expects one vertical "
            "FVM line with a constant x/r coordinate.\n"
            f"Found {len(unique_r)} radial values.\n"
            "Extract one line from the 2D workbook first."
        )

    fvm_points = np.column_stack(
        [
            r_fvm,
            z_fvm,
        ]
    )

    # TWO MATERIALS ONLY.
    inc_mask, nh3_mask = material_masks_numpy(
        r_fvm,
        z_fvm,
    )

    assigned_mask = inc_mask | nh3_mask

    if not np.all(assigned_mask):
        raise ValueError(
            f"{np.count_nonzero(~assigned_mask)} FVM point(s) "
            "are outside both Inconel and NH3 masks."
        )

    T_pinn = np.empty(
        len(T_fvm),
        dtype=np.float64,
    )

    if np.any(inc_mask):
        T_pinn[inc_mask] = evaluate_field(
            space,
            fvm_points[inc_mask],
            "inc",
        )

    if np.any(nh3_mask):
        T_pinn[nh3_mask] = evaluate_field(
            space,
            fvm_points[nh3_mask],
            "nh3",
        )

    relative_error_pct = (T_pinn - T_fvm) / T_fvm * 100.0

    absolute_relative_error_pct = np.abs(
        relative_error_pct,
    )

    rmse_pct = np.sqrt(
        np.mean(
            relative_error_pct**2,
        )
    )

    mae_pct = np.mean(
        absolute_relative_error_pct,
    )

    max_abs_relative_error_pct = np.max(
        absolute_relative_error_pct,
    )

    material = np.where(
        nh3_mask,
        "NH3",
        "Inconel",
    )

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "relative_error_pct": relative_error_pct,
            "absolute_relative_error_pct": absolute_relative_error_pct,
            "material": material,
        }
    )

    comparison_df.to_csv(
        str(comparison_csv),
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM VS PINN LINE RELATIVE ERROR: INCONEL + NH3")
    print("=" * 72)

    print(f"Excel file:\n{excel_path}")

    print(f"Number of FVM points: {len(T_fvm)}")

    print(f"Probe radius [m]: {r_fvm[0]:.8f}")

    print(f"Relative RMSE [%]: {rmse_pct:.8e}")

    print(f"Relative MAE [%]: {mae_pct:.8e}")

    print(f"Maximum absolute relative error [%]: {max_abs_relative_error_pct:.8e}")

    order = np.argsort(
        z_fvm,
    )

    z_sorted = z_fvm[order]
    relative_error_sorted = relative_error_pct[order]

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0),
    )

    ax.axhline(
        0.0,
        color="black",
        lw=1.0,
    )

    ax.plot(
        z_sorted * 1.0e3,
        relative_error_sorted,
        "-o",
        ms=3.0,
        lw=1.2,
        color="tab:blue",
    )

    ax.set_xlabel(
        "z [mm]",
    )

    ax.set_ylabel(
        "(PINN - FVM) / FVM [%]",
    )

    ax.set_title(f"FVM vs PINN relative error: r = {r_fvm[0] * 1.0e3:.3f} mm")

    ax.grid(
        True,
        alpha=0.30,
    )

    # Inconel/NH3 interface lines for this vertical probe.
    r_probe = float(r_fvm[0])

    if 0.0 <= r_probe <= R_NH3:
        z_nh3_bottom = float(
            z_bottom_ammonia_numpy(
                np.array(
                    [r_probe],
                    dtype=np.float64,
                )
            )[0]
        )

        for z_interface in (
            z_nh3_bottom,
            Z_NH3_TOP,
        ):
            ax.axvline(
                z_interface * 1.0e3,
                color="0.35",
                lw=1.0,
                ls="--",
                alpha=0.80,
            )

    ax.text(
        0.02,
        0.03,
        (
            f"Relative RMSE = {rmse_pct:.3e} %\n"
            f"Relative MAE = {mae_pct:.3e} %\n"
            f"Max |relative error| = "
            f"{max_abs_relative_error_pct:.3e} %"
        ),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.90,
        },
    )

    fig.tight_layout()

    fig.savefig(
        str(figure_path),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig,
    )

    print(f"Saved FVM/PINN line relative-error plot in:\n{figure_path}")

    print(f"Saved FVM/PINN line relative-error data in:\n{comparison_csv}")

    return comparison_df


def extract_fvm_vertical_line(
    full_excel_path,
    target_r=0.004,
    sheet_name="Sheet1",
    output_name="inconel_nh3_line_data_extracted.xlsx",
):
    """
    Extract FVM values on the available radius closest to target_r.

    The returned Excel file contains one constant-r vertical probe line
    and is compatible with plot_fvm_vs_pinn().
    """

    full_excel_path = Path(full_excel_path).resolve()

    if not full_excel_path.exists():
        raise FileNotFoundError(f"Full FVM workbook not found:\n{full_excel_path}")

    full_df = read_fvm_table(full_excel_path, sheet_name=sheet_name)

    required_columns = {"x", "y", "Temperature"}

    missing_columns = required_columns.difference(full_df.columns)

    if missing_columns:
        raise ValueError(
            "Full 2D FVM workbook is missing columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {list(full_df.columns)}"
        )

    full_df = full_df.dropna(subset=["x", "y", "Temperature"]).copy()

    radial_values = np.unique(full_df["x"].to_numpy(dtype=np.float64))
    radial_values = np.sort(radial_values)

    if radial_values.size == 0:
        raise ValueError("No radial coordinates were found in the FVM file.")

    selected_r = radial_values[np.argmin(np.abs(radial_values - target_r))]

    line_df = full_df[
        np.isclose(
            full_df["x"].to_numpy(dtype=np.float64),
            selected_r,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    ].copy()

    line_df = line_df.sort_values(by="y")

    if len(line_df) < 2:
        raise ValueError(
            "The extracted FVM line has fewer than two nodes. "
            f"Selected r = {selected_r:.10f} m."
        )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    line_file = (SAVE_DIR / output_name).resolve()

    line_df.to_excel(str(line_file), index=False)

    print("\n" + "=" * 72)
    print("EXTRACTED INCONEL-NH3 1D FVM LINE")
    print("=" * 72)
    print(f"Requested radius: {target_r * 1.0e3:.3f} mm")
    print(f"Selected FVM radius: {selected_r * 1.0e3:.6f} mm")
    print(f"Extracted FVM nodes: {len(line_df)}")
    print(f"Saved extracted FVM line:\n{line_file}")

    return line_file


# FVM versus PINN comparison along every available radial line.
# The corrected reference data (2D_Heat_data.csv) already provides many
# vertical lines (r = 0 to 35 mm in 1 mm steps), so instead of picking one
# probe radius (as extract_fvm_vertical_line/plot_fvm_vs_pinn above do),
# this compares all of them and grids the result into one figure plus a
# per-radius error-summary table. Ported from
# "2D_heat_Inconel_+_Ammonia_single_case.py".
def plot_fvm_vs_pinn_all_radii(
    space,
    df,
    output_name="fvm_vs_pinn_inconel_nh3_all_radii.png",
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    r_values = np.sort(df["x"].unique())
    n_lines = len(r_values)

    n_cols = 6
    n_rows = int(np.ceil(n_lines / n_cols))

    figure, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.6 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    summary_rows = []
    all_T_fvm = []
    all_T_pinn = []

    for i, r_probe in enumerate(r_values):
        line = df[np.isclose(df["x"], r_probe)].sort_values("y")
        z_line = line["y"].to_numpy(dtype=np.float64)
        T_fvm_line = line["Temperature"].to_numpy(dtype=np.float64)

        points = np.column_stack(
            [np.full_like(z_line, r_probe), z_line]
        )
        inc_mask, nh3_mask = material_masks_numpy(points[:, 0], points[:, 1])

        T_pinn_line = np.empty_like(T_fvm_line)
        if np.any(inc_mask):
            T_pinn_line[inc_mask] = evaluate_field(
                space, points[inc_mask], "inc",
            )
        if np.any(nh3_mask):
            T_pinn_line[nh3_mask] = evaluate_field(
                space, points[nh3_mask], "nh3",
            )

        error = T_pinn_line - T_fvm_line
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        max_abs_error = float(np.max(np.abs(error)))

        summary_rows.append(
            {
                "r_mm": float(r_probe) * 1.0e3,
                "n_points": len(z_line),
                "rmse_K": rmse,
                "mae_K": mae,
                "max_abs_error_K": max_abs_error,
            }
        )

        all_T_fvm.append(T_fvm_line)
        all_T_pinn.append(T_pinn_line)

        ax = axes[i]
        ax.plot(z_line * 1.0e3, T_fvm_line, "o", ms=2.5, color="black", label="FVM")
        ax.plot(z_line * 1.0e3, T_pinn_line, "-", lw=1.5, color="tab:red", label="PINN")
        ax.set_title(f"r={r_probe * 1.0e3:.0f} mm, RMSE={rmse:.2f} K", fontsize=8)
        ax.tick_params(labelsize=7)

        if 0.0 <= r_probe <= R_NH3:
            z_nh3_bottom = float(
                z_bottom_ammonia_numpy(np.array([r_probe], dtype=np.float64))[0]
            )
            for z_interface in (z_nh3_bottom, Z_NH3_TOP):
                ax.axvline(
                    z_interface * 1.0e3, color="0.5", lw=0.8, ls="--", alpha=0.70,
                )

    for ax in axes[n_lines:]:
        ax.axis("off")

    axes[0].legend(fontsize=7, loc="best")

    T_fvm_all = np.concatenate(all_T_fvm)
    T_pinn_all = np.concatenate(all_T_pinn)
    overall_error = T_pinn_all - T_fvm_all
    overall_rmse = float(np.sqrt(np.mean(overall_error**2)))
    overall_mae = float(np.mean(np.abs(overall_error)))

    figure.suptitle(
        f"FVM vs PINN temperature along all {n_lines} radial lines "
        f"(r = 0 to {r_values[-1] * 1.0e3:.0f} mm)\n"
        f"Overall RMSE = {overall_rmse:.3f} K, MAE = {overall_mae:.3f} K",
        fontsize=12,
    )

    output_path = SAVE_DIR / output_name
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = SAVE_DIR / "fvm_vs_pinn_inconel_nh3_all_radii_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    worst = summary_df.loc[summary_df["rmse_K"].idxmax()]

    print("\n" + "=" * 72)
    print("FVM VS PINN: ALL RADIAL LINES")
    print("=" * 72)
    print(f"Number of radial lines: {n_lines}")
    print(f"Overall RMSE [K]: {overall_rmse:.6e}")
    print(f"Overall MAE [K]: {overall_mae:.6e}")
    print(
        f"Worst line: r={worst['r_mm']:.1f} mm, "
        f"RMSE={worst['rmse_K']:.3f} K, "
        f"max|error|={worst['max_abs_error_K']:.3f} K"
    )
    print(f"Saved grid plot in:\n{output_path.resolve()}")
    print(f"Saved per-radius summary in:\n{summary_csv.resolve()}")

    return summary_df


# Training-loss history
def plot_loss_history(
    losses,
    output_name="loss_history.png",
):
    history_dict = losses.losses_history

    if not isinstance(
        history_dict,
        dict,
    ):
        raise TypeError(
            "Expected losses.losses_history to be a dictionary, "
            f"but got {type(history_dict)}."
        )

    if "total" not in history_dict:
        raise KeyError(
            "No 'total' entry found in losses.losses_history. "
            f"Available keys: {list(history_dict.keys())}"
        )

    total_loss = np.asarray(
        history_dict["total"],
        dtype=np.float64,
    ).reshape(
        -1,
    )

    n_epochs = total_loss.size

    epoch = np.arange(
        1,
        n_epochs + 1,
    )

    def get_history(key):
        if key not in history_dict:
            print(f"Warning: loss key '{key}' was not found. Using zeros.")

            return np.zeros(
                n_epochs,
                dtype=np.float64,
            )

        values = np.asarray(
            history_dict[key],
            dtype=np.float64,
        ).reshape(
            -1,
        )

        if values.size != n_epochs:
            raise ValueError(
                f"History length mismatch for '{key}': "
                f"{values.size} values, expected {n_epochs}."
            )

        return values

    def summed_history(keys):
        result = np.zeros(
            n_epochs,
            dtype=np.float64,
        )

        for key in keys:
            result += get_history(
                key,
            )

        return result

    def plot_positive(
        ax,
        x,
        y,
        label,
        color,
        linewidth=1.8,
        linestyle="-",
    ):
        mask = np.isfinite(y) & (y > 0.0)

        if np.any(mask):
            ax.semilogy(
                x[mask],
                y[mask],
                linestyle,
                lw=linewidth,
                color=color,
                label=label,
            )

    def save_loss_plot(
        title,
        output_filename,
        curves,
    ):
        figure, ax = plt.subplots(
            figsize=(8.0, 5.5),
        )

        for curve in curves:
            plot_positive(
                ax=ax,
                x=epoch,
                y=curve["values"],
                label=curve["label"],
                color=curve["color"],
                linewidth=curve.get(
                    "linewidth",
                    1.8,
                ),
                linestyle=curve.get(
                    "linestyle",
                    "-",
                ),
            )

        ax.set_xlabel(
            "Epoch",
        )

        ax.set_ylabel(
            "Weighted loss",
        )

        ax.set_title(
            title,
        )

        ax.grid(
            True,
            which="both",
            alpha=0.30,
        )

        ax.legend(
            loc="best",
            fontsize=9,
        )

        figure.tight_layout()

        figure_path = SAVE_DIR / output_filename

        figure.savefig(
            figure_path,
            dpi=250,
            bbox_inches="tight",
        )

        plt.close(
            figure,
        )

        print("Saved loss-history plot in:")

        print(
            figure_path.resolve(),
        )

    # Inconel-related loss grouping

    inconel_pde_loss = summed_history(
        [
            "inc_bottom",
            "inc_side",
            "inc_top",
            "inc_lower",
        ]
    )

    inconel_axis_loss = summed_history(
        [
            "axis_inc",
        ]
    )

    inconel_interface_loss = summed_history(
        [
            "interface_temperature",
            "interface_flux",
        ]
    )

    inconel_related_loss = inconel_pde_loss + inconel_axis_loss + inconel_interface_loss

    save_loss_plot(
        title="Inconel network loss history",
        output_filename="loss_history_inconel.png",
        curves=[
            {
                "values": inconel_pde_loss,
                "label": "Inconel interior PDE",
                "color": "tab:blue",
            },
            {
                "values": inconel_axis_loss,
                "label": "Inconel axis condition",
                "color": "tab:cyan",
            },
            {
                "values": inconel_interface_loss,
                "label": "Inconel-NH3 interfaces",
                "color": "tab:orange",
            },
            {
                "values": inconel_related_loss,
                "label": "Total Inconel-related loss",
                "color": "tab:red",
                "linewidth": 2.2,
            },
        ],
    )

    # NH3-related loss grouping
    nh3_pde_loss = summed_history(
        [
            "nh3_lower",
            "nh3_upper",
        ]
    )

    nh3_axis_loss = summed_history(
        [
            "axis_nh3",
        ]
    )

    nh3_interface_loss = summed_history(
        [
            "interface_temperature",
            "interface_flux",
        ]
    )

    nh3_related_loss = nh3_pde_loss + nh3_axis_loss + nh3_interface_loss

    save_loss_plot(
        title="NH3 network loss history",
        output_filename="loss_history_nh3.png",
        curves=[
            {
                "values": nh3_pde_loss,
                "label": "NH3 interior PDE",
                "color": "tab:blue",
            },
            {
                "values": nh3_axis_loss,
                "label": "NH3 axis condition",
                "color": "tab:cyan",
            },
            {
                "values": nh3_interface_loss,
                "label": "NH3-Inconel interfaces",
                "color": "tab:orange",
            },
            {
                "values": nh3_related_loss,
                "label": "Total NH3-related loss",
                "color": "tab:red",
                "linewidth": 2.2,
            },
        ],
    )

    # Global loss grouping
    interior_loss = summed_history(
        [
            "inc_bottom",
            "inc_side",
            "inc_top",
            "inc_lower",
            "nh3_lower",
            "nh3_upper",
        ]
    )

    boundary_loss = summed_history(
        [
            "axis_inc",
            "axis_nh3",
            "interface_temperature",
            "interface_flux",
        ]
    )

    save_loss_plot(
        title="Global PINN training-loss history",
        output_filename=output_name,
        curves=[
            {
                "values": interior_loss,
                "label": "All interior PDE losses",
                "color": "tab:blue",
            },
            {
                "values": boundary_loss,
                "label": "All boundary + interface losses",
                "color": "tab:orange",
            },
            {
                "values": total_loss,
                "label": "Total weighted loss",
                "color": "tab:red",
                "linewidth": 2.2,
            },
        ],
    )

    return {
        "epoch": epoch,
        "inconel_pde_loss": inconel_pde_loss,
        "inconel_axis_loss": inconel_axis_loss,
        "inconel_interface_loss": inconel_interface_loss,
        "inconel_related_loss": inconel_related_loss,
        "nh3_pde_loss": nh3_pde_loss,
        "nh3_axis_loss": nh3_axis_loss,
        "nh3_interface_loss": nh3_interface_loss,
        "nh3_related_loss": nh3_related_loss,
        "global_interior_loss": interior_loss,
        "global_boundary_loss": boundary_loss,
        "global_total_loss": total_loss,
    }


# Full 2D FVM/PINN direct-node error map
def plot_fvm_vs_pinn_2d_error(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_2d_error.png",
):
    df_raw = read_fvm_table(
        excel_path,
        sheet_name=sheet_name,
    )

    required_columns = {
        "x",
        "y",
        "Temperature",
    }

    missing_columns = required_columns.difference(
        df_raw.columns,
    )

    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    df = df_raw.dropna(
        subset=[
            "x",
            "y",
            "Temperature",
        ],
    ).copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after removing NaN values.")

    r_fvm = df["x"].to_numpy(
        dtype=np.float64,
    )

    z_fvm = df["y"].to_numpy(
        dtype=np.float64,
    )

    T_fvm = df["Temperature"].to_numpy(
        dtype=np.float64,
    )

    if {"T0", "T1"}.issubset(
        df.columns,
    ):
        T0_values = np.unique(
            df["T0"].to_numpy(
                dtype=np.float64,
            )
        )

        T1_values = np.unique(
            df["T1"].to_numpy(
                dtype=np.float64,
            )
        )

        if T0_values.size != 1:
            raise ValueError(f"FVM T0 contains multiple values: {T0_values}")

        if T1_values.size != 1:
            raise ValueError(f"FVM T1 contains multiple values: {T1_values}")

        if not np.isclose(
            T0_values[0],
            T0_K,
        ):
            raise ValueError(f"FVM/PINN T0 mismatch: FVM={T0_values[0]}, PINN={T0_K}")

        if not np.isclose(
            T1_values[0],
            T1,
        ):
            raise ValueError(f"FVM/PINN T1 mismatch: FVM={T1_values[0]}, PINN={T1}")

    fvm_points = np.column_stack(
        [
            r_fvm,
            z_fvm,
        ]
    )

    inc_mask, nh3_mask = material_masks_numpy(
        r_fvm,
        z_fvm,
    )

    assigned_mask = inc_mask | nh3_mask

    if not np.all(assigned_mask):
        outside_count = np.count_nonzero(~assigned_mask)

        raise ValueError(
            f"{outside_count} FVM nodes are outside both the "
            "Inconel and NH3 geometry masks. Check that the "
            "Excel coordinates are in metres and use the same "
            "geometry as this PINN model."
        )

    T_pinn = np.empty(
        len(T_fvm),
        dtype=np.float64,
    )

    if np.any(inc_mask):
        T_pinn[inc_mask] = evaluate_field(
            space,
            fvm_points[inc_mask],
            "inc",
        )

    if np.any(nh3_mask):
        T_pinn[nh3_mask] = evaluate_field(
            space,
            fvm_points[nh3_mask],
            "nh3",
        )

    signed_error_K = T_pinn - T_fvm
    absolute_error_K = np.abs(
        signed_error_K,
    )

    rmse_K = np.sqrt(
        np.mean(
            signed_error_K**2,
        )
    )

    mae_K = np.mean(
        absolute_error_K,
    )

    max_abs_error_K = np.max(
        absolute_error_K,
    )

    material = np.where(
        nh3_mask,
        "NH3",
        "Inconel",
    )

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "pinn_minus_fvm_K": signed_error_K,
            "absolute_error_K": absolute_error_K,
            "material": material,
        }
    )

    print("\n" + "=" * 72)
    print("FVM VS PINN 2D ERROR SUMMARY")
    print("=" * 72)
    print(f"Rows read from Excel: {len(df_raw)}")
    print(f"Valid FVM rows compared: {len(df)}")
    print(f"Unique radial coordinates: {np.unique(r_fvm).size}")
    print(f"Unique axial coordinates: {np.unique(z_fvm).size}")
    print(f"Global RMSE [K]: {rmse_K:.8e}")
    print(f"Global MAE [K]: {mae_K:.8e}")
    print(f"Global maximum absolute error [K]: {max_abs_error_K:.8e}")

    print("\nPER-MATERIAL ERROR METRICS")
    print("-" * 72)

    for material_name in (
        "Inconel",
        "NH3",
    ):
        mask = material == material_name

        if not np.any(mask):
            print(f"{material_name}: no FVM points found")
            continue

        local_error_K = signed_error_K[mask]
        local_absolute_error_K = absolute_error_K[mask]

        local_rmse_K = np.sqrt(
            np.mean(
                local_error_K**2,
            )
        )

        local_mae_K = np.mean(
            local_absolute_error_K,
        )

        local_max_error_K = np.max(
            local_absolute_error_K,
        )

        local_mean_signed_error_K = np.mean(
            local_error_K,
        )

        print(
            f"{material_name}: "
            f"N={np.count_nonzero(mask)}, "
            f"mean signed error="
            f"{local_mean_signed_error_K:.6e} K, "
            f"RMSE={local_rmse_K:.6e} K, "
            f"MAE={local_mae_K:.6e} K, "
            f"Max|error|={local_max_error_K:.6e} K"
        )

    signed_limit = np.max(
        np.abs(
            signed_error_K,
        )
    )

    signed_norm = TwoSlopeNorm(
        vmin=-signed_limit,
        vcenter=0.0,
        vmax=signed_limit,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 8.0),
        constrained_layout=True,
    )

    signed_plot = axes[0].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=signed_error_K,
        s=12,
        marker="s",
        cmap="coolwarm",
        norm=signed_norm,
        linewidths=0.0,
        edgecolors="none",
    )

    axes[0].set_title(
        r"Signed nodal error: "
        r"$T_{\mathrm{PINN}}-T_{\mathrm{FVM}}$"
    )

    axes[0].set_xlabel(
        "r [mm]",
    )

    axes[0].set_ylabel(
        "z [mm]",
    )

    axes[0].set_aspect(
        "equal",
    )

    axes[0].grid(
        True,
        alpha=0.20,
    )

    signed_colorbar = figure.colorbar(
        signed_plot,
        ax=axes[0],
    )

    signed_colorbar.set_label(
        "PINN - FVM [K]",
    )

    absolute_plot = axes[1].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=absolute_error_K,
        s=12,
        marker="s",
        cmap="magma",
        vmin=0.0,
        vmax=max_abs_error_K,
        linewidths=0.0,
        edgecolors="none",
    )

    axes[1].set_title(
        r"Absolute nodal error: "
        r"$|T_{\mathrm{PINN}}-T_{\mathrm{FVM}}|$"
    )

    axes[1].set_xlabel(
        "r [mm]",
    )

    axes[1].set_ylabel(
        "z [mm]",
    )

    axes[1].set_aspect(
        "equal",
    )

    axes[1].grid(
        True,
        alpha=0.20,
    )

    absolute_colorbar = figure.colorbar(
        absolute_plot,
        ax=axes[1],
    )

    absolute_colorbar.set_label(
        "Absolute error [K]",
    )

    figure.suptitle(
        (
            "Inconel--NH3 FVM--PINN 2D Nodal Temperature Error\n"
            f"Nodes = {len(T_fvm)}, "
            f"RMSE = {rmse_K:.3f} K, "
            f"MAE = {mae_K:.3f} K, "
            f"Max |error| = {max_abs_error_K:.3f} K"
        ),
        fontsize=14,
    )

    output_path = SAVE_DIR / output_name

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure,
    )

    csv_path = SAVE_DIR / "fvm_vs_pinn_2d_error_data.csv"

    comparison_df.to_csv(
        csv_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM--PINN 2D ERROR FILES")
    print("=" * 72)
    print(f"Saved direct-node error plot in:\n{output_path.resolve()}")
    print(f"Saved pointwise comparison table in:\n{csv_path.resolve()}")
    print(f"Rows written to comparison CSV: {len(comparison_df)}")

    return comparison_df


def plot_fvm_vs_pinn_2d_relative_error(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_2d_relative_error.png",
):
    """
    Plot the signed and absolute relative error
    (PINN - FVM) / FVM [%] over the full 2D FVM node set.
    """

    df_raw = read_fvm_table(
        excel_path,
        sheet_name=sheet_name,
    )

    required_columns = {
        "x",
        "y",
        "Temperature",
    }

    missing_columns = required_columns.difference(
        df_raw.columns,
    )

    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    df = df_raw.dropna(
        subset=[
            "x",
            "y",
            "Temperature",
        ],
    ).copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after removing NaN values.")

    r_fvm = df["x"].to_numpy(
        dtype=np.float64,
    )

    z_fvm = df["y"].to_numpy(
        dtype=np.float64,
    )

    T_fvm = df["Temperature"].to_numpy(
        dtype=np.float64,
    )

    fvm_points = np.column_stack(
        [
            r_fvm,
            z_fvm,
        ]
    )

    inc_mask, nh3_mask = material_masks_numpy(
        r_fvm,
        z_fvm,
    )

    assigned_mask = inc_mask | nh3_mask

    if not np.all(assigned_mask):
        outside_count = np.count_nonzero(~assigned_mask)

        raise ValueError(
            f"{outside_count} FVM nodes are outside both the "
            "Inconel and NH3 geometry masks. Check that the "
            "Excel coordinates are in metres and use the same "
            "geometry as this PINN model."
        )

    T_pinn = np.empty(
        len(T_fvm),
        dtype=np.float64,
    )

    if np.any(inc_mask):
        T_pinn[inc_mask] = evaluate_field(
            space,
            fvm_points[inc_mask],
            "inc",
        )

    if np.any(nh3_mask):
        T_pinn[nh3_mask] = evaluate_field(
            space,
            fvm_points[nh3_mask],
            "nh3",
        )

    relative_error_pct = (T_pinn - T_fvm) / T_fvm * 100.0

    absolute_relative_error_pct = np.abs(
        relative_error_pct,
    )

    rmse_pct = np.sqrt(
        np.mean(
            relative_error_pct**2,
        )
    )

    mae_pct = np.mean(
        absolute_relative_error_pct,
    )

    max_abs_relative_error_pct = np.max(
        absolute_relative_error_pct,
    )

    material = np.where(
        nh3_mask,
        "NH3",
        "Inconel",
    )

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "relative_error_pct": relative_error_pct,
            "absolute_relative_error_pct": absolute_relative_error_pct,
            "material": material,
        }
    )

    print("\n" + "=" * 72)
    print("FVM VS PINN 2D RELATIVE ERROR SUMMARY")
    print("=" * 72)
    print(f"Rows read from Excel: {len(df_raw)}")
    print(f"Valid FVM rows compared: {len(df)}")
    print(f"Global relative RMSE [%]: {rmse_pct:.8e}")
    print(f"Global relative MAE [%]: {mae_pct:.8e}")
    print(
        f"Global maximum absolute relative error [%]: {max_abs_relative_error_pct:.8e}"
    )

    print("\nPER-MATERIAL RELATIVE ERROR METRICS")
    print("-" * 72)

    for material_name in (
        "Inconel",
        "NH3",
    ):
        mask = material == material_name

        if not np.any(mask):
            print(f"{material_name}: no FVM points found")
            continue

        local_relative_error_pct = relative_error_pct[mask]
        local_absolute_relative_error_pct = absolute_relative_error_pct[mask]

        local_rmse_pct = np.sqrt(
            np.mean(
                local_relative_error_pct**2,
            )
        )

        local_mae_pct = np.mean(
            local_absolute_relative_error_pct,
        )

        local_max_error_pct = np.max(
            local_absolute_relative_error_pct,
        )

        local_mean_signed_error_pct = np.mean(
            local_relative_error_pct,
        )

        print(
            f"{material_name}: "
            f"N={np.count_nonzero(mask)}, "
            f"mean signed relative error="
            f"{local_mean_signed_error_pct:.6e} %, "
            f"RMSE={local_rmse_pct:.6e} %, "
            f"MAE={local_mae_pct:.6e} %, "
            f"Max|error|={local_max_error_pct:.6e} %"
        )

    signed_limit = np.max(
        np.abs(
            relative_error_pct,
        )
    )

    signed_norm = TwoSlopeNorm(
        vmin=-signed_limit,
        vcenter=0.0,
        vmax=signed_limit,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 8.0),
        constrained_layout=True,
    )

    signed_plot = axes[0].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=relative_error_pct,
        s=12,
        marker="s",
        cmap="coolwarm",
        norm=signed_norm,
        linewidths=0.0,
        edgecolors="none",
    )

    axes[0].set_title(
        r"Signed nodal relative error: "
        r"$(T_{\mathrm{PINN}}-T_{\mathrm{FVM}})/T_{\mathrm{FVM}}$"
    )

    axes[0].set_xlabel(
        "r [mm]",
    )

    axes[0].set_ylabel(
        "z [mm]",
    )

    axes[0].set_aspect(
        "equal",
    )

    axes[0].grid(
        True,
        alpha=0.20,
    )

    signed_colorbar = figure.colorbar(
        signed_plot,
        ax=axes[0],
    )

    signed_colorbar.set_label(
        "(PINN - FVM) / FVM [%]",
    )

    absolute_plot = axes[1].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=absolute_relative_error_pct,
        s=12,
        marker="s",
        cmap="magma",
        vmin=0.0,
        vmax=max_abs_relative_error_pct,
        linewidths=0.0,
        edgecolors="none",
    )

    axes[1].set_title(
        r"Absolute nodal relative error: "
        r"$|T_{\mathrm{PINN}}-T_{\mathrm{FVM}}|/T_{\mathrm{FVM}}$"
    )

    axes[1].set_xlabel(
        "r [mm]",
    )

    axes[1].set_ylabel(
        "z [mm]",
    )

    axes[1].set_aspect(
        "equal",
    )

    axes[1].grid(
        True,
        alpha=0.20,
    )

    absolute_colorbar = figure.colorbar(
        absolute_plot,
        ax=axes[1],
    )

    absolute_colorbar.set_label(
        "|(PINN - FVM) / FVM| [%]",
    )

    figure.suptitle(
        (
            "Inconel--NH3 FVM--PINN 2D Nodal Relative Temperature Error\n"
            f"Nodes = {len(T_fvm)}, "
            f"Relative RMSE = {rmse_pct:.3f} %, "
            f"Relative MAE = {mae_pct:.3f} %, "
            f"Max |relative error| = {max_abs_relative_error_pct:.3f} %"
        ),
        fontsize=14,
    )

    output_path = SAVE_DIR / output_name

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure,
    )

    csv_path = SAVE_DIR / "fvm_vs_pinn_2d_relative_error_data.csv"

    comparison_df.to_csv(
        csv_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM--PINN 2D RELATIVE ERROR FILES")
    print("=" * 72)
    print(f"Saved relative-error plot in:\n{output_path.resolve()}")
    print(f"Saved pointwise comparison table in:\n{csv_path.resolve()}")
    print(f"Rows written to comparison CSV: {len(comparison_df)}")

    return comparison_df


# Run: Inconel + NH3


def check_interface_hard_constraints(space, n_per_segment=7, verbose=True):
    """Max temperature and flux jump across Gamma. Both should be ~0."""
    points = []
    normals = []

    for a, b in zip(_IF_P0, _IF_P1):
        direction = b - a
        length = jnp.linalg.norm(direction)
        unit = direction / length
        normal = jnp.array([-unit[1], unit[0]])

        for t in jnp.linspace(0.1, 0.9, n_per_segment):
            point = a + t * direction
            oriented = jnp.where(
                in_nh3(
                    point[0] + 1.0e-7 * normal[0],
                    point[1] + 1.0e-7 * normal[1],
                ),
                1.0,
                -1.0,
            )
            points.append(point)
            normals.append(oriented * normal)

    points = jnp.stack(points)
    normals = jnp.stack(normals)

    t_inc = T_inc_eval(space, points)[:, 0]
    t_nh3 = T_nh3_eval(space, points)[:, 0]
    t_jump = float(jnp.max(jnp.abs(t_inc - t_nh3)))

    # The flux condition is a statement about the two ONE-SIDED limits, and
    # the distance function's gradient is undefined exactly on Gamma (dist^2
    # has a minimum there, so the chain rule through sqrt returns 0). So step
    # off by h on each side. The remaining mismatch is the O(h) term
    # chi'(h) * (G - B) * (k_inc - k_nh3) and falls linearly with h -- it is
    # zero in the limit, which is what "hard" means here.
    offset = 1.0e-9

    x_inc = points - offset * normals
    x_nh3 = points + offset * normals

    grad_inc = jax.vmap(jax.grad(lambda y: _temperature_at_point(space, y, "inc")[0]))(
        x_inc
    )

    grad_nh3 = jax.vmap(jax.grad(lambda y: _temperature_at_point(space, y, "nh3")[0]))(
        x_nh3
    )

    q_inc = k_inc_value(T_inc_eval(space, x_inc)[:, 0]) * jnp.sum(
        grad_inc * normals, axis=-1
    )
    q_nh3 = k_nh3_value(T_nh3_eval(space, x_nh3)[:, 0]) * jnp.sum(
        grad_nh3 * normals, axis=-1
    )

    q_jump = float(jnp.max(jnp.abs(q_inc - q_nh3)))
    q_scale = float(jnp.mean(jnp.abs(q_inc)))

    if verbose:
        print("\n" + "=" * 72)
        print("INTERFACE HARD-CONSTRAINT CHECK")
        print("=" * 72)
        print(f"max |T_inc - T_nh3| on Gamma         : {t_jump:.3e} K")
        print(f"max |k dT/dn jump| on Gamma          : {q_jump:.3e} W/m2")
        print(f"typical |k dT/dn| on Gamma           : {q_scale:.3e} W/m2")

    return t_jump, q_jump


if __name__ == "__main__":
    space = build_space(
        seed=42,
    )

    print(
        "Number of neural models:",
        len(space.models),
    )

    pinn = Projector(
        model,
        space,
        sampler,
        # optimizer="Adam",
        weights=W_COMMON,
    )

    key = jax.random.PRNGKey(
        42,
    )

    key, samples = sampler.sample(
        key,
        N_COLLOC,
        N_BC_COLLOC,
    )

    print("\n" + "=" * 72)
    print("SAMPLE KEYS")
    print("=" * 72)

    print(
        samples.keys(),
    )

    print("\n" + "=" * 72)
    print("INITIAL LOSSES")
    print("=" * 72)

    initial_losses = pinn.evaluate_losses(
        space,
        samples,
    )

    print(
        initial_losses,
    )

    start = timeit.default_timer()

    key, pinn = pinn.project(
        key,
        space,
        N_EPOCHS,
        N_COLLOC,
        N_BC_COLLOC,
        tqdm_desc="Inconel-NH3 conduction",
    )

    elapsed = timeit.default_timer() - start

    print("\n" + "=" * 72)
    print("FINAL-SPACE LOSSES ON FRESH SAMPLES")
    print("=" * 72)

    key, final_samples = sampler.sample(
        key,
        N_COLLOC,
        N_BC_COLLOC,
    )

    final_space_losses = pinn.evaluate_losses(
        pinn.space,
        final_samples,
    )

    print(
        final_space_losses,
    )

    plot_all_interior_residuals(
        pinn.space,
    )

    plot_loss_history(
        pinn.losses,
        output_name="loss_history.png",
    )

    print("\n" + "=" * 72)
    print("BEST LOSS")
    print("=" * 72)

    print(
        pinn.best_loss,
    )

    print(f"\nTraining time: {elapsed:.2f} s")

    # Persist the trained projector so the DG comparison (and any later
    # re-analysis) can reload it instead of retraining.
    check_interface_hard_constraints(pinn.space)

    pinn.save(
        "inconel_nh3",
        postfix="corrected_k_nh3",
    )

    print("\nSaved trained projector under scriptname 'inconel_nh3'.")

    # FVM validation: Inconel + NH3

    # FV reference with T_outer(z) imposed on the sidewall as well; the
    # matching (T0, T1) case is selected by read_fvm_table.
    PROJECT_DIR = Path(__file__).resolve().parents[1] / "data"
    FVM_FILE_2D = (
        PROJECT_DIR / "2D Inconel + Ammonia K(T) correct data" / "2D_Heat_data.csv"
    )

    print("\n" + "=" * 72)
    print("FVM VALIDATION FILE CHECK: INCONEL + NH3")
    print("=" * 72)

    print(f"Expected 2D FVM file:\n{FVM_FILE_2D.resolve()}")

    print(
        "File exists:",
        FVM_FILE_2D.exists(),
    )

    if not FVM_FILE_2D.exists():
        raise FileNotFoundError(
            "The 2D FVM data file was not found:\n"
            f"{FVM_FILE_2D.resolve()}\n\n"
            "Place 2D_Heat_data.csv next to this script, or replace "
            "FVM_FILE_2D with the correct absolute path."
        )

    # Full 2D direct-node FVM/PINN error comparison.

    print("\n" + "=" * 72)
    print("RUNNING FVM VS PINN 2D NODAL COMPARISON")
    print("=" * 72)

    comparison_2d_df = plot_fvm_vs_pinn_2d_error(
        pinn.space,
        excel_path=FVM_FILE_2D,
        sheet_name="Sheet1",
        output_name="fvm_vs_pinn_inconel_nh3_2d_error.png",
    )

    comparison_2d_relative_df = plot_fvm_vs_pinn_2d_relative_error(
        pinn.space,
        excel_path=FVM_FILE_2D,
        sheet_name="Sheet1",
        output_name="fvm_vs_pinn_inconel_nh3_2d_relative_error.png",
    )

    plot_solution(
        pinn.space,
        fvm_excel_path=FVM_FILE_2D,
        fvm_sheet_name="Sheet1",
    )

    # 1D FVM/PINN line comparison

    TARGET_LINE_RADIUS = 0.004

    FVM_FILE_LINE = extract_fvm_vertical_line(
        full_excel_path=FVM_FILE_2D,
        target_r=TARGET_LINE_RADIUS,
        sheet_name="Sheet1",
        output_name="inconel_nh3_line_data_extracted.xlsx",
    )

    print("\n" + "=" * 72)
    print("RUNNING FVM VS PINN 1D LINE COMPARISON")
    print("=" * 72)

    comparison_line_df = plot_fvm_vs_pinn(
        pinn.space,
        excel_path=FVM_FILE_LINE,
        sheet_name="Sheet1",
        output_name="fvm_vs_pinn_inconel_nh3_line_comparison.png",
    )

    comparison_line_relative_df = plot_fvm_vs_pinn_relative_error(
        pinn.space,
        excel_path=FVM_FILE_LINE,
        sheet_name="Sheet1",
        output_name="fvm_vs_pinn_inconel_nh3_line_relative_error.png",
    )

    # All-radii FVM/PINN comparison (grid over every radial line in
    # 2D_Heat_data.csv), same as "2D_heat_Inconel_+_Ammonia_single_case.py".
    print("\n" + "=" * 72)
    print("RUNNING FVM VS PINN ALL-RADII COMPARISON")
    print("=" * 72)

    fvm_df_all_radii = read_fvm_table(FVM_FILE_2D, sheet_name="Sheet1")

    comparison_all_radii_df = plot_fvm_vs_pinn_all_radii(
        pinn.space,
        fvm_df_all_radii,
    )

    print("\n" + "=" * 72)
    print("INCONEL + NH3 FVM VALIDATION COMPLETE")
    print("=" * 72)

    print("2D FVM/PINN error plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_nh3_2d_error.png").resolve())

    print("\n2D FVM/PINN relative-error plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_nh3_2d_relative_error.png").resolve())

    print("\nExtracted 1D FVM line data:")
    print(FVM_FILE_LINE.resolve())

    print("\n1D FVM/PINN line comparison plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_nh3_line_comparison.png").resolve())

    print("\n1D FVM/PINN line relative-error plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_nh3_line_relative_error.png").resolve())

    print("\nAll-radii FVM/PINN comparison plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_nh3_all_radii.png").resolve())

    print("\n" + "=" * 72)
    print("INCONEL + NH3 FVM VALIDATION COMPLETE")
    print("=" * 72)

    print("2D FVM/PINN error plot:")

    print((SAVE_DIR / "fvm_vs_pinn_inconel_nh3_2d_error.png").resolve())

    print("\nExtracted 1D FVM line data:")

    print("\n1D FVM/PINN line comparison plot:")

    print((SAVE_DIR / "fvm_vs_pinn_inconel_nh3_line_comparison.png").resolve())

# Sanity check of the hard-constrained interface construction
#
# Runs on import as well as on execution, and needs no trained state: both
# conditions hold for ANY network weights, which is the whole point.
