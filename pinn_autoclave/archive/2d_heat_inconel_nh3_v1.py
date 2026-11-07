import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
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
)
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import (
    BoundaryResidual,
    InteriorResidual,
)


jax.config.update("jax_enable_x64", True)

SAVE_DIR = Path("output_inc_nh3_temp_dependent")
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
Z_NH3_MIN = float(
    NH3_LOWER_Z[0]
)


# Height at which the lower polyline reaches the NH3
Z_NH3_MAX_LOWER = float(
    NH3_LOWER_Z[-1]
)

# Temperature-dependent thermal-conductivity correlations

CONDUCTIVITY_TEMPERATURE_UNIT = "K"


def conductivity_temperature(
    temperature_K,
):

    if CONDUCTIVITY_TEMPERATURE_UNIT == "K":
        return temperature_K

    if CONDUCTIVITY_TEMPERATURE_UNIT == "C":
        return temperature_K - 273.15

    raise ValueError(
        "CONDUCTIVITY_TEMPERATURE_UNIT must be either 'K' or 'C'."
    )


def k_inc_value(
    temperature_K,
):

    temperature_input = conductivity_temperature(
        temperature_K,
    )

    return (
        6.9995
        + 0.0151
        * temperature_input
    )


def k_nh3_value(
    temperature_K,
):

    temperature_input = conductivity_temperature(
        temperature_K,
    )

    return (
        0.3039497375
        - 0.00019097
        * temperature_input
        + 0.0000001887
        * temperature_input**2
    )

# External temperature boundary condition. T0_K/T1 are set by
# configure_boundary_condition() below (called once at import time with
# the default case, and again per case inside run_case()).
def T_outer(x):
    """
    Prescribed external Inconel temperature profile [K]:

        T_outer(z) = T0_K + T1 * z.
    """
    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    z = x[..., 1:2]

    return T0_K + z * T1

# Training settings
N_COLLOC = {
    # Inconel interior sampling regions
    "inc_bottom": 1000,
    "inc_side": 4000,
    "inc_top": 1500,
    "inc_lower": 3500,

    # NH3 interior sampling regions
    "nh3_upper": 5000,
    "nh3_lower": 2500,
}

N_EPOCHS = 200

LEARNING_RATE = 1.0e-4

HIDDEN = [32, 32, 32]

# Interface temperature residual scale [K].
INTERFACE_T_REF = 5

# Characteristic length [m].
L_REF = R_OUT

EPS_AXIS = 1.0e-7

# Dimensionless multiplier of the raw NH3 neural-network output.
NH3_CORRECTION_SCALE = 1.0

# Margin added around [T0, T0 + T1*H_OUT] when building the conductivity
# reference-scale grid below, so a different (T0, T1) case still gets a
# scaling window that actually brackets its own operating temperature.
T_SCALING_MARGIN = 20.0


def configure_boundary_condition(t0, t1):
    """
    (Re)configure every T0/T1-dependent global for a new boundary-condition
    case: the outer wall law itself (T0_K, T1), the Boussinesq-style
    reference scales derived from it (DELTA_T_REF, T_INTERNAL_CENTER/RANGE),
    and the conductivity-reference-scale window (T_K_MIN/MAX_FOR_SCALING,
    K_*_REF, FLUX_REF, PDE_R_WEIGHTED_REF) -- the window must bracket this
    case's own [T0, T0 + T1*H_OUT] range, not a range hardcoded for one
    specific case, otherwise the residual normalization would be evaluated
    on the wrong part of the k(T) curves.

    Call this BEFORE building a fresh model/space for the new case: every
    residual and post-processing hook reads these as module globals at
    call time, so a fresh AutoclaveConductionModel()/build_space() built
    after this call will pick up the new values.
    """

    global T0_K, T1, DELTA_T_REF
    global T_K_MIN_FOR_SCALING, T_K_MAX_FOR_SCALING
    global K_INC_SCALE_VALUES, K_NH3_SCALE_VALUES
    global K_INC_MAX_REF, K_NH3_MAX_REF, K_REF
    global FLUX_REF, PDE_R_WEIGHTED_REF
    global T_INTERNAL_CENTER, T_INTERNAL_RANGE

    T0_K = t0
    T1 = t1

    DELTA_T_REF = abs(T1) * H_OUT

    z_extremes = jnp.array([0.0, H_OUT], dtype=jnp.float64)
    T_extremes = T0_K + T1 * z_extremes
    T_K_MIN_FOR_SCALING = float(jnp.min(T_extremes)) - T_SCALING_MARGIN
    T_K_MAX_FOR_SCALING = float(jnp.max(T_extremes)) + T_SCALING_MARGIN

    T_K_SCALE_GRID = jnp.linspace(
        T_K_MIN_FOR_SCALING, T_K_MAX_FOR_SCALING, 2001,
    )

    K_INC_SCALE_VALUES = k_inc_value(T_K_SCALE_GRID)
    K_NH3_SCALE_VALUES = k_nh3_value(T_K_SCALE_GRID)

    # Fail early if the correlation/unit choice produces an unphysical
    # conductivity anywhere in this case's own operating range.
    if bool(jnp.any(K_INC_SCALE_VALUES <= 0.0)):
        raise ValueError(
            "k_inc_value() is <= 0 somewhere in "
            f"[{T_K_MIN_FOR_SCALING}, {T_K_MAX_FOR_SCALING}] K "
            f"for T0={t0}, T1={t1}."
        )

    if bool(jnp.any(K_NH3_SCALE_VALUES <= 0.0)):
        raise ValueError(
            "k_nh3_value() is <= 0 somewhere in "
            f"[{T_K_MIN_FOR_SCALING}, {T_K_MAX_FOR_SCALING}] K "
            f"for T0={t0}, T1={t1}."
        )

    K_INC_MAX_REF = float(jnp.max(K_INC_SCALE_VALUES))
    K_NH3_MAX_REF = float(jnp.max(K_NH3_SCALE_VALUES))

    # One fixed conservative global scale for all variable-k PDE and flux
    # residuals in this case.
    K_REF = max(K_INC_MAX_REF, K_NH3_MAX_REF)

    FLUX_REF = K_REF * DELTA_T_REF / L_REF

    # Reference scale for the r-weighted axisymmetric PDE residual.
    PDE_R_WEIGHTED_REF = FLUX_REF

    T_INTERNAL_CENTER = T0_K + 0.5 * T1 * H_OUT
    T_INTERNAL_RANGE = -T1 * H_OUT


# Module-level defaults so every function below has these globals defined
# at import time; configure_boundary_condition() is called again per case
# inside run_case().
T0_K = None
T1 = None
DELTA_T_REF = None
T_K_MIN_FOR_SCALING = None
T_K_MAX_FOR_SCALING = None
K_INC_SCALE_VALUES = None
K_NH3_SCALE_VALUES = None
K_INC_MAX_REF = None
K_NH3_MAX_REF = None
K_REF = None
FLUX_REF = None
PDE_R_WEIGHTED_REF = None
T_INTERNAL_CENTER = None
T_INTERNAL_RANGE = None

configure_boundary_condition(611.15, -300.0)

# Reference coordinates
XI_NH3 = R_NH3 / R_OUT


ETA_NH3_MIN = Z_NH3_MIN / H_OUT

ETA_NH3_MAX_LOWER = (
    Z_NH3_MAX_LOWER
    / H_OUT
)

ETA_NH3_TOP = Z_NH3_TOP / H_OUT


# Lower NH3/Inconel polyline in normalized reference coordinates.
XI_NH3_LOWER = NH3_LOWER_R / R_OUT

ETA_NH3_LOWER = NH3_LOWER_Z / H_OUT

# Main mapped autoclave domain
main_mapping = DomainMapping.rectangle(
    origin=jnp.array(
        [0.0, 0.0]
    ),
    x_dir=jnp.array(
        [R_OUT, 0.0]
    ),
    y_dir=jnp.array(
        [0.0, H_OUT]
    ),
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


# Attach all volumetric sampling regions to main_domain.
for domain in (
    inc_bottom,
    inc_side,
    inc_top,
    inc_lower,
    nh3_lower,
    nh3_upper,
):
    main_domain.add_subdomain(
        domain,
    )


# Boundary segments
#
# Note: the outer autoclave wall and the symmetry axis carry no soft
# residual -- they are hard constraints baked into
# inconel_post_processing/scaled_coordinates instead, so no boundary
# domains are declared for them here.

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
N_LOWER_INTERFACE_SEGMENTS = (
    len(NH3_LOWER_R) - 1
)


lower_temperature_domains = []
lower_flux_domains = []


for i in range(
    N_LOWER_INTERFACE_SEGMENTS
):
    p1 = (
        float(
            XI_NH3_LOWER[i]
        ),
        float(
            ETA_NH3_LOWER[i]
        ),
    )

    p2 = (
        float(
            XI_NH3_LOWER[i + 1]
        ),
        float(
            ETA_NH3_LOWER[i + 1]
        ),
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

    # NH3/Inconel temperature continuity.
    "interface_temperature": [
        domain_if_side_T.get_label(),
        domain_if_top_T.get_label(),
        *[
            domain.get_label()
            for domain in lower_temperature_domains
        ],
    ],

    # NH3/Inconel conductive-flux continuity.
    "interface_flux": [
        domain_if_side_q.get_label(),
        domain_if_top_q.get_label(),
        *[
            domain.get_label()
            for domain in lower_flux_domains
        ],
    ],
}


main_domain.set_boundaries_dict(
    BOUNDARY_GROUPS,
)


# Boundary collocation counts
N_BC_COLLOC = {
    group_name: {
        boundary_label: 300
        for boundary_label in boundary_labels
    }
    for group_name, boundary_labels in BOUNDARY_GROUPS.items()
}


# Interface samples.
for boundary_label in N_BC_COLLOC["interface_temperature"]:
    N_BC_COLLOC["interface_temperature"][
        boundary_label
    ] = 750


for boundary_label in N_BC_COLLOC["interface_flux"]:
    N_BC_COLLOC["interface_flux"][
        boundary_label
    ] = 750


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
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom_ammonia(r))
        & (z <= Z_NH3_MAX_LOWER)
    )


def in_nh3(r, z):

    return (
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom_ammonia(r))
        & (z <= Z_NH3_TOP)
    )


# Neural temperature trial functions
def scaled_coordinates(x):

    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    r_scaled = x[..., 0:1] / R_OUT
    z_scaled = x[..., 1:2] / H_OUT

    rho = r_scaled**2
    eta = z_scaled

    return jnp.concatenate(
        [
            rho,
            eta,
        ],
        axis=-1,
    )


def outer_distance_factor(
    x,
):

    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    r = x[..., 0:1]
    z = x[..., 1:2]

    d_side = (
        1.0
        - (
            r / R_OUT
        )**2
    )

    d_bottom = (
        z / H_OUT
    )

    d_top = (
        H_OUT
        - z
    ) / H_OUT

    return (
        d_side
        * d_bottom
        * d_top
    )


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
    d_bottom = (
        z
        - z_bottom
    ) / H_OUT

    d_top = (
        Z_NH3_TOP
        - z
    ) / H_OUT

    d_side = (
        R_NH3
        - r
    ) / R_OUT

    return (
        d_bottom
        * d_top
        * d_side
    )

def internal_reference_profile(x):

    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    z_scaled = x[..., 1:2] / H_OUT

    return (
        T_INTERNAL_CENTER
        + T_INTERNAL_RANGE
        * (
            0.5 - z_scaled
        )
    )

# post_processing[0]: Inconel field. Takes the RAW scalar network output
# (already evaluated on pre_processed coordinates by ApproximationSpace)
# and the ORIGINAL (unscaled) x, and returns the physical temperature.

def inconel_post_processing(theta_inc, x):
    return (
        T_outer(x)
        + DELTA_T_REF
        * outer_distance_factor(x)
        * theta_inc
    )


# post_processing[1]: NH3 field, same principle.

def nh3_post_processing(raw, x):
    return (
        internal_reference_profile(x)
        + NH3_CORRECTION_SCALE
        * T_INTERNAL_RANGE
        * raw
    )

# 5) evaluate_temperature / T_inc_eval / T_nh3_eval

def evaluate_temperature(space, index, x):
    x = jnp.asarray(x, dtype=jnp.float64)
    temperature = space.create_variables()[index]

    if x.ndim == 2:
        return jax.vmap(lambda x_one: temperature(space, x_one))(x)

    return temperature(space, x)


def T_inc_eval(space, x):
    return evaluate_temperature(space, 0, x)


def T_nh3_eval(space, x):
    return evaluate_temperature(space, 1, x)

# Parametric temperature fields
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
def axisymmetric_conduction(T, conductivity, space, x):
    """
    Return the nondimensional r-weighted axisymmetric
    steady-conduction residual.

    q_r = k * dT/dr
    q_z = k * dT/dz

    Residual:
        [r*d(q_r)/dr + q_r + r*d(q_z)/dz]
        / PDE_R_WEIGHTED_REF.
    """
    dT_dr = T.partial_derivative(
        "x",
        0,
    )

    dT_dz = T.partial_derivative(
        "x",
        1,
    )

    q_r = conductivity(T) * dT_dr
    q_z = conductivity(T) * dT_dz

    r = x[..., 0:1]

    raw_residual = (
        r
        * q_r.partial_derivative(
            "x",
            0,
        )(space, x)
        + q_r(space, x)
        + r
        * q_z.partial_derivative(
            "x",
            1,
        )(space, x)
    )

    return raw_residual / PDE_R_WEIGHTED_REF


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
        T_inc, _ = vars

        def residual(space, x):
            value = axisymmetric_conduction(
                T_inc,
                k_inc,
                space,
                x,
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
            T_inc.dims,
            residual,
            f_type=T_inc.f_type,
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
        _, T_nh3 = vars

        def residual(space, x):
            value = axisymmetric_conduction(
                T_nh3,
                k_nh3,
                space,
                x,
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
            T_nh3.dims,
            residual,
            f_type=T_nh3.f_type,
        )


# Boundary residual domain dictionaries
INTERFACE_T_DOMAINS = {
    domain_if_side_T.get_label(): domain_if_side_T,
    domain_if_top_T.get_label(): domain_if_top_T,
}


INTERFACE_Q_DOMAINS = {
    domain_if_side_q.get_label(): domain_if_side_q,
    domain_if_top_q.get_label(): domain_if_top_q,
}


for domain in lower_temperature_domains:
    INTERFACE_T_DOMAINS[
        domain.get_label()
    ] = domain


for domain in lower_flux_domains:
    INTERFACE_Q_DOMAINS[
        domain.get_label()
    ] = domain


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
                / FLUX_REF.
            """

            n_r = n[
                ...,
                0:1
            ]

            n_z = n[
                ...,
                1:2
            ]

            dT_inc_dn = (
                n_r
                * dT_inc_dr(
                    space,
                    x,
                )
                + n_z
                * dT_inc_dz(
                    space,
                    x,
                )
            )

            dT_nh3_dn = (
                n_r
                * dT_nh3_dr(
                    space,
                    x,
                )
                + n_z
                * dT_nh3_dz(
                    space,
                    x,
                )
            )

            k_inc_local = k_inc_field(
                space,
                x,
            )

            k_nh3_local = k_nh3_field(
                space,
                x,
            )

            flux_jump = (
                k_inc_local
                * dT_inc_dn
                - k_nh3_local
                * dT_nh3_dn
            )

            return (
                flux_jump
                / FLUX_REF
            )

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

            # NH3/Inconel interface conditions
            "interface_temperature": (
                InterfaceTemperatureResidual()
            ),
            "interface_flux": (
                InterfaceFluxResidual()
            ),
            }

        self.data_residuals = {}


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

    key_inc, key_nh3 = jax.random.split(
        key,
        2,
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

    return ApproximationSpace(
        {"x": 2},
        [
            (nn_inc, "scalar", None),
            (nn_nh3, "scalar", None),
        ],
        model_type="x",
        pre_processing=scaled_coordinates,
        post_processing=[
            inconel_post_processing,
            nh3_post_processing,
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

    # Softly enforces T_INC = T_NH3 on all NH3/Inconel interfaces.
    "interface_temperature": (1500.0,),

    # Remains a soft material-coupling condition.
    "interface_flux": (600.0,),
}


# Plotting and post-training diagnostics

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

    return (
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom)
        & (z <= Z_NH3_TOP)
    )


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
        raise ValueError(
            f"Unknown material '{material}'. "
            "Use 'inc' or 'nh3'."
        )

    return np.asarray(
        field(
            space,
            x,
        )
    ).reshape(
        -1,
    )


# Corrected CryS-MAS reference data (2D_Heat_data.csv). Replaces the old,
# inaccurate complete_data_inc_nh3.xlsx. The CSV bundles a sweep over 25
# (T0, T1) boundary-condition combinations; this loader keeps only the
# combination matching this script's own T0_K/T1 and returns a DataFrame
# with the plain x, y, Temperature columns the plotting functions expect.
def load_fvm_reference(csv_path, t0=T0_K, t1=T1):

    csv_path = Path(csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            "Corrected FVM reference CSV not found:\n"
            f"{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {"x", "y", "Temperature"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The FVM reference CSV is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    if {"T0", "T1"}.issubset(df.columns):
        mask = np.isclose(
            df["T0"].to_numpy(dtype=np.float64), t0,
        ) & np.isclose(
            df["T1"].to_numpy(dtype=np.float64), t1,
        )

        df = df.loc[mask]

        if len(df) == 0:
            raise ValueError(
                f"No rows in {csv_path} match T0={t0}, T1={t1}."
            )

    df = df[["x", "y", "Temperature"]].dropna().reset_index(drop=True)

    return df


# plot_solution: higher-resolution filled contour + isotherm lines, plus an
# optional third figure (temperature_field_vs_fvm_error.png) showing PINN
# error against the corrected FVM reference if fvm_df is given (as returned
# by load_fvm_reference()).

def plot_solution(
    space,
    n_r=400,
    n_z=1200,
    fvm_df=None,
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
            "k", lw=1.2,
        )
        ax.plot(
            np.array([0.0, R_NH3], dtype=np.float64) * 1.0e3,
            np.full(2, Z_NH3_TOP * 1.0e3),
            "k", lw=1.2,
        )
        ax.plot(
            np.asarray(NH3_LOWER_R, dtype=np.float64) * 1.0e3,
            np.asarray(NH3_LOWER_Z, dtype=np.float64) * 1.0e3,
            "k", lw=1.4,
        )
        ax.plot(
            np.full(2, R_OUT * 1.0e3),
            np.array([0.0, H_OUT], dtype=np.float64) * 1.0e3,
            "k", lw=1.2,
        )
        ax.plot(
            np.array([0.0, R_OUT], dtype=np.float64) * 1.0e3,
            np.zeros(2, dtype=np.float64),
            "k", lw=1.2,
        )
        ax.plot(
            np.array([0.0, R_OUT], dtype=np.float64) * 1.0e3,
            np.full(2, H_OUT * 1.0e3),
            "k", lw=1.2,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("z [mm]")

    # ── Plot 1: original filled contour (kept, higher resolution now) ──
    figure, ax = plt.subplots(figsize=(5.8, 10.0))

    contour = ax.contourf(
        rr * 1.0e3, zz * 1.0e3, temperature, levels=100, cmap="turbo",
    )
    colorbar = figure.colorbar(contour, ax=ax)
    colorbar.set_label("Temperature [K]")

    draw_geometry(ax)
    ax.set_title(
        "Axisymmetric Inconel--NH3 conduction\n"
        f"T0 = {T0_K:.1f} K, T1 = {T1:.1f} K/m"
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
        rr * 1.0e3, zz * 1.0e3, temperature, levels=100, cmap="turbo",
    )
    colorbar2 = figure2.colorbar(contour2, ax=ax2)
    colorbar2.set_label("Temperature [K]")

    line_levels = np.linspace(temperature.min(), temperature.max(), 20)
    lines = ax2.contour(
        rr * 1.0e3, zz * 1.0e3, temperature,
        levels=line_levels, colors="white", linewidths=0.6,
    )
    ax2.clabel(lines, inline=True, fontsize=6, fmt="%.0f")

    draw_geometry(ax2)
    ax2.set_title(
        "Axisymmetric Inconel--NH3 conduction\n"
        "Isotherms every "
        f"{(line_levels[1]-line_levels[0]):.2f} K"
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
    if fvm_df is not None:
        df_fvm = fvm_df
        r_fvm = df_fvm["x"].to_numpy(dtype=np.float64)
        z_fvm = df_fvm["y"].to_numpy(dtype=np.float64)
        T_fvm = df_fvm["Temperature"].to_numpy(dtype=np.float64)

        fvm_points = np.column_stack([r_fvm, z_fvm])
        inc_mask_fvm, nh3_mask_fvm = material_masks_numpy(r_fvm, z_fvm)

        T_pinn_fvm = np.empty(len(T_fvm), dtype=np.float64)
        if np.any(inc_mask_fvm):
            T_pinn_fvm[inc_mask_fvm] = evaluate_field(
                space, fvm_points[inc_mask_fvm], "inc",
            )
        if np.any(nh3_mask_fvm):
            T_pinn_fvm[nh3_mask_fvm] = evaluate_field(
                space, fvm_points[nh3_mask_fvm], "nh3",
            )

        error_fvm = T_pinn_fvm - T_fvm
        limit = np.max(np.abs(error_fvm))

        figure3, ax3 = plt.subplots(figsize=(5.8, 10.0))

        scatter = ax3.scatter(
            r_fvm * 1.0e3, z_fvm * 1.0e3,
            c=error_fvm, cmap="coolwarm",
            vmin=-limit, vmax=limit,
            s=14, marker="s", linewidths=0.0,
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
            )(
                x_one
            )[0]

            conductivity = conductivity_function(
                temperature,
            )

            return (
                conductivity
                * dT_dr
            )

        def axial_flux_at_one_point(
            x_one,
        ):
            temperature = temperature_at_one_point(
                x_one,
            )

            dT_dz = jax.grad(
                temperature_at_one_point,
            )(
                x_one
            )[1]

            conductivity = conductivity_function(
                temperature,
            )

            return (
                conductivity
                * dT_dz
            )

        radial_flux_gradient = jax.grad(
            radial_flux_at_one_point,
        )

        axial_flux_gradient = jax.grad(
            axial_flux_at_one_point,
        )

        def residual_at_one_point(
            x_one,
        ):
            r = x_one[
                0
            ]

            q_r = radial_flux_at_one_point(
                x_one,
            )

            dq_r_dr = radial_flux_gradient(
                x_one
            )[
                0
            ]

            dq_z_dz = axial_flux_gradient(
                x_one
            )[
                1
            ]

            raw_residual = (
                r
                * dq_r_dr
                + q_r
                + r
                * dq_z_dz
            )

            return (
                raw_residual
                / PDE_R_WEIGHTED_REF
            )

        return jax.vmap(
            residual_at_one_point,
        )(
            x
        )

    return residual_on_points


inc_conduction_residual = make_axisymmetric_residual(
    T_inc_eval,
    k_inc_value,
)

nh3_conduction_residual = make_axisymmetric_residual(
    T_nh3_eval,
    k_nh3_value,
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
        mask = (
            (r_np >= 0.0)
            & (r_np <= R_OUT)
            & (z_np >= 0.0)
            & (z_np <= Z_NH3_MIN)
        )

    elif residual_name == "inc_lower":
        z_polyline = z_bottom_ammonia_numpy(
            r_np,
        )

        mask = (
            (r_np >= 0.0)
            & (r_np <= R_OUT)
            & (z_np >= Z_NH3_MIN)
            & (z_np <= Z_NH3_MAX_LOWER)
            & (
                (r_np > R_NH3)
                | (z_np < z_polyline)
            )
        )

    elif residual_name == "inc_side":
        mask = (
            (r_np >= R_NH3)
            & (r_np <= R_OUT)
            & (z_np >= Z_NH3_MIN)
            & (z_np <= Z_NH3_TOP)
        )

    elif residual_name == "inc_top":
        mask = (
            (r_np >= 0.0)
            & (r_np <= R_OUT)
            & (z_np >= Z_NH3_TOP)
            & (z_np <= H_OUT)
        )

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
        raise ValueError(
            f"Unknown interior residual name: "
            f"{residual_name}"
        )

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
            raise ValueError(
                f"No positive residual values remain for "
                f"{residual_name}."
            )

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
            values[
                np.isfinite(values)
                & (values > 0.0)
            ]
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

    output_path = (
        SAVE_DIR
        / "all_interior_residual_maps.png"
    )

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure,
    )

    print(
        "Saved interior residual maps in:"
    )

    print(
        output_path.resolve(),
    )




# FVM versus PINN comparison along every available radial line.
# The corrected reference data (2D_Heat_data.csv) already provides 36
# vertical lines (r = 0 to 35 mm in 1 mm steps), so instead of picking one
# probe radius, this compares all of them and grids the result into three
# figures (temperature overlay, signed error, relative error) plus a
# per-radius error-summary table.
def _grid_axes(n_lines, n_cols=6):
    n_rows = int(np.ceil(n_lines / n_cols))
    figure, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.6 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_lines:]:
        ax.axis("off")
    return figure, axes


def plot_fvm_vs_pinn_all_radii(
    space,
    df,
    output_name="fvm_vs_pinn_inconel_nh3_all_radii.png",
    error_output_name="fvm_vs_pinn_inconel_nh3_all_radii_error.png",
    relative_error_output_name="fvm_vs_pinn_inconel_nh3_all_radii_relative_error.png",
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    r_values = np.sort(df["x"].unique())
    n_lines = len(r_values)

    figure_t, axes_t = _grid_axes(n_lines)
    figure_e, axes_e = _grid_axes(n_lines)
    figure_r, axes_r = _grid_axes(n_lines)

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
        relative_error_pct = error / T_fvm_line * 100.0

        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        max_abs_error = float(np.max(np.abs(error)))
        rmse_pct = float(np.sqrt(np.mean(relative_error_pct**2)))
        mae_pct = float(np.mean(np.abs(relative_error_pct)))

        summary_rows.append(
            {
                "r_mm": float(r_probe) * 1.0e3,
                "n_points": len(z_line),
                "rmse_K": rmse,
                "mae_K": mae,
                "max_abs_error_K": max_abs_error,
                "rmse_pct": rmse_pct,
                "mae_pct": mae_pct,
            }
        )

        all_T_fvm.append(T_fvm_line)
        all_T_pinn.append(T_pinn_line)

        z_mm = z_line * 1.0e3

        def draw_interfaces(ax):
            if 0.0 <= r_probe <= R_NH3:
                z_nh3_bottom = float(
                    z_bottom_ammonia_numpy(np.array([r_probe], dtype=np.float64))[0]
                )
                for z_interface in (z_nh3_bottom, Z_NH3_TOP):
                    ax.axvline(
                        z_interface * 1.0e3, color="0.5", lw=0.8, ls="--", alpha=0.70,
                    )

        ax_t = axes_t[i]
        ax_t.plot(z_mm, T_fvm_line, "o", ms=2.5, color="black", label="FVM")
        ax_t.plot(z_mm, T_pinn_line, "-", lw=1.5, color="tab:red", label="PINN")
        ax_t.set_title(f"r={r_probe * 1.0e3:.0f} mm, RMSE={rmse:.2f} K", fontsize=8)
        ax_t.tick_params(labelsize=7)
        draw_interfaces(ax_t)

        ax_e = axes_e[i]
        ax_e.axhline(0.0, color="black", lw=0.8)
        ax_e.plot(z_mm, error, "-o", ms=2.0, lw=1.0, color="tab:blue")
        ax_e.set_title(f"r={r_probe * 1.0e3:.0f} mm, RMSE={rmse:.2f} K", fontsize=8)
        ax_e.tick_params(labelsize=7)
        draw_interfaces(ax_e)

        ax_r = axes_r[i]
        ax_r.axhline(0.0, color="black", lw=0.8)
        ax_r.plot(z_mm, relative_error_pct, "-o", ms=2.0, lw=1.0, color="tab:green")
        ax_r.set_title(f"r={r_probe * 1.0e3:.0f} mm, RMSE={rmse_pct:.2f} %", fontsize=8)
        ax_r.tick_params(labelsize=7)
        draw_interfaces(ax_r)

    axes_t[0].legend(fontsize=7, loc="best")

    T_fvm_all = np.concatenate(all_T_fvm)
    T_pinn_all = np.concatenate(all_T_pinn)
    overall_error = T_pinn_all - T_fvm_all
    overall_rmse = float(np.sqrt(np.mean(overall_error**2)))
    overall_mae = float(np.mean(np.abs(overall_error)))
    overall_relative_error_pct = overall_error / T_fvm_all * 100.0
    overall_rmse_pct = float(np.sqrt(np.mean(overall_relative_error_pct**2)))
    overall_mae_pct = float(np.mean(np.abs(overall_relative_error_pct)))

    r_max_mm = r_values[-1] * 1.0e3

    figure_t.suptitle(
        f"FVM vs PINN temperature along all {n_lines} radial lines "
        f"(r = 0 to {r_max_mm:.0f} mm)\n"
        f"Overall RMSE = {overall_rmse:.3f} K, MAE = {overall_mae:.3f} K",
        fontsize=12,
    )
    figure_e.suptitle(
        f"PINN - FVM signed error along all {n_lines} radial lines "
        f"(r = 0 to {r_max_mm:.0f} mm)\n"
        f"Overall RMSE = {overall_rmse:.3f} K, MAE = {overall_mae:.3f} K",
        fontsize=12,
    )
    figure_r.suptitle(
        f"(PINN - FVM) / FVM relative error along all {n_lines} radial lines "
        f"(r = 0 to {r_max_mm:.0f} mm)\n"
        f"Overall RMSE = {overall_rmse_pct:.3f} %, MAE = {overall_mae_pct:.3f} %",
        fontsize=12,
    )

    output_path = SAVE_DIR / output_name
    error_output_path = SAVE_DIR / error_output_name
    relative_error_output_path = SAVE_DIR / relative_error_output_name

    figure_t.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure_t)
    figure_e.savefig(error_output_path, dpi=200, bbox_inches="tight")
    plt.close(figure_e)
    figure_r.savefig(relative_error_output_path, dpi=200, bbox_inches="tight")
    plt.close(figure_r)

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = SAVE_DIR / "fvm_vs_pinn_inconel_nh3_all_radii_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    worst = summary_df.loc[summary_df["rmse_K"].idxmax()]

    print("\n" + "=" * 72)
    print("FVM VS PINN: ALL RADIAL LINES")
    print("=" * 72)
    print(f"Number of radial lines: {n_lines}")
    print(f"Overall RMSE [K]: {overall_rmse:.6e}   Overall RMSE [%]: {overall_rmse_pct:.6e}")
    print(f"Overall MAE [K]: {overall_mae:.6e}   Overall MAE [%]: {overall_mae_pct:.6e}")
    print(
        f"Worst line: r={worst['r_mm']:.1f} mm, "
        f"RMSE={worst['rmse_K']:.3f} K, "
        f"max|error|={worst['max_abs_error_K']:.3f} K"
    )
    print(f"Saved temperature-overlay grid in:\n{output_path.resolve()}")
    print(f"Saved signed-error grid in:\n{error_output_path.resolve()}")
    print(f"Saved relative-error grid in:\n{relative_error_output_path.resolve()}")
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
            print(
                f"Warning: loss key '{key}' was not found. "
                "Using zeros."
            )

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
        mask = (
            np.isfinite(y)
            & (y > 0.0)
        )

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

        print(
            "Saved loss-history plot in:"
        )

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

    inconel_interface_loss = summed_history(
        [
            "interface_temperature",
            "interface_flux",
        ]
    )

    inconel_related_loss = (
        inconel_pde_loss
        + inconel_interface_loss
    )

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

    nh3_interface_loss = summed_history(
        [
            "interface_temperature",
            "interface_flux",
        ]
    )

    nh3_related_loss = (
        nh3_pde_loss
        + nh3_interface_loss
    )

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
        "inconel_interface_loss": inconel_interface_loss,
        "inconel_related_loss": inconel_related_loss,
        "nh3_pde_loss": nh3_pde_loss,
        "nh3_interface_loss": nh3_interface_loss,
        "nh3_related_loss": nh3_related_loss,
        "global_interior_loss": interior_loss,
        "global_boundary_loss": boundary_loss,
        "global_total_loss": total_loss,
    }


# Full 2D FVM/PINN direct-node error map
def plot_fvm_vs_pinn_2d_error(
    space,
    df,
    output_name="fvm_vs_pinn_2d_error.png",
):

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

    assigned_mask = (
    inc_mask
    | nh3_mask
    )

    if not np.all(
        assigned_mask
    ):
        outside_count = np.count_nonzero(
            ~assigned_mask
        )

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
    print(f"Valid FVM rows compared: {len(df)}")
    print(
        f"Unique radial coordinates: "
        f"{np.unique(r_fvm).size}"
    )
    print(
        f"Unique axial coordinates: "
        f"{np.unique(z_fvm).size}"
    )
    print(f"Global RMSE [K]: {rmse_K:.8e}")
    print(f"Global MAE [K]: {mae_K:.8e}")
    print(
        "Global maximum absolute error [K]: "
        f"{max_abs_error_K:.8e}"
    )

    print("\nPER-MATERIAL ERROR METRICS")
    print("-" * 72)

    for material_name in (
        "Inconel",
        "NH3",
    ):
        mask = material == material_name

        if not np.any(mask):
            print(
                f"{material_name}: no FVM points found"
            )
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

    csv_path = (
        SAVE_DIR
        / "fvm_vs_pinn_2d_error_data.csv"
    )

    comparison_df.to_csv(
        csv_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM--PINN 2D ERROR FILES")
    print("=" * 72)
    print(
        f"Saved direct-node error plot in:\n"
        f"{output_path.resolve()}"
    )
    print(
        f"Saved pointwise comparison table in:\n"
        f"{csv_path.resolve()}"
    )
    print(
        f"Rows written to comparison CSV: "
        f"{len(comparison_df)}"
    )

    return comparison_df


def plot_fvm_vs_pinn_2d_relative_error(
    space,
    df,
    output_name="fvm_vs_pinn_2d_relative_error.png",
):
    """
    Plot the signed and absolute relative error
    (PINN - FVM) / FVM [%] over the full 2D FVM node set.
    """

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

    assigned_mask = (
        inc_mask
        | nh3_mask
    )

    if not np.all(
        assigned_mask
    ):
        outside_count = np.count_nonzero(
            ~assigned_mask
        )

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

    relative_error_pct = (
        (T_pinn - T_fvm)
        / T_fvm
        * 100.0
    )

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
    print(f"Valid FVM rows compared: {len(df)}")
    print(f"Global relative RMSE [%]: {rmse_pct:.8e}")
    print(f"Global relative MAE [%]: {mae_pct:.8e}")
    print(
        "Global maximum absolute relative error [%]: "
        f"{max_abs_relative_error_pct:.8e}"
    )

    print("\nPER-MATERIAL RELATIVE ERROR METRICS")
    print("-" * 72)

    for material_name in (
        "Inconel",
        "NH3",
    ):
        mask = material == material_name

        if not np.any(mask):
            print(
                f"{material_name}: no FVM points found"
            )
            continue

        local_relative_error_pct = relative_error_pct[mask]
        local_absolute_relative_error_pct = (
            absolute_relative_error_pct[mask]
        )

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

    csv_path = (
        SAVE_DIR
        / "fvm_vs_pinn_2d_relative_error_data.csv"
    )

    comparison_df.to_csv(
        csv_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM--PINN 2D RELATIVE ERROR FILES")
    print("=" * 72)
    print(
        f"Saved relative-error plot in:\n"
        f"{output_path.resolve()}"
    )
    print(
        f"Saved pointwise comparison table in:\n"
        f"{csv_path.resolve()}"
    )
    print(
        f"Rows written to comparison CSV: "
        f"{len(comparison_df)}"
    )

    return comparison_df


# Run: Inconel + NH3

# Boundary-condition cases to run. Each entry retrains a fresh model from
# scratch under configure_boundary_condition(t0, t1) and writes its outputs
# to output_inc_nh3_temp_dependent/<label>/, so cases never overwrite each
# other. "t0_611_t1_m300" is the original/default case (matches the
# lower-left corner of the 2D_Heat_data.csv sweep exactly).
CASES = [
    {"t0": 611.15, "t1": -300.0, "label": "t0_611_t1_m300"},
    {"t0": 611.15, "t1": 300.0, "label": "t0_611_t1_p300"},
    {"t0": 811.15, "t1": -300.0, "label": "t0_811_t1_m300"},
    {"t0": 611.15, "t1": -150.0, "label": "t0_611_t1_m150"},
]


def run_case(t0, t1, label, n_epochs=N_EPOCHS, seed=42):
    """Build, train, and validate one (T0, T1) boundary-condition case."""

    global SAVE_DIR

    configure_boundary_condition(t0, t1)

    SAVE_DIR = Path("output_inc_nh3_temp_dependent") / label
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 72)
    print(f"# CASE {label}: T0={t0} K, T1={t1} K/m")
    print("#" * 72)

    space = build_space(seed=seed)
    print("Number of neural models:", len(space.models))

    case_model = AutoclaveConductionModel()

    pinn = Projector(
        case_model,
        space,
        sampler,
        weights=W_COMMON,
        learning_rate=LEARNING_RATE,
    )

    key = jax.random.PRNGKey(seed)
    key, samples = sampler.sample(key, N_COLLOC, N_BC_COLLOC)

    print("\n" + "=" * 72)
    print("INITIAL LOSSES")
    print("=" * 72)
    print(pinn.evaluate_losses(space, samples))

    start = timeit.default_timer()

    key, pinn = pinn.project(
        key,
        space,
        n_epochs,
        N_COLLOC,
        N_BC_COLLOC,
        tqdmdesc=f"Inconel-NH3 conduction [{label}]",
    )

    elapsed = timeit.default_timer() - start

    plot_all_interior_residuals(pinn.space)
    plot_loss_history(pinn.losses, output_name="loss_history.png")

    print("\n" + "=" * 72)
    print("BEST LOSS")
    print("=" * 72)
    print(pinn.best_loss)
    print(f"\nTraining time: {elapsed:.2f} s")

    # FVM validation against the corrected CryS-MAS reference
    # (2D_Heat_data.csv), filtered to this case's own (T0, T1).

    PROJECT_DIR = Path(__file__).resolve().parents[5]

    FVM_REFERENCE_CSV = (
        PROJECT_DIR
        / "2D Inconel + Ammonia K(T) correct data"
        / "2D_Heat_data.csv"
    )

    fvm_df = load_fvm_reference(FVM_REFERENCE_CSV, t0=T0_K, t1=T1)
    print(f"\nFVM reference rows matching T0={T0_K}, T1={T1}: {len(fvm_df)}")

    plot_fvm_vs_pinn_2d_error(
        pinn.space, fvm_df, output_name="fvm_vs_pinn_inconel_nh3_2d_error.png",
    )
    plot_fvm_vs_pinn_2d_relative_error(
        pinn.space, fvm_df,
        output_name="fvm_vs_pinn_inconel_nh3_2d_relative_error.png",
    )
    plot_solution(pinn.space, fvm_df=fvm_df)
    summary_df = plot_fvm_vs_pinn_all_radii(pinn.space, fvm_df)

    print(f"\nCase {label} complete. Outputs in:\n{SAVE_DIR.resolve()}")

    return {"label": label, "t0": t0, "t1": t1, "summary_df": summary_df}


if __name__ == "__main__":

    case_results = [
        run_case(case["t0"], case["t1"], case["label"], n_epochs=N_EPOCHS)
        for case in CASES
    ]

    print("\n" + "#" * 72)
    print("ALL CASES COMPLETE")
    print("#" * 72)
    for result in case_results:
        print(
            f"{result['label']}: T0={result['t0']} K, T1={result['t1']} K/m "
            f"-> output_inc_nh3_temp_dependent/{result['label']}/"
        )
