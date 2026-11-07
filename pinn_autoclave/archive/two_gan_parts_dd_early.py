"""
Fixed-case domain-decomposed PINN for steady axisymmetric heat conduction
in an Inconel autoclave with ammonia and GaN cavity.

Coordinates received by ScimBa residuals are physical:
    x = (r, z) [m]

Materials:
    k_inconel(T) = 6.9995 + 0.0151*T          W/(m K)
    k_ammonia(T) = cubic correlation           W/(m K)
    k_GaN        = 130.0                       W/(m K)

Networks (7): Inconel; NH3 in four z-slabs (a: cavity bottom .. top of the lower
GaN block, b: between the GaN blocks, c: alongside the upper GaN block, d: cap);
lower GaN block; upper GaN block.

Outer Inconel boundary:
    T_outer(z) = T0_C + 273.15 + z*T1  [K]


Geometry:
    outer vessel:
        0 <= r <= 0.035
        0 <= z <= 0.370

    ammonia:
        0 <= r <= 0.012
        z_bottom(r) <= z <= 0.335

    z_bottom(r):
        0.035 - sqrt(0.012^2 - r^2)     (lowest point z = 0.023 on the axis)

    GaN (both 0 <= r <= 0.005):
        lower: 0.040 <= z <= 0.120
        upper: 0.175 <= z <= 0.325

Geometry follows Autoclave.pcs (2D_Coupled_Heat_Interiour_Smaller_GaN_Top).

The physical lower ammonia arc is approximated with short Segment2D pieces
in reference coordinates so it remains circular after the main-domain mapping.
"""

import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

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
from scimba_jax.plots.plots_nd import plot_abstract_approx_space


jax.config.update("jax_enable_x64", True)

SAVE_DIR = Path("outputs_autoclave_NH3_GaN")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Physical geometry and properties
R_OUT = 0.035
H_OUT = 0.370

R_NH3 = 0.012
Z_NH3_MIN = 0.023  # lowest point of the cavity on the axis (Autoclave.pcs)
Z_ARC_CENTER = Z_NH3_MIN + R_NH3  # = 0.035: lower cavity is a quarter circle
Z_NH3_TOP = 0.335

K_GAN = 130.0  # W/(m K), GaN conductivity (constant)

CONDUCTIVITY_TEMPERATURE_UNIT = "K"


def conductivity_temperature(temperature_K):
    if CONDUCTIVITY_TEMPERATURE_UNIT == "K":
        return temperature_K
    if CONDUCTIVITY_TEMPERATURE_UNIT == "C":
        return temperature_K - 273.15
    raise ValueError("CONDUCTIVITY_TEMPERATURE_UNIT must be either 'K' or 'C'.")


def k_inc_value(temperature_K):
    """Inconel conductivity [W/(m K)]; same correlation as the Inconel + NH3 model."""
    temperature_input = conductivity_temperature(temperature_K)
    return 6.9995 + 0.0151 * temperature_input


def k_nh3_value(temperature_K):
    """Ammonia conductivity [W/(m K)]; same correlation as the Inconel + NH3 model."""
    temperature_input = conductivity_temperature(temperature_K)
    return (
        70.63350679
        - 0.19209961 * temperature_input
        + 0.00020083 * temperature_input**2
        - 0.00000007 * temperature_input**3
    )


def k_gan_value(temperature_K):
    """GaN conductivity [W/(m K)]: constant 130, returned with the shape of T."""
    return K_GAN * jnp.ones_like(temperature_K)


# Lower continuous GaN block:
# 0 <= r <= 5 mm
# 40 mm <= z <= 120 mm
R_GAN_LOWER = 0.005
Z_GAN_LOWER_MIN = 0.040
Z_GAN_LOWER_MAX = 0.120


# Separate upper GaN block:
# 0 <= r <= 5 mm
# 175 mm <= z <= 325 mm
R_GAN_UPPER = 0.005
Z_GAN_UPPER_MIN = 0.175
Z_GAN_UPPER_MAX = 0.325


# Fixed external thermal boundary condition
T0_K = 611.15
T1 = -300

T0_C = T0_K - 273.15
T_REF = T0_K
INTERFACE_T_REF = 5  # K


def T_outer(x):
    """Outer Inconel temperature target in K."""
    z = x[..., 1:2]
    return T0_K + z * T1


# Training settings

N_COLLOC = {
    "inc_bottom": 100,
    "inc_side": 1000,
    "inc_top": 100,
    "inc_lower": 300,
    "nh3_a_arc": 300,
    "nh3_a_side": 600,
    "nh3_b": 400,
    "nh3_c_side": 700,
    "nh3_d": 300,
    "gan_lower": 1000,
    "gan_upper": 1000,
}

# N_BC_COLLOC (boundary points per interface segment) is generated together with
# the interface segments below.

N_EPOCHS =100
LEARNING_RATE = 1.0e-4
HIDDEN = [16, 16, 16]        # Inconel and GaN networks
HIDDEN_NH3 = [16, 16, 16]    # each NH3 slab network

DELTA_T_REF = abs(T1) * H_OUT
CORRECTION_SCALE = DELTA_T_REF
L_REF = R_OUT

# Conductivity ranges over the anticipated temperature window
T_K_MIN_FOR_SCALING = 495.0
T_K_MAX_FOR_SCALING = 615.0
T_K_SCALE_GRID = jnp.linspace(T_K_MIN_FOR_SCALING, T_K_MAX_FOR_SCALING, 2001)

K_INC_SCALE_VALUES = k_inc_value(T_K_SCALE_GRID)
K_NH3_SCALE_VALUES = k_nh3_value(T_K_SCALE_GRID)

if bool(jnp.any(K_INC_SCALE_VALUES <= 0.0)):
    raise ValueError("Inconel conductivity is not positive in the scaling range.")
if bool(jnp.any(K_NH3_SCALE_VALUES <= 0.0)):
    raise ValueError("NH3 conductivity is not positive in the scaling range.")

K_INC_MAX_REF = float(jnp.max(K_INC_SCALE_VALUES))
K_NH3_MAX_REF = float(jnp.max(K_NH3_SCALE_VALUES))

# Residual scales.  Each material is normalized with its own conductivity, so the
# Inconel/NH3 residuals keep the same scale as in the Inconel + NH3 model and are
# not diluted by the (much larger) GaN conductivity.
K_REF = max(K_INC_MAX_REF, K_NH3_MAX_REF)
FLUX_REF = K_REF * DELTA_T_REF / L_REF          # Inconel/NH3 PDE and interface flux
PDE_R_WEIGHTED_REF = FLUX_REF

# GaN PDE: own scale.
FLUX_REF_GAN = K_GAN * DELTA_T_REF / L_REF
# NH3/GaN interface flux: normalized with the LOW-conductivity (NH3) side, so that a
# flux mismatch is not hidden by the large K_GAN.
FLUX_REF_GAN_INTERFACE = K_NH3_MAX_REF * DELTA_T_REF / L_REF

EPS_AXIS = 1.0e-7

# The r-weighted PDE residual is divided by max(r, R_WEIGHT_FLOOR): the residual is
# the true (unweighted) one for r > R_WEIGHT_FLOOR and only r-weighted in a thin
# strip next to the axis, so the small-r region (NH3, GaN) is not under-weighted.
R_WEIGHT_FLOOR = 0.0025  # m


# Reference coordinates of material geometry
XI_NH3 = R_NH3 / R_OUT

ETA_NH3_MIN = Z_NH3_MIN / H_OUT
ETA_ARC_CENTER = Z_ARC_CENTER / H_OUT
ETA_NH3_TOP = Z_NH3_TOP / H_OUT

XI_GAN_LOWER = R_GAN_LOWER / R_OUT
ETA_GAN_LOWER_MIN = Z_GAN_LOWER_MIN / H_OUT
ETA_GAN_LOWER_MAX = Z_GAN_LOWER_MAX / H_OUT


XI_GAN_UPPER = R_GAN_UPPER / R_OUT
ETA_GAN_UPPER_MIN = Z_GAN_UPPER_MIN / H_OUT
ETA_GAN_UPPER_MAX = Z_GAN_UPPER_MAX / H_OUT


# Main mapped autoclave domain
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
    bounds_postmap=[
        (0.0, R_OUT),
        (0.0, H_OUT),
    ],
)


# Volumetric subdomains
# These are in reference coordinates. Once attached to main_domain, ScimBa
# maps their samples into physical coordinates (r,z) automatically.

inc_bottom = Square2D(
    [(0.0, 1.0), (0.0, ETA_NH3_MIN)],
    is_main_domain=False,
    label_str="inc_bottom",
)

inc_side = Square2D(
    [(XI_NH3, 1.0), (ETA_ARC_CENTER, ETA_NH3_TOP)],
    is_main_domain=False,
    label_str="inc_side",
)

inc_top = Square2D(
    [(0.0, 1.0), (ETA_NH3_TOP, 1.0)],
    is_main_domain=False,
    label_str="inc_top",
)

# Lower rectangle includes both Inconel and the ammonia quarter-disk.
# A mask inside the Inconel PDE residual excludes the NH3 part.
inc_lower = Square2D(
    [(0.0, 1.0), (ETA_NH3_MIN, ETA_ARC_CENTER)],
    is_main_domain=False,
    label_str="inc_lower",
)


# NH3 sub-domains: the cavity is split in z into four slabs, each with its own
# network, at the top/bottom faces of the two GaN blocks:
#   a: bottom of the cavity .. top of the lower GaN block  (includes the arc)
#   b: between the two GaN blocks
#   c: alongside the upper GaN block
#   d: cap above the upper GaN block
# Each slab is a rectangle in reference coordinates; the PDE residual masks the
# true NH3 region (arc excluded, GaN excluded).

NH3_SLABS = {
    "a": (Z_NH3_MIN, Z_GAN_LOWER_MAX),
    "b": (Z_GAN_LOWER_MAX, Z_GAN_UPPER_MIN),
    "c": (Z_GAN_UPPER_MIN, Z_GAN_UPPER_MAX),
    "d": (Z_GAN_UPPER_MAX, Z_NH3_TOP),
}

# Sampling tiles of the NH3 cavity: (slab network, r_min, r_max, z_min, z_max).
# The tiles cover the true NH3 region without the GaN blocks, so (almost) no
# collocation point is wasted on a masked-out residual row.  The arc tile still
# contains the Inconel corner below the arc, which the residual masks.
NH3_TILES = {
    "nh3_a_arc": ("a", 0.0, R_NH3, Z_NH3_MIN, Z_GAN_LOWER_MIN),
    "nh3_a_side": ("a", R_GAN_LOWER, R_NH3, Z_GAN_LOWER_MIN, Z_GAN_LOWER_MAX),
    "nh3_b": ("b", 0.0, R_NH3, Z_GAN_LOWER_MAX, Z_GAN_UPPER_MIN),
    "nh3_c_side": ("c", R_GAN_UPPER, R_NH3, Z_GAN_UPPER_MIN, Z_GAN_UPPER_MAX),
    "nh3_d": ("d", 0.0, R_NH3, Z_GAN_UPPER_MAX, Z_NH3_TOP),
}

NH3_TILE_KEYS = {
    slab_name: [key for key, tile in NH3_TILES.items() if tile[0] == slab_name]
    for slab_name in NH3_SLABS
}

nh3_domains = {}

for tile_key, (_, tile_r_min, tile_r_max, tile_z_min, tile_z_max) in NH3_TILES.items():
    nh3_domains[tile_key] = Square2D(
        [
            (tile_r_min / R_OUT, tile_r_max / R_OUT),
            (tile_z_min / H_OUT, tile_z_max / H_OUT),
        ],
        is_main_domain=False,
        label_str=tile_key,
    )

gan_lower = Square2D(
    [(0.0, XI_GAN_LOWER), (ETA_GAN_LOWER_MIN, ETA_GAN_LOWER_MAX)],
    is_main_domain=False,
    label_str="gan_lower",
)

gan_upper = Square2D(
    [(0.0, XI_GAN_UPPER), (ETA_GAN_UPPER_MIN, ETA_GAN_UPPER_MAX)],
    is_main_domain=False,
    label_str="gan_upper",
)

for domain in (
    inc_bottom,
    inc_side,
    inc_top,
    inc_lower,
    *nh3_domains.values(),
    gan_lower,
    gan_upper,
):
    main_domain.add_subdomain(domain)


# Interface groups
# Every interface is described once and generates a temperature-continuity and a
# flux-continuity boundary group.  A group couples exactly two networks (a, b),
# because the boundary residual of a group is one function of (space, x, n).
# Segments are (tag, r1, z1, r2, z2, n_points) in PHYSICAL coordinates [m].

N_ARC_SEGMENTS = 32
ARC_BC_POINTS = 75

INTERFACE_SPECS = {
    # Inconel / NH3
    "inc_nh3_a": {
        "a": "inc",
        "b": "nh3_a",
        "flux_ref": FLUX_REF,
        "arc": True,
        "segments": [
            ("side", R_NH3, Z_ARC_CENTER, R_NH3, Z_GAN_LOWER_MAX, 100),
        ],
    },
    "inc_nh3_b": {
        "a": "inc",
        "b": "nh3_b",
        "flux_ref": FLUX_REF,
        "segments": [
            ("side", R_NH3, Z_GAN_LOWER_MAX, R_NH3, Z_GAN_UPPER_MIN, 60),
        ],
    },
    "inc_nh3_c": {
        "a": "inc",
        "b": "nh3_c",
        "flux_ref": FLUX_REF,
        "segments": [
            ("side", R_NH3, Z_GAN_UPPER_MIN, R_NH3, Z_GAN_UPPER_MAX, 100),
        ],
    },
    "inc_nh3_d": {
        "a": "inc",
        "b": "nh3_d",
        "flux_ref": FLUX_REF,
        "segments": [
            ("side", R_NH3, Z_GAN_UPPER_MAX, R_NH3, Z_NH3_TOP, 30),
            ("top", 0.0, Z_NH3_TOP, R_NH3, Z_NH3_TOP, 100),
        ],
    },
    # NH3 slab / NH3 slab (same material: continuity of T and flux)
    "nh3_ab": {
        "a": "nh3_a",
        "b": "nh3_b",
        "flux_ref": FLUX_REF_GAN_INTERFACE,
        "segments": [
            ("plane", R_GAN_LOWER, Z_GAN_LOWER_MAX, R_NH3, Z_GAN_LOWER_MAX, 60),
        ],
    },
    "nh3_bc": {
        "a": "nh3_b",
        "b": "nh3_c",
        "flux_ref": FLUX_REF_GAN_INTERFACE,
        "segments": [
            ("plane", R_GAN_UPPER, Z_GAN_UPPER_MIN, R_NH3, Z_GAN_UPPER_MIN, 40),
        ],
    },
    "nh3_cd": {
        "a": "nh3_c",
        "b": "nh3_d",
        "flux_ref": FLUX_REF_GAN_INTERFACE,
        "segments": [
            ("plane", R_GAN_UPPER, Z_GAN_UPPER_MAX, R_NH3, Z_GAN_UPPER_MAX, 40),
        ],
    },
    # Lower GaN block / NH3
    "gan_lower_nh3_a": {
        "a": "gan_lower",
        "b": "nh3_a",
        "flux_ref": FLUX_REF_GAN_INTERFACE,
        "segments": [
            ("side", R_GAN_LOWER, Z_GAN_LOWER_MIN, R_GAN_LOWER, Z_GAN_LOWER_MAX, 200),
            ("bottom", 0.0, Z_GAN_LOWER_MIN, R_GAN_LOWER, Z_GAN_LOWER_MIN, 120),
        ],
    },
    "gan_lower_nh3_b": {
        "a": "gan_lower",
        "b": "nh3_b",
        "flux_ref": FLUX_REF_GAN_INTERFACE,
        "segments": [
            ("top", 0.0, Z_GAN_LOWER_MAX, R_GAN_LOWER, Z_GAN_LOWER_MAX, 120),
        ],
    },
    # Upper GaN block / NH3
    "gan_upper_nh3_c": {
        "a": "gan_upper",
        "b": "nh3_c",
        "flux_ref": FLUX_REF_GAN_INTERFACE,
        "segments": [
            ("side", R_GAN_UPPER, Z_GAN_UPPER_MIN, R_GAN_UPPER, Z_GAN_UPPER_MAX, 250),
        ],
    },
    "gan_upper_nh3_b": {
        "a": "gan_upper",
        "b": "nh3_b",
        "flux_ref": FLUX_REF_GAN_INTERFACE,
        "segments": [
            ("bottom", 0.0, Z_GAN_UPPER_MIN, R_GAN_UPPER, Z_GAN_UPPER_MIN, 150),
        ],
    },
    "gan_upper_nh3_d": {
        "a": "gan_upper",
        "b": "nh3_d",
        "flux_ref": FLUX_REF_GAN_INTERFACE,
        "segments": [
            ("top", 0.0, Z_GAN_UPPER_MAX, R_GAN_UPPER, Z_GAN_UPPER_MAX, 150),
        ],
    },
}


def reference_segment(r1, z1, r2, z2, label, label_idx=None):
    """Segment2D in reference coordinates from physical end points."""
    kwargs = {
        "pt1": (float(r1 / R_OUT), float(z1 / H_OUT)),
        "pt2": (float(r2 / R_OUT), float(z2 / H_OUT)),
        "label_str": label,
    }

    if label_idx is not None:
        kwargs["label_idx"] = label_idx

    return Segment2D(**kwargs)


# Lower circular Inconel/NH3 interface approximated by short segments.
# Each reference-space segment maps to an actual physical circular-arc segment.
theta_values = np.linspace(-0.5 * np.pi, 0.0, N_ARC_SEGMENTS + 1)
r_arc = R_NH3 * np.cos(theta_values)
z_arc = Z_ARC_CENTER + R_NH3 * np.sin(theta_values)

# INTERFACE_DOMAINS[group]["T" or "q"] = {label: Segment2D}
INTERFACE_DOMAINS = {}
N_BC_COLLOC = {}
BOUNDARIES_DICT = {}

for group, spec in INTERFACE_SPECS.items():
    INTERFACE_DOMAINS[group] = {"T": {}, "q": {}}

    for kind in ("T", "q"):
        group_key = f"{group}_{kind}"

        N_BC_COLLOC[group_key] = {}
        BOUNDARIES_DICT[group_key] = []

        for tag, r1, z1, r2, z2, n_points in spec["segments"]:
            label = f"{group}_{tag}_{kind}"
            segment = reference_segment(r1, z1, r2, z2, label)

            INTERFACE_DOMAINS[group][kind][label] = segment
            N_BC_COLLOC[group_key][label] = n_points
            BOUNDARIES_DICT[group_key].append(label)
            main_domain.add_bc_domain(segment)

        if spec.get("arc", False):
            arc_label = f"{group}_arc_{kind}"

            for i in range(N_ARC_SEGMENTS):
                segment = reference_segment(
                    r_arc[i],
                    z_arc[i],
                    r_arc[i + 1],
                    z_arc[i + 1],
                    arc_label,
                    label_idx=i + 1,
                )

                label = segment.get_label()

                INTERFACE_DOMAINS[group][kind][label] = segment
                N_BC_COLLOC[group_key][label] = ARC_BC_POINTS
                main_domain.add_bc_domain(segment)

            BOUNDARIES_DICT[group_key].append(arc_label)

main_domain.set_boundaries_dict(BOUNDARIES_DICT)


# Geometry predicates in PHYSICAL coordinates

def z_bottom_ammonia(r):
    return Z_ARC_CENTER - jnp.sqrt(
        jnp.maximum(R_NH3**2 - r**2, 0.0)
    )


def in_lower_ammonia(r, z):
    return (
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom_ammonia(r))
        & (z <= Z_ARC_CENTER)
    )

def in_gan_lower(r, z):
    return (
        (r >= 0.0)
        & (r <= R_GAN_LOWER)
        & (z >= Z_GAN_LOWER_MIN)
        & (z <= Z_GAN_LOWER_MAX)
    )


def in_gan_upper(r, z):
    return (
        (r >= 0.0)
        & (r <= R_GAN_UPPER)
        & (z >= Z_GAN_UPPER_MIN)
        & (z <= Z_GAN_UPPER_MAX)
    )


def in_gan(r, z):
    return (
        in_gan_lower(r, z)
        | in_gan_upper(r, z)
    )


def in_nh3_full(r, z):
    return (
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom_ammonia(r))
        & (z <= Z_NH3_TOP)
    )


def in_nh3_without_gan(r, z):
    return (
        in_nh3_full(r, z)
        & (~in_gan(r, z))
    )


# Neural temperature fields
# Network input is physical (r,z); each network has its own pre-processing.
# Networks (index order): Inconel, NH3 slabs a-d, lower GaN block, upper GaN block.

NET_NAMES = ("inc", "nh3_a", "nh3_b", "nh3_c", "nh3_d", "gan_lower", "gan_upper")
NET_INDEX = {name: i for i, name in enumerate(NET_NAMES)}


def scaled_coordinates(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    r_scaled = x[..., 0:1] / R_OUT
    z_scaled = x[..., 1:2] / H_OUT
    rho = r_scaled**2
    eta = z_scaled
    return jnp.concatenate([rho, eta], axis=-1)


def make_block_pre_processing(r_max, z_min, z_max):
    """Map a block to the unit square: rho = (r/r_max)^2, eta in [0, 1]."""

    def pre_processing(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        rho = (x[..., 0:1] / r_max) ** 2
        eta = (x[..., 1:2] - z_min) / (z_max - z_min)
        return jnp.concatenate([rho, eta], axis=-1)

    return pre_processing


def outer_distance_factor(x):
    r = x[..., 0:1]
    z = x[..., 1:2]

    d_side = (R_OUT - r) / R_OUT
    d_bottom = z / H_OUT
    d_top = (H_OUT - z) / H_OUT

    return d_side * d_bottom * d_top


# Block extents of the block-local networks: (r_max, z_min, z_max)
NET_BLOCKS = {
    "nh3_a": (R_NH3, *NH3_SLABS["a"]),
    "nh3_b": (R_NH3, *NH3_SLABS["b"]),
    "nh3_c": (R_NH3, *NH3_SLABS["c"]),
    "nh3_d": (R_NH3, *NH3_SLABS["d"]),
    "gan_lower": (R_GAN_LOWER, Z_GAN_LOWER_MIN, Z_GAN_LOWER_MAX),
    "gan_upper": (R_GAN_UPPER, Z_GAN_UPPER_MIN, Z_GAN_UPPER_MAX),
}

# Correction scale [K] of a block network: outer-law temperature drop over the
# block height, but never below MIN_CORRECTION_SCALE (thin slabs would otherwise
# force large raw outputs).
MIN_CORRECTION_SCALE = 15.0  # K


def block_correction_scale(name):
    _, z_min, z_max = NET_BLOCKS[name]
    return max(abs(T1) * (z_max - z_min), MIN_CORRECTION_SCALE)


NH3_A_CORRECTION_SCALE = block_correction_scale("nh3_a")
NH3_B_CORRECTION_SCALE = block_correction_scale("nh3_b")
NH3_C_CORRECTION_SCALE = block_correction_scale("nh3_c")
NH3_D_CORRECTION_SCALE = block_correction_scale("nh3_d")
GAN_LOWER_CORRECTION_SCALE = block_correction_scale("gan_lower")
GAN_UPPER_CORRECTION_SCALE = block_correction_scale("gan_upper")


# post_processing[i]: field of network i. Takes the RAW scalar network output
# (already evaluated on pre_processed coordinates by ApproximationSpace)
# and the ORIGINAL (unscaled) x, and returns the physical temperature.
#
# post_processing[0]: Inconel field. The outer Dirichlet condition
# T = T_outer is imposed exactly through outer_distance_factor.

def inconel_post_processing(theta_inc, x):
    return T_outer(x) + DELTA_T_REF * outer_distance_factor(x) * theta_inc


# post_processing[1..4]: NH3 slab fields a-d, same principle. The NH3 fields
# do not impose a boundary condition: they start from the outer-law profile
# (identical to the reference profile used in the Inconel + NH3 model) and add a
# free correction.

def nh3_a_post_processing(raw, x):
    return T_outer(x) + NH3_A_CORRECTION_SCALE * raw


def nh3_b_post_processing(raw, x):
    return T_outer(x) + NH3_B_CORRECTION_SCALE * raw


def nh3_c_post_processing(raw, x):
    return T_outer(x) + NH3_C_CORRECTION_SCALE * raw


def nh3_d_post_processing(raw, x):
    return T_outer(x) + NH3_D_CORRECTION_SCALE * raw


# post_processing[5], post_processing[6]: lower and upper GaN block fields.

def gan_lower_post_processing(raw, x):
    return T_outer(x) + GAN_LOWER_CORRECTION_SCALE * raw


def gan_upper_post_processing(raw, x):
    return T_outer(x) + GAN_UPPER_CORRECTION_SCALE * raw


# In network-index order (see NET_NAMES)
POST_PROCESSING = {
    "inc": inconel_post_processing,
    "nh3_a": nh3_a_post_processing,
    "nh3_b": nh3_b_post_processing,
    "nh3_c": nh3_c_post_processing,
    "nh3_d": nh3_d_post_processing,
    "gan_lower": gan_lower_post_processing,
    "gan_upper": gan_upper_post_processing,
}


# evaluate_temperature / T_inc_eval / T_nh3_*_eval / T_gan_*_eval

def evaluate_temperature(space, index, x):
    x = jnp.asarray(x, dtype=jnp.float64)
    temperature = space.create_variables()[index]

    if x.ndim == 2:
        return jax.vmap(lambda x_one: temperature(space, x_one))(x)

    return temperature(space, x)


def T_inc_eval(space, x):
    return evaluate_temperature(space, NET_INDEX["inc"], x)


def T_nh3_a_eval(space, x):
    return evaluate_temperature(space, NET_INDEX["nh3_a"], x)


def T_nh3_b_eval(space, x):
    return evaluate_temperature(space, NET_INDEX["nh3_b"], x)


def T_nh3_c_eval(space, x):
    return evaluate_temperature(space, NET_INDEX["nh3_c"], x)


def T_nh3_d_eval(space, x):
    return evaluate_temperature(space, NET_INDEX["nh3_d"], x)


def T_gan_lower_eval(space, x):
    return evaluate_temperature(space, NET_INDEX["gan_lower"], x)


def T_gan_upper_eval(space, x):
    return evaluate_temperature(space, NET_INDEX["gan_upper"], x)


# Parametric temperature fields
T_INC = ParamScalarFunction(
    {"x": 2},
    T_inc_eval,
    f_type="x",
)

T_NH3_A = ParamScalarFunction(
    {"x": 2},
    T_nh3_a_eval,
    f_type="x",
)

T_NH3_B = ParamScalarFunction(
    {"x": 2},
    T_nh3_b_eval,
    f_type="x",
)

T_NH3_C = ParamScalarFunction(
    {"x": 2},
    T_nh3_c_eval,
    f_type="x",
)

T_NH3_D = ParamScalarFunction(
    {"x": 2},
    T_nh3_d_eval,
    f_type="x",
)

T_GAN_LOWER = ParamScalarFunction(
    {"x": 2},
    T_gan_lower_eval,
    f_type="x",
)

T_GAN_UPPER = ParamScalarFunction(
    {"x": 2},
    T_gan_upper_eval,
    f_type="x",
)

# Lookup by network name (used by the interface residuals and the plots)
FIELDS = {
    "inc": T_INC,
    "nh3_a": T_NH3_A,
    "nh3_b": T_NH3_B,
    "nh3_c": T_NH3_C,
    "nh3_d": T_NH3_D,
    "gan_lower": T_GAN_LOWER,
    "gan_upper": T_GAN_UPPER,
}

TEMPERATURE_EVAL = {
    "inc": T_inc_eval,
    "nh3_a": T_nh3_a_eval,
    "nh3_b": T_nh3_b_eval,
    "nh3_c": T_nh3_c_eval,
    "nh3_d": T_nh3_d_eval,
    "gan_lower": T_gan_lower_eval,
    "gan_upper": T_gan_upper_eval,
}


# Conductivities as ParamScalarFunction of the field
def make_conductivity(value_fn):
    def conductivity(T):
        return ParamScalarFunction(
            T.dims,
            lambda space, x: value_fn(T(space, x)),
            f_type=T.f_type,
        )

    return conductivity


k_inc = make_conductivity(k_inc_value)
k_nh3 = make_conductivity(k_nh3_value)
k_gan = make_conductivity(k_gan_value)


def conductivity_of(name):
    if name == "inc":
        return k_inc
    if name.startswith("nh3"):
        return k_nh3
    return k_gan


def conductivity_value_of(name):
    if name == "inc":
        return k_inc_value
    if name.startswith("nh3"):
        return k_nh3_value
    return k_gan_value


# r-weighted axisymmetric steady-conduction operator
#   [ r d(q_r)/dr + q_r + r d(q_z)/dz ] / (ref * max(r, R_WEIGHT_FLOOR) / L_REF),
#   q = k(T) grad T   (= true residual / (ref / L) for r > R_WEIGHT_FLOOR)

def axisymmetric_conduction(T, conductivity, space, x, ref=PDE_R_WEIGHTED_REF):
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

    return raw_residual / (ref * jnp.maximum(r, R_WEIGHT_FLOOR) / L_REF)


# Interior residuals

class InconelResidual(InteriorResidual):
    def __init__(self, domain, lower_box=False):
        super().__init__(domain=domain, size=1, model_type="x")
        self.lower_box = lower_box

    def construct_residual(self, *vars):
        T_inc = vars[NET_INDEX["inc"]]

        def residual(space, x):
            value = axisymmetric_conduction(T_inc, k_inc, space, x)

            if not self.lower_box:
                return value

            r = x[..., 0:1]
            z = x[..., 1:2]

            return jnp.where(in_lower_ammonia(r, z), 0.0, value)

        return ParamScalarFunction(T_inc.dims, residual, f_type=T_inc.f_type)


class AmmoniaResidual(InteriorResidual):
    """PDE residual of one NH3 slab network."""

    def __init__(self, domain, slab):
        super().__init__(domain=domain, size=1, model_type="x")
        self.net_index = NET_INDEX[f"nh3_{slab}"]  # slab network a-d

    def construct_residual(self, *vars):
        T_nh3 = vars[self.net_index]

        def residual(space, x):
            value = axisymmetric_conduction(T_nh3, k_nh3, space, x)

            r = x[..., 0:1]
            z = x[..., 1:2]

            # True NH3 only: arc region and GaN blocks are excluded.
            return jnp.where(in_nh3_without_gan(r, z), value, 0.0)

        return ParamScalarFunction(T_nh3.dims, residual, f_type=T_nh3.f_type)


class GaNResidual(InteriorResidual):
    def __init__(self, domain, block):
        super().__init__(domain=domain, size=1, model_type="x")

        if block not in ("lower", "upper"):
            raise ValueError(f"Unknown GaN block: {block}")

        self.net_index = NET_INDEX[f"gan_{block}"]
        self.mask_fn = in_gan_lower if block == "lower" else in_gan_upper

    def construct_residual(self, *vars):
        T_gan = vars[self.net_index]

        def residual(space, x):
            value = axisymmetric_conduction(
                T_gan, k_gan, space, x, ref=FLUX_REF_GAN
            )

            r = x[..., 0:1]
            z = x[..., 1:2]

            return jnp.where(self.mask_fn(r, z), value, 0.0)

        return ParamScalarFunction(T_gan.dims, residual, f_type=T_gan.f_type)


# Interface residuals
# Fields are evaluated as F(space, x) with the module-level ParamScalarFunctions
# in FIELDS; only the normal n comes from the boundary sampler.

def normal_derivative(T, space, x, n):
    """dT/dn = n_r dT/dr + n_z dT/dz, evaluated at (space, x)."""
    dT_dr = T.partial_derivative("x", 0)
    dT_dz = T.partial_derivative("x", 1)

    return n[..., 0:1] * dT_dr(space, x) + n[..., 1:2] * dT_dz(space, x)


def temperature_jump_residual(T_a, T_b):
    def residual(space, x, n):
        return (T_a(space, x) - T_b(space, x)) / INTERFACE_T_REF

    return ParamScalarFunction({"x": 2}, residual, f_type="x_n")


def flux_jump_residual(T_a, k_a, T_b, k_b, flux_ref):
    """[k_a(T_a) dT_a/dn - k_b(T_b) dT_b/dn] / flux_ref."""
    k_a_field = k_a(T_a)
    k_b_field = k_b(T_b)

    def residual(space, x, n):
        flux_a = k_a_field(space, x) * normal_derivative(T_a, space, x, n)
        flux_b = k_b_field(space, x) * normal_derivative(T_b, space, x, n)
        return (flux_a - flux_b) / flux_ref

    return ParamScalarFunction({"x": 2}, residual, f_type="x_n")


class InterfaceTemperatureResidual(BoundaryResidual):
    """T_a - T_b = 0 on all segments of one interface group."""

    def __init__(self, group):
        super().__init__(
            domain=INTERFACE_DOMAINS[group]["T"], size=1, model_type="x"
        )
        self.group = group

    def construct_residual(self, *vars):
        spec = INTERFACE_SPECS[self.group]
        return temperature_jump_residual(FIELDS[spec["a"]], FIELDS[spec["b"]])


class InterfaceFluxResidual(BoundaryResidual):
    """k_a dT_a/dn - k_b dT_b/dn = 0 on all segments of one interface group."""

    def __init__(self, group):
        super().__init__(
            domain=INTERFACE_DOMAINS[group]["q"], size=1, model_type="x"
        )
        self.group = group

    def construct_residual(self, *vars):
        spec = INTERFACE_SPECS[self.group]
        return flux_jump_residual(
            FIELDS[spec["a"]],
            conductivity_of(spec["a"]),
            FIELDS[spec["b"]],
            conductivity_of(spec["b"]),
            spec["flux_ref"],
        )


# Physical model

class AutoclaveConductionModel(AbstractPhysicalModel):
    def __init__(self):
        super().__init__(main_domain=main_domain)

        self.physical_residuals = {
            "inc_bottom": InconelResidual(inc_bottom),
            "inc_side": InconelResidual(inc_side),
            "inc_top": InconelResidual(inc_top),
            "inc_lower": InconelResidual(inc_lower, lower_box=True),
            "gan_lower": GaNResidual(gan_lower, block="lower"),
            "gan_upper": GaNResidual(gan_upper, block="upper"),
        }

        for tile_key, tile_domain in nh3_domains.items():
            self.physical_residuals[tile_key] = AmmoniaResidual(
                tile_domain, NH3_TILES[tile_key][0]
            )

        for group in INTERFACE_SPECS:
            self.physical_residuals[f"{group}_T"] = InterfaceTemperatureResidual(
                group
            )
            self.physical_residuals[f"{group}_q"] = InterfaceFluxResidual(group)

        self.data_residuals = {}


model = AutoclaveConductionModel()


# Sampler

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


# Neural network space
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
        return MLP(
            in_size=in_size,
            out_size=out_size,
            hidden_sizes=hidden_sizes,
            key=key,
        )


OUTPUT_LAYER_INIT_SCALE = 1.0e-3


def start_on_outer_law(network):
    """
    Shrink the output layer so the network starts (almost) exactly on the outer
    law: every field is T_outer + scale * raw with raw ~ 0.  Independent random
    networks would otherwise start with different offsets of several K, and the
    interface-continuity terms would dominate the initial loss.
    """
    last = network.layers[-1]
    last.weight = OUTPUT_LAYER_INIT_SCALE * last.weight

    if last.bias is not None:
        last.bias = jnp.zeros_like(last.bias)

    return network


def build_space(seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, len(NET_NAMES))

    list_models = []
    pre_processing = []
    post_processing = []

    for name, net_key in zip(NET_NAMES, keys):
        if name == "inc":
            hidden = HIDDEN
            pre_processing.append(scaled_coordinates)
            post_processing.append(inconel_post_processing)
        else:
            hidden = HIDDEN_NH3 if name.startswith("nh3") else HIDDEN
            pre_processing.append(make_block_pre_processing(*NET_BLOCKS[name]))
            post_processing.append(POST_PROCESSING[name])

        network = start_on_outer_law(build_mlp(2, 1, hidden, net_key))
        list_models.append((network, "scalar", None))

    return ApproximationSpace(
        {"x": 2},
        list_models,
        model_type="x",
        pre_processing=pre_processing,
        post_processing=post_processing,
    )


# Loss weights
# Interface weights are deliberately lower than in the first version (1500/600):
# with the r-weighted, per-material normalized residuals they no longer need to
# dominate the interior PDE terms.

W_INTERFACE_T = 300.0
W_INTERFACE_Q = 150.0

W_COMMON = {
    "inc_bottom": (150.0,),
    "inc_side": (350.0,),
    "inc_top": (200.0,),
    "inc_lower": (350.0,),
    "gan_lower": (350.0,),
    "gan_upper": (350.0,),
    "nh3_a_arc": (350.0,),
    "nh3_a_side": (350.0,),
    "nh3_b": (350.0,),
    "nh3_c_side": (350.0,),
    "nh3_d": (350.0,),
}

for _group in INTERFACE_SPECS:
    W_COMMON[f"{_group}_T"] = (W_INTERFACE_T,)
    W_COMMON[f"{_group}_q"] = (W_INTERFACE_Q,)


# Plotting and post-training diagnostics

MATERIAL_LABELS = {
    "inc": "Inconel",
    "nh3_a": "NH3 a",
    "nh3_b": "NH3 b",
    "nh3_c": "NH3 c",
    "nh3_d": "NH3 d",
    "gan_lower": "GaN lower",
    "gan_upper": "GaN upper",
}

MATERIAL_FIELDS = {name: FIELDS[name] for name in MATERIAL_LABELS}


def z_bottom_ammonia_numpy(r):
    r = np.asarray(r, dtype=np.float64)
    return Z_ARC_CENTER - np.sqrt(np.maximum(R_NH3**2 - r**2, 0.0))


def in_ammonia_numpy(r, z):
    """Mask of the whole physical ammonia cavity (GaN blocks included)."""
    r = np.asarray(r)
    z = np.asarray(z)

    return (
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom_ammonia_numpy(r))
        & (z <= Z_NH3_TOP)
    )


def in_gan_lower_numpy(r, z):
    r = np.asarray(r)
    z = np.asarray(z)
    return (
        (r >= 0.0)
        & (r <= R_GAN_LOWER)
        & (z >= Z_GAN_LOWER_MIN)
        & (z <= Z_GAN_LOWER_MAX)
    )


def in_gan_upper_numpy(r, z):
    r = np.asarray(r)
    z = np.asarray(z)
    return (
        (r >= 0.0)
        & (r <= R_GAN_UPPER)
        & (z >= Z_GAN_UPPER_MIN)
        & (z <= Z_GAN_UPPER_MAX)
    )


def material_masks_numpy(r, z):
    """
    Boolean masks per network, as a dict keyed like MATERIAL_LABELS.

    GaN takes precedence over NH3 because it occupies part of the cavity; the
    NH3 cavity is split into the z-slabs a-d (see NH3_SLABS).
    """
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    gan_lower_mask = in_gan_lower_numpy(r, z)
    gan_upper_mask = in_gan_upper_numpy(r, z)
    nh3_mask = in_ammonia_numpy(r, z) & ~(gan_lower_mask | gan_upper_mask)
    inc_mask = ~(gan_lower_mask | gan_upper_mask | nh3_mask)

    return {
        "inc": inc_mask,
        "nh3_a": nh3_mask & (z <= Z_GAN_LOWER_MAX),
        "nh3_b": nh3_mask & (z > Z_GAN_LOWER_MAX) & (z <= Z_GAN_UPPER_MIN),
        "nh3_c": nh3_mask & (z > Z_GAN_UPPER_MIN) & (z <= Z_GAN_UPPER_MAX),
        "nh3_d": nh3_mask & (z > Z_GAN_UPPER_MAX),
        "gan_lower": gan_lower_mask,
        "gan_upper": gan_upper_mask,
    }


def evaluate_field(space, points, material):
    """Evaluate one trained PINN temperature field [K] at physical points (N, 2)."""
    if material not in MATERIAL_FIELDS:
        raise ValueError(
            f"Unknown material '{material}'. Use one of {list(MATERIAL_FIELDS)}."
        )

    x = jnp.asarray(points, dtype=jnp.float64)

    return np.asarray(MATERIAL_FIELDS[material](space, x)).reshape(-1)


def evaluate_pinn_on_points(space, r, z):
    """Composite PINN temperature at (r, z); returns (T, material_masks dict)."""
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    points = np.column_stack([r, z])

    masks = material_masks_numpy(r, z)

    temperature = np.empty(len(r), dtype=np.float64)

    for material, mask in masks.items():
        if np.any(mask):
            temperature[mask] = evaluate_field(space, points[mask], material)

    return temperature, masks


def material_label_array(masks):
    labels = np.empty(len(next(iter(masks.values()))), dtype=object)

    for material, mask in masks.items():
        labels[mask] = MATERIAL_LABELS[material]

    return labels


def draw_geometry(ax):
    theta = np.linspace(-0.5 * np.pi, 0.0, 200)

    for r_max, z_min, z_max in (
        (R_GAN_LOWER, Z_GAN_LOWER_MIN, Z_GAN_LOWER_MAX),
        (R_GAN_UPPER, Z_GAN_UPPER_MIN, Z_GAN_UPPER_MAX),
    ):
        ax.plot(
            np.array([0.0, r_max, r_max, 0.0]) * 1.0e3,
            np.array([z_min, z_min, z_max, z_max]) * 1.0e3,
            "k",
            lw=1.6,
        )

    ax.plot(
        np.full(2, R_NH3 * 1.0e3),
        np.array([Z_ARC_CENTER, Z_NH3_TOP]) * 1.0e3,
        "k",
        lw=1.2,
    )
    ax.plot(
        np.array([0.0, R_NH3]) * 1.0e3,
        np.full(2, Z_NH3_TOP * 1.0e3),
        "k",
        lw=1.2,
    )
    ax.plot(
        R_NH3 * np.cos(theta) * 1.0e3,
        (Z_ARC_CENTER + R_NH3 * np.sin(theta)) * 1.0e3,
        "k",
        lw=1.4,
    )
    ax.plot(
        np.full(2, R_OUT * 1.0e3),
        np.array([0.0, H_OUT]) * 1.0e3,
        "k",
        lw=1.2,
    )
    ax.plot(np.array([0.0, R_OUT]) * 1.0e3, np.zeros(2), "k", lw=1.2)
    ax.plot(np.array([0.0, R_OUT]) * 1.0e3, np.full(2, H_OUT * 1.0e3), "k", lw=1.2)

    ax.set_aspect("equal")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")


# FVM data access

def load_fvm_table(path, sheet_name="Sheet1"):
    """
    Read a CryS-MAS table (xlsx or csv) with columns x (r [m]), y (z [m]),
    Temperature [K] and optional T0, T1.  If T0/T1 are present, only the rows
    matching the PINN outer law (T0_K, T1) are kept.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"FVM file does not exist:\n{path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    required_columns = {"x", "y", "Temperature"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The FVM file is missing required columns: "
            f"{sorted(missing_columns)}. Available columns are: {list(df.columns)}"
        )

    df = df.dropna(subset=["x", "y", "Temperature"]).copy()

    if {"T0", "T1"}.issubset(df.columns):
        keep = np.isclose(df["T0"].to_numpy(dtype=np.float64), T0_K) & np.isclose(
            df["T1"].to_numpy(dtype=np.float64), T1
        )

        if not np.any(keep):
            raise ValueError(
                f"No FVM rows match the PINN outer law T0={T0_K}, T1={T1}."
            )

        df = df.loc[keep].copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after cleaning.")

    return df


def fvm_arrays(df):
    return (
        df["x"].to_numpy(dtype=np.float64),
        df["y"].to_numpy(dtype=np.float64),
        df["Temperature"].to_numpy(dtype=np.float64),
    )


def error_statistics(error):
    return (
        float(np.sqrt(np.mean(error**2))),
        float(np.mean(np.abs(error))),
        float(np.max(np.abs(error))),
    )


def print_error_statistics(title, error, masks=None, unit="K"):
    rmse, mae, max_abs = error_statistics(error)

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Number of FVM points: {len(error)}")
    print(f"RMSE [{unit}]: {rmse:.8e}")
    print(f"MAE [{unit}]: {mae:.8e}")
    print(f"Maximum absolute error [{unit}]: {max_abs:.8e}")

    if masks is not None:
        for material, mask in masks.items():
            if np.any(mask):
                m_rmse, m_mae, m_max = error_statistics(error[mask])
                print(
                    f"  {MATERIAL_LABELS[material]:<10s} "
                    f"n={np.count_nonzero(mask):5d}  "
                    f"RMSE={m_rmse:.4e}  MAE={m_mae:.4e}  Max={m_max:.4e} {unit}"
                )


# FVM data in 2D with the domain (network) names

def plot_fvm_domain_map(
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_data_domains_2d.png",
):
    """
    Plot the FVM temperature nodes of a 2D table (columns x = r [m], y = z [m],
    Temperature [K]) together with the domain outlines and the name of every
    domain: Inconel, the four NH3 slabs a-d, and the two GaN blocks.  The
    dashed lines mark the planes where the NH3 cavity is split into slabs.
    """
    df = load_fvm_table(excel_path, sheet_name)
    r_fvm, z_fvm, T_fvm = fvm_arrays(df)
    masks = material_masks_numpy(r_fvm, z_fvm)

    print("\n" + "=" * 72)
    print("FVM DATA: NODES PER DOMAIN")
    print("=" * 72)
    print(f"File: {Path(excel_path).resolve()}")
    print(f"Total nodes: {len(T_fvm)}, radii: {np.unique(np.round(r_fvm, 12)).size}")
    print(f"Temperature range: {T_fvm.min():.2f} .. {T_fvm.max():.2f} K")

    for material, mask in masks.items():
        if np.any(mask):
            print(
                f"  {MATERIAL_LABELS[material]:<10s} n={np.count_nonzero(mask):5d}  "
                f"T = {T_fvm[mask].min():.2f} .. {T_fvm[mask].max():.2f} K"
            )

    fig, ax = plt.subplots(figsize=(9.0, 13.0))

    scatter = ax.scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=T_fvm,
        cmap="turbo",
        s=16,
        marker="s",
        linewidths=0.0,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02).set_label(
        "FVM temperature [K]"
    )

    draw_geometry(ax)

    # Planes where the NH3 cavity is split into slabs (outside the GaN blocks)
    for z_split, r_start in (
        (Z_GAN_LOWER_MAX, R_GAN_LOWER),
        (Z_GAN_UPPER_MIN, R_GAN_UPPER),
        (Z_GAN_UPPER_MAX, R_GAN_UPPER),
    ):
        ax.plot(
            [r_start * 1.0e3, R_NH3 * 1.0e3],
            [z_split * 1.0e3, z_split * 1.0e3],
            "k--",
            lw=1.0,
        )

    label_x = 42.0  # mm, to the right of the vessel (R_OUT = 35 mm)

    # (text, point inside the domain [r, z] in mm, label height z in mm)
    annotations = (
        ("GaN upper\n(z 175-325 mm)", (2.5, 250.0), 265.0),
        ("NH3 slab d\n(z 325-335 mm)", (6.0, 330.0), 330.0),
        ("NH3 slab c\n(z 175-325 mm)", (8.5, 200.0), 200.0),
        ("NH3 slab b\n(z 120-175 mm)", (7.0, 147.0), 147.0),
        ("GaN lower\n(z 40-120 mm)", (2.5, 90.0), 95.0),
        ("NH3 slab a\n(z 23-120 mm)", (9.0, 55.0), 55.0),
    )

    for text, xy, label_z in annotations:
        ax.annotate(
            text,
            xy=xy,
            xytext=(label_x, label_z),
            fontsize=9,
            va="center",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.4", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "0.15", "lw": 1.0},
        )

    for text, xy in (
        ("Inconel\n(bottom, z < 23 mm)", (24.0, 12.0)),
        ("Inconel\n(sidewall, r > 12 mm)", (24.0, 180.0)),
        ("Inconel\n(top, z > 335 mm)", (24.0, 355.0)),
    ):
        ax.text(
            xy[0],
            xy[1],
            text,
            fontsize=9,
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.4", "alpha": 0.85},
        )

    ax.set_xlim(-3.0, 70.0)
    ax.set_ylim(-5.0, H_OUT * 1.0e3 + 5.0)
    ax.set_title(
        "FVM temperature data and domain names\n"
        f"T0 = {T0_K:.2f} K, T1 = {T1:.1f} K/m, {len(T_fvm)} nodes"
    )

    fig.tight_layout()
    figure_path = SAVE_DIR / output_name
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved FVM domain map in:\n{figure_path.resolve()}")


# Temperature field plots

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

    temperature, _ = evaluate_pinn_on_points(space, rr.ravel(), zz.ravel())
    temperature = temperature.reshape(rr.shape)

    title = "Axisymmetric Inconel--NH3--GaN conduction"

    # Plot 1: filled contour
    figure, ax = plt.subplots(figsize=(5.8, 10.0))
    contour = ax.contourf(
        rr * 1.0e3, zz * 1.0e3, temperature, levels=100, cmap="turbo"
    )
    figure.colorbar(contour, ax=ax).set_label("Temperature [K]")
    draw_geometry(ax)
    ax.set_title(f"{title}\nT0 = {T0_K:.1f} K, T1 = {T1:.1f} K/m")
    figure.tight_layout()
    figure_path = SAVE_DIR / "temperature_field.png"
    figure.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print("Saved PINN temperature plot in:")
    print(figure_path.resolve())

    # Plot 2: filled contour + labeled isotherms (interface kinks are visible as
    # bent contour lines right at the black material boundaries)
    figure2, ax2 = plt.subplots(figsize=(5.8, 10.0))
    contour2 = ax2.contourf(
        rr * 1.0e3, zz * 1.0e3, temperature, levels=100, cmap="turbo"
    )
    figure2.colorbar(contour2, ax=ax2).set_label("Temperature [K]")

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
        f"{title}\nIsotherms every {(line_levels[1] - line_levels[0]):.2f} K"
    )
    figure2.tight_layout()
    figure2_path = SAVE_DIR / "temperature_field_isotherms.png"
    figure2.savefig(figure2_path, dpi=250, bbox_inches="tight")
    plt.close(figure2)

    print("Saved isotherm plot in:")
    print(figure2_path.resolve())

    # Plot 3: dense PINN-vs-FVM error field (needs a full 2D FVM table)
    if fvm_excel_path is not None:
        df_fvm = load_fvm_table(fvm_excel_path, fvm_sheet_name)
        r_fvm, z_fvm, T_fvm = fvm_arrays(df_fvm)

        if np.unique(np.round(r_fvm, 12)).size < 2:
            print(
                "FVM table holds a single vertical line; "
                "skipping the 2D PINN-vs-FVM error field."
            )
            return

        T_pinn_fvm, _ = evaluate_pinn_on_points(space, r_fvm, z_fvm)
        error_fvm = T_pinn_fvm - T_fvm
        limit = np.max(np.abs(error_fvm))
        rmse, mae, _ = error_statistics(error_fvm)

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
        figure3.colorbar(scatter, ax=ax3).set_label("PINN - FVM [K]")
        draw_geometry(ax3)
        ax3.set_title(
            "PINN error vs. FVM reference\n"
            f"RMSE = {rmse:.3f} K, MAE = {mae:.3f} K"
        )
        figure3.tight_layout()
        figure3_path = SAVE_DIR / "temperature_field_vs_fvm_error.png"
        figure3.savefig(figure3_path, dpi=250, bbox_inches="tight")
        plt.close(figure3)

        print("Saved PINN-vs-FVM error field in:")
        print(figure3_path.resolve())


# Interior PDE residual maps

def make_axisymmetric_residual(temperature_function, conductivity_function, ref):
    """r-weighted residual of one field, evaluated with jax.grad at many points."""

    def residual_on_points(space, x):
        x = jnp.asarray(x, dtype=jnp.float64)

        def temperature_at_one_point(x_one):
            return jnp.ravel(temperature_function(space, x_one))[0]

        def radial_flux(x_one):
            dT_dr = jax.grad(temperature_at_one_point)(x_one)[0]
            return conductivity_function(temperature_at_one_point(x_one)) * dT_dr

        def axial_flux(x_one):
            dT_dz = jax.grad(temperature_at_one_point)(x_one)[1]
            return conductivity_function(temperature_at_one_point(x_one)) * dT_dz

        def residual_at_one_point(x_one):
            r = x_one[0]
            raw_residual = (
                r * jax.grad(radial_flux)(x_one)[0]
                + radial_flux(x_one)
                + r * jax.grad(axial_flux)(x_one)[1]
            )
            return raw_residual / (ref * jnp.maximum(r, R_WEIGHT_FLOOR) / L_REF)

        return jax.vmap(residual_at_one_point)(x)

    return residual_on_points


CONDUCTION_RESIDUALS = {
    name: make_axisymmetric_residual(
        TEMPERATURE_EVAL[name],
        conductivity_value_of(name),
        FLUX_REF_GAN if name.startswith("gan") else PDE_R_WEIGHTED_REF,
    )
    for name in MATERIAL_LABELS
}


def plot_all_interior_residuals(space, n_r=151, n_z=251):
    # (name, title, material, r_min, r_max, z_min, z_max)
    residual_regions = (
        ("inc_bottom", "Inconel: bottom", "inc", 0.0, R_OUT, 0.0, Z_NH3_MIN),
        ("inc_lower", "Inconel: lower region", "inc", 0.0, R_OUT, Z_NH3_MIN, Z_ARC_CENTER),
        ("inc_side", "Inconel: side region", "inc", R_NH3, R_OUT, Z_ARC_CENTER, Z_NH3_TOP),
        ("inc_top", "Inconel: top", "inc", 0.0, R_OUT, Z_NH3_TOP, H_OUT),
        ("nh3_a", "NH3 a: bottom .. lower GaN top", "nh3_a", 0.0, R_NH3, *NH3_SLABS["a"]),
        ("nh3_b", "NH3 b: between GaN blocks", "nh3_b", 0.0, R_NH3, *NH3_SLABS["b"]),
        ("nh3_c", "NH3 c: alongside upper GaN", "nh3_c", 0.0, R_NH3, *NH3_SLABS["c"]),
        ("nh3_d", "NH3 d: cap", "nh3_d", 0.0, R_NH3, *NH3_SLABS["d"]),
        ("gan_lower", "GaN: lower block", "gan_lower", 0.0, R_GAN_LOWER, Z_GAN_LOWER_MIN, Z_GAN_LOWER_MAX),
        ("gan_upper", "GaN: upper block", "gan_upper", 0.0, R_GAN_UPPER, Z_GAN_UPPER_MIN, Z_GAN_UPPER_MAX),
    )

    plot_data = []

    for name, title, material, r_min, r_max, z_min, z_max in residual_regions:
        residual_function = CONDUCTION_RESIDUALS[material]

        r_plot_min = max(r_min, 10.0 * EPS_AXIS)

        r_values = np.linspace(r_plot_min, r_max, n_r)
        z_values = np.linspace(z_min, z_max, n_z)
        r_grid, z_grid = np.meshgrid(r_values, z_values, indexing="xy")

        x_eval = jnp.column_stack([r_grid.reshape(-1), z_grid.reshape(-1)])

        residual_values = np.asarray(residual_function(space, x_eval)).reshape(
            n_z, n_r
        )

        mask = material_masks_numpy(r_grid, z_grid)[material]
        absolute_values = np.abs(np.where(mask, residual_values, np.nan))

        positive_values = absolute_values[
            np.isfinite(absolute_values) & (absolute_values > 0.0)
        ]

        if positive_values.size == 0:
            raise ValueError(f"No positive residual values remain for {name}.")

        print(
            f"{name}: "
            f"min |R| = {np.min(positive_values):.3e}, "
            f"max |R| = {np.max(positive_values):.3e}, "
            f"mean |R| = {np.mean(positive_values):.3e}"
        )

        plot_data.append((name, title, absolute_values, r_plot_min, r_max, z_min, z_max))

    all_positive_values = np.concatenate(
        [v[np.isfinite(v) & (v > 0.0)] for _, _, v, _, _, _, _ in plot_data]
    )

    common_vmin = np.min(all_positive_values)
    common_vmax = np.max(all_positive_values)

    figure, axes = plt.subplots(2, 5, figsize=(24.0, 10.0), constrained_layout=True)

    image = None

    for ax, (name, title, absolute_values, r_min, r_max, z_min, z_max) in zip(
        axes.ravel(), plot_data
    ):
        plot_values = np.where(absolute_values > 0.0, absolute_values, np.nan)

        image = ax.pcolormesh(
            np.linspace(r_min * 1.0e3, r_max * 1.0e3, n_r),
            np.linspace(z_min * 1.0e3, z_max * 1.0e3, n_z),
            plot_values,
            shading="auto",
            cmap="magma",
            norm=LogNorm(vmin=common_vmin, vmax=common_vmax),
        )

        ax.set_title(f"{title}\n" r"$|\mathcal{R}_{\mathrm{PDE}}|$")
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("z [mm]")

    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88)
    colorbar.set_label(
        r"Absolute scaled PDE residual $|\mathcal{R}_{\mathrm{PDE}}|$"
    )

    output_path = SAVE_DIR / "all_interior_residual_maps.png"
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print("Saved interior residual maps in:")
    print(output_path.resolve())


# One-dimensional FVM versus PINN line comparison

def extract_fvm_vertical_line(
    full_excel_path,
    target_r=0.004,
    sheet_name="Sheet1",
    output_name="gan_line_data_extracted.xlsx",
):
    """Extract the FVM nodes at the radius closest to target_r into a line file."""
    df = load_fvm_table(full_excel_path, sheet_name)

    available_r = np.unique(df["x"].to_numpy(dtype=np.float64))
    selected_r = float(available_r[np.argmin(np.abs(available_r - target_r))])

    line_df = (
        df.loc[np.isclose(df["x"].to_numpy(dtype=np.float64), selected_r)]
        .sort_values("y")
        .reset_index(drop=True)
    )

    output_path = (SAVE_DIR / output_name).resolve()
    line_df.to_excel(output_path, index=False, sheet_name=sheet_name)

    print(f"Requested radius [m]: {target_r:.8f}")
    print(f"Selected radius [m]:  {selected_r:.8f}")
    print(f"Extracted {len(line_df)} FVM nodes.")
    print(f"Saved extracted line data in:\n{output_path}")

    return output_path


def _line_interface_positions(r_probe):
    """z positions [m] of material interfaces crossed by the vertical line at r_probe."""
    positions = []

    if 0.0 <= r_probe <= R_NH3:
        positions += [float(z_bottom_ammonia_numpy(np.array([r_probe]))[0]), Z_NH3_TOP]
    if 0.0 <= r_probe <= R_GAN_LOWER:
        positions += [Z_GAN_LOWER_MIN, Z_GAN_LOWER_MAX]
    if 0.0 <= r_probe <= R_GAN_UPPER:
        positions += [Z_GAN_UPPER_MIN, Z_GAN_UPPER_MAX]

    return positions


def _load_line(space, excel_path, sheet_name):
    df = load_fvm_table(excel_path, sheet_name)
    r_fvm, z_fvm, T_fvm = fvm_arrays(df)

    if not np.allclose(r_fvm, r_fvm[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(
            "A 1D line comparison expects a single radius "
            f"but found {np.unique(r_fvm).size}. Extract one line first."
        )

    T_pinn, masks = evaluate_pinn_on_points(space, r_fvm, z_fvm)

    return r_fvm, z_fvm, T_fvm, T_pinn, masks


def plot_fvm_vs_pinn(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_inconel_nh3_gan_line_comparison.png",
):
    """
    Compare FVM temperatures with the Inconel + NH3 + GaN PINN prediction along
    one vertical line (Excel columns: x = r [m], y = z [m], Temperature [K]).
    """
    r_fvm, z_fvm, T_fvm, T_pinn, masks = _load_line(space, excel_path, sheet_name)

    error_K = T_pinn - T_fvm
    rmse_K, mae_K, max_abs_error_K = error_statistics(error_K)

    figure_path = (SAVE_DIR / output_name).resolve()
    comparison_csv = (
        SAVE_DIR / "fvm_vs_pinn_inconel_nh3_gan_line_comparison.csv"
    ).resolve()

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "pinn_minus_fvm_K": error_K,
            "absolute_error_K": np.abs(error_K),
            "material": material_label_array(masks),
        }
    )
    comparison_df.to_csv(str(comparison_csv), index=False)

    print_error_statistics(
        "FVM VS PINN LINE COMPARISON: INCONEL + NH3 + GaN", error_K, masks
    )
    print(f"Probe radius [m]: {r_fvm[0]:.8f}")

    order = np.argsort(z_fvm)
    z_sorted = z_fvm[order] * 1.0e3

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(z_sorted, T_fvm[order], "o", ms=4.0, color="black", label="FVM")
    axes[0].plot(z_sorted, T_pinn[order], "-", lw=2.0, color="tab:red", label="PINN")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title(f"FVM vs PINN temperature: r = {r_fvm[0] * 1.0e3:.3f} mm")
    axes[0].grid(True, alpha=0.30)
    axes[0].legend()
    axes[0].text(
        0.02,
        0.03,
        (
            f"RMSE = {rmse_K:.3e} K\n"
            f"MAE = {mae_K:.3e} K\n"
            f"Max |error| = {max_abs_error_K:.3e} K"
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

    for z_interface in _line_interface_positions(float(r_fvm[0])):
        for ax in axes:
            ax.axvline(
                z_interface * 1.0e3, color="0.35", lw=1.0, ls="--", alpha=0.80
            )

    axes[1].axhline(0.0, color="black", lw=1.0)
    axes[1].plot(z_sorted, error_K[order], "-o", ms=3.0, lw=1.2, color="tab:blue")
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("PINN - FVM [K]")
    axes[1].grid(True, alpha=0.30)

    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved FVM/PINN line plot in:\n{figure_path}")
    print(f"Saved FVM/PINN line data in:\n{comparison_csv}")

    return comparison_df


def plot_fvm_vs_pinn_relative_error(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_inconel_nh3_gan_line_relative_error.png",
):
    """Relative error (PINN - FVM) / FVM [%] along one vertical line."""
    r_fvm, z_fvm, T_fvm, T_pinn, masks = _load_line(space, excel_path, sheet_name)

    if np.any(T_fvm == 0.0):
        raise ValueError("FVM temperature contains zeros; relative error undefined.")

    relative_error_pct = (T_pinn - T_fvm) / T_fvm * 100.0
    rmse_pct, mae_pct, max_abs_pct = error_statistics(relative_error_pct)

    figure_path = (SAVE_DIR / output_name).resolve()
    comparison_csv = (
        SAVE_DIR / "fvm_vs_pinn_inconel_nh3_gan_line_relative_error.csv"
    ).resolve()

    pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "relative_error_percent": relative_error_pct,
            "material": material_label_array(masks),
        }
    ).to_csv(str(comparison_csv), index=False)

    print_error_statistics(
        "RELATIVE ERROR LINE COMPARISON: INCONEL + NH3 + GaN",
        relative_error_pct,
        masks,
        unit="%",
    )

    order = np.argsort(z_fvm)
    z_sorted = z_fvm[order] * 1.0e3

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axhline(0.0, color="black", lw=1.0)
    ax.plot(z_sorted, relative_error_pct[order], "-o", ms=3.0, lw=1.4, color="tab:blue")

    for z_interface in _line_interface_positions(float(r_fvm[0])):
        ax.axvline(z_interface * 1.0e3, color="0.35", lw=1.0, ls="--", alpha=0.80)

    ax.set_xlabel("z [mm]")
    ax.set_ylabel(r"(PINN - FVM) / FVM [%]")
    ax.set_title(f"Relative error: r = {r_fvm[0] * 1.0e3:.3f} mm")
    ax.grid(True, alpha=0.30)
    ax.text(
        0.02,
        0.97,
        (
            f"RMSE = {rmse_pct:.3e} %\n"
            f"MAE = {mae_pct:.3e} %\n"
            f"Max |error| = {max_abs_pct:.3e} %"
        ),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.90,
        },
    )

    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved relative-error line plot in:\n{figure_path}")
    print(f"Saved relative-error line data in:\n{comparison_csv}")


# Loss history

def plot_loss_history(losses, output_name="loss_history.png"):
    history_dict = losses.losses_history

    if not isinstance(history_dict, dict):
        raise TypeError(
            "Expected losses.losses_history to be a dictionary, "
            f"but got {type(history_dict)}."
        )

    if "total" not in history_dict:
        raise KeyError(
            "No 'total' entry found in losses.losses_history. "
            f"Available keys: {list(history_dict.keys())}"
        )

    total_loss = np.asarray(history_dict["total"], dtype=np.float64).reshape(-1)
    n_epochs = total_loss.size
    epoch = np.arange(1, n_epochs + 1)

    def get_history(key):
        if key not in history_dict:
            print(f"Warning: loss key '{key}' was not found. Using zeros.")
            return np.zeros(n_epochs, dtype=np.float64)

        values = np.asarray(history_dict[key], dtype=np.float64).reshape(-1)

        if values.size != n_epochs:
            raise ValueError(
                f"History length mismatch for '{key}': "
                f"{values.size} values, expected {n_epochs}."
            )

        # The history stores the raw per-term MSE; the total is the weighted sum.
        return values * float(W_COMMON[key][0])

    def summed_history(keys):
        result = np.zeros(n_epochs, dtype=np.float64)
        for key in keys:
            result += get_history(key)
        return result

    def save_loss_plot(title, output_filename, curves):
        figure, ax = plt.subplots(figsize=(8.0, 5.5))

        for curve in curves:
            values = curve["values"]
            mask = np.isfinite(values) & (values > 0.0)

            if np.any(mask):
                ax.semilogy(
                    epoch[mask],
                    values[mask],
                    curve.get("linestyle", "-"),
                    lw=curve.get("linewidth", 1.8),
                    color=curve["color"],
                    label=curve["label"],
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted loss")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.30)
        ax.legend(loc="best", fontsize=9)
        figure.tight_layout()

        figure_path = SAVE_DIR / output_filename
        figure.savefig(figure_path, dpi=250, bbox_inches="tight")
        plt.close(figure)

        print("Saved loss-history plot in:")
        print(figure_path.resolve())

    inc_pde_keys = ["inc_bottom", "inc_side", "inc_top", "inc_lower"]

    def group_keys(*groups):
        keys = []
        for group in groups:
            keys += [f"{group}_T", f"{group}_q"]
        return keys

    def groups_touching(net_name):
        return [
            group
            for group, spec in INTERFACE_SPECS.items()
            if net_name in (spec["a"], spec["b"])
        ]

    def network_plot(title, filename, pde_label, pde_keys, interface_label, net_name):
        pde_loss = summed_history(pde_keys)
        interface_loss = summed_history(group_keys(*groups_touching(net_name)))
        related_loss = pde_loss + interface_loss

        save_loss_plot(
            title=title,
            output_filename=filename,
            curves=[
                {"values": pde_loss, "label": pde_label, "color": "tab:blue"},
                {"values": interface_loss, "label": interface_label, "color": "tab:orange"},
                {
                    "values": related_loss,
                    "label": "Total related loss",
                    "color": "tab:red",
                    "linewidth": 2.2,
                },
            ],
        )

        return pde_loss, interface_loss, related_loss

    histories = {}

    histories["inc"] = network_plot(
        "Inconel network loss history",
        "loss_history_inconel.png",
        "Inconel interior PDE",
        inc_pde_keys,
        "Inconel-NH3 interfaces",
        "inc",
    )

    for slab_name in NH3_SLABS:
        histories[f"nh3_{slab_name}"] = network_plot(
            f"NH3 slab {slab_name} network loss history",
            f"loss_history_nh3_{slab_name}.png",
            f"NH3 {slab_name} interior PDE",
            NH3_TILE_KEYS[slab_name],
            f"NH3 {slab_name} interfaces",
            f"nh3_{slab_name}",
        )

    for block in ("lower", "upper"):
        histories[f"gan_{block}"] = network_plot(
            f"GaN {block}-block network loss history",
            f"loss_history_gan_{block}.png",
            f"GaN {block} interior PDE",
            [f"gan_{block}"],
            f"GaN {block}-NH3 interfaces",
            f"gan_{block}",
        )

    all_pde_keys = (
        inc_pde_keys
        + list(NH3_TILES)
        + ["gan_lower", "gan_upper"]
    )
    interior_loss = summed_history(all_pde_keys)
    boundary_loss = summed_history(group_keys(*INTERFACE_SPECS))

    save_loss_plot(
        title="Global PINN training-loss history",
        output_filename=output_name,
        curves=[
            {"values": interior_loss, "label": "All interior PDE losses", "color": "tab:blue"},
            {"values": boundary_loss, "label": "All interface losses", "color": "tab:orange"},
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
        "histories": histories,
        "global_interior_loss": interior_loss,
        "global_boundary_loss": boundary_loss,
        "global_total_loss": total_loss,
    }


# Full 2D FVM/PINN direct-node error maps (need a 2D FVM table)

def _load_2d(space, excel_path, sheet_name):
    df = load_fvm_table(excel_path, sheet_name)
    r_fvm, z_fvm, T_fvm = fvm_arrays(df)

    if np.unique(np.round(r_fvm, 12)).size < 2:
        raise ValueError(
            "The 2D comparison needs an FVM table with several radii; "
            "this file holds a single vertical line."
        )

    T_pinn, masks = evaluate_pinn_on_points(space, r_fvm, z_fvm)

    return r_fvm, z_fvm, T_fvm, T_pinn, masks


def _scatter_error_panels(r_fvm, z_fvm, values, masks, label, output_path, title):
    """Signed (left) and absolute (right) 2D scatter panels."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 10.0), constrained_layout=True)

    limit = np.max(np.abs(values))

    signed = axes[0].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=values,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        s=14,
        marker="s",
        linewidths=0.0,
    )
    fig.colorbar(signed, ax=axes[0], shrink=0.85).set_label(f"Signed error {label}")

    absolute = axes[1].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=np.abs(values),
        cmap="magma",
        vmin=0.0,
        vmax=limit,
        s=14,
        marker="s",
        linewidths=0.0,
    )
    fig.colorbar(absolute, ax=axes[1], shrink=0.85).set_label(f"Absolute error {label}")

    rmse, mae, max_abs = error_statistics(values)

    for ax in axes:
        draw_geometry(ax)

    axes[0].set_title(f"{title}\nsigned")
    axes[1].set_title(
        f"{title}\nRMSE = {rmse:.3e}, MAE = {mae:.3e}, max = {max_abs:.3e}"
    )

    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 2D error plot in:\n{output_path}")


def plot_fvm_vs_pinn_2d_error(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_inconel_nh3_gan_2d_error.png",
):
    r_fvm, z_fvm, T_fvm, T_pinn, masks = _load_2d(space, excel_path, sheet_name)

    error_K = T_pinn - T_fvm

    print_error_statistics("FVM VS PINN 2D NODAL COMPARISON [K]", error_K, masks)

    pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "pinn_minus_fvm_K": error_K,
            "material": material_label_array(masks),
        }
    ).to_csv(SAVE_DIR / (Path(output_name).stem + ".csv"), index=False)

    _scatter_error_panels(
        r_fvm,
        z_fvm,
        error_K,
        masks,
        "PINN - FVM [K]",
        SAVE_DIR / output_name,
        "PINN - FVM [K]",
    )


def plot_fvm_vs_pinn_2d_relative_error(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_inconel_nh3_gan_2d_relative_error.png",
):
    r_fvm, z_fvm, T_fvm, T_pinn, masks = _load_2d(space, excel_path, sheet_name)

    if np.any(T_fvm == 0.0):
        raise ValueError("FVM temperature contains zeros; relative error undefined.")

    relative_error_pct = (T_pinn - T_fvm) / T_fvm * 100.0

    print_error_statistics(
        "FVM VS PINN 2D RELATIVE ERROR [%]", relative_error_pct, masks, unit="%"
    )

    pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "relative_error_percent": relative_error_pct,
            "material": material_label_array(masks),
        }
    ).to_csv(SAVE_DIR / (Path(output_name).stem + ".csv"), index=False)

    _scatter_error_panels(
        r_fvm,
        z_fvm,
        relative_error_pct,
        masks,
        "(PINN - FVM) / FVM [%]",
        SAVE_DIR / output_name,
        "Relative error (PINN - FVM) / FVM [%]",
    )


# Run

if __name__ == "__main__":
    space = build_space(seed=42)

    print("Number of neural models:", len(space.models))

    PROJECT_DIR = Path(__file__).resolve().parents[1] / "data"
    FVM_FILE = PROJECT_DIR / "complete_data_two_gan.xlsx"

    # FVM data with the domain names (before training, so it can be inspected early)
    if FVM_FILE.exists():
        plot_fvm_domain_map(FVM_FILE)

    pinn = Projector(
        model,
        space,
        sampler,
        # optimizer="Adam",
        # The default ENG setting (matrix_regularization=1e-6, log-grid line search)
        # diverged on this problem.  Armijo line search with adaptive
        # Levenberg-Marquardt damping (raised whenever a step fails to reduce the
        # loss) is stable.  LEARNING_RATE is not used by the ENG optimizer.
        linesearch="armijo",
        adaptive_matrix_regularization=True,
        matrix_regularization=1.0e-3,
        weights=W_COMMON,
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

    key, pinn = pinn.project(
        key,
        space,
        N_EPOCHS,
        N_COLLOC,
        N_BC_COLLOC,
        tqdmdesc="Inconel-NH3-GaN conduction",
    )

    elapsed = timeit.default_timer() - start

    print("\n" + "=" * 72)
    print("FINAL-SPACE LOSSES ON FRESH SAMPLES")
    print("=" * 72)

    key, final_samples = sampler.sample(key, N_COLLOC, N_BC_COLLOC)
    print(pinn.evaluate_losses(pinn.space, final_samples))

    plot_all_interior_residuals(pinn.space)

    plot_loss_history(pinn.losses, output_name="loss_history.png")

    print("\n" + "=" * 72)
    print("BEST LOSS")
    print("=" * 72)
    print(pinn.best_loss)

    print(f"\nTraining time: {elapsed:.2f} s")

    # FVM validation: Inconel + NH3 + GaN

    print("\n" + "=" * 72)
    print("FVM VALIDATION FILE CHECK: INCONEL + NH3 + GaN")
    print("=" * 72)
    print(f"Expected FVM file:\n{FVM_FILE.resolve()}")
    print("File exists:", FVM_FILE.exists())

    if not FVM_FILE.exists():
        raise FileNotFoundError(
            f"The FVM Excel file was not found:\n{FVM_FILE.resolve()}"
        )

    fvm_table = load_fvm_table(FVM_FILE, "Sheet1")
    fvm_is_2d = np.unique(np.round(fvm_table["x"].to_numpy(dtype=np.float64), 12)).size > 1

    plot_solution(
        pinn.space,
        fvm_excel_path=FVM_FILE if fvm_is_2d else None,
        fvm_sheet_name="Sheet1",
    )

    TARGET_LINE_RADIUS = 0.004

    if fvm_is_2d:
        print("\n" + "=" * 72)
        print("RUNNING FVM VS PINN 2D NODAL COMPARISON")
        print("=" * 72)

        plot_fvm_vs_pinn_2d_error(
            pinn.space,
            excel_path=FVM_FILE,
            sheet_name="Sheet1",
        )

        plot_fvm_vs_pinn_2d_relative_error(
            pinn.space,
            excel_path=FVM_FILE,
            sheet_name="Sheet1",
        )

        FVM_FILE_LINE = extract_fvm_vertical_line(
            full_excel_path=FVM_FILE,
            target_r=TARGET_LINE_RADIUS,
            sheet_name="Sheet1",
            output_name="gan_line_data_extracted.xlsx",
        )
    else:
        print(
            "\nThe FVM file holds a single vertical line: "
            "using it directly for the 1D comparison "
            "(2D error maps need a full-domain FVM table)."
        )
        FVM_FILE_LINE = FVM_FILE

    print("\n" + "=" * 72)
    print("RUNNING FVM VS PINN 1D LINE COMPARISON")
    print("=" * 72)

    plot_fvm_vs_pinn(
        pinn.space,
        excel_path=FVM_FILE_LINE,
        sheet_name="Sheet1",
        output_name="fvm_vs_pinn_inconel_nh3_gan_line_comparison.png",
    )

    plot_fvm_vs_pinn_relative_error(
        pinn.space,
        excel_path=FVM_FILE_LINE,
        sheet_name="Sheet1",
        output_name="fvm_vs_pinn_inconel_nh3_gan_line_relative_error.png",
    )

    print("\n" + "=" * 72)
    print("INCONEL + NH3 + GaN FVM VALIDATION COMPLETE")
    print("=" * 72)
    print(f"Saved plot(s) in:\n{SAVE_DIR.resolve()}")
