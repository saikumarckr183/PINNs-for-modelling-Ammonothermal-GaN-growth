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
from scimba_jax.plots.plots_nd import plot_abstract_approx_space


jax.config.update("jax_enable_x64", True)

SAVE_DIR = Path("outputs_autoclave_one_GaN_part")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# CryS-MAS geometry: outer autoclave
R_OUT = 0.035
H_OUT = 0.370


# CryS-MAS geometry: NH3 cavity
R_NH3 = 0.012
Z_NH3_TOP = 0.335


# CryS-MAS lower NH3 boundary.
# The points are listed in physical coordinates [m] as (r, z).
# They form a piecewise-linear curve from the axis to the NH3 side wall.
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


# Keep these values because the rectangular sampling envelopes need them.
Z_NH3_MIN = float(NH3_LOWER_Z[0])
Z_NH3_MAX_LOWER = float(NH3_LOWER_Z[-1])


# Material thermal conductivities [W/(m K)], from the CryS-MAS material
# database (Autoclave.mat).
def k_inc_value(temperature_K):
    """Inconel 718 conductivity [W/(m K)]."""
    return 6.9995 + 0.0151 * temperature_K


def k_nh3_value(temperature_K):
    """Ammonia conductivity [W/(m K)]."""
    return (
        0.3039497375
        - 0.00019097 * temperature_K
        + 0.0000001887 * temperature_K**2
    )


K_GAN = 130.0  # W/(m K)


def k_gan_value(temperature_K):
    return K_GAN * jnp.ones_like(temperature_K)


# CryS-MAS geometry: one continuous GaN region.
# 0 <= r <= 5 mm
# 40 mm <= z <= 325 mm
R_GAN = 0.005
Z_GAN_MIN = 0.040
Z_GAN_MAX = 0.325
Z_GAN_MID = 0.5 * (Z_GAN_MIN + Z_GAN_MAX)


# Fixed external thermal boundary condition
T0_K = 611.15
T1 = -300

T0_C = T0_K - 273.15
T_REF = T0_K
INTERFACE_T_REF = 1.0  # K


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
    # nh3_upper is the thin annular sleeve (r in [R_GAN, R_NH3]) that has to
    # simultaneously satisfy the Inconel-side interface AND both GaN-facing
    # interfaces 285 mm apart -- diagnosed as under-resolved right at r=R_GAN
    # (NH3's own field was flattened out radially there vs FVM), which GaN's
    # affine slope inherits directly since it's read off NH3's flux at that
    # boundary. Bumped well past "gan" itself given it covers a larger,
    # more constrained region.
    "nh3_upper": 3000,
    "nh3_lower": 1500,
    "gan": 2000,
}

N_EPOCHS = 100
LEARNING_RATE = 1.0e-4
HIDDEN = [24, 24, 24]

DELTA_T_REF = abs(T1) * H_OUT

# Internal reference profile for the NH3/GaN fields: the same linear law as
# T_outer(z), but written in center/range form so each interior material can
# be given its own independent correction-scale knob below (same convention
# as the plain Inconel+NH3 script's NH3_CORRECTION_SCALE).
T_INTERNAL_CENTER = T0_K + 0.5 * T1 * H_OUT
T_INTERNAL_RANGE = -T1 * H_OUT

NH3_CORRECTION_SCALE = 1.0
GAN_CORRECTION_SCALE = 1.0

# GaN's field is modeled as affine in z (see gan_center_post_processing /
# gan_slope_post_processing below): T_gan(r,z) = center(r) + slope(r)*(z - Z_GAN_MID).
GAN_SLOPE_SCALE = DELTA_T_REF / (Z_GAN_MAX - Z_GAN_MIN)

# Small hidden size for GaN's center/slope sub-networks: these represent a
# mild, expected-to-be-nearly-uniform r-dependence, not a full field, so
# they intentionally get far less capacity than the Inconel/NH3 networks.
GAN_COMPONENT_HIDDEN = [8]

# NH3's steepest real gradient sits in the thin lower sliver between
# Z_NH3_MIN and Z_NH3_MAX_LOWER. NH3_WARP_BOOST is the local input-space
# resolution multiplier the network gets there.
NH3_WARP_BOOST = 8.0

L_REF = R_OUT

# Conductivity reference scale, scanned over the anticipated operating
# temperature window so it reflects the real k(T) curves rather than a
# single constant.
T_K_MIN_FOR_SCALING = 495.0
T_K_MAX_FOR_SCALING = 615.0
T_K_SCALE_GRID = jnp.linspace(T_K_MIN_FOR_SCALING, T_K_MAX_FOR_SCALING, 2001)

K_INC_SCALE_VALUES = k_inc_value(T_K_SCALE_GRID)
K_NH3_SCALE_VALUES = k_nh3_value(T_K_SCALE_GRID)

if bool(jnp.any(K_INC_SCALE_VALUES <= 0.0)):
    raise ValueError(
        "Inconel conductivity is not positive in the scaling range "
        f"[{T_K_MIN_FOR_SCALING}, {T_K_MAX_FOR_SCALING}] K."
    )

if bool(jnp.any(K_NH3_SCALE_VALUES <= 0.0)):
    raise ValueError(
        "NH3 conductivity is not positive in the scaling range "
        f"[{T_K_MIN_FOR_SCALING}, {T_K_MAX_FOR_SCALING}] K."
    )

K_INC_MAX_REF = float(jnp.max(K_INC_SCALE_VALUES))
K_NH3_MAX_REF = float(jnp.max(K_NH3_SCALE_VALUES))
K_REF = max(K_INC_MAX_REF, K_NH3_MAX_REF, K_GAN)

# Reference scale for INTERFACE flux-continuity residuals. These compare
# k_a*grad(T_a) against k_b*grad(T_b) at the same physical location, so both
# sides must be normalized by the same constant -- but that constant must
# match the scale the TRUE physical flux actually has, which is set by the
# poorly-conducting side (NH3, on both interfaces here), not the
# well-conducting one. Confirmed empirically: with the old K_REF-based scale
# (built from GaN's k=130), a converged run showed k_nh3*dT_nh3/dn = 77 W/m^2
# against k_gan*dT_gan/dn = 38991 W/m^2 at the GaN/NH3 face -- a ~500x real
# flux violation that the loss saw as ~0.006, because dividing by a
# reference ~500x too large hid it. Same failure mode Victor's file
# documents for the Inconel/NH3 interface (a 60x version of the same bug,
# from the same root cause).
FLUX_REF = K_NH3_MAX_REF * DELTA_T_REF / L_REF

# Per-material reference scales for the INTERIOR PDE residual. Ammonia's
# real conductivity (~0.26 W/(m K)) is ~500x smaller than GaN's (130 W/(m K)),
# so normalizing its interior residual by the shared K_REF-based FLUX_REF
# makes that residual trivially tiny for almost any smooth (including
# wrong) network output -- it stops providing real training signal. Each
# material's interior residual is instead normalized by its own local flux
# scale, so the residual only vanishes when that material's own physics is
# actually satisfied.
PDE_R_WEIGHTED_REF_INC = K_INC_MAX_REF * DELTA_T_REF / L_REF
PDE_R_WEIGHTED_REF_NH3 = K_NH3_MAX_REF * DELTA_T_REF / L_REF
PDE_R_WEIGHTED_REF_GAN = K_GAN * DELTA_T_REF / L_REF

# Used only to keep the interior-residual diagnostic plot's evaluation grid
# a hair off r=0 (jax.grad-based residual, unlike the r-weighted PDE
# residual above, which needs no such floor). Same convention as the plain
# Inconel+NH3 script.
EPS_AXIS = 1.0e-7


# Reference coordinates for the rectangular main-domain mapping.
XI_NH3 = R_NH3 / R_OUT

ETA_NH3_MIN = Z_NH3_MIN / H_OUT
ETA_NH3_MAX_LOWER = Z_NH3_MAX_LOWER / H_OUT
ETA_NH3_TOP = Z_NH3_TOP / H_OUT

XI_GAN = R_GAN / R_OUT
ETA_GAN_MIN = Z_GAN_MIN / H_OUT
ETA_GAN_MAX = Z_GAN_MAX / H_OUT


# CryS-MAS lower NH3 polyline converted to reference coordinates.
XI_NH3_LOWER = NH3_LOWER_R / R_OUT
ETA_NH3_LOWER = NH3_LOWER_Z / H_OUT


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


# Volumetric sampling subdomains.
# The lower NH3 boundary is curved/piecewise-linear, so rectangular
# envelopes are used and the physical material masks later remove
# inactive samples.

# Inconel below the lowest NH3 point on the symmetry axis.
inc_bottom = Square2D(
    [(0.0, 1.0), (0.0, ETA_NH3_MIN)],
    is_main_domain=False,
    label_str="inc_bottom",
)


# Inconel outside the NH3 cylindrical side wall.
inc_side = Square2D(
    [(XI_NH3, 1.0), (ETA_NH3_MIN, ETA_NH3_TOP)],
    is_main_domain=False,
    label_str="inc_side",
)


# Inconel above the NH3 cavity.
inc_top = Square2D(
    [(0.0, 1.0), (ETA_NH3_TOP, 1.0)],
    is_main_domain=False,
    label_str="inc_top",
)


# Lower envelope contains both Inconel and part of the lower NH3 region.
# InconelResidual masks out the physical NH3 portion.
inc_lower = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MIN, ETA_NH3_MAX_LOWER)],
    is_main_domain=False,
    label_str="inc_lower",
)


# Upper NH3 rectangle. It includes GaN points, which are excluded
# by in_nh3_without_gan() inside AmmoniaResidual.
nh3_upper = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MAX_LOWER, ETA_NH3_TOP)],
    is_main_domain=False,
    label_str="nh3_upper",
)


# Lower NH3 rectangular envelope. The physical lower boundary mask
# retains only the CryS-MAS piecewise-linear cavity region.
nh3_lower = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MIN, ETA_NH3_MAX_LOWER)],
    is_main_domain=False,
    label_str="nh3_lower",
)


# One continuous CryS-MAS GaN region.
gan = Square2D(
    [
        (0.0, XI_GAN),
        (ETA_GAN_MIN, ETA_GAN_MAX),
    ],
    is_main_domain=False,
    label_str="gan",
)


for domain in (
    inc_bottom,
    inc_side,
    inc_top,
    inc_lower,
    nh3_upper,
    nh3_lower,
    gan,
):
    main_domain.add_subdomain(domain)


# Surfacic domains in REFERENCE coordinates
# They are mapped into physical coordinates when add_bc_domain() is called.

# CryS-MAS NH3/Inconel cylindrical sidewall interface.
# Physical location:
# r = 0.012 m
# z = 0.035 m to z = 0.335 m
#
# ETA_NH3_MAX_LOWER = 0.035 / H_OUT
# ETA_NH3_TOP       = 0.335 / H_OUT

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


domain_if_top_T = Segment2D(
    pt1=(0.0, ETA_NH3_TOP),
    pt2=(XI_NH3, ETA_NH3_TOP),
    label_str="if_top_T",
)

domain_if_top_q = Segment2D(
    pt1=(0.0, ETA_NH3_TOP),
    pt2=(XI_NH3, ETA_NH3_TOP),
    label_str="if_top_q",
)

# CryS-MAS one-GaN-block interfaces.
# The block is:
# 0 <= r <= 0.005 m
# 0.040 <= z <= 0.325 m

# Cylindrical GaN side surface:
# r = 0.005 m, 0.040 <= z <= 0.325 m
domain_gan_side_T = Segment2D(
    pt1=(XI_GAN, ETA_GAN_MIN),
    pt2=(XI_GAN, ETA_GAN_MAX),
    label_str="gan_side_T",
)

domain_gan_side_q = Segment2D(
    pt1=(XI_GAN, ETA_GAN_MIN),
    pt2=(XI_GAN, ETA_GAN_MAX),
    label_str="gan_side_q",
)


# Lower GaN horizontal surface:
# z = 0.040 m, 0 <= r <= 0.005 m
domain_gan_bottom_T = Segment2D(
    pt1=(0.0, ETA_GAN_MIN),
    pt2=(XI_GAN, ETA_GAN_MIN),
    label_str="gan_bottom_T",
)

domain_gan_bottom_q = Segment2D(
    pt1=(0.0, ETA_GAN_MIN),
    pt2=(XI_GAN, ETA_GAN_MIN),
    label_str="gan_bottom_q",
)


# Upper GaN horizontal surface:
# z = 0.325 m, 0 <= r <= 0.005 m
domain_gan_top_T = Segment2D(
    pt1=(0.0, ETA_GAN_MAX),
    pt2=(XI_GAN, ETA_GAN_MAX),
    label_str="gan_top_T",
)

domain_gan_top_q = Segment2D(
    pt1=(0.0, ETA_GAN_MAX),
    pt2=(XI_GAN, ETA_GAN_MAX),
    label_str="gan_top_q",
)

# CryS-MAS lower NH3/Inconel interface.
#
# It is a polyline defined by NH3_LOWER_R and NH3_LOWER_Z.
# Each adjacent pair becomes one physical interface segment.

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

    lower_temperature_domains.append(
        Segment2D(
            pt1=p1,
            pt2=p2,
            label_str="if_lower_T",
            label_idx=i + 1,
        )
    )

    lower_flux_domains.append(
        Segment2D(
            pt1=p1,
            pt2=p2,
            label_str="if_lower_q",
            label_idx=i + 1,
        )
    )


for boundary in (
    domain_gan_side_T,
    domain_gan_side_q,
    domain_gan_bottom_T,
    domain_gan_bottom_q,
    domain_gan_top_T,
    domain_gan_top_q,

    domain_if_side_T,
    domain_if_side_q,
    domain_if_top_T,
    domain_if_top_q,

    *lower_temperature_domains,
    *lower_flux_domains,
):
    main_domain.add_bc_domain(boundary)

# Boundary groups.
#
# Always use get_label() for the segmented lower CryS-MAS
# polyline. The six individual Segment2D objects can have
# generated labels such as if_lower_T_1, if_lower_T_2, etc.

BOUNDARY_GROUPS = {
    "gan_interface_temperature": [
        domain_gan_side_T.get_label(),
        domain_gan_bottom_T.get_label(),
        domain_gan_top_T.get_label(),
    ],

    "gan_interface_flux": [
        domain_gan_side_q.get_label(),
        domain_gan_bottom_q.get_label(),
        domain_gan_top_q.get_label(),
    ],

    "interface_temperature": [
        domain_if_side_T.get_label(),
        domain_if_top_T.get_label(),
        *[
            domain.get_label()
            for domain in lower_temperature_domains
        ],
    ],

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
    BOUNDARY_GROUPS
)

# Boundary collocation counts.
# Construct entries from exactly the labels registered above.
# This prevents missing-key errors in sampler.sample().

N_BC_COLLOC = {
    group_name: {
        boundary_label: 150
        for boundary_label in boundary_labels
    }
    for group_name, boundary_labels in BOUNDARY_GROUPS.items()
}


# Sample GaN/NH3 material interfaces more densely -- this is exactly the
# boundary GaN's affine slope is read off of, and where NH3's field was
# diagnosed as under-resolved (flattened out radially vs FVM right at
# r=R_GAN).
for group_name in (
    "gan_interface_temperature",
    "gan_interface_flux",
):
    for boundary_label in N_BC_COLLOC[group_name]:
        N_BC_COLLOC[group_name][boundary_label] = 500


# Optional: inspect exactly what will be sent to the sampler.
print("\n" + "=" * 72)
print("BOUNDARY GROUPS")
print("=" * 72)

for group_name, boundary_labels in BOUNDARY_GROUPS.items():
    print(f"\n{group_name}:")
    for boundary_label in boundary_labels:
        print(f"  {boundary_label}")


print("\n" + "=" * 72)
print("N_BC_COLLOC")
print("=" * 72)

for group_name, counts in N_BC_COLLOC.items():
    print(f"\n{group_name}:")
    for boundary_label, n_points in counts.items():
        print(f"  {boundary_label}: {n_points}")


# Geometry predicates in PHYSICAL coordinates

def z_bottom_ammonia(r):
    """
    CryS-MAS lower NH3 boundary z(r), defined by the PCS polyline.

    The implementation uses linear interpolation between the supplied
    physical geometry points.
    """
    r = jnp.asarray(r)

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
    """
    Physical NH3 points in the CryS-MAS lower-envelope region.
    """
    return (
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom_ammonia(r))
        & (z <= Z_NH3_MAX_LOWER)
    )

def in_gan(r, z):
    """
    CryS-MAS one-block GaN region.
    """
    return (
        (r >= 0.0)
        & (r <= R_GAN)
        & (z >= Z_GAN_MIN)
        & (z <= Z_GAN_MAX)
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
# Network input is (rho, eta) = ((r/R_OUT)^2, z/H_OUT) -- see
# scaled_coordinates(), registered as this space's pre_processing -- so
# every field is structurally even in r and needs no separate axis regularity residual.

def scaled_coordinates(x):
    x = jnp.asarray(x, dtype=jnp.float64)

    r_scaled = x[..., 0:1] / R_OUT
    z_scaled = x[..., 1:2] / H_OUT

    rho = r_scaled**2
    eta = z_scaled

    return jnp.concatenate([rho, eta], axis=-1)


# NH3-only input warp: a smooth (Gaussian-density), monotonic reparametrization
# of z that locally stretches the thin lower-ammonia sliver [Z_NH3_MIN,
# Z_NH3_MAX_LOWER] -- only ~3% of the vessel height, yet where the real FVM
# solution has its steepest gradient (ammonia's conductivity is tiny). This
# gives NH3's own network extra resolution there without touching Inconel's
# or GaN's input encoding. Built as a normalized integral of a
# constant-plus-Gaussian density, so it stays smooth everywhere (no kinks
# that would show up as artifacts in gradient-based PDE residuals).
NH3_WARP_CENTER = 0.5 * (Z_NH3_MIN + Z_NH3_MAX_LOWER)
NH3_WARP_SIGMA = 0.5 * (Z_NH3_MAX_LOWER - Z_NH3_MIN)


def _nh3_warp_cumulative_density(z):
    gaussian_integral = NH3_WARP_SIGMA * jnp.sqrt(jnp.pi / 2.0) * jax.scipy.special.erf(
        (z - NH3_WARP_CENTER) / (NH3_WARP_SIGMA * jnp.sqrt(2.0))
    )
    return z + (NH3_WARP_BOOST - 1.0) * gaussian_integral


def nh3_z_warp(z):
    z = jnp.asarray(z, dtype=jnp.float64)

    numerator = _nh3_warp_cumulative_density(z) - _nh3_warp_cumulative_density(
        jnp.zeros_like(z)
    )
    denominator = _nh3_warp_cumulative_density(
        jnp.full_like(z, H_OUT)
    ) - _nh3_warp_cumulative_density(jnp.zeros_like(z))

    return numerator / denominator


def nh3_scaled_coordinates(x):
    x = jnp.asarray(x, dtype=jnp.float64)

    r_scaled = x[..., 0:1] / R_OUT
    rho = r_scaled**2
    eta_warped = nh3_z_warp(x[..., 1:2])

    return jnp.concatenate([rho, eta_warped], axis=-1)


# GaN-only input: rho = (r/R_OUT)^2 alone, z dropped entirely. GaN's field is
# built explicitly affine in z in post-processing below, so the network only
# needs to learn how the two affine parameters (center, slope) vary with r --
# expected to be mild, since GaN's radius is tiny and its conductivity is
# uniform and far higher than its neighbors'.
def gan_component_preprocessing(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    r_scaled = x[..., 0:1] / R_OUT

    return r_scaled**2


def outer_distance_factor(x):
    x = jnp.asarray(x, dtype=jnp.float64)

    r = x[..., 0:1]
    z = x[..., 1:2]

    d_side = 1.0 - (r / R_OUT) ** 2
    d_bottom = z / H_OUT
    d_top = (H_OUT - z) / H_OUT

    return d_side * d_bottom * d_top


# post_processing[0]: Inconel field -- hard-constrained to T_outer(x) on
# the outer wall (outer_distance_factor vanishes there).
def inconel_post_processing(theta_inc, x):
    return T_outer(x) + DELTA_T_REF * outer_distance_factor(x) * theta_inc


def internal_reference_profile(x):
    """Same linear law as T_outer(z), in center/range form."""
    x = jnp.asarray(x, dtype=jnp.float64)
    z_scaled = x[..., 1:2] / H_OUT

    return T_INTERNAL_CENTER + T_INTERNAL_RANGE * (0.5 - z_scaled)


# post_processing[1]/[2]: NH3 and GaN fields. Not hard-constrained to any
# interface value -- that continuity is enforced by the soft
# interface_temperature/interface_flux and gan_interface_* residuals. Each
# material gets its own correction-scale knob (NH3_CORRECTION_SCALE,
# GAN_CORRECTION_SCALE).
def nh3_post_processing(raw, x):
    return internal_reference_profile(x) + NH3_CORRECTION_SCALE * T_INTERNAL_RANGE * raw


# post_processing[2]/[3]: GaN's field is forced to be exactly affine in z --
# T_gan(r,z) = center(r) + slope(r)*(z - Z_GAN_MID) -- which is the exact
# family of solutions for a constant-conductivity, negligible-radial-flux
# block (see combine_gan_field below). "center" and "slope" are each a
# separate small network, fed only rho (via gan_component_preprocessing).
def gan_center_post_processing(raw, x):
    return internal_reference_profile(x) + GAN_CORRECTION_SCALE * T_INTERNAL_RANGE * raw


def gan_slope_post_processing(raw, x):
    return GAN_SLOPE_SCALE * raw


def combine_gan_field(T_gan_center, T_gan_slope):
    """
    Build the physical GaN field T_gan(r,z) = center(r) + slope(r)*(z -
    Z_GAN_MID) from its two affine-component ParamScalarFunctions. Any
    residual/eval code that needs GaN's temperature calls this once on the
    (T_gan_center, T_gan_slope) pair taken from space.create_variables().

    Accepts *args so this works both inside InteriorResidual (vars called as
    (space, x)) and BoundaryResidual (vars called as (space, x, n) after
    with_normals()) contexts identically.
    """

    def fn(space, x, *args):
        z = x[..., 1:2]
        return (
            T_gan_center(space, x, *args)
            + T_gan_slope(space, x, *args) * (z - Z_GAN_MID)
        )

    return ParamScalarFunction(T_gan_center.dims, fn, f_type=T_gan_center.f_type)


# evaluate_temperature/T_inc_eval/T_nh3_eval/T_gan_eval go through
# space.create_variables(), i.e. the SAME pre_processing/post_processing
# composition used by every physical residual -- so training and
# plotting/FVM-comparison see the identical field, not two disconnected
# functions of the same raw network output.
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


def T_gan_eval(space, x):
    x = jnp.asarray(x, dtype=jnp.float64)
    T_gan_center, T_gan_slope = space.create_variables()[2:4]
    T_gan = combine_gan_field(T_gan_center, T_gan_slope)

    if x.ndim == 2:
        return jax.vmap(lambda x_one: T_gan(space, x_one))(x)

    return T_gan(space, x)


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

T_GAN = ParamScalarFunction(
    {"x": 2},
    T_gan_eval,
    f_type="x",
)


def k_inc(T):
    return ParamScalarFunction(
        T.dims,
        lambda space, x: k_inc_value(T(space, x)),
        f_type=T.f_type,
    )


def k_nh3(T):
    return ParamScalarFunction(
        T.dims,
        lambda space, x: k_nh3_value(T(space, x)),
        f_type=T.f_type,
    )


def k_gan(T):
    return ParamScalarFunction(
        T.dims,
        lambda space, x: k_gan_value(T(space, x)),
        f_type=T.f_type,
    )


# Axisymmetric steady-conduction operator, r-weighted to avoid dividing by r
# near the symmetry axis (same convention as the other conduction scripts):
#   r*d(q_r)/dr + q_r + r*d(q_z)/dz = 0, normalized by the CALLER's own
# material-specific pde_ref (PDE_R_WEIGHTED_REF_INC/NH3/GAN) -- see the
# comment above those constants for why a shared reference would leave the
# low-conductivity NH3 residual with almost no training signal.

def axisymmetric_conduction(T, conductivity, space, x, pde_ref):
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

    return raw_residual / pde_ref


# Interior residuals

class InconelResidual(InteriorResidual):
    def __init__(self, domain, lower_box=False):
        super().__init__(
            domain=domain,
            size=1,
            model_type="x",
        )

        self.lower_box = lower_box

    def construct_residual(self, *vars):
        T_inc, _, _, _ = vars

        def residual(space, x):
            value = axisymmetric_conduction(
                T_inc,
                k_inc,
                space,
                x,
                PDE_R_WEIGHTED_REF_INC,
            )

            if not self.lower_box:
                return value

            r = x[..., 0:1]
            z = x[..., 1:2]

            return jnp.where(
                in_lower_ammonia(r, z),
                0.0,
                value,
            )

        return ParamScalarFunction(
            T_inc.dims,
            residual,
            f_type=T_inc.f_type,
        )


class AmmoniaResidual(InteriorResidual):
    def __init__(self, domain, lower_box=False):
        super().__init__(
            domain=domain,
            size=1,
            model_type="x",
        )

        self.lower_box = lower_box

    def construct_residual(self, *vars):
        _, T_nh3, _, _ = vars

        def residual(space, x):
            value = axisymmetric_conduction(
                T_nh3,
                k_nh3,
                space,
                x,
                PDE_R_WEIGHTED_REF_NH3,
            )

            r = x[..., 0:1]
            z = x[..., 1:2]

            if not self.lower_box:
                return jnp.where(
                    in_nh3_without_gan(r, z),
                    value,
                    0.0,
                )

            r = x[..., 0:1]
            z = x[..., 1:2]

            return jnp.where(
                in_nh3_without_gan(r, z),
                value,
                0.0,
            )

        return ParamScalarFunction(
            T_nh3.dims,
            residual,
            f_type=T_nh3.f_type,
        )

class GaNResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(
            domain=domain,
            size=1,
            model_type="x",
        )

    def construct_residual(self, *vars):
        _, _, T_gan_center, T_gan_slope = vars
        T_gan = combine_gan_field(T_gan_center, T_gan_slope)

        def residual(space, x):
            value = axisymmetric_conduction(
                T_gan,
                k_gan,
                space,
                x,
                PDE_R_WEIGHTED_REF_GAN,
            )

            r = x[..., 0:1]
            z = x[..., 1:2]

            return jnp.where(
                in_gan(r, z),
                value,
                0.0,
            )

        return ParamScalarFunction(
            T_gan.dims,
            residual,
            f_type=T_gan.f_type,
        )

# Boundary residuals
# BoundaryResidual converts field functions from f_type="x" to f_type="x_n".
# Inside boundary residuals, evaluate fields as T(space, x, n).

INTERFACE_T_DOMAINS = {
    "if_side_T": domain_if_side_T,
    "if_top_T": domain_if_top_T,
}

INTERFACE_Q_DOMAINS = {
    "if_side_q": domain_if_side_q,
    "if_top_q": domain_if_top_q,
}

for domain in lower_temperature_domains:
    INTERFACE_T_DOMAINS[domain.get_label()] = domain

for domain in lower_flux_domains:
    INTERFACE_Q_DOMAINS[domain.get_label()] = domain

GAN_INTERFACE_T_DOMAINS = {
    "gan_side_T": domain_gan_side_T,
    "gan_bottom_T": domain_gan_bottom_T,
    "gan_top_T": domain_gan_top_T,
}


GAN_INTERFACE_Q_DOMAINS = {
    "gan_side_q": domain_gan_side_q,
    "gan_bottom_q": domain_gan_bottom_q,
    "gan_top_q": domain_gan_top_q,
}


class InterfaceTemperatureResidual(BoundaryResidual):
    def __init__(self):
        super().__init__(
            domain=INTERFACE_T_DOMAINS,
            size=1,
            model_type="x",
        )

    def construct_residual(self, *vars):
        T_inc, T_nh3, _, _ = vars

        def residual(space, x, n):
            return (
                T_inc(space, x, n)
                - T_nh3(space, x, n)
            ) / INTERFACE_T_REF

        return ParamScalarFunction(
            T_inc.dims,
            residual,
            f_type=T_inc.f_type,
        )

class GaNInterfaceTemperatureResidual(BoundaryResidual):
    def __init__(self):
        super().__init__(
            domain=GAN_INTERFACE_T_DOMAINS,
            size=1,
            model_type="x",
        )


    def construct_residual(self, *vars):
        _, T_nh3, T_gan_center, T_gan_slope = vars
        T_gan = combine_gan_field(T_gan_center, T_gan_slope)


        def residual(space, x, n):
            return (
                T_gan(space, x, n)
                - T_nh3(space, x, n)
            ) / INTERFACE_T_REF


        return ParamScalarFunction(
            T_gan.dims,
            residual,
            f_type=T_gan.f_type,
        )


class InterfaceFluxResidual(BoundaryResidual):
    def __init__(self):
        super().__init__(
            domain=INTERFACE_Q_DOMAINS,
            size=1,
            model_type="x",
        )

    def construct_residual(self, *vars):
        T_inc, T_nh3, _, _ = vars

        d_inc_dr = T_inc.partial_derivative("x", 0)
        d_inc_dz = T_inc.partial_derivative("x", 1)

        d_nh3_dr = T_nh3.partial_derivative("x", 0)
        d_nh3_dz = T_nh3.partial_derivative("x", 1)

        k_inc_field = k_inc(T_inc)
        k_nh3_field = k_nh3(T_nh3)

        def residual(space, x, n):
            n_r = n[..., 0:1]
            n_z = n[..., 1:2]

            grad_inc_normal = (
                n_r * d_inc_dr(space, x, n)
                + n_z * d_inc_dz(space, x, n)
            )

            grad_nh3_normal = (
                n_r * d_nh3_dr(space, x, n)
                + n_z * d_nh3_dz(space, x, n)
            )

            k_inc_local = k_inc_field(space, x)
            k_nh3_local = k_nh3_field(space, x)

            return (
                k_inc_local * grad_inc_normal
                - k_nh3_local * grad_nh3_normal
            ) / FLUX_REF

        return ParamScalarFunction(
            T_inc.dims,
            residual,
            f_type=T_inc.f_type,
        )

class GaNInterfaceFluxResidual(BoundaryResidual):
    def __init__(self):
        super().__init__(
            domain=GAN_INTERFACE_Q_DOMAINS,
            size=1,
            model_type="x",
        )


    def construct_residual(self, *vars):
        _, T_nh3, T_gan_center, T_gan_slope = vars
        T_gan = combine_gan_field(T_gan_center, T_gan_slope)


        d_nh3_dr = T_nh3.partial_derivative(
            "x",
            0,
        )

        d_nh3_dz = T_nh3.partial_derivative(
            "x",
            1,
        )

        d_gan_dr = T_gan.partial_derivative(
            "x",
            0,
        )

        d_gan_dz = T_gan.partial_derivative(
            "x",
            1,
        )

        k_nh3_field = k_nh3(T_nh3)
        k_gan_field = k_gan(T_gan)

        def residual(space, x, n):
            n_r = n[..., 0:1]
            n_z = n[..., 1:2]

            grad_nh3_normal = (
                n_r * d_nh3_dr(space, x, n)
                + n_z * d_nh3_dz(space, x, n)
            )

            grad_gan_normal = (
                n_r * d_gan_dr(space, x, n)
                + n_z * d_gan_dz(space, x, n)
            )

            k_nh3_local = k_nh3_field(space, x)
            k_gan_local = k_gan_field(space, x)

            return (
                k_gan_local * grad_gan_normal
                - k_nh3_local * grad_nh3_normal
            ) / FLUX_REF


        return ParamScalarFunction(
            T_gan.dims,
            residual,
            f_type=T_gan.f_type,
        )

# Physical model

class AutoclaveConductionModel(AbstractPhysicalModel):
    def __init__(self):
        super().__init__(main_domain=main_domain)

        self.physical_residuals = {
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
            "nh3_upper": AmmoniaResidual(
                nh3_upper,
            ),
            "nh3_lower": AmmoniaResidual(
                nh3_lower,
                lower_box=True,
            ),
            "gan": GaNResidual(gan,),
            "gan_interface_temperature": (
                GaNInterfaceTemperatureResidual()
            ),
            "gan_interface_flux": (
                GaNInterfaceFluxResidual()
            ),
            "interface_temperature": (
                InterfaceTemperatureResidual()
            ),
            "interface_flux": (
                InterfaceFluxResidual()
            ),
        }

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


def build_space(seed=42):
    key = jax.random.PRNGKey(seed)

    key_inc, key_nh3, key_gan_center, key_gan_slope = jax.random.split(key, 4)

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

    # GaN: two small r-only networks (see gan_component_preprocessing /
    # combine_gan_field) instead of one full (rho, eta) network -- this makes
    # GaN's field exactly affine in z by construction, removing the
    # degenerate-slope ambiguity a free-form network + interior PDE residual
    # alone cannot resolve for a constant-conductivity material.
    nn_gan_center = build_mlp(
        in_size=1,
        out_size=1,
        hidden_sizes=GAN_COMPONENT_HIDDEN,
        key=key_gan_center,
    )

    nn_gan_slope = build_mlp(
        in_size=1,
        out_size=1,
        hidden_sizes=GAN_COMPONENT_HIDDEN,
        key=key_gan_slope,
    )

    return ApproximationSpace(
        {"x": 2},
        [
            (nn_inc, "scalar", None),
            (nn_nh3, "scalar", None),
            (nn_gan_center, "scalar", None),
            (nn_gan_slope, "scalar", None),
        ],
        model_type="x",
        pre_processing=[
            scaled_coordinates,
            nh3_scaled_coordinates,
            gan_component_preprocessing,
            gan_component_preprocessing,
        ],
        post_processing=[
            inconel_post_processing,
            nh3_post_processing,
            gan_center_post_processing,
            gan_slope_post_processing,
        ],
    )


W_COMMON = {
    "inc_bottom": (10.0,),
    "inc_side": (10.0,),
    "inc_top": (10.0,),
    "inc_lower": (10.0,),
    "nh3_upper": (10.0,),
    "nh3_lower": (10.0,),
    "gan": (10.0,),
    "gan_interface_temperature": (100.0,),
    "gan_interface_flux": (100.0,),
    "interface_temperature": (100.0,),
    "interface_flux": (100.0,),
}


# Plotting

def z_bottom_ammonia_numpy(r):
    """
    Evaluate the CryS-MAS lower NH3 boundary using the supplied
    piecewise-linear geometry points.

    Parameters
    ----------
    r : array-like
        Radial coordinate(s) [m].

    Returns
    -------
    ndarray
        Lower NH3 boundary axial coordinate z_bottom(r) [m].
    """
    r = np.asarray(
        r,
        dtype=np.float64,
    )

    return np.interp(
        np.clip(
            r,
            float(NH3_LOWER_R[0]),
            float(NH3_LOWER_R[-1]),
        ),
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
    """
    Return a Boolean mask for all points in the physical NH3 cavity,
    including points occupied by the internal GaN block.
    """
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


def in_lower_ammonia_numpy(r, z):
    """
    Return a Boolean mask for the physical NH3 points in the CryS-MAS
    lower-envelope region (r <= R_NH3, below the upper cavity).
    """
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    return (
        (r >= 0.0)
        & (r <= R_NH3)
        & (z >= z_bottom_ammonia_numpy(r))
        & (z <= Z_NH3_MAX_LOWER)
    )


def in_gan_numpy(r, z):
    """
    Return a Boolean mask for the single continuous CryS-MAS GaN block.

    GaN geometry:
        0 <= r <= R_GAN
        Z_GAN_MIN <= z <= Z_GAN_MAX
    """
    r = np.asarray(
        r,
        dtype=np.float64,
    )

    z = np.asarray(
        z,
        dtype=np.float64,
    )

    return (
        (r >= 0.0)
        & (r <= R_GAN)
        & (z >= Z_GAN_MIN)
        & (z <= Z_GAN_MAX)
    )


def in_nh3_without_gan_numpy(r, z):
    """
    Return a Boolean mask for NH3 only.

    The GaN block lies inside the NH3 cavity, so GaN points are
    explicitly removed from the NH3 material mask.
    """
    return (
        in_ammonia_numpy(r, z)
        & (~in_gan_numpy(r, z))
    )


def material_masks_numpy(r, z):
    """
    Return Boolean masks in the order:

        Inconel mask,
        NH3 mask,
        GaN mask.

    GaN has higher priority than NH3 because GaN occupies a part of
    the NH3 cavity.
    """
    gan_mask = in_gan_numpy(
        r,
        z,
    )

    nh3_mask = in_nh3_without_gan_numpy(
        r,
        z,
    )

    inc_mask = ~(
        gan_mask
        | nh3_mask
    )

    return (
        inc_mask,
        nh3_mask,
        gan_mask,
    )


def evaluate_field(space, points, material):
    """
    Evaluate one trained PINN temperature field.

    Parameters
    ----------
    space : ApproximationSpace
        Trained PINN approximation space, normally pinn.space.
    points : ndarray, shape (N, 2)
        Physical coordinates [r, z] in m.
    material : {"inc", "nh3", "gan"}
        Field to evaluate.

    Returns
    -------
    ndarray, shape (N,)
        PINN temperatures [K].
    """
    x = jnp.asarray(
        points,
        dtype=jnp.float64,
    )

    if material == "inc":
        field = T_INC

    elif material == "nh3":
        field = T_NH3

    elif material == "gan":
        field = T_GAN

    else:
        raise ValueError(
            f"Unknown material '{material}'. "
            "Use 'inc', 'nh3', or 'gan'."
        )

    return np.asarray(
        field(
            space,
            x,
        )
    ).reshape(-1)


def draw_geometry(ax):
    """Draw the CryS-MAS one-GaN material-interface outlines on ax."""
    # GaN block outline.
    gan_r = np.array([0.0, R_GAN, R_GAN, 0.0, 0.0], dtype=np.float64)
    gan_z = np.array(
        [Z_GAN_MIN, Z_GAN_MIN, Z_GAN_MAX, Z_GAN_MAX, Z_GAN_MIN],
        dtype=np.float64,
    )
    ax.plot(gan_r * 1.0e3, gan_z * 1.0e3, "k", lw=1.8, label="GaN boundary")

    # NH3/Inconel cylindrical sidewall interface.
    ax.plot(
        np.full(2, R_NH3 * 1.0e3),
        np.array([float(NH3_LOWER_Z[-1]), Z_NH3_TOP], dtype=np.float64) * 1.0e3,
        "k", lw=1.4,
    )

    # NH3/Inconel upper interface.
    ax.plot(
        np.array([0.0, R_NH3], dtype=np.float64) * 1.0e3,
        np.full(2, Z_NH3_TOP * 1.0e3),
        "k", lw=1.4,
    )

    # Piecewise-linear lower NH3/Inconel interface.
    ax.plot(
        np.asarray(NH3_LOWER_R, dtype=np.float64) * 1.0e3,
        np.asarray(NH3_LOWER_Z, dtype=np.float64) * 1.0e3,
        "k", lw=1.4,
    )

    # Outer Inconel boundaries.
    ax.plot(
        np.full(2, R_OUT * 1.0e3),
        np.array([0.0, H_OUT], dtype=np.float64) * 1.0e3,
        "k", lw=1.4,
    )
    ax.plot(
        np.array([0.0, R_OUT], dtype=np.float64) * 1.0e3,
        np.zeros(2, dtype=np.float64),
        "k", lw=1.4,
    )
    ax.plot(
        np.array([0.0, R_OUT], dtype=np.float64) * 1.0e3,
        np.full(2, H_OUT * 1.0e3),
        "k", lw=1.4,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")


# plot_solution: higher-resolution filled contour + isotherm lines, plus an
# optional third figure (temperature_field_vs_fvm_error.png) showing PINN
# error against the corrected FVM reference if fvm_df is given (as returned
# by load_fvm_reference()). Same three-figure convention as the plain
# Inconel+NH3 script, extended to the GaN field/mask.
def plot_solution(space, n_r=400, n_z=1200, fvm_df=None):
    """
    Plot the full three-material CryS-MAS one-GaN PINN temperature field.

    Field selection:
    - T_INC in the Inconel vessel
    - T_NH3 in the ammonia cavity excluding GaN
    - T_GAN in the one continuous GaN block
    """
    r = np.linspace(0.0, R_OUT, n_r)
    z = np.linspace(0.0, H_OUT, n_z)

    rr, zz = np.meshgrid(r, z, indexing="xy")

    points = np.column_stack([rr.ravel(), zz.ravel()])

    inc_mask, nh3_mask, gan_mask = material_masks_numpy(
        points[:, 0], points[:, 1],
    )

    temperature = np.empty(points.shape[0], dtype=np.float64)

    if np.any(inc_mask):
        temperature[inc_mask] = evaluate_field(space, points[inc_mask], "inc")

    if np.any(nh3_mask):
        temperature[nh3_mask] = evaluate_field(space, points[nh3_mask], "nh3")

    if np.any(gan_mask):
        temperature[gan_mask] = evaluate_field(space, points[gan_mask], "gan")

    temperature = temperature.reshape(rr.shape)

    # ── Plot 1: filled contour ──
    figure, ax = plt.subplots(figsize=(5.8, 10.0))

    contour = ax.contourf(
        rr * 1.0e3, zz * 1.0e3, temperature, levels=100, cmap="turbo",
    )
    colorbar = figure.colorbar(contour, ax=ax)
    colorbar.set_label("Temperature [K]")

    draw_geometry(ax)
    ax.set_title(
        "CryS-MAS one-GaN conduction\n"
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
        "CryS-MAS one-GaN conduction\n"
        f"Isotherms every {(line_levels[1] - line_levels[0]):.2f} K"
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
        r_fvm = fvm_df["x"].to_numpy(dtype=np.float64)
        z_fvm = fvm_df["y"].to_numpy(dtype=np.float64)
        T_fvm = fvm_df["Temperature"].to_numpy(dtype=np.float64)

        fvm_points = np.column_stack([r_fvm, z_fvm])
        inc_mask_fvm, nh3_mask_fvm, gan_mask_fvm = material_masks_numpy(
            r_fvm, z_fvm,
        )

        T_pinn_fvm = np.empty(len(T_fvm), dtype=np.float64)
        if np.any(inc_mask_fvm):
            T_pinn_fvm[inc_mask_fvm] = evaluate_field(
                space, fvm_points[inc_mask_fvm], "inc",
            )
        if np.any(nh3_mask_fvm):
            T_pinn_fvm[nh3_mask_fvm] = evaluate_field(
                space, fvm_points[nh3_mask_fvm], "nh3",
            )
        if np.any(gan_mask_fvm):
            T_pinn_fvm[gan_mask_fvm] = evaluate_field(
                space, fvm_points[gan_mask_fvm], "gan",
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


def make_axisymmetric_residual(temperature_function, conductivity_function, pde_ref):
    """
    Build a standalone (space, x) -> residual function for the scaled,
    r-weighted axisymmetric conduction PDE, evaluated via jax.grad directly
    on temperature_function -- independent of the ParamScalarFunction/space
    residual machinery, so it can be plotted on an arbitrary diagnostic grid.
    Normalized by the same material-specific pde_ref used during training
    (PDE_R_WEIGHTED_REF_INC/NH3/GAN), so the diagnostic map matches what the
    optimizer actually sees.
    """

    def residual_on_points(space, x):
        x = jnp.asarray(x, dtype=jnp.float64)

        def temperature_at_one_point(x_one):
            return jnp.ravel(temperature_function(space, x_one))[0]

        def radial_flux_at_one_point(x_one):
            temperature = temperature_at_one_point(x_one)
            dT_dr = jax.grad(temperature_at_one_point)(x_one)[0]
            conductivity = conductivity_function(temperature)
            return conductivity * dT_dr

        def axial_flux_at_one_point(x_one):
            temperature = temperature_at_one_point(x_one)
            dT_dz = jax.grad(temperature_at_one_point)(x_one)[1]
            conductivity = conductivity_function(temperature)
            return conductivity * dT_dz

        radial_flux_gradient = jax.grad(radial_flux_at_one_point)
        axial_flux_gradient = jax.grad(axial_flux_at_one_point)

        def residual_at_one_point(x_one):
            r = x_one[0]
            q_r = radial_flux_at_one_point(x_one)
            dq_r_dr = radial_flux_gradient(x_one)[0]
            dq_z_dz = axial_flux_gradient(x_one)[1]

            raw_residual = r * dq_r_dr + q_r + r * dq_z_dz

            return raw_residual / pde_ref

        return jax.vmap(residual_at_one_point)(x)

    return residual_on_points


inc_conduction_residual = make_axisymmetric_residual(
    T_inc_eval, k_inc_value, PDE_R_WEIGHTED_REF_INC,
)
nh3_conduction_residual = make_axisymmetric_residual(
    T_nh3_eval, k_nh3_value, PDE_R_WEIGHTED_REF_NH3,
)
gan_conduction_residual = make_axisymmetric_residual(
    T_gan_eval, k_gan_value, PDE_R_WEIGHTED_REF_GAN,
)


def apply_interior_material_mask(residual_name, r_grid, z_grid, values):
    """
    Return PDE residual values inside the correct physical material
    region and NaN outside that region.
    """
    r_np = np.asarray(r_grid, dtype=np.float64)
    z_np = np.asarray(z_grid, dtype=np.float64)
    values_np = np.asarray(values, dtype=np.float64)

    if residual_name in ("inc_bottom", "inc_side", "inc_top", "gan"):
        # These rectangular envelopes are entirely one physical material.
        mask = np.ones_like(r_np, dtype=bool)

    elif residual_name == "inc_lower":
        mask = ~in_lower_ammonia_numpy(r_np, z_np)

    elif residual_name == "nh3_lower":
        mask = in_lower_ammonia_numpy(r_np, z_np)

    elif residual_name == "nh3_upper":
        mask = in_nh3_without_gan_numpy(r_np, z_np)

    else:
        raise ValueError(f"Unknown interior residual name: {residual_name}")

    return np.where(mask, values_np, np.nan)


def plot_all_interior_residuals(space, n_r=151, n_z=251):
    residual_regions = (
        ("inc_bottom", "Inconel: bottom", inc_conduction_residual,
         0.0, R_OUT, 0.0, Z_NH3_MIN),
        ("inc_lower", "Inconel: lower polyline region", inc_conduction_residual,
         0.0, R_NH3, Z_NH3_MIN, Z_NH3_MAX_LOWER),
        ("inc_side", "Inconel: side region", inc_conduction_residual,
         R_NH3, R_OUT, Z_NH3_MIN, Z_NH3_TOP),
        ("inc_top", "Inconel: top", inc_conduction_residual,
         0.0, R_OUT, Z_NH3_TOP, H_OUT),
        ("nh3_lower", "NH3: lower polyline region", nh3_conduction_residual,
         0.0, R_NH3, Z_NH3_MIN, Z_NH3_MAX_LOWER),
        ("nh3_upper", "NH3: upper cavity region", nh3_conduction_residual,
         0.0, R_NH3, Z_NH3_MAX_LOWER, Z_NH3_TOP),
        ("gan", "GaN block", gan_conduction_residual,
         0.0, R_GAN, Z_GAN_MIN, Z_GAN_MAX),
    )

    plot_data = []

    for (residual_name, title, residual_function,
         r_min, r_max, z_min, z_max) in residual_regions:

        r_plot_min = max(r_min, 10.0 * EPS_AXIS)

        r_values = jnp.linspace(r_plot_min, r_max, n_r)
        z_values = jnp.linspace(z_min, z_max, n_z)
        r_grid, z_grid = jnp.meshgrid(r_values, z_values, indexing="xy")

        x_eval = jnp.column_stack([r_grid.reshape(-1), z_grid.reshape(-1)])

        residual_values = residual_function(space, x_eval)
        residual_values = jnp.asarray(residual_values).reshape(n_z, n_r)

        masked_values = apply_interior_material_mask(
            residual_name, r_grid, z_grid, residual_values,
        )

        absolute_values = np.abs(masked_values)

        positive_values = absolute_values[
            np.isfinite(absolute_values) & (absolute_values > 0.0)
        ]

        if positive_values.size == 0:
            raise ValueError(
                f"No positive residual values remain for {residual_name}."
            )

        print(
            f"{residual_name}: "
            f"min |R| = {np.min(positive_values):.3e}, "
            f"max |R| = {np.max(positive_values):.3e}, "
            f"mean |R| = {np.mean(positive_values):.3e}"
        )

        plot_data.append(
            (residual_name, title, absolute_values,
             float(r_plot_min), float(r_max), float(z_min), float(z_max))
        )

    all_positive_values = np.concatenate(
        [
            values[np.isfinite(values) & (values > 0.0)]
            for (_, _, values, _, _, _, _) in plot_data
        ]
    )

    common_vmin = np.min(all_positive_values)
    common_vmax = np.max(all_positive_values)

    n_regions = len(plot_data)
    n_cols = 4
    n_rows = int(np.ceil(n_regions / n_cols))

    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 5.0 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    image = None

    for ax, (residual_name, title, absolute_values,
             r_min, r_max, z_min, z_max) in zip(axes, plot_data):

        plot_values = np.where(absolute_values > 0.0, absolute_values, np.nan)

        image = ax.pcolormesh(
            np.linspace(r_min * 1.0e3, r_max * 1.0e3, n_r),
            np.linspace(z_min * 1.0e3, z_max * 1.0e3, n_z),
            plot_values,
            shading="auto", cmap="magma",
            norm=LogNorm(vmin=common_vmin, vmax=common_vmax),
        )

        ax.set_title(f"{title}\n" r"$|\mathcal{R}_{\mathrm{PDE}}|$")
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("z [mm]")

    for ax in axes[n_regions:]:
        ax.axis("off")

    colorbar = figure.colorbar(image, ax=axes.tolist(), shrink=0.88)
    colorbar.set_label(r"Absolute scaled PDE residual $|\mathcal{R}_{\mathrm{PDE}}|$")

    output_path = SAVE_DIR / "all_interior_residual_maps.png"
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print("Saved interior residual maps in:")
    print(output_path.resolve())


# Corrected CryS-MAS reference data (2D_Heat_data.csv, one-GaN geometry).
# Same format used for the plain Inconel+NH3 and two-GaN cases: a sweep
# over 25 (T0, T1) combinations, 36 radial lines each. This loader keeps
# only the combination matching this script's own (T0_K, T1).
def load_fvm_reference(csv_path, t0=T0_K, t1=T1):
    csv_path = Path(csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Corrected FVM reference CSV not found:\n{csv_path}"
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

    return df[["x", "y", "Temperature"]].dropna().reset_index(drop=True)


# FVM versus PINN comparison along every available radial line (36 lines,
# r = 0 to 35 mm in 1 mm steps), gridded into three figures (temperature
# overlay, signed error, relative error) plus a per-radius error-summary
# table -- same pattern used for the plain Inconel+NH3 script.
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
    output_name="fvm_vs_pinn_all_radii.png",
    error_output_name="fvm_vs_pinn_all_radii_error.png",
    relative_error_output_name="fvm_vs_pinn_all_radii_relative_error.png",
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
        inc_mask, nh3_mask, gan_mask = material_masks_numpy(
            points[:, 0], points[:, 1],
        )

        T_pinn_line = np.empty_like(T_fvm_line)
        if np.any(inc_mask):
            T_pinn_line[inc_mask] = evaluate_field(
                space, points[inc_mask], "inc",
            )
        if np.any(nh3_mask):
            T_pinn_line[nh3_mask] = evaluate_field(
                space, points[nh3_mask], "nh3",
            )
        if np.any(gan_mask):
            T_pinn_line[gan_mask] = evaluate_field(
                space, points[gan_mask], "gan",
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

        def draw_interfaces(ax, r_probe=r_probe):
            if 0.0 <= r_probe <= R_NH3:
                z_nh3_bottom = float(
                    z_bottom_ammonia_numpy(np.array([r_probe], dtype=np.float64))[0]
                )
                for z_interface in (z_nh3_bottom, Z_NH3_TOP):
                    ax.axvline(
                        z_interface * 1.0e3, color="0.5", lw=0.8, ls="--", alpha=0.70,
                    )
            if 0.0 <= r_probe <= R_GAN:
                for z_interface in (Z_GAN_MIN, Z_GAN_MAX):
                    ax.axvline(
                        z_interface * 1.0e3, color="0.3", lw=0.8, ls=":", alpha=0.70,
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

    summary_csv = SAVE_DIR / "fvm_vs_pinn_all_radii_summary.csv"
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


def plot_fvm_vs_pinn_2d_error(
    space,
    df,
    output_name="fvm_vs_pinn_one_gan_2d_error.png",
):
    r_fvm = df["x"].to_numpy(dtype=np.float64)
    z_fvm = df["y"].to_numpy(dtype=np.float64)
    T_fvm = df["Temperature"].to_numpy(dtype=np.float64)

    fvm_points = np.column_stack([r_fvm, z_fvm])

    inc_mask, nh3_mask, gan_mask = material_masks_numpy(r_fvm, z_fvm)

    assigned_mask = inc_mask | nh3_mask | gan_mask

    if not np.all(assigned_mask):
        outside_count = np.count_nonzero(~assigned_mask)
        raise ValueError(
            f"{outside_count} FVM nodes are outside the Inconel, NH3, and "
            "GaN geometry masks. Check that the CSV coordinates are in "
            "metres and use the same geometry as this PINN model."
        )

    T_pinn = np.empty(len(T_fvm), dtype=np.float64)

    if np.any(inc_mask):
        T_pinn[inc_mask] = evaluate_field(space, fvm_points[inc_mask], "inc")
    if np.any(nh3_mask):
        T_pinn[nh3_mask] = evaluate_field(space, fvm_points[nh3_mask], "nh3")
    if np.any(gan_mask):
        T_pinn[gan_mask] = evaluate_field(space, fvm_points[gan_mask], "gan")

    signed_error_K = T_pinn - T_fvm
    absolute_error_K = np.abs(signed_error_K)

    rmse_K = np.sqrt(np.mean(signed_error_K**2))
    mae_K = np.mean(absolute_error_K)
    max_abs_error_K = np.max(absolute_error_K)

    material = np.select(
        [gan_mask, nh3_mask, inc_mask],
        ["GaN", "NH3", "Inconel"],
        default="Unknown",
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
    print(f"Unique radial coordinates: {np.unique(r_fvm).size}")
    print(f"Unique axial coordinates: {np.unique(z_fvm).size}")
    print(f"Global RMSE [K]: {rmse_K:.8e}")
    print(f"Global MAE [K]: {mae_K:.8e}")
    print(f"Global maximum absolute error [K]: {max_abs_error_K:.8e}")

    print("\nPER-MATERIAL ERROR METRICS")
    print("-" * 72)

    for material_name in ("Inconel", "NH3", "GaN"):
        mask = material == material_name

        if not np.any(mask):
            print(f"{material_name}: no FVM points found")
            continue

        local_error_K = signed_error_K[mask]
        local_absolute_error_K = absolute_error_K[mask]

        local_rmse_K = np.sqrt(np.mean(local_error_K**2))
        local_mae_K = np.mean(local_absolute_error_K)
        local_max_error_K = np.max(local_absolute_error_K)
        local_mean_signed_error_K = np.mean(local_error_K)

        print(
            f"{material_name}: "
            f"N={np.count_nonzero(mask)}, "
            f"mean signed error={local_mean_signed_error_K:.6e} K, "
            f"RMSE={local_rmse_K:.6e} K, "
            f"MAE={local_mae_K:.6e} K, "
            f"Max|error|={local_max_error_K:.6e} K"
        )

    signed_limit = np.max(np.abs(signed_error_K))
    signed_norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0.0, vmax=signed_limit)

    figure, axes = plt.subplots(1, 2, figsize=(15.0, 8.0), constrained_layout=True)

    signed_plot = axes[0].scatter(
        r_fvm * 1.0e3, z_fvm * 1.0e3,
        c=signed_error_K, s=12, marker="s",
        cmap="coolwarm", norm=signed_norm,
        linewidths=0.0, edgecolors="none",
    )
    axes[0].set_title(r"Signed nodal error: $T_{\mathrm{PINN}}-T_{\mathrm{FVM}}$")
    axes[0].set_xlabel("r [mm]")
    axes[0].set_ylabel("z [mm]")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.20)
    signed_colorbar = figure.colorbar(signed_plot, ax=axes[0])
    signed_colorbar.set_label("PINN - FVM [K]")

    absolute_plot = axes[1].scatter(
        r_fvm * 1.0e3, z_fvm * 1.0e3,
        c=absolute_error_K, s=12, marker="s",
        cmap="magma", vmin=0.0, vmax=max_abs_error_K,
        linewidths=0.0, edgecolors="none",
    )
    axes[1].set_title(r"Absolute nodal error: $|T_{\mathrm{PINN}}-T_{\mathrm{FVM}}|$")
    axes[1].set_xlabel("r [mm]")
    axes[1].set_ylabel("z [mm]")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.20)
    absolute_colorbar = figure.colorbar(absolute_plot, ax=axes[1])
    absolute_colorbar.set_label("Absolute error [K]")

    figure.suptitle(
        (
            "CryS-MAS one-GaN FVM--PINN 2D Nodal Temperature Error\n"
            f"Nodes = {len(T_fvm)}, "
            f"RMSE = {rmse_K:.3f} K, "
            f"MAE = {mae_K:.3f} K, "
            f"Max |error| = {max_abs_error_K:.3f} K"
        ),
        fontsize=14,
    )

    output_path = SAVE_DIR / output_name
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    csv_path = SAVE_DIR / "fvm_vs_pinn_2d_error_data.csv"
    comparison_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 72)
    print("FVM--PINN 2D ERROR FILES")
    print("=" * 72)
    print(f"Saved direct-node error plot in:\n{output_path.resolve()}")
    print(f"Saved pointwise comparison table in:\n{csv_path.resolve()}")
    print(f"Rows written to comparison CSV: {len(comparison_df)}")

    return comparison_df


def plot_fvm_vs_pinn_2d_relative_error(
    space,
    df,
    output_name="fvm_vs_pinn_one_gan_2d_relative_error.png",
):
    """
    Plot the signed and absolute relative error
    (PINN - FVM) / FVM [%] over the full 2D FVM node set.
    """
    r_fvm = df["x"].to_numpy(dtype=np.float64)
    z_fvm = df["y"].to_numpy(dtype=np.float64)
    T_fvm = df["Temperature"].to_numpy(dtype=np.float64)

    fvm_points = np.column_stack([r_fvm, z_fvm])

    inc_mask, nh3_mask, gan_mask = material_masks_numpy(r_fvm, z_fvm)

    assigned_mask = inc_mask | nh3_mask | gan_mask

    if not np.all(assigned_mask):
        outside_count = np.count_nonzero(~assigned_mask)
        raise ValueError(
            f"{outside_count} FVM nodes are outside the Inconel, NH3, and "
            "GaN geometry masks. Check that the CSV coordinates are in "
            "metres and use the same geometry as this PINN model."
        )

    T_pinn = np.empty(len(T_fvm), dtype=np.float64)

    if np.any(inc_mask):
        T_pinn[inc_mask] = evaluate_field(space, fvm_points[inc_mask], "inc")
    if np.any(nh3_mask):
        T_pinn[nh3_mask] = evaluate_field(space, fvm_points[nh3_mask], "nh3")
    if np.any(gan_mask):
        T_pinn[gan_mask] = evaluate_field(space, fvm_points[gan_mask], "gan")

    relative_error_pct = (T_pinn - T_fvm) / T_fvm * 100.0
    absolute_relative_error_pct = np.abs(relative_error_pct)

    rmse_pct = np.sqrt(np.mean(relative_error_pct**2))
    mae_pct = np.mean(absolute_relative_error_pct)
    max_abs_relative_error_pct = np.max(absolute_relative_error_pct)

    material = np.select(
        [gan_mask, nh3_mask, inc_mask],
        ["GaN", "NH3", "Inconel"],
        default="Unknown",
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
    print(f"Global maximum absolute relative error [%]: {max_abs_relative_error_pct:.8e}")

    print("\nPER-MATERIAL RELATIVE ERROR METRICS")
    print("-" * 72)

    for material_name in ("Inconel", "NH3", "GaN"):
        mask = material == material_name

        if not np.any(mask):
            print(f"{material_name}: no FVM points found")
            continue

        local_relative_error_pct = relative_error_pct[mask]
        local_absolute_relative_error_pct = absolute_relative_error_pct[mask]

        local_rmse_pct = np.sqrt(np.mean(local_relative_error_pct**2))
        local_mae_pct = np.mean(local_absolute_relative_error_pct)
        local_max_error_pct = np.max(local_absolute_relative_error_pct)
        local_mean_signed_error_pct = np.mean(local_relative_error_pct)

        print(
            f"{material_name}: "
            f"N={np.count_nonzero(mask)}, "
            f"mean signed relative error={local_mean_signed_error_pct:.6e} %, "
            f"RMSE={local_rmse_pct:.6e} %, "
            f"MAE={local_mae_pct:.6e} %, "
            f"Max|error|={local_max_error_pct:.6e} %"
        )

    signed_limit = np.max(np.abs(relative_error_pct))
    signed_norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0.0, vmax=signed_limit)

    figure, axes = plt.subplots(1, 2, figsize=(15.0, 8.0), constrained_layout=True)

    signed_plot = axes[0].scatter(
        r_fvm * 1.0e3, z_fvm * 1.0e3,
        c=relative_error_pct, s=12, marker="s",
        cmap="coolwarm", norm=signed_norm,
        linewidths=0.0, edgecolors="none",
    )
    axes[0].set_title(
        r"Signed nodal relative error: $(T_{\mathrm{PINN}}-T_{\mathrm{FVM}})/T_{\mathrm{FVM}}$"
    )
    axes[0].set_xlabel("r [mm]")
    axes[0].set_ylabel("z [mm]")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.20)
    signed_colorbar = figure.colorbar(signed_plot, ax=axes[0])
    signed_colorbar.set_label("(PINN - FVM) / FVM [%]")

    absolute_plot = axes[1].scatter(
        r_fvm * 1.0e3, z_fvm * 1.0e3,
        c=absolute_relative_error_pct, s=12, marker="s",
        cmap="magma", vmin=0.0, vmax=max_abs_relative_error_pct,
        linewidths=0.0, edgecolors="none",
    )
    axes[1].set_title(
        r"Absolute nodal relative error: $|T_{\mathrm{PINN}}-T_{\mathrm{FVM}}|/T_{\mathrm{FVM}}$"
    )
    axes[1].set_xlabel("r [mm]")
    axes[1].set_ylabel("z [mm]")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.20)
    absolute_colorbar = figure.colorbar(absolute_plot, ax=axes[1])
    absolute_colorbar.set_label("|(PINN - FVM) / FVM| [%]")

    figure.suptitle(
        (
            "CryS-MAS one-GaN FVM--PINN 2D Nodal Relative Temperature Error\n"
            f"Nodes = {len(T_fvm)}, "
            f"Relative RMSE = {rmse_pct:.3f} %, "
            f"Relative MAE = {mae_pct:.3f} %, "
            f"Max |relative error| = {max_abs_relative_error_pct:.3f} %"
        ),
        fontsize=14,
    )

    output_path = SAVE_DIR / output_name
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    return comparison_df


# Training loss-history plotting

def plot_loss_history(
    losses,
    output_name="loss_history.png",
):
    """
    Save one loss plot for each neural material field:

    1. Inconel network:
       - Inconel PDE residuals
       - Inconel symmetry-axis residual
       - Inconel/NH3 temperature interface
       - Inconel/NH3 flux interface

    2. NH3 network:
       - NH3 PDE residuals
       - NH3 symmetry-axis residual
       - Inconel/NH3 temperature interface
       - Inconel/NH3 flux interface
       - GaN/NH3 temperature interface
       - GaN/NH3 flux interface

    3. GaN network:
       - GaN PDE residual
       - GaN symmetry-axis residual
       - GaN/NH3 temperature interface
       - GaN/NH3 flux interface

    It also saves the original combined loss plot.
    """
    history_dict = losses.losses_history

    if not isinstance(
        history_dict,
        dict,
    ):
        raise TypeError(
            "Expected losses.losses_history to be a "
            "dictionary, but got "
            f"{type(history_dict)}."
        )

    if "total" not in history_dict:
        raise KeyError(
            "No 'total' entry found in "
            "losses.losses_history. "
            f"Available keys: "
            f"{list(history_dict.keys())}"
        )

    total_loss = np.asarray(
        history_dict["total"],
        dtype=np.float64,
    ).reshape(-1)

    n_epochs = total_loss.size

    epoch = np.arange(
        1,
        n_epochs + 1,
    )

    # Helper: retrieve a loss history. Missing residuals are
    # treated as zero so plotting remains robust if you later
    # rename/remove a residual.
    def get_history(key):
        if key not in history_dict:
            print(
                f"Warning: loss key '{key}' was not found. "
                "Using zeros for this term."
            )

            return np.zeros(
                n_epochs,
                dtype=np.float64,
            )

        values = np.asarray(
            history_dict[key],
            dtype=np.float64,
        ).reshape(-1)

        if values.size != n_epochs:
            raise ValueError(
                f"History length mismatch for '{key}': "
                f"{values.size} values, expected "
                f"{n_epochs}."
            )

        return values

    # Helper: sum several residual histories.
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

    # Residual ownership by neural field

    # T_INC / Inconel network:
    # - all Inconel PDE regions
    # - Inconel axis condition
    # - Inconel-NH3 temperature and flux continuity

    # The interface losses involve both networks, so they are
    # intentionally included in both corresponding plots.

    inconel_pde_keys = [
        "inc_bottom",
        "inc_side",
        "inc_top",
        "inc_lower",
    ]

    inconel_interface_keys = [
        "interface_temperature",
        "interface_flux",
    ]

    inconel_all_keys = (
        inconel_pde_keys
        + inconel_interface_keys
    )

    # T_NH3 / ammonia network:
    # - both NH3 PDE envelopes
    # - NH3 axis conditions
    # - Inconel-NH3 interfaces
    # - GaN-NH3 interfaces

    nh3_pde_keys = [
        "nh3_lower",
        "nh3_upper",
    ]

    nh3_inc_interface_keys = [
        "interface_temperature",
        "interface_flux",
    ]

    nh3_gan_interface_keys = [
        "gan_interface_temperature",
        "gan_interface_flux",
    ]

    nh3_all_keys = (
        nh3_pde_keys
        + nh3_inc_interface_keys
        + nh3_gan_interface_keys
    )

    # T_GAN / GaN network:
    # - one continuous GaN interior PDE region
    # - GaN-NH3 temperature and flux interfaces

    gan_pde_keys = [
        "gan",
    ]

    gan_interface_keys = [
        "gan_interface_temperature",
        "gan_interface_flux",
    ]

    gan_all_keys = (
        gan_pde_keys
        + gan_interface_keys
    )

    # Construct the summed curves for all three neural fields.
    inconel_pde_loss = summed_history(
        inconel_pde_keys,
    )

    inconel_interface_loss = summed_history(
        inconel_interface_keys,
    )

    inconel_total_related_loss = summed_history(
        inconel_all_keys,
    )

    nh3_pde_loss = summed_history(
        nh3_pde_keys,
    )

    nh3_inc_interface_loss = summed_history(
        nh3_inc_interface_keys,
    )

    nh3_gan_interface_loss = summed_history(
        nh3_gan_interface_keys,
    )

    nh3_total_related_loss = summed_history(
        nh3_all_keys,
    )

    gan_pde_loss = summed_history(
        gan_pde_keys,
    )

    gan_interface_loss = summed_history(
        gan_interface_keys,
    )

    gan_total_related_loss = summed_history(
        gan_all_keys,
    )

    # Helper: plot only finite positive values on log scale.
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

    # Helper: save a single material-network loss plot.
    def save_network_plot(
        title,
        output_filename,
        curves,
    ):
        fig, ax = plt.subplots(
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

        fig.tight_layout()

        figure_path = (
            SAVE_DIR
            / output_filename
        )

        fig.savefig(
            figure_path,
            dpi=250,
            bbox_inches="tight",
        )

        plt.close(
            fig,
        )

        print(
            "Saved network loss plot in:\n"
            f"{figure_path.resolve()}"
        )

    # Plot 1: Inconel network
    save_network_plot(
        title="Inconel network loss history",
        output_filename="loss_history_inconel.png",
        curves=[
            {
                "values": inconel_pde_loss,
                "label": "Inconel interior PDE loss",
                "color": "tab:blue",
            },
            {
                "values": inconel_interface_loss,
                "label": "Inconel-NH3 interface loss",
                "color": "tab:orange",
            },
            {
                "values": inconel_total_related_loss,
                "label": "Total Inconel-related loss",
                "color": "tab:red",
                "linewidth": 2.2,
            },
        ],
    )

    # Plot 2: NH3 network
    save_network_plot(
        title="NH3 network loss history",
        output_filename="loss_history_nh3.png",
        curves=[
            {
                "values": nh3_pde_loss,
                "label": "NH3 interior PDE loss",
                "color": "tab:blue",
            },
            {
                "values": nh3_inc_interface_loss,
                "label": "NH3-Inconel interface loss",
                "color": "tab:orange",
            },
            {
                "values": nh3_gan_interface_loss,
                "label": "NH3-GaN interface loss",
                "color": "tab:purple",
            },
            {
                "values": nh3_total_related_loss,
                "label": "Total NH3-related loss",
                "color": "tab:red",
                "linewidth": 2.2,
            },
        ],
    )

    # Plot 3: GaN network
    save_network_plot(
        title="GaN network loss history",
        output_filename="loss_history_gan.png",
        curves=[
            {
                "values": gan_pde_loss,
                "label": "GaN interior PDE loss",
                "color": "tab:blue",
            },
            {
                "values": gan_interface_loss,
                "label": "GaN-NH3 interface loss",
                "color": "tab:orange",
            },
            {
                "values": gan_total_related_loss,
                "label": "Total GaN-related loss",
                "color": "tab:red",
                "linewidth": 2.2,
            },
        ],
    )

    # Optional fourth plot: global training loss.
    # This preserves the behavior of your original function.
    all_interior_keys = [
        "inc_bottom",
        "inc_side",
        "inc_top",
        "inc_lower",
        "nh3_upper",
        "nh3_lower",
        "gan",
    ]

    all_boundary_keys = [
        "gan_interface_temperature",
        "gan_interface_flux",
        "interface_temperature",
        "interface_flux",
    ]

    all_interior_loss = summed_history(
        all_interior_keys,
    )

    all_boundary_loss = summed_history(
        all_boundary_keys,
    )

    save_network_plot(
        title="Global PINN training-loss history",
        output_filename=output_name,
        curves=[
            {
                "values": all_interior_loss,
                "label": "All interior PDE losses",
                "color": "tab:blue",
            },
            {
                "values": all_boundary_loss,
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
        "inconel_total_related_loss": inconel_total_related_loss,

        "nh3_pde_loss": nh3_pde_loss,
        "nh3_inc_interface_loss": nh3_inc_interface_loss,
        "nh3_gan_interface_loss": nh3_gan_interface_loss,
        "nh3_total_related_loss": nh3_total_related_loss,

        "gan_pde_loss": gan_pde_loss,
        "gan_interface_loss": gan_interface_loss,
        "gan_total_related_loss": gan_total_related_loss,

        "global_interior_loss": all_interior_loss,
        "global_boundary_loss": all_boundary_loss,
        "global_total_loss": total_loss,
    }


# Three-network diagnostic for the one-GaN geometry

def print_three_field_values(space):
    """
    Evaluate all three neural fields at representative points.

    The physical-material column shows which field should be selected
    by the one-GaN CryS-MAS geometry. The other two field values are
    printed for diagnostic purposes only.
    """
    test_points = np.array(
        [
            [0.003, 0.030],
            [0.003, 0.100],
            [0.003, 0.200],
            [0.003, 0.330],
        ],
        dtype=np.float64,
    )

    T_inc_test = evaluate_field(
        space,
        test_points,
        "inc",
    )

    T_nh3_test = evaluate_field(
        space,
        test_points,
        "nh3",
    )

    T_gan_test = evaluate_field(
        space,
        test_points,
        "gan",
    )

    print("\n" + "=" * 72)
    print("THREE-FIELD DIAGNOSTIC: CRYS-MAS ONE-GAN GEOMETRY")
    print("=" * 72)

    print(
        "r [m]      z [m]      "
        "T_inc [K]      "
        "T_nh3 [K]     "
        "T_gan [K]     "
        "physical material"
    )

    for point, T_inc_i, T_nh3_i, T_gan_i in zip(
        test_points,
        T_inc_test,
        T_nh3_test,
        T_gan_test,
    ):
        inc_mask, nh3_mask, gan_mask = material_masks_numpy(
            np.array(
                [point[0]],
                dtype=np.float64,
            ),
            np.array(
                [point[1]],
                dtype=np.float64,
            ),
        )

        if gan_mask[0]:
            physical_material = "GaN"

        elif nh3_mask[0]:
            physical_material = "NH3"

        elif inc_mask[0]:
            physical_material = "Inconel"

        else:
            physical_material = "Unknown"

        print(
            f"{point[0]:.6f}   "
            f"{point[1]:.6f}   "
            f"{T_inc_i:12.6f}   "
            f"{T_nh3_i:12.6f}   "
            f"{T_gan_i:12.6f}   "
            f"{physical_material}"
        )


# Run

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
        optimizer="RATATAM-LM",
        weights=W_COMMON,
        learning_rate=LEARNING_RATE,
    )

    key = jax.random.PRNGKey(
        42,
    )

    key, samples = sampler.sample(
        key,
        N_COLLOC,
        N_BC_COLLOC,
    )

    print("Sample keys:")
    print(samples.keys())

    print("\nInitial losses:")
    print(
        pinn.evaluate_losses(
            space,
            samples,
        )
    )

    start = timeit.default_timer()

    key, pinn = pinn.project(
        key,
        space,
        N_EPOCHS,
        N_COLLOC,
        N_BC_COLLOC,
        tqdmdesc="CryS-MAS one-GaN conduction",
    )

    elapsed = (
        timeit.default_timer()
        - start
    )

    plot_all_interior_residuals(
        pinn.space,
    )

    plot_loss_history(
        pinn.losses,
    )

    print("\nBest loss:")
    print(pinn.best_loss)

    print(
        f"\nTraining time: "
        f"{elapsed:.2f} s"
    )

    print_three_field_values(
        pinn.space,
    )

    PROJECT_DIR = (
        Path(__file__)
        .resolve()
        .parents[5]
    )

    FVM_REFERENCE_CSV = (
        PROJECT_DIR
        / "2D Inconel + Ammonia + 1 GaN K(T) correct data"
        / "2D_Heat_data.csv"
    )

    print("\n" + "=" * 72)
    print("FVM VALIDATION FILE CHECK: INCONEL + NH3 + 1 GaN")
    print("=" * 72)
    print(f"Reference CSV:\n{FVM_REFERENCE_CSV.resolve()}")

    fvm_df = load_fvm_reference(FVM_REFERENCE_CSV, t0=T0_K, t1=T1)
    print(f"Rows matching T0={T0_K}, T1={T1}: {len(fvm_df)}")

    plot_fvm_vs_pinn_2d_error(pinn.space, fvm_df)
    plot_fvm_vs_pinn_2d_relative_error(pinn.space, fvm_df)

    plot_solution(pinn.space, fvm_df=fvm_df)

    plot_fvm_vs_pinn_all_radii(pinn.space, fvm_df)

    print("\nSaved plot(s) in:")
    print(
        SAVE_DIR.resolve(),
    )
