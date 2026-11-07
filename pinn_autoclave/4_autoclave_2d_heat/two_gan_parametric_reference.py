"""Inconel/NH3 autoclave with TWO GaN pieces, PARAMETRIC in the boundary data.

The networks take mu = (T0, T1) as a second input and one training covers the whole box
MU_BOUNDS -- every case of 2D_Heat_data.csv ("2D Inconel + NH3 + 2 GaN K(T)
extended": 21 x 21 = 441 (T0, T1) cases).

Built from "2D_heat_one_GaN_Part_parametric.py". The GaN is now a list of
pieces (GAN_PARTS): a lower one, r <= 5 mm, 40-120 mm, and an upper one,
r <= 10 mm, 175-325 mm, which leaves a 2 mm NH3 gap to the cavity wall.
Every per-piece object is a list over the pieces: a GaN network, a pair of
interface traces, an interface band, a lead table, an interior residual and
an energy balance each. The leading order of the pieces is solved jointly,
and together with the refraction fields (see "Leading-order profile inside
the cavity" and _solve_case).

What depended on the two constants is now a function of mu, per collocation
point: T_outer(z; mu), every reference scale, the correction budgets (signed
with -T1), and the leading order. The leading-order solves (rod sources, refraction fields)
are linear in T1 and smooth in T0; they are solved on a Chebyshev grid over the box and
interpolated (lead_weights), and every leading-order field is a fixed basis of
unit-source fields with mu-dependent weights.

T0_K / T1 below are only the case the diagnostics and plots are drawn for
(set_case); the model itself never reads them.
"""

import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.special import j0 as _np_j0, j1 as _np_j1, jn_zeros

from scimba_jax.domains.domain_mapping import DomainMapping
from scimba_jax.domains.meshless_domains.domains_2d import Square2D
from scimba_jax.nonlinear_approximation.approximation_spaces.approximation_spaces import (
    ApproximationSpace,
)
from scimba_jax.nonlinear_approximation.integration.monte_carlo import (
    DomainSampler,
    TensorizedSampler,
)
from scimba_jax.nonlinear_approximation.integration.monte_carlo_parameters import (
    UniformParametricSampler,
)
from scimba_jax.nonlinear_approximation.model_class.funcparam_vectorial import (
    ParamScalarFunction,
)
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import InteriorResidual

jax.config.update("jax_enable_x64", True)

SAVE_DIR = Path("output_inc_nh3_two_gan_parametric")
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
    [0.0000, 0.0031, 0.0060, 0.0085, 0.0104, 0.0116, 0.0120], dtype=jnp.float64
)


NH3_LOWER_Z = jnp.array(
    [0.0230, 0.0234, 0.0246, 0.0265, 0.0290, 0.0319, 0.0350], dtype=jnp.float64
)


# Lowest NH3 point, located on the symmetry axis.
Z_NH3_MIN = float(NH3_LOWER_Z[0])


# Height at which the lower polyline reaches the NH3
Z_NH3_MAX_LOWER = float(NH3_LOWER_Z[-1])


# CryS-MAS geometry: two GaN pieces on the axis of the NH3 cavity, each
# (radius, z_min, z_max) [m]. The upper one leaves a 2 mm NH3 gap to the
# cavity wall. Every per-piece quantity below is a list over this one.
GAN_PARTS = [(0.005, 0.040, 0.120), (0.010, 0.175, 0.325)]

N_RODS = len(GAN_PARTS)

ROD_R = [part[0] for part in GAN_PARTS]
ROD_Z_MIN = [part[1] for part in GAN_PARTS]
ROD_Z_MAX = [part[2] for part in GAN_PARTS]
ROD_Z_MID = [0.5 * (part[1] + part[2]) for part in GAN_PARTS]

# Temperature-dependent thermal-conductivity correlations

CONDUCTIVITY_TEMPERATURE_UNIT = "K"


def conductivity_temperature(temperature_K):
    if CONDUCTIVITY_TEMPERATURE_UNIT == "K":
        return temperature_K

    if CONDUCTIVITY_TEMPERATURE_UNIT == "C":
        return temperature_K - 273.15

    raise ValueError("CONDUCTIVITY_TEMPERATURE_UNIT must be either 'K' or 'C'.")


def k_inc_value(temperature_K):
    temperature_input = conductivity_temperature(temperature_K)

    return 6.9995 + 0.0151 * temperature_input


def k_nh3_value(temperature_K):
    temperature_input = conductivity_temperature(temperature_K)

    return (
        0.3039497375
        - 0.00019097 * temperature_input
        + 0.0000001887 * temperature_input**2
    )


# GaN conductivity [W/(m K)]
GAN_CONDUCTIVITY = 130.0

def k_gan_value(temperature_K):
    return GAN_CONDUCTIVITY * jnp.ones_like(temperature_K)


# External temperature boundary condition, as a PARAMETER
# mu = (T0 [K], T1 [K/m]) is an input of every network. The box covers every
# case of 2D_Heat_data.csv: T0 in {611.15, ..., 811.15} K, T1 in {-300, ...,
# 300} K/m.
MU_BOUNDS = [(611.15, 811.15), (-300.0, 300.0)]

# Affine map of the box onto [-1, 1]^2, for the network inputs.
MU_CENTER = jnp.array([0.5 * (lo + hi) for lo, hi in MU_BOUNDS])
MU_HALF_WIDTH = jnp.array([0.5 * (hi - lo) for lo, hi in MU_BOUNDS])

# The case the diagnostics and plots are drawn for (NOT read by the model).
T0_K = 611.15
T1 = -300.0


def set_case(t0, t1):
    """Select the (T0, T1) case the diagnostics and plots evaluate."""
    global T0_K, T1
    T0_K, T1 = float(t0), float(t1)


def case_mu():
    """mu of the current diagnostic case, shape (2,)."""
    return jnp.array([T0_K, T1], dtype=jnp.float64)


def mu_t0(mu):
    return mu[..., 0:1]


def mu_t1(mu):
    return mu[..., 1:2]


def mu_features(mu):
    """Network input for the parameters: (T0, T1) mapped onto [-1, 1]^2."""
    return (jnp.asarray(mu, dtype=jnp.float64) - MU_CENTER) / MU_HALF_WIDTH


def T_outer(x, mu):
    """
    Prescribed external Inconel temperature profile [K]:

        T_outer(z) = T0 + T1 * z.

    Imposed on the bottom (z = 0), top (z = H_OUT) AND outer sidewall
    (r = R_OUT), as in the FV reference 2D_Heat_data.csv.
    """
    x = jnp.asarray(x, dtype=jnp.float64)

    z = x[..., 1:2]

    return mu_t0(mu) + z * mu_t1(mu)


# Geometry predicates in PHYSICAL coordinates
def z_bottom_ammonia(r):
    r = jnp.asarray(r, dtype=jnp.float64)

    r_clipped = jnp.clip(r, NH3_LOWER_R[0], NH3_LOWER_R[-1])

    return jnp.interp(r_clipped, NH3_LOWER_R, NH3_LOWER_Z)


def in_lower_ammonia(r, z):
    return (
        (r >= 0.0) & (r <= R_NH3) & (z >= z_bottom_ammonia(r)) & (z <= Z_NH3_MAX_LOWER)
    )


def in_nh3(r, z):
    """The whole NH3 cavity, GaN block included (the cavity side of Gamma_1)."""
    return (r >= 0.0) & (r <= R_NH3) & (z >= z_bottom_ammonia(r)) & (z <= Z_NH3_TOP)


def in_rod(i, r, z):
    """GaN piece ``i``."""
    return (r >= 0.0) & (r <= ROD_R[i]) & (z >= ROD_Z_MIN[i]) & (z <= ROD_Z_MAX[i])


def in_gan(r, z):
    """Any GaN piece."""
    inside = in_rod(0, r, z)

    for i in range(1, N_RODS):
        inside = inside | in_rod(i, r, z)

    return inside


# Material registry
# ``region`` is a predicate on (r, z); ``None`` marks the matrix material
# that fills whatever no other entry claims. Entries are tested in order,
# first match wins, so the GaN (which sits inside the NH3 cavity) goes
# BEFORE the NH3, and the NH3 BEFORE the Inconel matrix.
MATERIAL_SPECS = [
    {
        "name": "gan",
        # Autoclave.mat, GaN
        "k": [GAN_CONDUCTIVITY],
        "region": lambda r, z: in_gan(r, z),
    },
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
T_TABLE_MAX = 1200.0
_T_TABLE = jnp.linspace(T_TABLE_MIN, T_TABLE_MAX, 12001)


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
    assumption about the degree of k -- the constant GaN law inverts as
    readily as the quadratic NH3 one.
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
    spec["name"]: Material(spec["name"], spec["k"], spec["region"])
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


# Nondimensional reference scales, per case
# The NORMALISATIONS use the swing |T1| H_OUT floored at 1 K, so that a T1 = 0
# case (uniform temperature, also present in 2D_Heat_data.csv) does not
# divide by zero. The correction BUDGETS use the signed swing -T1 H_OUT:
# +111 K on the validated case T1 = -300 K/m (so every budget there is exactly
# the validated one), and 0 at T1 = 0, where the exact solution is the
# uniform T0 -- the leading order itself.
MIN_SWING_K = 1.0

def delta_t_ref(mu):
    """Imposed swing |T1| H_OUT, floored at MIN_SWING_K [K] (normalisations)."""
    return jnp.maximum(jnp.abs(mu_t1(mu)) * H_OUT, MIN_SWING_K)


def signed_swing(mu):
    """-T1 H_OUT [K]: the correction budgets scale with it."""
    return -mu_t1(mu) * H_OUT


def t_center(mu):
    """Mid-range temperature T0 + T1 H_OUT / 2 of case mu [K]."""
    return mu_t0(mu) + 0.5 * mu_t1(mu) * H_OUT


# Characteristic length [m].
L_REF = R_OUT

# Leading-order profile: Inconel
# All three outer faces carry the same Dirichlet datum T_outer(z), and the
# sidewall pins the Inconel to it: on the FV data the Inconel departs from
# T_outer by at most 1.7 K, and the cavity wall (Gamma_1) by 0.3 K. T_outer
# is therefore the leading behaviour of the Inconel and of the Gamma_1 trace.
def axial_coordinate(z):
    """Axial coordinate handed to the NETWORKS."""
    return z / H_OUT


def T_axial_profile(z, mu):
    """The leading-order temperature profile T_outer(z) of case mu."""
    return mu_t0(mu) + mu_t1(mu) * z


def matrix_potential_base(z, mu):
    """Matrix Kirchhoff potential of the leading profile, K_matrix(T_outer(z)).

    Equal to K_matrix(T_outer) on every outer face, so a correction multiplied
    by a factor vanishing on those faces leaves all three Dirichlet
    conditions exact.
    """
    return MATRIX.kirchhoff(T_axial_profile(z, mu))


# Leading-order profile inside the cavity: the GaN pieces as heat sources
#
# T_outer(z) is NOT the leading behaviour inside the cavity once GaN is
# present. GaN conducts ~500x better than NH3, so each piece short-circuits
# heat along the axis: on the FV reference (T0 = 611.15 K, T1 = -300 K/m) the
# lower piece departs from T_outer by up to 10.7 K, the upper one by up to
# 16.2 K, and across the 2 mm gap beside the upper piece the NH3 drops by up
# to 14 K -- an order of magnitude beyond the ~1 K correction budgets this
# construction relies on.
#
# The leading order is therefore built as an exact solution of the NH3 heat
# equation. Each piece exchanges heat with the NH3 through its side (a source
# of line density q(z) in a shell just inside r = R) and its two end faces
# (thin slab sources just inside the piece), within a cylinder whose wall
# r = R_NH3 is held at T_outer, with odd images about the cavity roof and
# floor. In Fourier-Bessel form
#
#     D(r, z) = sum_n J0(k_n r) A_n(z),   k_n = j_{0,n} / R_NH3,
#
# every term is harmonic (J0(k r) A(z) with A'' = k^2 A), so D satisfies the
# (Kirchhoff-linearized) NH3 equation EXACTLY away from the sources -- in the
# whole NH3, the pockets beyond the ends and between the pieces included --
# and vanishes on the cavity sidewall and roof.
#
# The sources are not guessed: they are solved for, all pieces together
# (_solve_rods, further down), so that the NH3 field on each piece's surface
# equals the piece's own temperature, the piece conducting axially
# (k_gan A T_rod'' = q) and passing its end-face heat through its ends.
# Geometry, conductivities and boundary data only, no reference data.
#
# Compared with the one-rod script, three things changed, each needed by the
# short lower piece and by the wide upper one (and each also improves the
# one-rod lead, 0.080 -> 0.040 K RMS against FV):
#
#   End faces   N_END_PROFILES radial profiles per face, the face matched
#               pointwise at _ROD_FACE_POINTS radii. One profile matched in
#               the mean left the pockets beyond the upper piece 1.1 K RMS off
#               (the face spans 10 of the cavity's 12 mm); now 0.16 K.
#   Corners     the side is matched from _ROD_CORNER_CLEARANCE * R of the
#               corners, at a fixed spacing, by REGULARIZED least squares.
#               Plain least squares on 158 points per piece (the one-rod
#               setting) put points 0.5 mm from the corners of the short
#               piece and lost 25 % of both departures.
#   Wall        the pieces sit in the NH3 refraction field too (the cavity
#               wall is not at T_outer: up to 1 K beside the upper piece),
#               so the rod solve and the refraction fits are iterated
#               together (_solve_case).
N_ROD_MODES = 30

# Parametric version. The lead depends on the case mu = (T0, T1) through T1
# (linearly, up to the conductivities) and through the NH3 conductivity at
# each piece's mid-height temperature (the GaN one is constant). It is solved
# on a Chebyshev grid over the parameter box and interpolated (lead_weights,
# further down), so every field is a FIXED basis of unit-source fields with
# mu-dependent weights. For that, the fin rates of the source basis are
# fixed: one pair at the validated case (611.15 K, -300 K/m) and one at the
# top of the range over the box, which lets the solve span every rate in
# between.
_GAN_K_ROD = float(MATERIALS["gan"].k(611.15))

_BESSEL_ZEROS = jn_zeros(0, N_ROD_MODES)

_K_MODES_NP = _BESSEL_ZEROS / R_NH3

# Mode normalisation WITHOUT the NH3 conductivity: the field of a source q is
# (q / k_nh3) times the unit field, so 1/k_nh3 sits in the weights.
_MODE_NORM_NP = 1.0 / (
    2.0 * np.pi * _K_MODES_NP * R_NH3**2 * _np_j1(_BESSEL_ZEROS) ** 2
)

_K_MODES = jnp.asarray(_K_MODES_NP)
_MODE_NORM = jnp.asarray(_MODE_NORM_NP)

# Radial shape of the side sources: (r/R)^8 (1 - r^2/R^2)^3, inside the
# piece. Only the field OUTSIDE it (in the NH3) is ever used, and there any
# source confined to r < R gives a harmonic field -- but its truncated Bessel
# series must converge fast enough not to leak into the NH3:
#
#   ring on r = R (a delta in r)      NH3 PDE residual 0.2 median, 11 max
#   (1 - r^2/R^2)^3, whole piece      no leak, but the field at the surface
#                                     cannot follow the ends: GaN 3.4 K off
#   (r/R)^8 (1 - r^2/R^2)^3           residual 1e-3 median, 0.03 p99
#
# (the last one peaks at 0.76 R and vanishes with two derivatives at the
# surface: a shell just inside it, smooth enough for 30 modes).
_SOURCE_PROFILE_POWER = 3

_SOURCE_PROFILE_SHELL = 8

# End faces: slabs this thick (in R) just inside the piece -- a
# zero-thickness disk would leave a kink of the truncated series in the NH3
# beside it -- with radial profiles (r/R)^(2j) (1 - r^2/R^2)^3, j = 0 ..
# N_END_PROFILES - 1.
_ROD_END_SLAB = 0.5

N_END_PROFILES = 6

# Axial source basis on the side: (rate mu, centre c) for exp(mu (z - c)),
# mu = 0 being the constant. The end-layer rates sit between consecutive
# k_n (j_{0,n} = 2.40, 5.52, 8.65), so no integral below is singular.
_ROD_END_RATES = np.array([1.2, 4.0, 7.1]) / R_NH3


# Evaluation of the leading-order fields at collocation points
#
# Every source is exp(mu (z' - c)) on an interval [a, b], and its mode
# amplitude at z is _layer_integral. Written source by source (63 sources per
# NH3 point here, each with three images, under two more derivatives and the
# parameter Jacobian) the traced program no longer compiled: XLA ran out of
# 40 GB before the first epoch, whatever the number of collocation points.
#
# For sources sharing [a, b] the integral factorises: with E_a = exp(-k |z - a|)
# and E_b = exp(-k |z - b|),
#
#     I = E_a alpha + E_b beta + [a < z < b] exp(mu (z - c)) gamma
#
#     alpha = exp(mu (a - c)) / (k - mu)   (z <= a),  -exp(mu (a - c)) / (k + mu)  (z > a)
#     beta  = exp(mu (b - c)) / (k + mu)   (z >= b),  -exp(mu (b - c)) / (k - mu)  (z < b)
#     gamma = 2 k / (k^2 - mu^2)
#
# so the weighted sum over the sources of a group costs two exponentials per
# mode and one per source, and four small matrix products that do not depend
# on z. Same numbers as _layer_integral, to round-off.
class _LayerGroup:
    """sum_s w_s radial[s, n] I(z; k_n, mu_s, c_s, a, b) for sources sharing [a, b]."""

    def __init__(self, k, rates, centres, a, b, radial):
        k = np.asarray(k, dtype=np.float64)[None, :]
        mu = np.asarray(rates, dtype=np.float64)[:, None]
        c = np.asarray(centres, dtype=np.float64)[:, None]

        e_a = np.exp(mu * (a - c))
        e_b = np.exp(mu * (b - c))

        self.a, self.b = float(a), float(b)
        self.k = jnp.asarray(k[0])
        self.mu = jnp.asarray(mu[:, 0])
        self.c = jnp.asarray(c[:, 0])

        self.alpha_below = jnp.asarray(radial * e_a / (k - mu))
        self.alpha_above = jnp.asarray(-radial * e_a / (k + mu))
        self.beta_above = jnp.asarray(radial * e_b / (k + mu))
        self.beta_below = jnp.asarray(-radial * e_b / (k - mu))
        self.gamma = jnp.asarray(radial * 2.0 * k / (k**2 - mu**2))

    def amplitude(self, weights, images):
        """Mode amplitudes (modes,) for source ``weights`` (sources,), summed
        over ``images``: [(sign, z)] with z of shape (1,)."""
        alpha_below = weights @ self.alpha_below
        alpha_above = weights @ self.alpha_above
        beta_above = weights @ self.beta_above
        beta_below = weights @ self.beta_below

        total = 0.0

        for sign, z in images:
            below, above = z <= self.a, z >= self.b
            between = ~(below | above)

            # Each distance and exponent is written per branch (no abs, no
            # clip), so a derivative taken exactly on a or b is the one of the
            # branch selected there, and no discarded branch overflows.
            e_a = jnp.exp(-self.k * jnp.where(below, self.a - z, z - self.a))
            e_b = jnp.exp(-self.k * jnp.where(above, z - self.b, self.b - z))

            z_in = jnp.where(between, z, self.a)
            inside = (weights * jnp.exp(self.mu * (z_in - self.c))) @ self.gamma

            total = total + sign * (
                e_a * jnp.where(below, alpha_below, alpha_above)
                + e_b * jnp.where(above, beta_above, beta_below)
                + jnp.where(between, inside, 0.0)
            )

        return total


def _radial_weights(radius, profile):
    """Bessel coefficients of a radial profile on [0, radius], unit total."""
    nodes, weights = np.polynomial.legendre.leggauss(400)
    r = 0.5 * radius * (nodes + 1.0)
    w = 0.5 * radius * weights
    values = profile(r / radius)

    return np.sum(w * r * values * _np_j0(np.outer(_K_MODES_NP, r)), axis=1) / np.sum(
        w * r * values
    )


def _fin_rate(i, temperature_K):
    """Fin rate of piece i (radial annulus resistance vs axial conduction)."""
    return float(
        np.sqrt(
            2.0
            * float(MATERIALS["nh3"].k(temperature_K))
            / (_GAN_K_ROD * ROD_R[i] ** 2 * np.log(R_NH3 / ROD_R[i]))
        )
    )


def _rod_setup(i):
    """Everything fixed about piece i: its source basis and radial weights."""
    radius, z_min, z_max, z_mid = ROD_R[i], ROD_Z_MIN[i], ROD_Z_MAX[i], ROD_Z_MID[i]

    rate = _fin_rate(i, 611.15 - 300.0 * z_mid)
    rate_high = _fin_rate(
        i, max(t0 + t1 * z_mid for t0 in MU_BOUNDS[0] for t1 in MU_BOUNDS[1])
    )

    basis = (
        [(0.0, z_mid), (rate, z_mid), (-rate, z_mid)]
        + [(rate_high, z_mid), (-rate_high, z_mid)]
        + [(-end_rate, z_min) for end_rate in _ROD_END_RATES]
        + [(end_rate, z_max) for end_rate in _ROD_END_RATES]
    )

    shell = _radial_weights(
        radius,
        lambda x: x**_SOURCE_PROFILE_SHELL * (1.0 - x**2) ** _SOURCE_PROFILE_POWER,
    )
    ends = np.stack(
        [
            _radial_weights(
                radius,
                lambda x, j=j: x ** (2 * j) * (1.0 - x**2) ** _SOURCE_PROFILE_POWER,
            )
            for j in range(N_END_PROFILES)
        ]
    )

    slab = _ROD_END_SLAB * radius
    n_basis = len(basis)
    slab_weights = _MODE_NORM_NP * ends / slab
    zeros = np.zeros(N_END_PROFILES)

    # (slice of the piece's source weights, group): side, bottom, top.
    groups = [
        (
            slice(0, n_basis),
            _LayerGroup(
                _K_MODES_NP,
                [mu for mu, _ in basis],
                [c for _, c in basis],
                z_min,
                z_max,
                np.tile(_MODE_NORM_NP * shell, (n_basis, 1)),
            ),
        ),
        (
            slice(n_basis, n_basis + N_END_PROFILES),
            _LayerGroup(_K_MODES_NP, zeros, zeros, z_min, z_min + slab, slab_weights),
        ),
        (
            slice(n_basis + N_END_PROFILES, n_basis + 2 * N_END_PROFILES),
            _LayerGroup(_K_MODES_NP, zeros, zeros, z_max - slab, z_max, slab_weights),
        ),
    ]

    return {
        "basis": basis,
        "groups": groups,
        "area": np.pi * radius**2,
        "slab": slab,
        # Radial weights of the side shell (modes,) and of the end profiles
        # (profiles, modes), numpy and jax.
        np: (shell, ends),
        jnp: (jnp.asarray(shell), jnp.asarray(ends)),
    }


ROD = [_rod_setup(i) for i in range(N_RODS)]

# Sources of a piece: its side basis, then the bottom and the top profiles.
N_ROD_BASIS = [len(rod["basis"]) for rod in ROD]

N_ROD_SOURCES = [n_basis + 2 * N_END_PROFILES for n_basis in N_ROD_BASIS]

# Solution vector of a piece: [source strengths, a, b], T_rod - T_outer being
# the double integral of q / (k_gan A) plus a + b (z - z_mid).
N_ROD_UNKNOWNS = [n_sources + 2 for n_sources in N_ROD_SOURCES]

# The rod field has one channel per source of every piece. Its weights are
# the source strengths divided by k_nh3 at the piece.
_ROD_CHANNEL_OFFSETS = np.concatenate([[0], np.cumsum(N_ROD_SOURCES)])

ROD_CHANNELS = [
    slice(int(_ROD_CHANNEL_OFFSETS[i]), int(_ROD_CHANNEL_OFFSETS[i + 1]))
    for i in range(N_RODS)
]

N_ROD_CHANNELS = int(_ROD_CHANNEL_OFFSETS[-1])


def _layer_integral(xp, z, k, mu, c, a, b):
    """int_a^b exp(mu (z' - c)) exp(-k |z - z'|) dz'.

    Each branch is evaluated on z clipped to its own range, so no exponent
    ever overflows -- not even in the branch jnp.where discards, whose
    gradient would otherwise turn into NaN.
    """
    e_a = mu * (a - c)
    e_b = mu * (b - c)

    z_lo = xp.minimum(z, a)
    z_hi = xp.maximum(z, b)
    z_in = xp.clip(z, a, b)

    below = (xp.exp(e_a - k * (a - z_lo)) - xp.exp(e_b - k * (b - z_lo))) / (k - mu)
    above = (xp.exp(e_b - k * (z_hi - b)) - xp.exp(e_a - k * (z_hi - a))) / (k + mu)
    inside = (xp.exp(mu * (z_in - c)) - xp.exp(e_a - k * (z_in - a))) / (k + mu) + (
        xp.exp(mu * (z_in - c)) - xp.exp(e_b - k * (b - z_in))
    ) / (k - mu)

    return xp.where(z <= a, below, xp.where(z >= b, above, inside))


def _with_images(fn, z):
    """Odd images about the cavity roof and floor planes."""
    return fn(z) - fn(2.0 * Z_NH3_TOP - z) - fn(2.0 * Z_NH3_MIN - z)


def _rod_source_amplitudes(xp, i, z):
    """A_n(z) of every unit source of piece i: (..., sources, modes); z (..., 1).

    The end profiles of a face share one axial amplitude, so the extra
    profiles cost no extra exponential.
    """
    rod = ROD[i]
    z_min, z_max, slab = ROD_Z_MIN[i], ROD_Z_MAX[i], rod["slab"]
    k, norm = (_K_MODES_NP, _MODE_NORM_NP) if xp is np else (_K_MODES, _MODE_NORM)
    shell, ends = rod[xp]

    side = [
        _with_images(
            lambda zz, mu=mu, c=c: _layer_integral(xp, zz, k, mu, c, z_min, z_max), z
        )
        for mu, c in rod["basis"]
    ]
    bottom = _with_images(
        lambda zz: _layer_integral(xp, zz, k, 0.0, z_min, z_min, z_min + slab), z
    )
    top = _with_images(
        lambda zz: _layer_integral(xp, zz, k, 0.0, z_max, z_max - slab, z_max), z
    )

    return xp.stack(
        [norm * shell * amplitude for amplitude in side]
        + [norm * profile * bottom / slab for profile in ends]
        + [norm * profile * top / slab for profile in ends],
        axis=-2,
    )


def _rod_mode_amplitudes(i, z, weights):
    """A_n(z) of piece i for given source weights: (modes,); z of shape (1,).

    The same sum as weights @ _rod_source_amplitudes(jnp, i, z), group by
    group (_LayerGroup), with the odd images about the cavity roof and floor.
    """
    images = [(1.0, z), (-1.0, 2.0 * Z_NH3_TOP - z), (-1.0, 2.0 * Z_NH3_MIN - z)]

    return sum(
        group.amplitude(weights[sources], images) for sources, group in ROD[i]["groups"]
    )


def _rod_departure_columns(xp, i, z):
    """dev(z) = double integral of q / (k_gan A) + a + b (z - z_mid), per unknown
    (the side basis, then a and b)."""
    rod, z_mid = ROD[i], ROD_Z_MID[i]
    columns = []

    for mu, c in rod["basis"]:
        if mu == 0.0:
            columns.append((z - z_mid) ** 2 / 2.0)
        else:
            columns.append(xp.exp(mu * (z - c)) / mu**2)

    columns = [column / (_GAN_K_ROD * rod["area"]) for column in columns]

    return columns + [xp.ones_like(z), z - z_mid]


def _rod_slope_columns(xp, i, z):
    rod, z_mid = ROD[i], ROD_Z_MID[i]
    columns = []

    for mu, c in rod["basis"]:
        if mu == 0.0:
            columns.append(z - z_mid)
        else:
            columns.append(xp.exp(mu * (z - c)) / mu)

    columns = [column / (_GAN_K_ROD * rod["area"]) for column in columns]

    return columns + [xp.zeros_like(z), xp.ones_like(z)]


def _bessel_j0_j1(x):
    """J0 and J1 for x >= 0 (Numerical Recipes rational fits, ~1e-8)."""
    y = x * x

    small_j0 = (
        57568490574.0
        + y
        * (
            -13362590354.0
            + y
            * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456)))
        )
    ) / (
        57568490411.0
        + y
        * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))))
    )
    small_j1 = (
        x
        * (
            72362614232.0
            + y
            * (
                -7895059235.0
                + y
                * (
                    242396853.1
                    + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606))
                )
            )
        )
    ) / (
        144725228442.0
        + y
        * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))))
    )

    x_large = jnp.maximum(x, 8.0)
    w = 8.0 / x_large
    v = w * w
    amplitude = jnp.sqrt(0.636619772 / x_large)

    p0 = 1.0 + v * (
        -0.1098628627e-2
        + v * (0.2734510407e-4 + v * (-0.2073370639e-5 + v * 0.2093887211e-6))
    )
    q0 = -0.1562499995e-1 + v * (
        0.1430488765e-3
        + v * (-0.6911147651e-5 + v * (0.7621095161e-6 - v * 0.934935152e-7))
    )
    p1 = 1.0 + v * (
        0.183105e-2
        + v * (-0.3516396496e-4 + v * (0.2457520174e-5 + v * -0.240337019e-6))
    )
    q1 = 0.04687499995 + v * (
        -0.2002690873e-3
        + v * (0.8449199096e-5 + v * (-0.88228987e-6 + v * 0.105787412e-6))
    )

    phase0 = x_large - 0.785398164
    phase1 = x_large - 2.356194491

    large_j0 = amplitude * (jnp.cos(phase0) * p0 - w * jnp.sin(phase0) * q0)
    large_j1 = amplitude * (jnp.cos(phase1) * p1 - w * jnp.sin(phase1) * q1)

    return jnp.where(x < 8.0, small_j0, large_j0), jnp.where(
        x < 8.0, small_j1, large_j1
    )


# The derivatives are given by the exact identities J0' = -J1 and
# J1' = J0 - J1/x, not by differentiating the fits: every mode
# J0(k r) A(z) then stays harmonic to round-off under autodiff too.
@jax.custom_jvp
def bessel_j0(x):
    return _bessel_j0_j1(x)[0]


@jax.custom_jvp
def bessel_j1(x):
    return _bessel_j0_j1(x)[1]


@bessel_j0.defjvp
def _bessel_j0_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return bessel_j0(x), -bessel_j1(x) * dx


@bessel_j1.defjvp
def _bessel_j1_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    x_safe = jnp.where(x < 1.0e-3, 1.0, x)
    j1_over_x = jnp.where(x < 1.0e-3, 0.5 - x * x / 16.0, bessel_j1(x_safe) / x_safe)
    return bessel_j1(x), (bessel_j0(x) - j1_over_x) * dx


def rod_field_departure(r, z, weights):
    """D(r, z): the departure field of the GaN pieces in the NH3 [K]; r, z of
    shape (1,).

    ``weights`` (N_ROD_CHANNELS,): source strengths / k_nh3 of the case
    (rod_field_weights). Combined at the amplitude level, so one Bessel sum
    whatever the number of sources.
    """
    amplitudes = sum(
        _rod_mode_amplitudes(i, z, weights[ROD_CHANNELS[i]]) for i in range(N_RODS)
    )

    return jnp.sum(bessel_j0(_K_MODES * r) * amplitudes, axis=-1, keepdims=True)


def _rod_unit_fields(p):
    """All rod channels with unit weight at one point p (2,): (N_ROD_CHANNELS,).

    The field is linear in the weights, so its channels are its Jacobian --
    through the grouped evaluation, NOT _rod_source_amplitudes: the lead
    tables take the gradient of these channels at points lying exactly on the
    GaN end faces, which are source edges, and there _layer_integral (written
    with min / max / clip) differentiates to HALF the slope.
    """
    return jax.jacfwd(lambda weights: rod_field_departure(p[0:1], p[1:2], weights)[0])(
        jnp.zeros(N_ROD_CHANNELS)
    )


def rod_departure(i, z, mu):
    """T_rod - T_outer on piece i, radially uniform (GaN is a strong conductor)."""
    z_rod = jnp.clip(z, ROD_Z_MIN[i], ROD_Z_MAX[i])
    solution = rod_solution(i, mu)
    n_basis, n_sources = N_ROD_BASIS[i], N_ROD_SOURCES[i]

    # The side strengths, then a and b (the end-face strengths lie between).
    coefficients = [solution[..., j : j + 1] for j in range(n_basis)] + [
        solution[..., n_sources : n_sources + 1],
        solution[..., n_sources + 1 : n_sources + 2],
    ]

    return sum(
        coefficient * column
        for coefficient, column in zip(
            coefficients, _rod_departure_columns(jnp, i, z_rod)
        )
    )


def T_rod_profile(i, z, mu):
    """Leading-order temperature of GaN piece i, case mu."""
    return T_axial_profile(z, mu) + rod_departure(i, z, mu)


def nh3_leading_profile(x, mu):
    """Leading-order NH3 temperature: T_outer, the departure field of the GaN
    pieces and the refraction field of the cavity (nh3_refraction_departure,
    below)."""
    x = jnp.asarray(x, dtype=jnp.float64)

    return (
        T_axial_profile(x[..., 1:2], mu)
        + rod_field_departure(x[..., 0:1], x[..., 1:2], rod_field_weights(mu))
        + nh3_refraction_departure(x, mu)
    )


# Scale of the axial heat flow [W]: a uniform Inconel column at the mid-range
# temperature under the imposed gradient.
def flux_balance_ref(mu):
    return jnp.pi * R_OUT**2 * MATRIX.k(t_center(mu)) * delta_t_ref(mu) / H_OUT


# Training settings
#
# Victor's Inconel/NH3 budget (8 750 points x 400 epochs), plus the GaN boxes.
# nh3_upper is raised from 2 500 to 4 000: its sampling box now contains the
# GaN pieces (masked out of the NH3 residual), so this keeps the NH3 around
# them -- where their departure decays -- at the one-rod density.
# (16 000 points -- nh3 5 000 / 2 000, gan 4 000 -- steadied the loss history
# but did not change the result, and cost ~40 % more time per epoch.)
N_COLLOC = {
    # Inconel interior sampling regions
    "inc_bottom": 500,
    "inc_side": 2000,
    "inc_top": 750,
    "inc_lower": 1750,
    # NH3 interior sampling regions
    "nh3_upper": 4000,
    "nh3_lower": 1250,
    # GaN interior sampling regions, one per piece (the one-rod density)
    "gan_0": 400,
    "gan_1": 1000,
    # Integral constraint: each sample is a whole cross section.
    "flux_balance": 40,
    # Global energy balances: every sample evaluates the same integral.
    "rod_energy_0": 2,
    "rod_energy_1": 2,
    "cavity_energy": 2,
}


# Measured on 12 FV cases (T0 in {611.15, 711.15, 811.15}, T1 in {-300, -150,
# 150, 300}), seed 42, 2254 s:
#
#                     validated (611.15, -300)   mean      worst
#   leading order            0.0435 K           0.0336 K  0.0480 K
#   50 epochs                0.0421 K           0.0337 K  0.0485 K
#
# All 441 FV cases, script run: mean 0.023 K, worst 0.049 K (the 416 cases
# between the 25 of the old table: 0.023 / 0.049 K).
#
# Training keeps the leading order's accuracy and does not improve on it.
# The loss stops falling near epoch 26 (4.7e5 -> 4.5e4), 94 % of it the
# interior residual of the upper GaN piece (gan_1): its temperatures are
# right (0.05-0.08 K), but the 1-D rod profile plus parabola is not an exact
# solution near the ends of a piece this wide. That term is not what holds
# the result back: with its weight cut from 350 to 3.5 the loss is balanced
# (2.2e3) and the FV error is no better (validated 0.0452 K, mean 0.0350 K).
N_EPOCHS = 50

LEARNING_RATE = 1.0e-4

# ENG (the Projector default optimizer) with a strong-Wolfe line search and
# adaptive Tikhonov regularization. With the default logarithmic-grid search
# ENG takes the best grid step even when it INCREASES the loss, and on this
# case it stalled then diverged to NaN within 20-40 epochs; Armijo diverged
# too. Strong-Wolfe decreases monotonically in expectation.
ENG_OPTIONS = {"linesearch": "strong-wolfe", "adaptive_matrix_regularization": True}

HIDDEN = [24, 24, 24]

# Width of the four interface-trace networks (functions on a curve).
HIDDEN_TRACE = [16, 16]


# Conductivity reference for residual normalization.
#
# The "anticipated" temperature window of a case -- its two Dirichlet
# extremes plus a safety margin -- is used only to pick reference
# conductivity SCALES below (not the Kirchhoff inverse-lookup table). It is
# checked once here for the whole parameter box.
T_SCALING_MARGIN_K = 20.0

_BOX_T_MIN = min(t0 + min(0.0, t1 * H_OUT) for t0 in MU_BOUNDS[0] for t1 in MU_BOUNDS[1])
_BOX_T_MAX = max(t0 + max(0.0, t1 * H_OUT) for t0 in MU_BOUNDS[0] for t1 in MU_BOUNDS[1])

T_K_MIN_FOR_SCALING = _BOX_T_MIN - T_SCALING_MARGIN_K
T_K_MAX_FOR_SCALING = _BOX_T_MAX + T_SCALING_MARGIN_K

if T_K_MIN_FOR_SCALING < T_TABLE_MIN or T_K_MAX_FOR_SCALING > T_TABLE_MAX:
    raise ValueError(
        "The anticipated temperature window "
        f"[{T_K_MIN_FOR_SCALING}, {T_K_MAX_FOR_SCALING}] K (from MU_BOUNDS, "
        f"H_OUT and a {T_SCALING_MARGIN_K} K margin) falls outside the "
        f"Kirchhoff inverse-lookup table range [{T_TABLE_MIN}, "
        f"{T_TABLE_MAX}] K. Widen T_TABLE_MIN/T_TABLE_MAX."
    )

T_K_SCALE_GRID = jnp.linspace(T_K_MIN_FOR_SCALING, T_K_MAX_FOR_SCALING, 2001)

# Fail early if the correlation/unit choice produces an
# unphysical conductivity anywhere in the box.
if bool(jnp.any(k_inc_value(T_K_SCALE_GRID) <= 0.0)):
    raise ValueError()

if bool(jnp.any(k_nh3_value(T_K_SCALE_GRID) <= 0.0)):
    raise ValueError()


def k_nh3_max_ref(mu):
    """Largest k_nh3 over the anticipated window of case mu.

    k_nh3 is a convex quadratic, so its maximum over an interval sits at one
    of the two ends -- the value the non-parametric script took as the max
    over a 2001-point grid of that interval.
    """
    t_bottom = mu_t0(mu)
    t_top = mu_t0(mu) + mu_t1(mu) * H_OUT
    low = jnp.minimum(t_bottom, t_top) - T_SCALING_MARGIN_K
    high = jnp.maximum(t_bottom, t_top) + T_SCALING_MARGIN_K

    return jnp.maximum(k_nh3_value(low), k_nh3_value(high))


# Reference scale for the interface flux traces, on BOTH interfaces.
#
# Built from k_nh3, NOT from a shared global conductivity scale dominated by
# the strong conductor. Both conditions are k_a dT_a/dn = k_nh3 dT_nh3/dn,
# and both sides equal the TRUE flux crossing the interface, which is set by
# the poorly conducting side -- the NH3 on Gamma_1 (Inconel/NH3) as on
# Gamma_2 (NH3/GaN). It is the scale of the flux-trace networks, so it takes
# the signed swing like every other correction budget.
def interface_flux_scale(mu):
    return k_nh3_max_ref(mu) * signed_swing(mu) / L_REF


# Reference scale for the r-weighted Laplacian of the potential.
#
# Each material's residual is normalised by its own potential span over the
# imposed temperature swing, k_m(T_center) * DELTA_T_REF (exactly
# K_m(T_bottom) - K_m(T_top) for the linear Inconel and constant GaN laws,
# within 1e-4 for the NH3). A shared span would leave the NH3 residual ~60x
# too loose.
#
# EXCEPT the GaN: it has no Dirichlet face, and its temperature level is set
# only by the heat it trades with the NH3. Normalised by its own k, an error
# in the rod's axial heat balance is invisible (5 K off FV at a loss of
# 1e-3); measured on the NH3 scale -- the energy imbalance that matters for
# the heat it exchanges -- it is not.
def potential_pde_ref(name, mu):
    return MATERIALS[name].k(t_center(mu)) * delta_t_ref(mu) / L_REF


# Residual normalisation of each material, per case (the GaN on the NH3 scale).
PDE_REF_MATERIAL = {"inc": MATRIX_MATERIAL, "nh3": "nh3", "gan": "nh3"}


EPS_AXIS = 1.0e-7

# Reference coordinates
XI_NH3 = R_NH3 / R_OUT


ETA_NH3_MIN = Z_NH3_MIN / H_OUT

ETA_NH3_MAX_LOWER = Z_NH3_MAX_LOWER / H_OUT

ETA_NH3_TOP = Z_NH3_TOP / H_OUT

# Main mapped autoclave domain
main_mapping = DomainMapping.rectangle(
    origin=jnp.array([0.0, 0.0]),
    x_dir=jnp.array([R_OUT, 0.0]),
    y_dir=jnp.array([0.0, H_OUT]),
)


main_domain = Square2D([(0.0, 1.0), (0.0, 1.0)], is_main_domain=True, label_str="main")


main_domain.set_mapping(main_mapping, bounds_postmap=[(0.0, R_OUT), (0.0, H_OUT)])

# Volumetric sampling subdomains
inc_bottom = Square2D(
    [(0.0, 1.0), (0.0, ETA_NH3_MIN)], is_main_domain=False, label_str="inc_bottom"
)

# Inconel outside the NH3 sidewall:
inc_side = Square2D(
    [(XI_NH3, 1.0), (ETA_NH3_MIN, ETA_NH3_TOP)],
    is_main_domain=False,
    label_str="inc_side",
)

# Inconel above NH3:
inc_top = Square2D(
    [(0.0, 1.0), (ETA_NH3_TOP, 1.0)], is_main_domain=False, label_str="inc_top"
)


# Lower-envelope rectangle:
inc_lower = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MIN, ETA_NH3_MAX_LOWER)],
    is_main_domain=False,
    label_str="inc_lower",
)

# NH3 sampling subdomains
nh3_lower = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MIN, ETA_NH3_MAX_LOWER)],
    is_main_domain=False,
    label_str="nh3_lower",
)


# Upper NH3 cavity (contains the GaN pieces, masked out of the NH3 residual):
nh3_upper = Square2D(
    [(0.0, XI_NH3), (ETA_NH3_MAX_LOWER, ETA_NH3_TOP)],
    is_main_domain=False,
    label_str="nh3_upper",
)


# GaN pieces:
gan_domains = [
    Square2D(
        [(0.0, ROD_R[i] / R_OUT), (ROD_Z_MIN[i] / H_OUT, ROD_Z_MAX[i] / H_OUT)],
        is_main_domain=False,
        label_str=f"gan_{i}",
    )
    for i in range(N_RODS)
]


flux_balance = Square2D(
    [(0.0, 1.0), (0.0, 1.0)], is_main_domain=False, label_str="flux_balance"
)

rod_energy_domains = [
    Square2D(
        [(0.0, 1.0), (0.0, 1.0)], is_main_domain=False, label_str=f"rod_energy_{i}"
    )
    for i in range(N_RODS)
]

cavity_energy = Square2D(
    [(0.0, 1.0), (0.0, 1.0)], is_main_domain=False, label_str="cavity_energy"
)


# Attach all volumetric sampling regions to main_domain.
for domain in (
    inc_bottom,
    inc_side,
    inc_top,
    inc_lower,
    nh3_lower,
    nh3_upper,
    *gan_domains,
    flux_balance,
    *rod_energy_domains,
    cavity_energy,
):
    main_domain.add_subdomain(domain)


# No boundary segments/domains are registered: the outer Dirichlet wall and
# all the interface conditions (temperature and flux, on every interface)
# are enforced EXACTLY by construction in the trial functions below, so there
# is nothing left for a boundary residual to penalize. The sampler is
# therefore interior-only (bc=False below).


# Neural temperature trial functions
#
# Every network input is its spatial features followed by mu_features(mu)
# (the wrappers after interface_trace_features_2).
def _scaled_coordinates(x):
    x = jnp.asarray(x, dtype=jnp.float64)

    r_scaled = x[..., 0:1] / R_OUT
    z_scaled = axial_coordinate(x[..., 1:2])

    # Quadratic in r/R_OUT, so any network of rho satisfies dT/dr = 0 on the
    # symmetry axis. The sidewall is Dirichlet, carried by
    # outer_distance_factor.
    rho = r_scaled**2

    eta = z_scaled

    return jnp.concatenate([rho, eta], axis=-1)


# End-layer input features
#
# Every correction the leading order still needs sits in a thin layer at an
# end: the NH3 pockets below and above the rod (10-17 mm of a 312 mm cavity),
# and the GaN just inside its end faces (a piece is 2-D within ~R of its
# ends, which the 1-D rod profile is not). On an axial input normalised over
# the whole cavity or rod such a layer spans 0.5-5 % of the range, which a
# small tanh network barely resolves. Each network touching an end therefore
# also gets exp(-distance to each end / length): O(1) variation across the
# layer, flat beyond it.
#
# OFF by default: with them the loss fell ~5x lower, but the networks used
# the extra freedom along near-null directions of the strong form (the whole
# rod or cavity shifting by 0.5-1 K), and the FV error rose (0.26 -> 0.42-0.46
# K RMS over 60 ENG epochs). Kept as a switch for further experiments.
END_LAYER_FEATURES = False


def end_layer_features(z, z_low, z_high, length):
    if not END_LAYER_FEATURES:
        return []

    return [
        jnp.exp(-jnp.abs(z - z_low) / length),
        jnp.exp(-jnp.abs(z_high - z) / length),
    ]


def network_input_sizes():
    """Input sizes of the networks: Inconel, NH3, one GaN field per piece, the
    Gamma_1 traces (U, W), then the (U, W) traces of every piece.

    Each also takes the two parameter features.
    """
    extra = 2 if END_LAYER_FEATURES else 0
    n_mu = len(MU_BOUNDS)

    return [2 + n_mu] + [2 + extra + n_mu] * (N_MODELS - 1)


# The NH3 and GaN networks get their own scaling.
#
# scaled_coordinates normalizes on the FULL autoclave, so a network living in
# the cavity would only ever see a sliver of its radial input range, squeezed
# against 0 where tanh is close to linear. Normalizing each on its own region
# spreads both inputs over [0, 1]. rho stays quadratic in r so that dT/dr = 0
# on the symmetry axis remains automatic (no axis residual is imposed).
def _nh3_scaled_coordinates(x):
    x = jnp.asarray(x, dtype=jnp.float64)

    r_scaled = x[..., 0:1] / R_NH3

    sigma = axial_coordinate(x[..., 1:2])
    sigma_min = axial_coordinate(jnp.asarray(Z_NH3_MIN))
    sigma_max = axial_coordinate(jnp.asarray(Z_NH3_TOP))

    z_scaled = (sigma - sigma_min) / (sigma_max - sigma_min)

    rho = r_scaled**2
    eta = z_scaled

    ends = end_layer_features(x[..., 1:2], Z_NH3_MIN, Z_NH3_TOP, R_NH3)

    return jnp.concatenate([rho, eta] + ends, axis=-1)


def _gan_scaled_coordinates(i):
    """Coordinates of the GaN network of piece i, normalised on the piece."""
    radius, z_min, z_max = ROD_R[i], ROD_Z_MIN[i], ROD_Z_MAX[i]

    def features(x):
        x = jnp.asarray(x, dtype=jnp.float64)

        rho = (x[..., 0:1] / radius) ** 2

        eta = (x[..., 1:2] - z_min) / (z_max - z_min)

        ends = end_layer_features(x[..., 1:2], z_min, z_max, radius)

        return jnp.concatenate([rho, eta] + ends, axis=-1)

    return features


ENVELOPE_LENGTH = R_OUT


def outer_distance_factor(x):
    x = jnp.asarray(x, dtype=jnp.float64)

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
    # the parabola z (H_OUT - z) / H_OUT^2: with a Dirichlet sidewall,
    # perturbations decay over ~R_OUT, and the parabola starved the correction
    # at the cavity ends (Inconel/NH3 case: 0.078 -> 0.060-0.071 K RMS).
    d_bottom = jnp.tanh(z / ENVELOPE_LENGTH)

    d_top = jnp.tanh((H_OUT - z) / ENVELOPE_LENGTH)

    d_side = 1.0 - (r / R_OUT) ** 2

    return d_bottom * d_top * d_side


# The material interfaces, as ordered lists of segments
#
# Gamma_1 (Inconel/NH3) runs from the axis at the cavity floor, along the
# lower polyline, up the cylindrical sidewall and back along the top face to
# the axis. The NH3/GaN interface of each piece runs from the axis along its
# bottom face, up its side and back along its top face to the axis. All end
# on r = 0, which is the symmetry axis and not a material interface: open
# curves.
_IF1_POINTS = [
    (float(NH3_LOWER_R[i]), float(NH3_LOWER_Z[i])) for i in range(len(NH3_LOWER_R))
] + [(R_NH3, Z_NH3_TOP), (0.0, Z_NH3_TOP)]

_ROD_IF_POINTS = [
    [
        (0.0, ROD_Z_MIN[i]),
        (ROD_R[i], ROD_Z_MIN[i]),
        (ROD_R[i], ROD_Z_MAX[i]),
        (0.0, ROD_Z_MAX[i]),
    ]
    for i in range(N_RODS)
]

_IF1_P0 = jnp.array(_IF1_POINTS[:-1], dtype=jnp.float64)
_IF1_P1 = jnp.array(_IF1_POINTS[1:], dtype=jnp.float64)

_ROD_IF_P0 = [jnp.array(points[:-1], dtype=jnp.float64) for points in _ROD_IF_POINTS]
_ROD_IF_P1 = [jnp.array(points[1:], dtype=jnp.float64) for points in _ROD_IF_POINTS]


def _sample_polyline(points, per_segment=400):
    p = np.asarray(points, dtype=np.float64)
    t = np.linspace(0.0, 1.0, per_segment)[:, None]

    return np.concatenate([p[i] + t * (p[i + 1] - p[i]) for i in range(len(p) - 1)])


def _distance_to_polyline(samples, points):
    p = np.asarray(points, dtype=np.float64)
    p0, p1 = p[:-1], p[1:]
    d = p1 - p0

    t = np.einsum("nsk,sk->ns", samples[:, None, :] - p0, d) / np.sum(d * d, axis=-1)
    projected = p0 + np.clip(t, 0.0, 1.0)[..., None] * d

    return np.min(np.linalg.norm(samples[:, None, :] - projected, axis=-1), axis=1)


# Half-widths of the bands in which the interface ansaetze are active [m].
#
# Each band must stay clear of the outer boundaries (so the outer Dirichlet
# hard constraints are untouched), must not meet itself across its own
# region, and no two bands may meet each other: in the NH3, between two
# interfaces, chi_a * chi_b = 0 is what makes the blend below a partition.
# All clearances come from the geometry; with a common safety factor of 0.4
# two facing bands together use at most 80 % of the gap between them.
#
# The upper piece sits 2 mm from the cavity wall, so its band and the one of
# Gamma_1 are 0.8 mm (one-rod script: 2.0 and 2.8 mm).
ROD_GAP = [
    float(np.min(_distance_to_polyline(_sample_polyline(points), _IF1_POINTS)))
    for points in _ROD_IF_POINTS
]

# Clearance between consecutive pieces along the axis.
_ROD_TO_ROD = [
    min(
        [abs(ROD_Z_MIN[i] - ROD_Z_MAX[j]) for j in range(N_RODS) if j != i]
        + [abs(ROD_Z_MIN[j] - ROD_Z_MAX[i]) for j in range(N_RODS) if j != i]
        + [np.inf]
    )
    for i in range(N_RODS)
]

GAMMA_GAP = min(ROD_GAP)

_GAMMA1_TO_OUTER_MARGIN = min(R_OUT - R_NH3, Z_NH3_MIN, H_OUT - Z_NH3_TOP)

INTERFACE_BAND_SAFETY = 0.4

INTERFACE_BAND_1 = INTERFACE_BAND_SAFETY * min(
    _GAMMA1_TO_OUTER_MARGIN, R_NH3, GAMMA_GAP
)

ROD_BAND = [
    INTERFACE_BAND_SAFETY * min(ROD_GAP[i], ROD_R[i], _ROD_TO_ROD[i])
    for i in range(N_RODS)
]

# Guard so that d(sqrt)/dx stays finite for a point sitting exactly on Gamma.
_IF_EPS2 = 1.0e-24


def polyline_projection(x, p0, p1):
    """Closest point of a polyline to ``x``, and the distance to it.

    Returns ``(distance, projected_point)``. The projection is what makes the
    construction work: a function of the projected point is constant along
    the normal, so its normal derivative vanishes and the flux trace is the
    ONLY thing that sets dT/dn on either side.
    """
    d = p1 - p0
    len2 = jnp.sum(d * d, axis=-1)

    t = jnp.sum((x - p0) * d, axis=-1) / len2
    t = jnp.clip(t, 0.0, 1.0)

    proj = p0 + t[:, None] * d
    diff = x - proj
    dist2 = jnp.sum(diff * diff, axis=-1)

    nearest = jnp.argmin(dist2)

    return (jnp.sqrt(dist2[nearest] + _IF_EPS2), proj[nearest])


def interface_phi_1(x):
    """Signed distance to Gamma_1, positive inside the NH3 cavity."""
    distance, projected = polyline_projection(x, _IF1_P0, _IF1_P1)

    sign = jnp.where(in_nh3(x[0], x[1]), 1.0, -1.0)

    return sign * distance, projected


def interface_phi_rod(i, x):
    """Signed distance to the surface of piece i, positive in the NH3, negative
    in the GaN."""
    distance, projected = polyline_projection(x, _ROD_IF_P0[i], _ROD_IF_P1[i])

    sign = jnp.where(in_rod(i, x[0], x[1]), -1.0, 1.0)

    return sign * distance, projected


def interface_cutoff(phi, band):
    """Band cutoff: 1 on Gamma, 0 beyond ``band``, flat at both ends.

    ``chi'(0) = 0`` is essential. The normal derivative of chi * G would
    otherwise contribute chi' * G at the interface and destroy the flux
    identity that the whole construction is built on.

    Cubed, not squared as in the Inconel/NH3 script: (1 - q^2)^2 has
    chi'' = 8 / band^2 just inside the band edge and 0 just outside, a jump
    that the PDE residual sees as a spike of 8 (G - B) / band^2 all along the
    edge. With the two interfaces only 7 mm apart the bands are 2-3x narrower
    here, which made that spike dominate the loss (every one of the largest
    residuals sat at |phi| = band). (1 - q^2)^3 is C2 at the edge.
    """
    q = phi / band

    return jnp.where(jnp.abs(q) < 1.0, (1.0 - q * q) ** 3, 0.0)


def _interface_trace_features_1(x):
    """Network input for the two Gamma_1 traces: the projected point.

    QUADRATIC in the projected radius: along the cavity's top and bottom
    faces the projection is (r, const), so a trace linear in r_gamma would
    have a nonzero radial derivative on the symmetry axis. Squaring makes the
    traces even in r, so dU/dr = dW/dr = 0 at r = 0 by construction.
    """
    x = jnp.asarray(x, dtype=jnp.float64)

    _, projected = interface_phi_1(x)

    ends = end_layer_features(projected[1:2], Z_NH3_MIN, Z_NH3_TOP, R_NH3)

    return jnp.concatenate(
        [(projected[0:1] / R_NH3) ** 2, projected[1:2] / H_OUT] + ends, axis=-1
    )


def _interface_trace_features_rod(i):
    """Network input for the two traces of piece i, normalized on the piece."""
    radius, z_min, z_max = ROD_R[i], ROD_Z_MIN[i], ROD_Z_MAX[i]

    def features(x):
        x = jnp.asarray(x, dtype=jnp.float64)

        _, projected = interface_phi_rod(i, x)

        ends = end_layer_features(projected[1:2], z_min, z_max, radius)

        return jnp.concatenate(
            [(projected[0:1] / radius) ** 2, (projected[1:2] - z_min) / (z_max - z_min)]
            + ends,
            axis=-1,
        )

    return features


def _with_mu(features):
    """Pre-processing (x, mu): the spatial features, then the parameter ones."""

    def pre_processing(x, mu):
        return jnp.concatenate([features(x), mu_features(mu)], axis=-1)

    return pre_processing


scaled_coordinates = _with_mu(_scaled_coordinates)
nh3_scaled_coordinates = _with_mu(_nh3_scaled_coordinates)
gan_scaled_coordinates = [
    _with_mu(_gan_scaled_coordinates(i)) for i in range(N_RODS)
]
interface_trace_features_1 = _with_mu(_interface_trace_features_1)
interface_trace_features_rod = [
    _with_mu(_interface_trace_features_rod(i)) for i in range(N_RODS)
]


# Interface leads from the harmonic field, tabulated along both interfaces
#
# Each interface ansatz K(U(P)) + phi W is blended into its material's
# background by chi, whose second derivative amplifies any mismatch between
# the two by ~1/band^2. With the NH3 background built on the harmonic field
# T_outer + D, the leads of BOTH interface traces are therefore taken from
# that same field, as its first-order Taylor expansion about the projected
# point P:
#
#     U lead = T_outer(P) + D(P)
#     W lead = grad K_nh3(T_outer + D)(P) . grad(phi)(x)   (GaN surfaces)
#
# so that the ansatz and the background differ by O(phi^2) only. Hand-made
# rod traces (rod profile, fin-model wall flux) left a residual of up to 70 at
# the rod corners, which the networks could not cancel.
#
# In a corner quadrant of a GaN surface (outside the convex piece, beyond a corner)
# every point projects onto the vertex and grad(phi) is the radial direction
# from it; elsewhere it is the segment normal. W is then continuous and
# constant along every normal ray, which is all the flux identity needs.
#
# D and grad D on the interfaces are functions of the position along them
# only. Evaluated directly they cost a 30-mode Bessel sum (and its gradient)
# at every collocation point, differentiated twice more by the PDE residual:
# most of the cost of a residual evaluation, and minutes of XLA compilation.
# They are therefore tabulated once: every segment is cut into panels no
# longer than _LEAD_PANEL, with Chebyshev interpolants of degree _LEAD_DEGREE
# in the segment parameter on each (error < 1e-8 of the leads).
_LEAD_PANEL = 0.6 * min(ROD_R)

_LEAD_DEGREE = 12


class InterfaceLeadTable:
    """D and grad D at the projection of a point on a polyline, tabulated.

    Parametric version: D = sum_c weight_c(mu) Phi_c is a combination of fixed
    unit fields (the channels returned by ``fields``), so each channel is
    tabulated once and the weights of the case are applied at evaluation.
    """

    def __init__(self, p0, p1, left, fields):
        self.p0 = jnp.asarray(p0)
        self.p1 = jnp.asarray(p1)

        p0, p1 = np.asarray(p0), np.asarray(p1)
        tangents = (p1 - p0) / np.linalg.norm(p1 - p0, axis=-1, keepdims=True)
        rotation = (
            np.array([[0.0, -1.0], [1.0, 0.0]])
            if left
            else np.array([[0.0, 1.0], [-1.0, 0.0]])
        )
        # Unit normal of each segment, pointing into the NH3.
        self.normals = jnp.asarray(tangents @ rotation.T)

        lengths = np.linalg.norm(p1 - p0, axis=-1)
        self.panels = np.maximum(np.ceil(lengths / _LEAD_PANEL), 1).astype(int)
        self.offsets = np.concatenate([[0], np.cumsum(self.panels)[:-1]])

        nodes = np.cos(np.pi * (np.arange(_LEAD_DEGREE + 1) + 0.5) / (_LEAD_DEGREE + 1))
        points = []
        for s, n_panels in enumerate(self.panels):
            for j in range(n_panels):
                t = (j + 0.5 * (nodes + 1.0)) / n_panels
                points.append(p0[s] + t[:, None] * (p1[s] - p0[s]))
        points = jnp.asarray(np.concatenate(points))

        def values_and_gradients(p):
            # (channels, 3): Phi_c, dPhi_c/dr, dPhi_c/dz.
            return jnp.concatenate(
                [fields(p)[:, None], jax.jacfwd(fields)(p)], axis=-1
            )

        values = np.asarray(jax.vmap(values_and_gradients)(points))
        n_channels = values.shape[1]
        values = values.reshape(-1, _LEAD_DEGREE + 1, n_channels * 3)

        # (rows, degree + 1, channels, 3): Chebyshev coefficients of every
        # channel's value and gradient.
        self.coefficients = jnp.asarray(
            np.stack(
                [
                    np.polynomial.chebyshev.chebfit(nodes, v, _LEAD_DEGREE).reshape(
                        _LEAD_DEGREE + 1, n_channels, 3
                    )
                    for v in values
                ]
            )
        )

    def __call__(self, x, weights):
        """(D(P), grad D(P), P, n) for one point ``x`` and channel ``weights``.

        n = grad(phi) in the NH3.
        """
        d = self.p1 - self.p0
        t_raw = jnp.sum((x - self.p0) * d, axis=-1) / jnp.sum(d * d, axis=-1)
        t = jnp.clip(t_raw, 0.0, 1.0)
        dist2 = jnp.sum((x - (self.p0 + t[:, None] * d)) ** 2, axis=-1)

        s = jnp.argmin(dist2)
        projected = self.p0[s] + t[s] * d[s]

        n_panels = jnp.asarray(self.panels)[s]
        position = t[s] * n_panels
        j = jnp.clip(jnp.floor(position), 0, n_panels - 1)
        y = 2.0 * (position - j) - 1.0

        # Clenshaw recurrence, on the channels combined with the case weights
        c = self.coefficients[jnp.asarray(self.offsets)[s] + j.astype(int)]
        c = jnp.einsum("kcj,c->kj", c, weights)
        b1 = b2 = jnp.zeros(3)
        for k in range(_LEAD_DEGREE, 0, -1):
            b1, b2 = 2.0 * y * b1 - b2 + c[k], b1
        values = y * b1 - b2 + c[0]

        # Beyond a vertex, grad(phi) is radial from it; the in-NH3 sign is the
        # segment normal's, a corner quadrant lying on the NH3 side.
        offset = x - projected
        radial = offset / jnp.sqrt(jnp.sum(offset * offset) + _IF_EPS2)
        radial = radial * jnp.sign(jnp.sum(radial * self.normals[s]) + 1.0e-30)
        in_corner = (t_raw[s] < 0.0) | (t_raw[s] > 1.0)
        normal = jnp.where(in_corner, radial, self.normals[s])

        return values[0:1], values[1:3], projected, normal


def interface_leads(table, x, mu):
    """Leading-order (U, W) of an interface trace at the projection of ``x``.

    U = T_outer(P) + D(P) and W = k_nh3 grad(T_outer + D)(P) . n: the value
    and normal derivative, in NH3 Kirchhoff units, of the NH3 background lead
    (D is the tabulated NH3 departure: rod field plus refraction field), for
    case mu.
    """
    departure, gradient, projected, normal = table(x, nh3_departure_weights(mu))

    u = T_axial_profile(projected[1:2], mu) + departure
    gradient_t = gradient + jnp.concatenate([jnp.zeros(1), mu_t1(mu)])

    w = MATERIALS["nh3"].k(u) * jnp.sum(gradient_t * normal, keepdims=True)

    return u, w


# Leading-order profile: heat refracting around the cavity
#
# The Inconel carries the axial heat flux k_inc |T1| (~4600 W/m2), but the NH3
# cavity is nearly insulating: heat piles up under its floor and is starved
# above its roof. On the FV data the Inconel departs from T_outer by +0.7 K at
# the floor and -1.1 K at the roof, the cavity wall by up to 2 K -- the NH3
# pockets inherited that as a 1.1 K RMS lead error, which no network learned.
#
# Two harmonic perturbations model it, both Bessel series in the whole
# cylinder r <= R_OUT, 0 <= z <= H_OUT that vanish on its three outer faces
# (J0(lam_m R_OUT) = 0, odd images about z = 0 and z = H_OUT):
#
#   Inconel  delta_K_inc, driven by sources INSIDE the cavity (so harmonic in
#            the Inconel), fitted so that on Gamma_1 the Inconel carries the
#            flux the NH3 lead carries:  d(K_inc(T_outer) + delta_K)/dn = W.
#   NH3      delta_T_nh3, driven by sources OUTSIDE the cavity (so harmonic in
#            the NH3), fitted to the wall temperature delta_K_inc / k_inc.
#
# Sources: a sheet under the roof and one over the floor (radial profiles
# (r/R_NH3)^(2j) (1 - r^2/R_NH3^2)^3), and a cylindrical shell along the
# sidewall (radial profile x^8 (1 - x^2)^3, axial density 1 and end layers).
# Like the rod sources they are smooth: a ring or a point source leaks its
# truncated series into the field. Geometry, conductivities and boundary
# data only.
#
# With GaN pieces close to the wall the flux through it follows each piece's
# departure (up to 1750 W/m2 across the 2 mm gap, against an axial 4600 W/m2
# in the Inconel), which densities spanning the whole cavity cannot follow:
# both shells also carry each piece's own axial basis over its span. Against
# FV, with them: Inconel 0.13 -> 0.02 K RMS.
CAVITY_MODES = 60

_CAVITY_SHEET_PROFILES = 4

# Distance of the sheets from the roof/floor, and width of the outer shell
# (3 mm and 8 mm here), relative to the cavity so that they follow the
# geometry. The outer shell must stay inside the Inconel.
_CAVITY_SOURCE_OFFSET = 0.25 * R_NH3

_CAVITY_SHELL_WIDTH = min(2.0 * R_NH3 / 3.0, 0.5 * (R_OUT - R_NH3))

if not (
    Z_NH3_MIN - _CAVITY_SOURCE_OFFSET > 0.0
    and Z_NH3_TOP + _CAVITY_SOURCE_OFFSET < H_OUT
):
    raise ValueError("The refraction sheets must lie inside the Inconel.")

_CAVITY_LAM_NP = jn_zeros(0, CAVITY_MODES) / R_OUT

_CAVITY_LAM = jnp.asarray(_CAVITY_LAM_NP)

_CAVITY_NORM_NP = 2.0 / (R_OUT**2 * _np_j1(_CAVITY_LAM_NP * R_OUT) ** 2)


def _cavity_radial_weights(r0, r1, sheet_power=None):
    """Bessel coefficients of a unit radial profile on [r0, r1]."""
    nodes, weights = np.polynomial.legendre.leggauss(400)
    r = r0 + 0.5 * (r1 - r0) * (nodes + 1.0)
    w = 0.5 * (r1 - r0) * weights

    if sheet_power is None:
        x = (r - r0) / (r1 - r0)
        profile = x**8 * (1.0 - x**2) ** 3
    else:
        profile = (r / r1) ** (2 * sheet_power) * (1.0 - (r / r1) ** 2) ** 3

    return _CAVITY_NORM_NP * np.sum(
        w * r * profile * _np_j0(np.outer(_CAVITY_LAM_NP, r)), axis=1
    )


def _cavity_sources(inside):
    """[(radial weights, kind, parameters)]: sheets at z = s, shells in z."""
    offset = _CAVITY_SOURCE_OFFSET

    if inside:
        sheets = (Z_NH3_TOP - offset, Z_NH3_MAX_LOWER + offset)
        shell_weights = _cavity_radial_weights(0.0, R_NH3)
        z_a, z_b = Z_NH3_MAX_LOWER, Z_NH3_TOP
    else:
        sheets = (Z_NH3_TOP + offset, Z_NH3_MIN - offset)
        shell_weights = _cavity_radial_weights(R_NH3, R_NH3 + _CAVITY_SHELL_WIDTH)
        z_a, z_b = Z_NH3_MIN, Z_NH3_TOP

    sources = [
        (_cavity_radial_weights(0.0, R_NH3, sheet_power=j), "sheet", (s,))
        for s in sheets
        for j in range(_CAVITY_SHEET_PROFILES)
    ]

    axial = [(0.0, 0.5 * (z_a + z_b))]
    axial += [(-rate, z_a) for rate in _ROD_END_RATES]
    axial += [(rate, z_b) for rate in _ROD_END_RATES]

    sources += [(shell_weights, "shell", (mu, c, z_a, z_b)) for mu, c in axial]

    for i in range(N_RODS):
        sources += [
            (shell_weights, "shell", (mu, c, ROD_Z_MIN[i], ROD_Z_MAX[i]))
            for mu, c in ROD[i]["basis"]
        ]

    return sources


def _cavity_amplitude(xp, z, lam, kind, parameters):
    """Axial amplitude of every mode, with the odd images; z of shape (..., 1)."""
    if kind == "sheet":
        (s,) = parameters
        fn = lambda zz: xp.exp(-lam * xp.abs(zz - s)) / (2.0 * lam)
    else:
        mu, c, a, b = parameters
        fn = lambda zz: _layer_integral(xp, zz, lam, mu, c, a, b) / (2.0 * lam)

    return fn(z) - fn(-z) - fn(2.0 * H_OUT - z)


def _cavity_field(xp, j0, lam, r, z, sources, coefs):
    """sum over sources of coef * sum_m J0(lam_m r) w_m A_m(z); r, z (..., 1)."""
    total = sum(
        coef * xp.asarray(weights) * _cavity_amplitude(xp, z, lam, kind, parameters)
        for coef, (weights, kind, parameters) in zip(coefs, sources)
    )

    return xp.sum(j0(lam * r) * total, axis=-1, keepdims=True)


def _cavity_columns(sources, r, z):
    """Field of every unit source at points (r, z), numpy."""
    r, z = np.asarray(r)[:, None], np.asarray(z)[:, None]

    return np.concatenate(
        [
            _cavity_field(np, _np_j0, _CAVITY_LAM_NP, r, z, [source], [1.0])
            for source in sources
        ],
        axis=1,
    )


_INC_REFRACTION_SOURCES = _cavity_sources(inside=True)

_NH3_REFRACTION_SOURCES = _cavity_sources(inside=False)


def _refraction_fit_points():
    """Gamma_1 collocation, every R_NH3 / 12 (1 mm here), unit normals into NH3."""
    points, normals = [], []
    for p0, p1 in zip(np.asarray(_IF1_P0), np.asarray(_IF1_P1)):
        length = float(np.linalg.norm(p1 - p0))
        n = max(int(np.ceil(length / (R_NH3 / 12.0))), 2)
        points.append(p0 + ((np.arange(n) + 0.5) / n)[:, None] * (p1 - p0))
        tangent = (p1 - p0) / length
        normals.append(np.repeat([[-tangent[1], tangent[0]]], n, axis=0))

    return np.concatenate(points), np.concatenate(normals)


def _regularized_lstsq_operator(matrix, regularization=1.0e-8):
    """The linear map rhs -> _regularized_lstsq(matrix, rhs), as a matrix."""
    scale = np.linalg.norm(matrix, axis=0)
    scaled = matrix / scale

    return (
        np.linalg.solve(
            scaled.T @ scaled + regularization * np.eye(scaled.shape[1]), scaled.T
        )
        / scale[:, None]
    )


def _nh3_refraction_unit_fields(p):
    """All NH3 refraction channels with unit weight at one point p (2,) (the
    Jacobian of the grouped field in its weights, as _rod_unit_fields)."""
    return jax.jacfwd(
        lambda weights: _cavity_group_field(_NH3_REFRACTION_GROUPS, p, weights)[0]
    )(jnp.zeros(len(_NH3_REFRACTION_SOURCES)))


def _values_and_gradients(fields, points):
    """(points, channels, 3): every channel's value, d/dr and d/dz."""

    def one(p):
        return jnp.concatenate([fields(p)[:, None], jax.jacfwd(fields)(p)], axis=-1)

    return np.asarray(jax.vmap(one)(jnp.asarray(points)))


# The fit is linear in the data of the case: precompute every matrix once.
_FIT_POINTS, _FIT_NORMALS = _refraction_fit_points()

_FIT_ROD_FIELDS = _values_and_gradients(_rod_unit_fields, _FIT_POINTS)

_H_FIT = 1.0e-6

# Inconel: d(delta_K)/dn = W - k_inc(T_outer) T1 n_z on Gamma_1.
_INC_FLUX_OPERATOR = _regularized_lstsq_operator(
    (
        _cavity_columns(_INC_REFRACTION_SOURCES, *(_FIT_POINTS + _H_FIT * _FIT_NORMALS).T)
        - _cavity_columns(
            _INC_REFRACTION_SOURCES, *(_FIT_POINTS - _H_FIT * _FIT_NORMALS).T
        )
    )
    / (2.0 * _H_FIT)
)

_INC_WALL_COLUMNS = _cavity_columns(_INC_REFRACTION_SOURCES, *_FIT_POINTS.T)

# NH3: the NH3 lead (T_outer + rod field + delta_T_nh3) meets the Inconel
# wall temperature on Gamma_1.
_NH3_WALL_COLUMNS = _cavity_columns(_NH3_REFRACTION_SOURCES, *_FIT_POINTS.T)

_NH3_WALL_OPERATOR = _regularized_lstsq_operator(_NH3_WALL_COLUMNS)

# Normal derivative of every NH3 refraction source on Gamma_1: its share of
# the flux the NH3 lead carries through the wall.
_NH3_WALL_NORMAL = (
    _cavity_columns(_NH3_REFRACTION_SOURCES, *(_FIT_POINTS + _H_FIT * _FIT_NORMALS).T)
    - _cavity_columns(_NH3_REFRACTION_SOURCES, *(_FIT_POINTS - _H_FIT * _FIT_NORMALS).T)
) / (2.0 * _H_FIT)


# Matching points of the rod solve: along the side from _ROD_CORNER_CLEARANCE
# * R of each corner, every _ROD_SIDE_SPACING, and on each end face at
# _ROD_FACE_POINTS radii from the axis to the corner. Measured on the FV data
# of this case and of the one-rod case (RMSE of the lead, K):
#
#   end profiles  clearance  face out to   two pieces   one rod
#        1          0.5 R    (mean)          0.176       0.078
#        4          0.5 R     0.85 R         0.098       0.084
#        4          0.2 R     0.95 R         0.065       0.059
#        6          0.05 R    R              0.040       0.040
#
# Matching right up to the corners needs the regularization: the corner
# carries a flux singularity that no smooth source reproduces, and plain
# least squares then trades the whole solution for the corner points.
_ROD_CORNER_CLEARANCE = 0.05

_ROD_SIDE_SPACING = 0.001

_ROD_FACE_POINTS = 12

# The two end-face heat balances of each piece are exact relations, not fits.
_ROD_HEAT_ROW_WEIGHT = 100.0


def _rod_matching_points(i):
    radius, z_min, z_max = ROD_R[i], ROD_Z_MIN[i], ROD_Z_MAX[i]

    z_a = z_min + _ROD_CORNER_CLEARANCE * radius
    z_b = z_max - _ROD_CORNER_CLEARANCE * radius
    z_side = np.linspace(z_a, z_b, int(round((z_b - z_a) / _ROD_SIDE_SPACING)) + 1)
    r_face = radius * np.linspace(0.0, 1.0, _ROD_FACE_POINTS)

    return (
        np.concatenate([np.full_like(z_side, radius), r_face, r_face]),
        np.concatenate([z_side, np.full_like(r_face, z_min), np.full_like(r_face, z_max)]),
    )


_ROD_MATCH = [_rod_matching_points(i) for i in range(N_RODS)]

# Field of every unit source of piece j at the matching points of piece i
# (without the 1 / k_nh3 of the case).
_ROD_MATCH_FIELDS = [
    [
        np.einsum(
            "pm,psm->ps",
            _np_j0(np.outer(r, _K_MODES_NP)),
            _rod_source_amplitudes(np, j, z[:, None]),
        )
        for j in range(N_RODS)
    ]
    for r, z in _ROD_MATCH
]

_ROD_MATCH_DEPARTURE = [
    np.stack(_rod_departure_columns(np, i, z), axis=1)
    for i, (_, z) in enumerate(_ROD_MATCH)
]

# The NH3 refraction field the pieces also sit in, per unit source.
_ROD_MATCH_REFRACTION = [
    _cavity_columns(_NH3_REFRACTION_SOURCES, r, z) for r, z in _ROD_MATCH
]

_ROD_UNKNOWN_OFFSETS = np.concatenate([[0], np.cumsum(N_ROD_UNKNOWNS)])


def _solve_rods(k_nh3, t1, refraction):
    """Solution vectors [source strengths, a, b] of every piece, one case.

    k_nh3: NH3 conductivity at each piece's mid-height temperature; t1: the
    imposed gradient; refraction: weights of the NH3 refraction field. On
    every matching point the NH3 lead (all pieces' fields plus the refraction
    field) equals the piece's departure.
    """
    offsets = _ROD_UNKNOWN_OFFSETS
    n_total = int(offsets[-1])

    blocks, rhs = [], []

    for i in range(N_RODS):
        n_basis, n_sources = N_ROD_BASIS[i], N_ROD_SOURCES[i]
        start = int(offsets[i])

        rows = np.zeros((len(_ROD_MATCH[i][0]), n_total))
        for j in range(N_RODS):
            first = int(offsets[j])
            rows[:, first : first + N_ROD_SOURCES[j]] = (
                _ROD_MATCH_FIELDS[i][j] / k_nh3[j]
            )

        departure = _ROD_MATCH_DEPARTURE[i]
        rows[:, start : start + n_basis] -= departure[:, :n_basis]
        rows[:, start + n_sources : start + n_sources + 2] -= departure[:, n_basis:]

        blocks.append(rows)
        rhs.append(-_ROD_MATCH_REFRACTION[i] @ refraction)

        # End-face heat: the bottom strengths sum to k A T_rod'(z_min), the
        # top ones to -k A T_rod'(z_max).
        k_a = _GAN_K_ROD * ROD[i]["area"]

        for z_end, sign, first in (
            (ROD_Z_MIN[i], 1.0, n_basis),
            (ROD_Z_MAX[i], -1.0, n_basis + N_END_PROFILES),
        ):
            slope = np.stack(_rod_slope_columns(np, i, np.array([z_end])), axis=1)[0]

            row = np.zeros(n_total)
            row[start : start + n_basis] = sign * slope[:n_basis]
            row[start + n_sources : start + n_sources + 2] = sign * slope[n_basis:]
            row[start + first : start + first + N_END_PROFILES] = -1.0 / k_a

            blocks.append(_ROD_HEAT_ROW_WEIGHT * row[None, :])
            rhs.append(_ROD_HEAT_ROW_WEIGHT * np.array([-sign * t1]))

    solution = _regularized_lstsq_operator(np.concatenate(blocks)) @ np.concatenate(rhs)

    return [solution[int(offsets[i]) : int(offsets[i + 1])] for i in range(N_RODS)]


# Passes of the coupled solve below (the wall temperature moves by less than
# 1e-3 K after the second).
_LEAD_PASSES = 4


def _solve_case(t0, t1):
    """Every leading-order weight of ONE case: [solution of each piece, rod
    field weights, Inconel refraction weights, NH3 refraction weights].

    The three solves depend on each other -- the pieces sit in the NH3
    refraction field, the Inconel carries the flux the NH3 lead passes through
    the wall, the NH3 refraction field follows the Inconel wall temperature --
    and are iterated from a wall at T_outer.
    """
    k_nh3 = [float(MATERIALS["nh3"].k(t0 + t1 * z_mid)) for z_mid in ROD_Z_MID]

    t_wall = t0 + t1 * _FIT_POINTS[:, 1]
    k_wall = np.asarray(k_inc_value(t_wall))

    nh3 = np.zeros(len(_NH3_REFRACTION_SOURCES))

    for _ in range(_LEAD_PASSES):
        rods = _solve_rods(k_nh3, t1, nh3)
        rod_weights = np.concatenate(
            [rods[i][: N_ROD_SOURCES[i]] / k_nh3[i] for i in range(N_RODS)]
        )

        # The NH3 lead on Gamma_1: the flux it carries through the cavity
        # wall, which the Inconel's refraction field matches.
        field = np.einsum("pcj,c->pj", _FIT_ROD_FIELDS, rod_weights)
        u = t_wall + field[:, 0] + _NH3_WALL_COLUMNS @ nh3
        normal_gradient = (
            np.sum((field[:, 1:3] + np.array([0.0, t1])) * _FIT_NORMALS, axis=1)
            + _NH3_WALL_NORMAL @ nh3
        )
        flux = np.asarray(k_nh3_value(u)) * normal_gradient

        target = flux - k_wall * t1 * _FIT_NORMALS[:, 1]

        inconel = _INC_FLUX_OPERATOR @ target
        wall = _INC_WALL_COLUMNS @ inconel / k_wall
        nh3 = _NH3_WALL_OPERATOR @ (wall - field[:, 0])

    return np.concatenate(rods + [rod_weights, inconel, nh3])


# Layout of the weight vector returned by _solve_case / lead_weights.
_W_ROD = [
    slice(int(_ROD_UNKNOWN_OFFSETS[i]), int(_ROD_UNKNOWN_OFFSETS[i + 1]))
    for i in range(N_RODS)
]
_W_ROD_FIELD = slice(_W_ROD[-1].stop, _W_ROD[-1].stop + N_ROD_CHANNELS)
_W_INC = slice(_W_ROD_FIELD.stop, _W_ROD_FIELD.stop + len(_INC_REFRACTION_SOURCES))
_W_NH3 = slice(_W_INC.stop, _W_INC.stop + len(_NH3_REFRACTION_SOURCES))


# Leading-order weights over the parameter box
#
# Every weight is proportional to T1 up to the smooth (T0, T1) dependence of
# the conductivities: the rod solve exactly (at fixed T_ref), the refraction
# fits through k(T) along the wall. So weight / T1 is a smooth function on the
# box -- finite at T1 = 0 -- solved exactly at the nodes of a tensor Chebyshev
# grid (an even number in T1, so no node sits on T1 = 0) and interpolated.
# Measured against the exact solve on every case of 2D_Heat_data.csv: see
# check_lead_interpolation.
LEAD_GRID = (6, 8)


def _chebyshev_nodes(n):
    return np.cos(np.pi * (np.arange(n) + 0.5) / n)


def _build_lead_table():
    nodes = [_chebyshev_nodes(n) for n in LEAD_GRID]
    center, half = np.asarray(MU_CENTER), np.asarray(MU_HALF_WIDTH)

    scaled = np.stack(
        [
            np.stack(
                [
                    _solve_case(center[0] + half[0] * s0, center[1] + half[1] * s1)
                    / (center[1] + half[1] * s1)
                    for s1 in nodes[1]
                ]
            )
            for s0 in nodes[0]
        ]
    )

    # Values at the nodes -> 2-D Chebyshev coefficients (n0, n1, weights).
    v0 = np.polynomial.chebyshev.chebvander(nodes[0], LEAD_GRID[0] - 1)
    v1 = np.polynomial.chebyshev.chebvander(nodes[1], LEAD_GRID[1] - 1)

    return np.einsum("ai,bj,ijw->abw", np.linalg.inv(v0), np.linalg.inv(v1), scaled)


_LEAD_TABLE = jnp.asarray(_build_lead_table())


def _chebyshev_values(s, n):
    """T_0(s) .. T_{n-1}(s), stacked on a new last axis."""
    values = [jnp.ones_like(s), s]
    for _ in range(n - 2):
        values.append(2.0 * s * values[-1] - values[-2])

    return jnp.stack(values[:n], axis=-1)


def lead_weights(mu):
    """All leading-order weights of case mu, shape (..., n_weights)."""
    s = mu_features(mu)
    c0 = _chebyshev_values(s[..., 0], LEAD_GRID[0])
    c1 = _chebyshev_values(s[..., 1], LEAD_GRID[1])

    return mu_t1(mu) * jnp.einsum("...a,...b,abw->...w", c0, c1, _LEAD_TABLE)


def rod_solution(i, mu):
    """[source strengths, a, b] of piece i, case mu."""
    return lead_weights(mu)[..., _W_ROD[i]]


def rod_field_weights(mu):
    """Weights of the rod's field channels: source strengths / k_nh3."""
    return lead_weights(mu)[..., _W_ROD_FIELD]


def nh3_departure_weights(mu):
    """Weights of the NH3 departure channels: rod field, then refraction."""
    weights = lead_weights(mu)

    return jnp.concatenate([weights[..., _W_ROD_FIELD], weights[..., _W_NH3]], axis=-1)


def _cavity_groups(sources):
    """The sources of a refraction field, grouped for evaluation: sheets by
    position, shells by axial interval. [(source indices, kind, group)]."""
    members = {}

    for index, (_, kind, parameters) in enumerate(sources):
        key = (kind, parameters[0]) if kind == "sheet" else (kind, *parameters[2:])
        members.setdefault(key, []).append(index)

    groups = []

    for key, indices in members.items():
        # The 1 / (2 lam) of _cavity_amplitude goes with the radial weights.
        radial = np.stack([sources[s][0] for s in indices]) / (2.0 * _CAVITY_LAM_NP)

        if key[0] == "sheet":
            group = (key[1], jnp.asarray(radial))
        else:
            group = _LayerGroup(
                _CAVITY_LAM_NP,
                [sources[s][2][0] for s in indices],
                [sources[s][2][1] for s in indices],
                key[1],
                key[2],
                radial,
            )

        groups.append((jnp.asarray(indices), key[0], group))

    return groups


_INC_REFRACTION_GROUPS = _cavity_groups(_INC_REFRACTION_SOURCES)

_NH3_REFRACTION_GROUPS = _cavity_groups(_NH3_REFRACTION_SOURCES)


def _cavity_group_field(groups, x, weights):
    """The same field as _cavity_field with these ``weights`` (sources,), at
    one point x (2,), evaluated group by group."""
    r, z = x[0:1], x[1:2]
    images = [(1.0, z), (-1.0, -z), (-1.0, 2.0 * H_OUT - z)]

    amplitudes = 0.0

    for indices, kind, group in groups:
        if kind == "sheet":
            position, radial = group
            amplitudes = amplitudes + (weights[indices] @ radial) * sum(
                sign * jnp.exp(-_CAVITY_LAM * jnp.abs(zz - position))
                for sign, zz in images
            )
        else:
            amplitudes = amplitudes + group.amplitude(weights[indices], images)

    return jnp.sum(bessel_j0(_CAVITY_LAM * r) * amplitudes, axis=-1, keepdims=True)


def inconel_refraction_potential(x, mu):
    """delta_K_inc(x) [W/m], zero on the three outer faces; x of shape (2,)."""
    x = jnp.asarray(x, dtype=jnp.float64)

    return _cavity_group_field(
        _INC_REFRACTION_GROUPS, x, lead_weights(mu)[..., _W_INC]
    )


def nh3_refraction_departure(x, mu):
    """delta_T_nh3(x) [K], harmonic in the cavity; x of shape (2,)."""
    x = jnp.asarray(x, dtype=jnp.float64)

    return _cavity_group_field(
        _NH3_REFRACTION_GROUPS, x, lead_weights(mu)[..., _W_NH3]
    )


def _nh3_departure_unit_fields(p):
    """NH3 departure channels at one point: rod field, then refraction."""
    return jnp.concatenate([_rod_unit_fields(p), _nh3_refraction_unit_fields(p)])


N_NH3_DEPARTURE_CHANNELS = N_ROD_CHANNELS + len(_NH3_REFRACTION_SOURCES)

GAMMA1_LEADS = InterfaceLeadTable(
    _IF1_POINTS[:-1], _IF1_POINTS[1:], left=True, fields=_nh3_departure_unit_fields
)

ROD_LEADS = [
    InterfaceLeadTable(
        points[:-1], points[1:], left=False, fields=_nh3_departure_unit_fields
    )
    for points in _ROD_IF_POINTS
]


# Correction budgets [K], as FRACTIONS of the imposed swing
#
# Each budget is the scale of a network's departure from its leading-order
# profile. The departures of the linear leading-order problem scale with the
# imposed swing, so the budgets are stored as fractions of the swing of the
# case they were sized on -- (T0, T1) = (611.15 K, -300 K/m), 111 K -- and
# multiply the signed swing -T1 H_OUT of the sample's own case:
#
#   Inconel background  0.5 K  (Victor's validated value)
#   Gamma_1 trace U_1   1.0 K  (Victor: 3.0 K. With the refraction lead the
#   NH3 background      1.0 K   wall and pockets are within ~0.3 K RMS of FV
#                               (Victor: 2.0 K); at 3.0 / 2.0 K training
#                               spent the slack drifting the rod by 0.5 K)
#   GaN backgrounds     0.2 K  (rod lead within 0.05 K of FV; tight on
#   GaN traces U        0.5 K   purpose: at 0.5 / 1.0 K training spent the
#                               slack tilting the whole rod by 1-2 K)
_BUDGET_SIZING_DELTA_T_K = 300.0 * 0.370

INC_CORRECTION_FRACTION = 0.5 / _BUDGET_SIZING_DELTA_T_K

INTERFACE_U1_FRACTION = 1.0 / _BUDGET_SIZING_DELTA_T_K

NH3_TRACE_CORR_FRACTION = 1.0 / _BUDGET_SIZING_DELTA_T_K

GAN_TRACE_CORR_FRACTION = 0.2 / _BUDGET_SIZING_DELTA_T_K

INTERFACE_U2_FRACTION = 0.5 / _BUDGET_SIZING_DELTA_T_K


# outer_distance_factor peaks at 1, so the Inconel budget is the scale of the
# Inconel background correction in kelvin. Small on purpose: the strong form
# is nearly blind to smooth error, so slack in this budget is spent on error
# almost for free.
def interface_u1_scale(mu):
    return INTERFACE_U1_FRACTION * signed_swing(mu)


def interface_u2_scale(mu):
    return INTERFACE_U2_FRACTION * signed_swing(mu)


# The background budgets expressed as potentials: a correction of X kelvin is
# a correction of roughly X * k(T) in K. Keeping them small matters because
# any mismatch between an interface ansatz and its background is multiplied
# by chi'' inside the band.
def inc_correction_scale_k(mu):
    return INC_CORRECTION_FRACTION * signed_swing(mu) * MATRIX.k(t_center(mu))


def nh3_trace_corr_k(mu):
    return (
        NH3_TRACE_CORR_FRACTION * signed_swing(mu) * MATERIALS["nh3"].k(t_center(mu))
    )


def gan_trace_corr_k(mu):
    return (
        GAN_TRACE_CORR_FRACTION * signed_swing(mu) * MATERIALS["gan"].k(t_center(mu))
    )


def rod_parabola(i, x, mu):
    """Fin parabola of the GaN potential of piece i, -q(z) (r^2 - R^2) / (4 A),
    scalar.

    It carries the side flux q / (2 pi R) the piece gives off, and cancels
    the Laplacian of the axial profile (k_gan A T_rod'' = q). q is evaluated
    UNclipped: on an end face z sits exactly on the clip bound, where autodiff
    returns half the slope, which broke the exact flux continuity there.
    """
    z = x[1]
    solution = rod_solution(i, mu)

    q = sum(
        solution[j] * (1.0 if rate == 0.0 else jnp.exp(rate * (z - c)))
        for j, (rate, c) in enumerate(ROD[i]["basis"])
    )

    return -q * (x[0] ** 2 - ROD_R[i] ** 2) / (4.0 * ROD[i]["area"])


# Positions of the raw models in the approximation space: the Inconel and NH3
# fields, one GaN field per piece, the two traces of Gamma_1, then the two
# traces of every piece.
IDX_INC, IDX_NH3 = 0, 1
IDX_GAN = [2 + i for i in range(N_RODS)]
IDX_U1, IDX_W1 = 2 + N_RODS, 3 + N_RODS
IDX_U_ROD = [4 + N_RODS + 2 * i for i in range(N_RODS)]
IDX_W_ROD = [5 + N_RODS + 2 * i for i in range(N_RODS)]

N_MODELS = 4 + 3 * N_RODS


def identity_post_processing(raw, x, mu):
    """All the models are combined by hand in _potential_at_point."""
    return raw


# Trial functions with ALL FOUR interface conditions hard-constrained
#
# For an interface Gamma, let phi be the signed distance to it, P(x) the
# closest point of Gamma, chi a band cutoff with chi(0) = 1 and chi'(0) = 0,
# and let
#
#     U(P)  be a shared interface TEMPERATURE trace   [K]
#     W(P)  be a shared interface NORMAL FLUX trace   dK/dn
#
# both functions of the projection alone. In Kirchhoff variables, per
# material m on either side,
#
#     G_m(x) = K_m(U) + phi * W
#     K_m(x) = chi * G_m + (1 - chi) * B_m(x)
#
# with B_m the material's background. On Gamma phi = 0 and chi = 1, so
# K_m^{-1}(K_m) = U on both sides: temperature continuity is exact. For the
# normal derivative chi'(0) = 0 kills the cutoff term and dU/dn = dW/dn = 0,
# leaving dK_m/dn = W on both sides: flux continuity is exact too.
#
# Gamma_1 (Inconel/NH3) and the surface of every GaN piece each get their
# own pair of traces. The NH3 touches them all, and its bands never overlap
# (chi_a chi_b = 0), so it blends all the ansaetze into its background as a
# partition of unity.
# Away from every band each field falls back to its background: the Inconel
# one carries the outer Dirichlet hard constraints, the NH3 and GaN ones
# never meet the outer wall.


def background_inconel(raw, x, mu):
    """Inconel background POTENTIAL: Dirichlet on all three outer faces, hard.

    K is monotone in T, so pinning K to K(T_outer) on a Dirichlet face pins T
    to T_outer exactly.
    """
    return (
        matrix_potential_base(x[..., 1:2], mu)
        + inconel_refraction_potential(x, mu)
        + inc_correction_scale_k(mu) * outer_distance_factor(x) * raw
    )


def _potential_at_point(space, x, mu, material):
    """Kirchhoff potential of one material at ONE point ``x`` of shape (2,),
    for ONE case ``mu`` of shape (2,).

    Working in K rather than T makes the interior equation the LINEAR,
    constant-coefficient Laplace equation ``Delta K = 0`` for every material,
    and flux continuity ``k dT/dn`` continuous becomes ``dK/dn`` continuous,
    which is linear in the network outputs.

    ``material`` is "inc", "nh3", "gan<i>" for GaN piece i, or "gan" for
    whichever piece is nearest in z (diagnostics and cross-section integrals).

    Only the models a material actually needs are evaluated: the Inconel
    never meets a GaN surface, the GaN never meets Gamma_1.
    """
    if material == "gan":
        value = _potential_at_point(space, x, mu, f"gan{N_RODS - 1}")

        for i in range(N_RODS - 2, -1, -1):
            split = 0.5 * (ROD_Z_MAX[i] + ROD_Z_MIN[i + 1])
            value = jnp.where(
                x[1] < split, _potential_at_point(space, x, mu, f"gan{i}"), value
            )

        return value

    variables = space.create_variables()

    def raw(index):
        return variables[index](space, x, mu)

    def gamma_1_terms():
        phi_1, _ = interface_phi_1(x)
        phi_1 = jnp.reshape(phi_1, (1,))
        chi_1 = interface_cutoff(phi_1, INTERFACE_BAND_1)

        u_1, w_1 = interface_leads(GAMMA1_LEADS, x, mu)

        u_1 = u_1 + interface_u1_scale(mu) * raw(IDX_U1)
        w_1 = w_1 + interface_flux_scale(mu) * raw(IDX_W1)

        return phi_1, chi_1, u_1, w_1

    def rod_terms(i):
        phi_2, projected_2 = interface_phi_rod(i, x)
        phi_2 = jnp.reshape(phi_2, (1,))
        chi_2 = interface_cutoff(phi_2, ROD_BAND[i])

        # The flux lead comes from the harmonic field, the temperature lead
        # from the rod: the GaN, ~500x more conductive, is isothermal across
        # its end faces (within 0.05 K on FV), where the harmonic field is
        # not -- a lead varying by 1 K there is, for the GaN, a heat flow
        # hundreds of times what the rod exchanges.
        _, w_2 = interface_leads(ROD_LEADS[i], x, mu)

        u_2 = T_rod_profile(i, projected_2[1:2], mu) + interface_u2_scale(mu) * raw(
            IDX_U_ROD[i]
        )
        w_2 = w_2 + interface_flux_scale(mu) * raw(IDX_W_ROD[i])

        return phi_2, projected_2, chi_2, u_2, w_2

    if material == "inc":
        phi_1, chi_1, u_1, w_1 = gamma_1_terms()

        near_1 = MATRIX.kirchhoff(u_1) + phi_1 * w_1

        return chi_1 * near_1 + (1.0 - chi_1) * background_inconel(
            raw(IDX_INC), x, mu
        )

    if material.startswith("gan"):
        i = int(material[3:])

        phi_2, projected_2, chi_2, u_2, w_2 = rod_terms(i)

        gan_material = MATERIALS["gan"]

        # The rod profile plus the fin's radial parabola (rod_parabola). The
        # interface ansatz carries the parabola minus its first-order Taylor
        # expansion about P: zero, with zero normal derivative, on Gamma_2 --
        # both interface conditions untouched -- but it keeps the ansatz and
        # the background consistent to second order across the end faces,
        # where the parabola does not vanish.
        _, _, _, normal_2 = ROD_LEADS[i](x, jnp.zeros(N_NH3_DEPARTURE_CHANNELS))
        parabola_p, parabola_grad_p = jax.value_and_grad(
            lambda p: rod_parabola(i, p, mu)
        )(projected_2)

        near_2 = (
            gan_material.kirchhoff(u_2)
            + phi_2 * w_2
            + rod_parabola(i, x, mu)
            - parabola_p
            - phi_2 * jnp.sum(parabola_grad_p * normal_2)
        )
        background = (
            gan_material.kirchhoff(T_rod_profile(i, x[..., 1:2], mu))
            + rod_parabola(i, x, mu)
            + gan_trace_corr_k(mu) * raw(IDX_GAN[i])
        )

        return chi_2 * near_2 + (1.0 - chi_2) * background

    # NH3: in the band of Gamma_1, of one GaN surface, or in none.
    phi_1, chi_1, u_1, w_1 = gamma_1_terms()

    nh3_material = MATERIALS["nh3"]

    blend = chi_1 * (nh3_material.kirchhoff(u_1) + phi_1 * w_1)
    chi_total = chi_1

    for i in range(N_RODS):
        phi_2, _, chi_2, u_2, w_2 = rod_terms(i)

        blend = blend + chi_2 * (nh3_material.kirchhoff(u_2) + phi_2 * w_2)
        chi_total = chi_total + chi_2

    background = nh3_material.kirchhoff(
        nh3_leading_profile(x, mu)
    ) + nh3_trace_corr_k(mu) * raw(IDX_NH3)

    return blend + (1.0 - chi_total) * background


def _material_of(material):
    """Registry name of a trial-function material ("gan<i>" -> "gan")."""
    if material == "inc":
        return MATRIX_MATERIAL

    return "gan" if material.startswith("gan") else material


def _temperature_at_point(space, x, mu, material):
    """Physical temperature: invert the material's Kirchhoff potential."""
    return MATERIALS[_material_of(material)].kirchhoff_inverse(
        _potential_at_point(space, x, mu, material)
    )


def _evaluate_pointwise(space, x, mu, material, fn):
    """fn at one point, or vmapped over a batch; a single mu is broadcast."""
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)

    if x.ndim == 2:
        if mu.ndim == 1:
            mu = jnp.broadcast_to(mu, (x.shape[0], mu.shape[0]))

        return jax.vmap(lambda x_one, mu_one: fn(space, x_one, mu_one, material))(
            x, mu
        )

    return fn(space, x, mu, material)


def evaluate_temperature(space, material, x, mu):
    return _evaluate_pointwise(space, x, mu, material, _temperature_at_point)


def evaluate_potential(space, material, x, mu):
    return _evaluate_pointwise(space, x, mu, material, _potential_at_point)


def T_inc_eval(space, x, mu):
    return evaluate_temperature(space, "inc", x, mu)


def T_nh3_eval(space, x, mu):
    return evaluate_temperature(space, "nh3", x, mu)


def T_gan_eval(space, x, mu):
    return evaluate_temperature(space, "gan", x, mu)


def K_inc_eval(space, x, mu):
    return evaluate_potential(space, "inc", x, mu)


def K_nh3_eval(space, x, mu):
    return evaluate_potential(space, "nh3", x, mu)


def K_gan_eval(space, x, mu):
    return evaluate_potential(space, "gan", x, mu)


FIELD_DIMS = {"x": 2, "mu": 2}

K_INC = ParamScalarFunction(FIELD_DIMS, K_inc_eval, f_type="x_mu")


K_NH3 = ParamScalarFunction(FIELD_DIMS, K_nh3_eval, f_type="x_mu")


K_GAN = ParamScalarFunction(FIELD_DIMS, K_gan_eval, f_type="x_mu")


def _rod_potential_field(i):
    """Kirchhoff potential of GaN piece i, as a differentiable field."""

    def evaluate(space, x, mu):
        return evaluate_potential(space, f"gan{i}", x, mu)

    return ParamScalarFunction(FIELD_DIMS, evaluate, f_type="x_mu")


K_GAN_ROD = [_rod_potential_field(i) for i in range(N_RODS)]


T_INC = ParamScalarFunction(FIELD_DIMS, T_inc_eval, f_type="x_mu")


T_NH3 = ParamScalarFunction(FIELD_DIMS, T_nh3_eval, f_type="x_mu")


T_GAN = ParamScalarFunction(FIELD_DIMS, T_gan_eval, f_type="x_mu")


# r-weighted axisymmetric steady-conduction operator
def axisymmetric_potential_laplacian(K, space, x, mu, reference):
    """r-weighted axisymmetric Laplacian of a Kirchhoff potential.

        div(k grad T) = div(grad K) = Laplace(K)

    so the interior equation is  d_r(r d_r K) + r d_zz K = 0,  in which no
    conductivity appears at all. One operator for every material; only the
    ``reference`` it is normalised by is per material.
    """
    dK_dr = K.partial_derivative("x", 0)
    dK_dz = K.partial_derivative("x", 1)

    r = x[..., 0:1]

    raw_residual = (
        r * dK_dr.partial_derivative("x", 0)(space, x, mu)
        + dK_dr(space, x, mu)
        + r * dK_dz.partial_derivative("x", 1)(space, x, mu)
    )

    return raw_residual / reference


# The two corners of each GaN piece, (R, z_min) and (R, z_max), are
# convex corners of a near-isothermal body: the true field has a gradient
# singularity there, which neither the smooth leading order nor the networks
# can represent. Left in the loss, the strong-form residual within a
# millimetre of them dominated it, and its sampling noise drowned the global
# balance of the rod -- training let the whole rod drift by 1 K. Both interface
# conditions stay exact there by construction; only the interior PDE is not
# collocated within ROD_CORNER_EXCLUSION of the corners (0.2 R, but no more
# than half the gap to the cavity wall: 1 mm for both pieces here).
ROD_CORNER_EXCLUSION = [
    min(0.2 * ROD_R[i], 0.5 * ROD_GAP[i]) for i in range(N_RODS)
]


def away_from_rod_corners(r, z):
    away = jnp.ones_like(r, dtype=bool)

    for i in range(N_RODS):
        for z_end in (ROD_Z_MIN[i], ROD_Z_MAX[i]):
            away = away & (
                (r - ROD_R[i]) ** 2 + (z - z_end) ** 2 >= ROD_CORNER_EXCLUSION[i] ** 2
            )

    return away


# Interior residuals
class InconelResidual(InteriorResidual):
    def __init__(self, domain, lower_box=False):
        super().__init__(domain=domain, size=1, model_type="x_mu")

        self.lower_box = lower_box

    def construct_residual(self, *vars):
        # NOT unpacked from *vars: the space holds seven raw models with
        # identity post-processing, so vars[0] is a bare network output. The
        # physical potential is assembled in K_inc_eval, which K_INC wraps.
        K_field = K_INC

        def residual(space, x, mu):
            value = axisymmetric_potential_laplacian(
                K_field, space, x, mu, potential_pde_ref(PDE_REF_MATERIAL["inc"], mu)
            )

            if not self.lower_box:
                return value

            r = x[..., 0:1]
            z = x[..., 1:2]

            return jnp.where(in_lower_ammonia(r, z), 0.0, value)

        return ParamScalarFunction(K_field.dims, residual, f_type=K_field.f_type)


class AmmoniaResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        # See InconelResidual: the assembled field, not vars[1].
        K_field = K_NH3

        def residual(space, x, mu):
            value = axisymmetric_potential_laplacian(
                K_field, space, x, mu, potential_pde_ref(PDE_REF_MATERIAL["nh3"], mu)
            )

            r = x[..., 0:1]
            z = x[..., 1:2]

            # The NH3 proper: the cavity minus the GaN pieces, for both the
            # lower and the upper sampling envelopes.
            keep = material_indicator("nh3", r, z) & away_from_rod_corners(r, z)

            return jnp.where(keep, value, 0.0)

        return ParamScalarFunction(K_field.dims, residual, f_type=K_field.f_type)


class GaNResidual(InteriorResidual):
    """Interior equation in GaN piece ``rod``."""

    def __init__(self, domain, rod):
        super().__init__(domain=domain, size=1, model_type="x_mu")

        self.rod = rod

    def construct_residual(self, *vars):
        # See InconelResidual: the assembled field of the piece, not vars.
        K_field = K_GAN_ROD[self.rod]

        def residual(space, x, mu):
            value = axisymmetric_potential_laplacian(
                K_field, space, x, mu, potential_pde_ref(PDE_REF_MATERIAL["gan"], mu)
            )

            r = x[..., 0:1]
            z = x[..., 1:2]

            return jnp.where(
                in_rod(self.rod, r, z) & away_from_rod_corners(r, z), value, 0.0
            )

        return ParamScalarFunction(K_field.dims, residual, f_type=K_field.f_type)


# Slab energy balance (integral / weak-form constraint)
#
# Q(z) = integral of -k dT/dz 2 pi r dr is the axial heat flow through the
# cross section at height z. With T prescribed on r = R_OUT it is NOT
# independent of z: the difference crosses the sidewall. What holds is the
# energy balance of the slab between z_ref and z:
#
#     Q(z) - Q(z_ref) = integral_{z_ref}^{z} 2 pi R_OUT k dT/dr|_{R_OUT} dz'
#
# (heat entering through the wall, outward normal +r). In Kirchhoff
# variables k grad T = grad K, so both sides are plain derivatives of the
# assembled potentials -- of all three materials across the cross section.

USE_FLUX_BALANCE = True

_N_QUAD_PANEL = 16

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_N_QUAD_PANEL)


def _panel(a, b):
    """Map the Gauss-Legendre rule from [-1, 1] onto [a, b]."""
    return (0.5 * (b - a) * _GL_NODES + 0.5 * (a + b), 0.5 * (b - a) * _GL_WEIGHTS)


# Panels split at every GaN radius and at the cavity wall: the integrand has
# a kink at each material interface, and a single rule across it would not
# integrate accurately.
_Q_BREAKS = sorted({0.0, *ROD_R, R_NH3, R_OUT})

_Q_PANELS = [_panel(a, b) for a, b in zip(_Q_BREAKS[:-1], _Q_BREAKS[1:])]

_Q_R = jnp.array(np.concatenate([p[0] for p in _Q_PANELS]), dtype=jnp.float64)
_Q_W = jnp.array(np.concatenate([p[1] for p in _Q_PANELS]), dtype=jnp.float64)

# Reference height every other cross-section is compared against. Using a
# DIFFERENCE avoids needing to know the true Q, which keeps this constraint
# free of any reference data.
Z_FLUX_REFERENCE = 0.5 * H_OUT

# Gauss-Legendre rule on [-1, 1] for the sidewall integral, mapped per sample
# onto [Z_FLUX_REFERENCE, z].
_N_QUAD_SIDE = 24

_GL_SIDE_NODES, _GL_SIDE_WEIGHTS = np.polynomial.legendre.leggauss(_N_QUAD_SIDE)
_GL_SIDE_NODES = jnp.array(_GL_SIDE_NODES, dtype=jnp.float64)
_GL_SIDE_WEIGHTS = jnp.array(_GL_SIDE_WEIGHTS, dtype=jnp.float64)


def _cross_section_flow(dK_inc_dz, dK_nh3_dz, dK_gan_dz, space, z, mu):
    """Q(z) = integral of -dK/dz 2 pi r dr over the cross section, case mu."""
    points = jnp.stack([_Q_R, jnp.full_like(_Q_R, z)], axis=-1)

    def one_point(x_one):
        r_one, z_one = x_one[0], x_one[1]

        gradient = jnp.where(
            in_gan(r_one, z_one),
            dK_gan_dz(space, x_one, mu),
            jnp.where(
                in_nh3(r_one, z_one),
                dK_nh3_dz(space, x_one, mu),
                dK_inc_dz(space, x_one, mu),
            ),
        )

        return -gradient[0]

    integrand = jax.vmap(one_point)(points)

    return jnp.sum(_Q_W * integrand * 2.0 * jnp.pi * _Q_R)


def _sidewall_inflow(dK_inc_dr, space, z_a, z_b, mu):
    """Heat entering through r = R_OUT between z_a and z_b (oriented), case mu."""
    z_nodes = 0.5 * (z_b - z_a) * _GL_SIDE_NODES + 0.5 * (z_a + z_b)
    weights = 0.5 * (z_b - z_a) * _GL_SIDE_WEIGHTS

    points = jnp.stack([jnp.full_like(z_nodes, R_OUT), z_nodes], axis=-1)

    wall_flux = jax.vmap(lambda x_one: dK_inc_dr(space, x_one, mu)[0])(points)

    return jnp.sum(weights * wall_flux) * 2.0 * jnp.pi * R_OUT


class FluxBalanceResidual(InteriorResidual):
    """Q(z) - Q(z_ref) - sidewall inflow = 0, normalised by the axial flow scale."""

    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        dK_inc_dr = K_INC.partial_derivative("x", 0)
        dK_inc_dz = K_INC.partial_derivative("x", 1)
        dK_nh3_dz = K_NH3.partial_derivative("x", 1)
        dK_gan_dz = K_GAN.partial_derivative("x", 1)

        def residual(space, x, mu):
            z = x[..., 1]

            here = _cross_section_flow(dK_inc_dz, dK_nh3_dz, dK_gan_dz, space, z, mu)
            there = _cross_section_flow(
                dK_inc_dz, dK_nh3_dz, dK_gan_dz, space, Z_FLUX_REFERENCE, mu
            )
            side = _sidewall_inflow(dK_inc_dr, space, Z_FLUX_REFERENCE, z, mu)

            return jnp.reshape((here - there - side) / flux_balance_ref(mu), (1,))

        return ParamScalarFunction(FIELD_DIMS, residual, f_type="x_mu")


# Global energy balances of the GaN pieces and of the cavity
#
# Neither a GaN piece nor the NH3 cavity (pieces included) holds a heat
# source, so the net heat through each boundary vanishes:
#
#     piece:   integral over its surface of its W dA = 0
#     cavity:  integral over Gamma_1 of W_1 dA = 0
#
# (W = dK/dn, the flux trace, so these are exact heat flows in W). The PDE
# residuals imply both, but only through their integral over the whole
# region: an unresolved layer, a band edge or the excluded rod corners can
# hide a net source that costs the pointwise loss almost nothing -- training
# then shifted the whole rod by 0.5-0.8 K. Imposed directly, a 0.05 W
# imbalance is already visible.
USE_ENERGY_BALANCE = True

# 1 W on the sizing case (the rod trades ~0.3 W through each end); heat
# flows scale with the imposed swing, and so does the reference (floored at a
# 1 K swing, like every normalisation).
def energy_balance_ref(mu):
    return 1.0 * delta_t_ref(mu) / _BUDGET_SIZING_DELTA_T_K

_GL8_NODES, _GL8_WEIGHTS = np.polynomial.legendre.leggauss(8)


def _interface_quadrature(points, panel):
    """Nodes on a polyline and weights for integral f dA, dA = 2 pi r ds."""
    nodes, weights = [], []

    for p0, p1 in zip(points[:-1], points[1:]):
        p0, p1 = np.asarray(p0), np.asarray(p1)
        length = float(np.linalg.norm(p1 - p0))
        n_panels = max(int(np.ceil(length / panel)), 1)

        for j in range(n_panels):
            t = (j + 0.5 * (_GL8_NODES + 1.0)) / n_panels
            pts = p0 + t[:, None] * (p1 - p0)
            nodes.append(pts)
            weights.append(
                0.5 * _GL8_WEIGHTS * length / n_panels * 2.0 * np.pi * pts[:, 0]
            )

    return jnp.asarray(np.concatenate(nodes)), jnp.asarray(np.concatenate(weights))


_GAMMA1_QUADRATURE = _interface_quadrature(_IF1_POINTS, 5.0 * R_NH3 / 12.0)

_ROD_QUADRATURE = [
    _interface_quadrature(_ROD_IF_POINTS[i], 0.6 * ROD_R[i]) for i in range(N_RODS)
]


def interface_heat(space, quadrature, table, index, mu):
    """Integral of the flux trace W = lead + flux scale * raw [W], case mu."""
    points, weights = quadrature
    variables = space.create_variables()

    def trace(x_one):
        _, lead = interface_leads(table, x_one, mu)
        return (lead + interface_flux_scale(mu) * variables[index](space, x_one, mu))[0]

    return jnp.sum(weights * jax.vmap(trace)(points))


class EnergyBalanceResidual(InteriorResidual):
    """Net heat through Gamma_1 (``rod`` None) or through the surface of GaN
    piece ``rod``, in energy_balance_ref.

    Each sample carries its own case mu; its position is irrelevant.
    """

    def __init__(self, domain, rod=None):
        super().__init__(domain=domain, size=1, model_type="x_mu")

        self.rod = rod

    def construct_residual(self, *vars):
        if self.rod is None:
            quadrature, table, index = _GAMMA1_QUADRATURE, GAMMA1_LEADS, IDX_W1
        else:
            quadrature = _ROD_QUADRATURE[self.rod]
            table, index = ROD_LEADS[self.rod], IDX_W_ROD[self.rod]

        def residual(space, x, mu):
            heat = interface_heat(space, quadrature, table, index, mu)

            return jnp.reshape(
                heat / energy_balance_ref(mu)[0] + 0.0 * x[..., 0], (1,)
            )

        return ParamScalarFunction(FIELD_DIMS, residual, f_type="x_mu")


# Physical model
class AutoclaveConductionModel(AbstractPhysicalModel):
    def __init__(self):
        super().__init__(main_domain=main_domain)

        self.physical_residuals = {
            # Inconel interior PDE residuals
            "inc_bottom": InconelResidual(inc_bottom),
            "inc_side": InconelResidual(inc_side),
            "inc_top": InconelResidual(inc_top),
            "inc_lower": InconelResidual(inc_lower, lower_box=True),
            # NH3 interior PDE residuals
            "nh3_upper": AmmoniaResidual(nh3_upper),
            "nh3_lower": AmmoniaResidual(nh3_lower),
        }

        # GaN interior PDE residuals, one per piece
        for i in range(N_RODS):
            self.physical_residuals[f"gan_{i}"] = GaNResidual(gan_domains[i], rod=i)

        if USE_FLUX_BALANCE:
            self.physical_residuals["flux_balance"] = FluxBalanceResidual(flux_balance)

        if USE_ENERGY_BALANCE:
            for i in range(N_RODS):
                self.physical_residuals[f"rod_energy_{i}"] = EnergyBalanceResidual(
                    rod_energy_domains[i], rod=i
                )
            self.physical_residuals["cavity_energy"] = EnergyBalanceResidual(
                cavity_energy
            )

        self.data_residuals = {}


model = AutoclaveConductionModel()


# Collocation sampler
domain_sampler = DomainSampler(main_domain, main_domain_fully_covered=True)


# Every collocation point also draws its own case mu, uniformly in MU_BOUNDS.
parameter_sampler = UniformParametricSampler(MU_BOUNDS)

sampler = TensorizedSampler(
    [domain_sampler, parameter_sampler], model_type="x_mu", bc=False, ic=False
)


# Neural-network approximation space
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
        # Compatibility fallback for ScimBa/JAX versions where MLP
        # does not expose the activation keyword argument.
        return MLP(
            in_size=in_size, out_size=out_size, hidden_sizes=hidden_sizes, key=key
        )


# Scale of the networks' output layer at initialisation. The leading order is
# already within 0.26 K RMS of FV; freshly initialised networks put the loss
# 1000x above the lead's own residual (in the GaN, where a random correction
# is a large heat source), and ENG spent its first tens of epochs undoing
# that. Shrunk, training starts AT the leading order.
OUTPUT_INIT_SCALE = 0.01


def _shrink_output_layer(network, scale):
    last = network.layers[-1]
    object.__setattr__(last, "weight", scale * last.weight)

    if last.bias is not None:
        object.__setattr__(last, "bias", scale * last.bias)

    return network


def build_space(seed=42):
    keys = jax.random.split(jax.random.PRNGKey(seed), N_MODELS)

    # The fields (Inconel, NH3, one per GaN piece), then the two traces of
    # Gamma_1 and the two of every piece. The traces live on a curve, so they
    # are 1-D objects and need far less capacity than the fields.
    hidden = [HIDDEN] * (2 + N_RODS) + [HIDDEN_TRACE] * (2 + 2 * N_RODS)

    networks = [
        _shrink_output_layer(
            build_mlp(in_size=n, out_size=1, hidden_sizes=h, key=k), OUTPUT_INIT_SCALE
        )
        for n, h, k in zip(network_input_sizes(), hidden, keys)
    ]

    return ApproximationSpace(
        dict(FIELD_DIMS),
        [(network, "scalar", None) for network in networks],
        model_type="x_mu",
        pre_processing=[
            scaled_coordinates,
            nh3_scaled_coordinates,
            *gan_scaled_coordinates,
            interface_trace_features_1,
            interface_trace_features_1,
            *[
                features
                for i in range(N_RODS)
                for features in [interface_trace_features_rod[i]] * 2
            ],
        ],
        # All the raw outputs are combined by hand in _potential_at_point,
        # which is the only place that can see more than one model at a time.
        post_processing=[identity_post_processing] * N_MODELS,
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

for _i in range(N_RODS):
    W_COMMON[f"gan_{_i}"] = (350.0,)

if USE_FLUX_BALANCE:
    W_COMMON["flux_balance"] = (100.0,)

if USE_ENERGY_BALANCE:
    for _i in range(N_RODS):
        W_COMMON[f"rod_energy_{_i}"] = (1.0e4,)
    W_COMMON["cavity_energy"] = (1.0e4,)


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
    r = np.asarray(r, dtype=np.float64)

    r_clipped = np.clip(r, float(NH3_LOWER_R[0]), float(NH3_LOWER_R[-1]))

    return np.interp(
        r_clipped,
        np.asarray(NH3_LOWER_R, dtype=np.float64),
        np.asarray(NH3_LOWER_Z, dtype=np.float64),
    )


def in_ammonia_numpy(r, z):
    r = np.asarray(r, dtype=np.float64)

    z = np.asarray(z, dtype=np.float64)

    z_bottom = z_bottom_ammonia_numpy(r)

    return (r >= 0.0) & (r <= R_NH3) & (z >= z_bottom) & (z <= Z_NH3_TOP)


def in_gan_numpy(r, z):
    r = np.asarray(r, dtype=np.float64)

    z = np.asarray(z, dtype=np.float64)

    inside = np.zeros(np.broadcast(r, z).shape, dtype=bool)

    for radius, z_min, z_max in GAN_PARTS:
        inside |= (r >= 0.0) & (r <= radius) & (z >= z_min) & (z <= z_max)

    return inside


def material_masks_numpy(r, z):
    """Physical material of each point: (Inconel, NH3, GaN) masks, a partition."""
    gan_mask = in_gan_numpy(r, z)

    nh3_mask = in_ammonia_numpy(r, z) & ~gan_mask

    inc_mask = ~(nh3_mask | gan_mask)

    return (inc_mask, nh3_mask, gan_mask)


# Final neural-field evaluation helper
def evaluate_field(space, points, material, mu=None):
    """Temperature of one material at points, for case mu (default: set_case)."""
    x = jnp.asarray(points, dtype=jnp.float64)

    if mu is None:
        mu = case_mu()

    if material == "inc":
        field = T_INC

    elif material == "nh3":
        field = T_NH3

    elif material == "gan":
        field = T_GAN

    else:
        raise ValueError(f"Unknown material '{material}'. Use 'inc', 'nh3' or 'gan'.")

    return np.asarray(field(space, x, mu)).reshape(-1)


def evaluate_pinn_on_points(space, points, mu=None):
    """PINN temperature at arbitrary points, each from its own material's field.

    Returns the temperatures and the material label of every point.
    """
    points = np.asarray(points, dtype=np.float64)

    masks = material_masks_numpy(points[:, 0], points[:, 1])

    temperature = np.empty(points.shape[0], dtype=np.float64)
    label = np.empty(points.shape[0], dtype=object)

    for mask, material, name in zip(
        masks, ("inc", "nh3", "gan"), ("Inconel", "NH3", "GaN")
    ):
        if np.any(mask):
            temperature[mask] = evaluate_field(space, points[mask], material, mu)

        label[mask] = name

    return temperature, label


def gan_interface_heights(r_probe):
    """Heights of the GaN end faces crossed by a vertical probe line."""
    return tuple(
        z_end
        for radius, z_min, z_max in GAN_PARTS
        if 0.0 <= r_probe <= radius
        for z_end in (z_min, z_max)
    )


# plot_solution: filled contour + isotherm lines
def plot_solution(space, n_r=400, n_z=1200):
    r = np.linspace(0.0, R_OUT, n_r)
    z = np.linspace(0.0, H_OUT, n_z)

    rr, zz = np.meshgrid(r, z, indexing="xy")

    points = np.column_stack([rr.ravel(), zz.ravel()])

    temperature, _ = evaluate_pinn_on_points(space, points)

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
        for radius, z_min, z_max in GAN_PARTS:
            ax.plot(
                np.array([0.0, radius, radius, 0.0], dtype=np.float64) * 1.0e3,
                np.array([z_min, z_min, z_max, z_max], dtype=np.float64) * 1.0e3,
                "k",
                lw=1.2,
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

    # Plot 1: original filled contour
    figure, ax = plt.subplots(figsize=(5.8, 10.0))

    contour = ax.contourf(rr * 1.0e3, zz * 1.0e3, temperature, levels=100, cmap="turbo")
    colorbar = figure.colorbar(contour, ax=ax)
    colorbar.set_label("Temperature [K]")

    draw_geometry(ax)
    ax.set_title(
        f"Axisymmetric Inconel--NH3--GaN conduction\nT0 = {T0_K:.1f} K, T1 = {T1:.1f} K/m"
    )

    figure.tight_layout()
    figure_path = SAVE_DIR / "temperature_field.png"
    figure.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved PINN temperature plot in: {figure_path.resolve()}")

    # Plot 2: filled contour + labeled isotherm lines
    # Isotherms make interface kinks/discontinuities visible directly
    figure2, ax2 = plt.subplots(figsize=(5.8, 10.0))

    contour2 = ax2.contourf(
        rr * 1.0e3, zz * 1.0e3, temperature, levels=100, cmap="turbo"
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
        "Axisymmetric Inconel--NH3--GaN conduction\n"
        "Isotherms every "
        f"{(line_levels[1] - line_levels[0]):.2f} K"
    )

    figure2.tight_layout()
    figure2_path = SAVE_DIR / "temperature_field_isotherms.png"
    figure2.savefig(figure2_path, dpi=250, bbox_inches="tight")
    plt.close(figure2)

    print(f"Saved isotherm plot in: {figure2_path.resolve()}")


def make_axisymmetric_residual(temperature_function, conductivity_function, material):
    """Temperature-form residual map of the CURRENT case (set_case)."""

    def residual_on_points(space, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        mu = case_mu()
        pde_ref = potential_pde_ref(PDE_REF_MATERIAL[material], mu)[0]

        def temperature_at_one_point(x_one):
            return jnp.ravel(temperature_function(space, x_one, mu))[0]

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


# The residual MAPS are still drawn from the temperature form, which stays
# valid and is the more readable diagnostic; only the trained loss moved to
# the potential. Each normalises by its material's potential span, as the loss.
inc_conduction_residual = make_axisymmetric_residual(T_inc_eval, k_inc_value, "inc")

nh3_conduction_residual = make_axisymmetric_residual(T_nh3_eval, k_nh3_value, "nh3")

gan_conduction_residual = make_axisymmetric_residual(T_gan_eval, k_gan_value, "gan")


def apply_interior_material_mask(residual_name, r_grid, z_grid, values):
    """
    Return PDE residual values inside the correct physical material
    region and NaN outside that region.
    """
    r_np = np.asarray(r_grid, dtype=np.float64)

    z_np = np.asarray(z_grid, dtype=np.float64)

    values_np = np.asarray(values, dtype=np.float64)

    if residual_name == "inc_bottom":
        mask = (r_np >= 0.0) & (r_np <= R_OUT) & (z_np >= 0.0) & (z_np <= Z_NH3_MIN)

    elif residual_name == "inc_lower":
        z_polyline = z_bottom_ammonia_numpy(r_np)

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
        z_polyline = z_bottom_ammonia_numpy(r_np)

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
            & ~in_gan_numpy(r_np, z_np)
        )

    elif residual_name == "gan":
        mask = in_gan_numpy(r_np, z_np)

    else:
        raise ValueError(f"Unknown interior residual name: {residual_name}")

    return np.where(mask, values_np, np.nan)


def plot_all_interior_residuals(space, n_r=151, n_z=251):
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
        (
            "gan",
            "GaN pieces",
            gan_conduction_residual,
            0.0,
            max(ROD_R),
            min(ROD_Z_MIN),
            max(ROD_Z_MAX),
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
        r_plot_min = max(r_min, 10.0 * EPS_AXIS)

        r_values = jnp.linspace(r_plot_min, r_max, n_r)

        z_values = jnp.linspace(z_min, z_max, n_z)

        r_grid, z_grid = jnp.meshgrid(r_values, z_values, indexing="xy")

        x_eval = jnp.column_stack([r_grid.reshape(-1), z_grid.reshape(-1)])

        residual_values = residual_function(space, x_eval)

        residual_values = jnp.asarray(residual_values).reshape(n_z, n_r)

        masked_values = apply_interior_material_mask(
            residual_name, r_grid, z_grid, residual_values
        )

        absolute_values = np.abs(masked_values)

        positive_values = absolute_values[
            np.isfinite(absolute_values) & (absolute_values > 0.0)
        ]

        if positive_values.size == 0:
            raise ValueError(f"No positive residual values remain for {residual_name}.")

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
            for (_, _, values, _, _, _, _) in plot_data
        ]
    )

    common_vmin = np.min(all_positive_values)

    common_vmax = np.max(all_positive_values)

    figure, axes = plt.subplots(2, 4, figsize=(20.0, 10.0), constrained_layout=True)

    image = None

    for ax, (residual_name, title, absolute_values, r_min, r_max, z_min, z_max) in zip(
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

    for ax in axes.ravel()[len(plot_data) :]:
        ax.axis("off")

    colorbar = figure.colorbar(
        image, ax=axes.ravel()[: len(plot_data)].tolist(), shrink=0.88
    )

    colorbar.set_label(
        r"Absolute scaled PDE residual " r"$|\mathcal{R}_{\mathrm{PDE}}|$"
    )

    output_path = SAVE_DIR / "all_interior_residual_maps.png"

    figure.savefig(output_path, dpi=250, bbox_inches="tight")

    plt.close(figure)

    print(f"Saved interior residual maps in: {output_path.resolve()}")


# One-dimensional FVM versus PINN line comparison
# One-dimensional FVM/PINN line comparison
def plot_fvm_vs_pinn(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_inconel_nh3_gan_line_comparison.png",
):
    """
    Compare FVM temperatures with the Inconel + NH3 + GaN PINN prediction.

    Required Excel columns:
    - x            radial coordinate r [m]
    - y            axial coordinate z [m]
    - Temperature  FVM temperature [K]

    The Excel input must contain one vertical line:
    all x values must have the same radial coordinate.
    """

    excel_path = Path(excel_path).resolve()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    figure_path = (SAVE_DIR / output_name).resolve()

    comparison_csv = (
        SAVE_DIR / "fvm_vs_pinn_inconel_nh3_gan_line_comparison.csv"
    ).resolve()

    if not excel_path.exists():
        raise FileNotFoundError(f"1D FVM line Excel file does not exist:\n{excel_path}")

    df = read_fvm_table(excel_path, sheet_name=sheet_name)

    required_columns = {"x", "y", "Temperature"}

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df.dropna(subset=["x", "y", "Temperature"]).copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after removing NaN values.")

    r_fvm = df["x"].to_numpy(dtype=np.float64)

    z_fvm = df["y"].to_numpy(dtype=np.float64)

    T_fvm = df["Temperature"].to_numpy(dtype=np.float64)

    # This must be a line at one radius, not the full 2D data.
    if not np.allclose(r_fvm, r_fvm[0], rtol=1.0e-10, atol=1.0e-12):
        unique_r = np.unique(r_fvm)

        raise ValueError(
            "plot_fvm_vs_pinn() expects one vertical FVM line "
            "with a constant x/r coordinate.\n"
            f"Found {len(unique_r)} radial values.\n"
            "Extract one line from the 2D workbook first."
        )

    fvm_points = np.column_stack([r_fvm, z_fvm])

    T_pinn, material = evaluate_pinn_on_points(space, fvm_points)

    error_K = T_pinn - T_fvm

    abs_error_K = np.abs(error_K)

    rmse_K = np.sqrt(np.mean(error_K**2))

    mae_K = np.mean(abs_error_K)

    max_abs_error_K = np.max(abs_error_K)

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

    comparison_df.to_csv(str(comparison_csv), index=False)

    order = np.argsort(z_fvm)

    z_sorted = z_fvm[order]
    T_fvm_sorted = T_fvm[order]
    T_pinn_sorted = T_pinn[order]
    error_sorted = error_K[order]

    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 8.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    axes[0].plot(
        z_sorted * 1.0e3, T_fvm_sorted, "o", ms=4.0, color="black", label="FVM"
    )

    axes[0].plot(
        z_sorted * 1.0e3, T_pinn_sorted, "-", lw=2.0, color="tab:red", label="PINN"
    )

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
            z_bottom_ammonia_numpy(np.array([r_probe], dtype=np.float64))[0]
        )

        for z_interface in (z_nh3_bottom, Z_NH3_TOP) + gan_interface_heights(r_probe):
            axes[0].axvline(
                z_interface * 1.0e3, color="0.35", lw=1.0, ls="--", alpha=0.80
            )

            axes[1].axvline(
                z_interface * 1.0e3, color="0.35", lw=1.0, ls="--", alpha=0.80
            )

    axes[1].axhline(0.0, color="black", lw=1.0)

    axes[1].plot(z_sorted * 1.0e3, error_sorted, "-o", ms=3.0, lw=1.2, color="tab:blue")

    axes[1].set_xlabel("z [mm]")

    axes[1].set_ylabel("PINN - FVM [K]")

    axes[1].grid(True, alpha=0.30)

    fig.tight_layout()

    fig.savefig(str(figure_path), dpi=250, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved FVM/PINN line plot in: {figure_path}")

    return comparison_df


def plot_fvm_vs_pinn_relative_error(
    space,
    excel_path,
    sheet_name="Sheet1",
    output_name="fvm_vs_pinn_inconel_nh3_gan_line_relative_error.png",
):
    """
    Plot the relative error (PINN - FVM) / FVM [%] along a single
    vertical FVM line.
    """
    excel_path = Path(excel_path).resolve()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    figure_path = (SAVE_DIR / output_name).resolve()

    comparison_csv = (
        SAVE_DIR / "fvm_vs_pinn_inconel_nh3_gan_line_relative_error.csv"
    ).resolve()

    if not excel_path.exists():
        raise FileNotFoundError(f"1D FVM line Excel file does not exist:\n{excel_path}")

    df = read_fvm_table(excel_path, sheet_name=sheet_name)

    required_columns = {"x", "y", "Temperature"}

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df.dropna(subset=["x", "y", "Temperature"]).copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after removing NaN values.")

    r_fvm = df["x"].to_numpy(dtype=np.float64)

    z_fvm = df["y"].to_numpy(dtype=np.float64)

    T_fvm = df["Temperature"].to_numpy(dtype=np.float64)

    # This must be a line at one radius, not the full 2D data.
    if not np.allclose(r_fvm, r_fvm[0], rtol=1.0e-10, atol=1.0e-12):
        unique_r = np.unique(r_fvm)

        raise ValueError(
            "plot_fvm_vs_pinn_relative_error() expects one vertical "
            "FVM line with a constant x/r coordinate.\n"
            f"Found {len(unique_r)} radial values.\n"
            "Extract one line from the 2D workbook first."
        )

    fvm_points = np.column_stack([r_fvm, z_fvm])

    T_pinn, material = evaluate_pinn_on_points(space, fvm_points)

    relative_error_pct = (T_pinn - T_fvm) / T_fvm * 100.0

    absolute_relative_error_pct = np.abs(relative_error_pct)

    rmse_pct = np.sqrt(np.mean(relative_error_pct**2))

    mae_pct = np.mean(absolute_relative_error_pct)

    max_abs_relative_error_pct = np.max(absolute_relative_error_pct)

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

    comparison_df.to_csv(str(comparison_csv), index=False)

    order = np.argsort(z_fvm)

    z_sorted = z_fvm[order]
    relative_error_sorted = relative_error_pct[order]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.axhline(0.0, color="black", lw=1.0)

    ax.plot(
        z_sorted * 1.0e3, relative_error_sorted, "-o", ms=3.0, lw=1.2, color="tab:blue"
    )

    ax.set_xlabel("z [mm]")

    ax.set_ylabel("(PINN - FVM) / FVM [%]")

    ax.set_title(f"FVM vs PINN relative error: r = {r_fvm[0] * 1.0e3:.3f} mm")

    ax.grid(True, alpha=0.30)

    # Inconel/NH3 interface lines for this vertical probe.
    r_probe = float(r_fvm[0])

    if 0.0 <= r_probe <= R_NH3:
        z_nh3_bottom = float(
            z_bottom_ammonia_numpy(np.array([r_probe], dtype=np.float64))[0]
        )

        for z_interface in (z_nh3_bottom, Z_NH3_TOP) + gan_interface_heights(r_probe):
            ax.axvline(z_interface * 1.0e3, color="0.35", lw=1.0, ls="--", alpha=0.80)

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

    fig.savefig(str(figure_path), dpi=250, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved FVM/PINN line relative-error plot in: {figure_path}")

    return comparison_df


def extract_fvm_vertical_line(
    full_excel_path,
    target_r=0.004,
    sheet_name="Sheet1",
    output_name="inconel_nh3_gan_line_data_extracted.xlsx",
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

    print(f"Saved extracted FVM line (r={selected_r * 1.0e3:.3f} mm) in: {line_file}")

    return line_file


# FVM versus PINN comparison along every available radial line.
# The corrected reference data (2D_Heat_data.csv) already provides many
# vertical lines (r = 0 to 35 mm in 1 mm steps), so instead of picking one
# probe radius, this compares all of them and grids the result into one figure
def plot_fvm_vs_pinn_all_radii(
    space, df, output_name="fvm_vs_pinn_inconel_nh3_gan_all_radii.png"
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    r_values = np.sort(df["x"].unique())
    n_lines = len(r_values)

    n_cols = 6
    n_rows = int(np.ceil(n_lines / n_cols))

    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.0 * n_cols, 2.6 * n_rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    summary_rows = []
    all_T_fvm = []
    all_T_pinn = []

    for i, r_probe in enumerate(r_values):
        line = df[np.isclose(df["x"], r_probe)].sort_values("y")
        z_line = line["y"].to_numpy(dtype=np.float64)
        T_fvm_line = line["Temperature"].to_numpy(dtype=np.float64)

        points = np.column_stack([np.full_like(z_line, r_probe), z_line])
        T_pinn_line, _ = evaluate_pinn_on_points(space, points)

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
            for z_interface in (z_nh3_bottom, Z_NH3_TOP) + gan_interface_heights(
                r_probe
            ):
                ax.axvline(
                    z_interface * 1.0e3, color="0.5", lw=0.8, ls="--", alpha=0.70
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

    summary_csv = SAVE_DIR / "fvm_vs_pinn_inconel_nh3_gan_all_radii_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved all-radii grid plot in: {output_path.resolve()}")

    return summary_df


# Training-loss history
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

        return values

    def summed_history(keys):
        result = np.zeros(n_epochs, dtype=np.float64)

        for key in keys:
            result += get_history(key)

        return result

    def plot_positive(ax, x, y, label, color, linewidth=1.8, linestyle="-"):
        mask = np.isfinite(y) & (y > 0.0)

        if np.any(mask):
            ax.semilogy(
                x[mask], y[mask], linestyle, lw=linewidth, color=color, label=label
            )

    def save_loss_plot(title, output_filename, curves):
        figure, ax = plt.subplots(figsize=(8.0, 5.5))

        for curve in curves:
            plot_positive(
                ax=ax,
                x=epoch,
                y=curve["values"],
                label=curve["label"],
                color=curve["color"],
                linewidth=curve.get("linewidth", 1.8),
                linestyle=curve.get("linestyle", "-"),
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

        print(f"Saved loss-history plot in: {figure_path.resolve()}")

    # Every outer-wall and interface condition is exact by construction (no
    # boundary residual exists any more), so the only loss groups left are
    # each network's own interior PDE terms -- there is no separate
    # axis/interface curve to plot alongside them.
    inconel_pde_loss = summed_history(
        ["inc_bottom", "inc_side", "inc_top", "inc_lower"]
    )

    save_loss_plot(
        title="Inconel network loss history",
        output_filename="loss_history_inconel.png",
        curves=[
            {
                "values": inconel_pde_loss,
                "label": "Inconel interior PDE",
                "color": "tab:blue",
            }
        ],
    )

    nh3_pde_loss = summed_history(["nh3_lower", "nh3_upper"])

    save_loss_plot(
        title="NH3 network loss history",
        output_filename="loss_history_nh3.png",
        curves=[
            {"values": nh3_pde_loss, "label": "NH3 interior PDE", "color": "tab:blue"}
        ],
    )

    gan_pde_loss = summed_history([f"gan_{i}" for i in range(N_RODS)])

    save_loss_plot(
        title="GaN network loss history",
        output_filename="loss_history_gan.png",
        curves=[
            {"values": gan_pde_loss, "label": "GaN interior PDE", "color": "tab:blue"}
        ],
    )

    interior_loss = inconel_pde_loss + nh3_pde_loss + gan_pde_loss

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
        "nh3_pde_loss": nh3_pde_loss,
        "gan_pde_loss": gan_pde_loss,
        "global_interior_loss": interior_loss,
        "global_total_loss": total_loss,
    }


# Full 2D FVM/PINN direct-node error map
def plot_fvm_vs_pinn_2d_error(
    space, excel_path, sheet_name="Sheet1", output_name="fvm_vs_pinn_2d_error.png"
):
    df_raw = read_fvm_table(excel_path, sheet_name=sheet_name)

    required_columns = {"x", "y", "Temperature"}

    missing_columns = required_columns.difference(df_raw.columns)

    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    df = df_raw.dropna(subset=["x", "y", "Temperature"]).copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after removing NaN values.")

    r_fvm = df["x"].to_numpy(dtype=np.float64)

    z_fvm = df["y"].to_numpy(dtype=np.float64)

    T_fvm = df["Temperature"].to_numpy(dtype=np.float64)

    if {"T0", "T1"}.issubset(df.columns):
        T0_values = np.unique(df["T0"].to_numpy(dtype=np.float64))

        T1_values = np.unique(df["T1"].to_numpy(dtype=np.float64))

        if T0_values.size != 1:
            raise ValueError(f"FVM T0 contains multiple values: {T0_values}")

        if T1_values.size != 1:
            raise ValueError(f"FVM T1 contains multiple values: {T1_values}")

        if not np.isclose(T0_values[0], T0_K):
            raise ValueError(f"FVM/PINN T0 mismatch: FVM={T0_values[0]}, PINN={T0_K}")

        if not np.isclose(T1_values[0], T1):
            raise ValueError(f"FVM/PINN T1 mismatch: FVM={T1_values[0]}, PINN={T1}")

    fvm_points = np.column_stack([r_fvm, z_fvm])

    T_pinn, material = evaluate_pinn_on_points(space, fvm_points)

    signed_error_K = T_pinn - T_fvm
    absolute_error_K = np.abs(signed_error_K)

    rmse_K = np.sqrt(np.mean(signed_error_K**2))

    mae_K = np.mean(absolute_error_K)

    max_abs_error_K = np.max(absolute_error_K)

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

    signed_limit = np.max(np.abs(signed_error_K))

    signed_norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0.0, vmax=signed_limit)

    figure, axes = plt.subplots(1, 2, figsize=(15.0, 8.0), constrained_layout=True)

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

    axes[0].set_title(r"Signed nodal error: " r"$T_{\mathrm{PINN}}-T_{\mathrm{FVM}}$")

    axes[0].set_xlabel("r [mm]")

    axes[0].set_ylabel("z [mm]")

    axes[0].set_aspect("equal")

    axes[0].grid(True, alpha=0.20)

    signed_colorbar = figure.colorbar(signed_plot, ax=axes[0])

    signed_colorbar.set_label("PINN - FVM [K]")

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
        r"Absolute nodal error: " r"$|T_{\mathrm{PINN}}-T_{\mathrm{FVM}}|$"
    )

    axes[1].set_xlabel("r [mm]")

    axes[1].set_ylabel("z [mm]")

    axes[1].set_aspect("equal")

    axes[1].grid(True, alpha=0.20)

    absolute_colorbar = figure.colorbar(absolute_plot, ax=axes[1])

    absolute_colorbar.set_label("Absolute error [K]")

    figure.suptitle(
        (
            "Inconel--NH3--GaN FVM--PINN 2D Nodal Temperature Error\n"
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

    print(f"Saved 2D error plot in: {output_path.resolve()}")

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

    df_raw = read_fvm_table(excel_path, sheet_name=sheet_name)

    required_columns = {"x", "y", "Temperature"}

    missing_columns = required_columns.difference(df_raw.columns)

    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    df = df_raw.dropna(subset=["x", "y", "Temperature"]).copy()

    if len(df) == 0:
        raise ValueError("No valid FVM rows remain after removing NaN values.")

    r_fvm = df["x"].to_numpy(dtype=np.float64)

    z_fvm = df["y"].to_numpy(dtype=np.float64)

    T_fvm = df["Temperature"].to_numpy(dtype=np.float64)

    fvm_points = np.column_stack([r_fvm, z_fvm])

    T_pinn, material = evaluate_pinn_on_points(space, fvm_points)

    relative_error_pct = (T_pinn - T_fvm) / T_fvm * 100.0

    absolute_relative_error_pct = np.abs(relative_error_pct)

    rmse_pct = np.sqrt(np.mean(relative_error_pct**2))

    mae_pct = np.mean(absolute_relative_error_pct)

    max_abs_relative_error_pct = np.max(absolute_relative_error_pct)

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

    signed_limit = np.max(np.abs(relative_error_pct))

    signed_norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0.0, vmax=signed_limit)

    figure, axes = plt.subplots(1, 2, figsize=(15.0, 8.0), constrained_layout=True)

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

    axes[0].set_xlabel("r [mm]")

    axes[0].set_ylabel("z [mm]")

    axes[0].set_aspect("equal")

    axes[0].grid(True, alpha=0.20)

    signed_colorbar = figure.colorbar(signed_plot, ax=axes[0])

    signed_colorbar.set_label("(PINN - FVM) / FVM [%]")

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

    axes[1].set_xlabel("r [mm]")

    axes[1].set_ylabel("z [mm]")

    axes[1].set_aspect("equal")

    axes[1].grid(True, alpha=0.20)

    absolute_colorbar = figure.colorbar(absolute_plot, ax=axes[1])

    absolute_colorbar.set_label("|(PINN - FVM) / FVM| [%]")

    figure.suptitle(
        (
            "Inconel--NH3--GaN FVM--PINN 2D Nodal Relative Temperature Error\n"
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

    csv_path = SAVE_DIR / "fvm_vs_pinn_2d_relative_error_data.csv"

    comparison_df.to_csv(csv_path, index=False)

    print(f"Saved 2D relative-error plot in: {output_path.resolve()}")

    return comparison_df


# Run: Inconel + NH3 + GaN


def _interface_jumps(space, p0s, p1s, plus_side, minus, plus, n_per_segment, mu):
    """Max temperature and flux jump across one interface, case mu. Both ~0.

    ``minus``/``plus`` name the materials on either side; ``plus_side`` is the
    predicate of the ``plus`` side, used to orient the normals toward it.
    """
    points = []
    normals = []

    for a, b in zip(p0s, p1s):
        direction = b - a
        unit = direction / jnp.linalg.norm(direction)
        normal = jnp.array([-unit[1], unit[0]])

        for t in jnp.linspace(0.1, 0.9, n_per_segment):
            point = a + t * direction
            oriented = jnp.where(
                plus_side(point[0] + 1.0e-7 * normal[0], point[1] + 1.0e-7 * normal[1]),
                1.0,
                -1.0,
            )
            points.append(point)
            normals.append(oriented * normal)

    points = jnp.stack(points)
    normals = jnp.stack(normals)

    t_minus = evaluate_temperature(space, minus, points, mu)[:, 0]
    t_plus = evaluate_temperature(space, plus, points, mu)[:, 0]
    t_jump = float(jnp.max(jnp.abs(t_minus - t_plus)))

    # The flux condition is a statement about the two ONE-SIDED limits, and
    # the distance function's gradient is undefined exactly on Gamma. So step
    # off by h on each side; the remaining mismatch is O(h), zero in the
    # limit, which is what "hard" means here.
    offset = 1.0e-9

    def one_sided_flux(material, x_side):
        grad = jax.vmap(
            jax.grad(lambda y: _temperature_at_point(space, y, mu, material)[0])
        )(x_side)
        temperature = evaluate_temperature(space, material, x_side, mu)[:, 0]
        conductivity = MATERIALS[_material_of(material)].k(temperature)

        return conductivity * jnp.sum(grad * normals, axis=-1)

    q_minus = one_sided_flux(minus, points - offset * normals)
    q_plus = one_sided_flux(plus, points + offset * normals)

    q_jump = float(jnp.max(jnp.abs(q_minus - q_plus)))
    q_scale = float(jnp.mean(jnp.abs(q_minus)))

    return t_jump, q_jump, q_scale


def check_interface_hard_constraints(space, n_per_segment=7, verbose=True, mu=None):
    """Temperature and flux jumps across Gamma_1 and every GaN surface."""
    if mu is None:
        mu = case_mu()

    interfaces = {
        "Inconel/NH3": (_IF1_P0, _IF1_P1, in_nh3, "inc", "nh3"),
    }

    for i in range(N_RODS):
        interfaces[f"NH3/GaN piece {i}"] = (
            _ROD_IF_P0[i],
            _ROD_IF_P1[i],
            lambda r, z, i=i: ~in_rod(i, r, z),
            f"gan{i}",
            "nh3",
        )

    results = {}

    for label, (p0s, p1s, plus_side, minus, plus) in interfaces.items():
        t_jump, q_jump, q_scale = _interface_jumps(
            space, p0s, p1s, plus_side, minus, plus, n_per_segment, mu
        )

        results[label] = (t_jump, q_jump)

        if verbose:
            print(
                f"Interface hard constraint ({label}): max|T jump|={t_jump:.3e} K, "
                f"max|flux jump|={q_jump:.3e} W/m2 (typical flux {q_scale:.3e} W/m2)"
            )

    return results


def fvm_errors_all_cases(space, fvm_file):
    """FV error of the ONE trained model on every (T0, T1) case of the table."""
    df = pd.read_csv(fvm_file)
    rows = []

    for (t0, t1), case in df.groupby(["T0", "T1"]):
        points = case[["x", "y"]].to_numpy(dtype=np.float64)
        t_fvm = case["Temperature"].to_numpy(dtype=np.float64)
        mu = jnp.array([t0, t1], dtype=jnp.float64)

        t_pinn, label = evaluate_pinn_on_points(space, points, mu)
        error = t_pinn - t_fvm

        row = {
            "T0": float(t0),
            "T1": float(t1),
            "rmse_K": float(np.sqrt(np.mean(error**2))),
            "mae_K": float(np.mean(np.abs(error))),
            "max_abs_error_K": float(np.max(np.abs(error))),
        }
        for name in ("Inconel", "NH3", "GaN"):
            mask = label == name
            row[f"rmse_{name}_K"] = float(np.sqrt(np.mean(error[mask] ** 2)))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_case_errors(summary, output_name="fvm_vs_pinn_all_cases_rmse.png"):
    """RMSE of the parametric model on every FV case, as a (T0, T1) map."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    table = summary.pivot(index="T1", columns="T0", values="rmse_K").sort_index(
        ascending=False
    )

    # 21 x 21 cases: label every other tick, write values only on a small grid.
    step = 2 if max(table.shape) > 10 else 1

    figure, ax = plt.subplots(figsize=(10.0, 8.0) if step > 1 else (7.0, 4.8))
    image = ax.imshow(table.to_numpy(), cmap="viridis", aspect="auto")
    figure.colorbar(image, ax=ax, label="RMSE against FV [K]")

    ax.set_xticks(
        range(0, table.shape[1], step),
        [f"{v:.2f}" for v in table.columns[::step]],
        rotation=45,
    )
    ax.set_yticks(
        range(0, table.shape[0], step), [f"{v:+.0f}" for v in table.index[::step]]
    )
    ax.set_xlabel("T0 [K]")
    ax.set_ylabel("T1 [K/m]")

    if step == 1:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(
                    j, i, f"{table.to_numpy()[i, j]:.3f}", ha="center",
                    va="center", color="white", fontsize=8,
                )

    ax.set_title(
        "One parametric PINN, all FV cases\n"
        f"mean RMSE {summary['rmse_K'].mean():.3f} K, "
        f"worst {summary['rmse_K'].max():.3f} K"
    )

    figure.tight_layout()
    output_path = SAVE_DIR / output_name
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved all-cases error map in: {output_path.resolve()}")


if __name__ == "__main__":
    space = build_space(seed=42)

    print("Number of neural models:", len(space.models))

    pinn = Projector(
        model,
        space,
        sampler,
        weights=W_COMMON,
        **ENG_OPTIONS,
    )

    key = jax.random.PRNGKey(42)

    # The initial-loss report draws its OWN sample, so training starts from
    # PRNGKey(42) itself, the stream the settings were measured with.
    _, samples = sampler.sample(jax.random.PRNGKey(0), N_COLLOC, {})

    print("Sample keys:", samples.keys())

    initial_losses = pinn.evaluate_losses(space, samples)

    print("Initial losses:", initial_losses)

    start = timeit.default_timer()

    key, pinn = pinn.project(
        key, space, N_EPOCHS, N_COLLOC, {}, tqdm_desc="Inconel-NH3-two-GaN parametric"
    )

    elapsed = timeit.default_timer() - start

    plot_loss_history(pinn.losses, output_name="loss_history.png")

    print("Best loss:", pinn.best_loss)
    print(f"Training time: {elapsed:.2f} s")

    pinn.save("inconel_nh3_two_gan", postfix="parametric")

    print(
        "Saved trained projector under scriptname 'inconel_nh3_two_gan', "
        "'parametric'."
    )

    # FVM validation: Inconel + NH3 + 2 GaN. FV reference with T_outer(z)
    # imposed on the sidewall as well; the matching (T0, T1) case is selected
    # by read_fvm_table.
    PROJECT_DIR = Path(__file__).resolve().parents[1] / "data"
    FVM_FILE_2D = (
        PROJECT_DIR
        / "2D Inconel + NH3 + 2 GaN K(T) extended"
        / "2D_Heat_data.csv"
    )

    if not FVM_FILE_2D.exists():
        raise FileNotFoundError(
            "The 2D FVM data file was not found:\n"
            f"{FVM_FILE_2D.resolve()}\n\n"
            "Replace FVM_FILE_2D with the correct absolute path."
        )

    # The ONE trained model on every (T0, T1) case of the FV table.
    case_summary = fvm_errors_all_cases(pinn.space, FVM_FILE_2D)
    case_summary.to_csv(SAVE_DIR / "fvm_vs_pinn_all_cases_summary.csv", index=False)
    print(case_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    plot_case_errors(case_summary)

    # The 25 cases of the old table against the 416 between them.
    t0_steps = (case_summary["T0"] - 611.15) / 50.0
    t1_steps = case_summary["T1"] / 150.0
    on_old_grid = np.isclose(t0_steps, np.round(t0_steps)) & np.isclose(
        t1_steps, np.round(t1_steps)
    )
    nonzero = ~np.isclose(case_summary["T1"], 0.0)
    for name, rows in (
        ("all cases", case_summary),
        ("old 25-case grid", case_summary[on_old_grid]),
        ("between the old cases", case_summary[~on_old_grid]),
        ("all cases, T1 != 0", case_summary[nonzero]),
    ):
        print(
            f"{name:>24}: n={len(rows):3d}  mean RMSE {rows['rmse_K'].mean():.4f} K"
            f"  worst {rows['rmse_K'].max():.4f} K"
        )

    # Full diagnostics for one case -- the validated one by default; call
    # set_case(T0, T1) before this block for any other.
    set_case(611.15, -300.0)

    plot_all_interior_residuals(pinn.space)

    check_interface_hard_constraints(pinn.space)

    plot_fvm_vs_pinn_2d_error(
        pinn.space,
        excel_path=FVM_FILE_2D,
        sheet_name="Sheet1",
        output_name="fvm_vs_pinn_inconel_nh3_gan_2d_error.png",
    )

    plot_fvm_vs_pinn_2d_relative_error(
        pinn.space,
        excel_path=FVM_FILE_2D,
        sheet_name="Sheet1",
        output_name="fvm_vs_pinn_inconel_nh3_gan_2d_relative_error.png",
    )

    plot_solution(pinn.space)

    # 1D FVM/PINN line comparison. r = 4 mm runs through the GaN block.
    TARGET_LINE_RADIUS = 0.004

    FVM_FILE_LINE = extract_fvm_vertical_line(
        full_excel_path=FVM_FILE_2D,
        target_r=TARGET_LINE_RADIUS,
        sheet_name="Sheet1",
        output_name="inconel_nh3_gan_line_data_extracted.xlsx",
    )

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

    # All-radii FVM/PINN comparison (grid over every radial line in
    # 2D_Heat_data.csv).
    fvm_df_all_radii = read_fvm_table(FVM_FILE_2D, sheet_name="Sheet1")

    plot_fvm_vs_pinn_all_radii(pinn.space, fvm_df_all_radii)
