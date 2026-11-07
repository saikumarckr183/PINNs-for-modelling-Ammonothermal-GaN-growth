r"""Steady heat conduction in an Inconel/NH3 autoclave holding two GaN pieces.

Axisymmetric (r, z) problem with temperature-dependent conductivities:

.. math::

    \nabla \cdot (k_m(T) \nabla T) = 0 \quad \text{in each material } m,

with :math:`T = T_0 + T_1 z` on the whole outer wall, and temperature and
normal heat flux continuous across the two material interfaces (Inconel/NH3
and NH3/GaN). The model is parametric: the networks take
:math:`\mu = (T_0, T_1)` as an input and one training covers every case of
the finite-volume table.

Variant of the "minimal" script whose networks output the Kirchhoff potential
:math:`K_m(T) = \int k_m \, dT` instead of the temperature. Everything else
(geometry, network inputs, sampling, loss terms and weights, optimizer) is
identical to the temperature version. Formulation:

* one MLP per material, :math:`K_m = K_m(T_{wall}(z)) + \sigma_m(\mu) \, D_m \, N_m`
  with :math:`\sigma_m = T_1 H \, k_m(T_c)` and :math:`T_c = T_0 + T_1 H / 2`.
  The wall law is only the boundary data extended inwards; everything else is
  learned.
* :math:`D_m` is the wall factor for the Inconel, which vanishes on the outer
  wall so that :math:`K = K(T_{out})` holds exactly there (and hence
  :math:`T = T_{out}`, since :math:`K_m` is increasing). It is 1 for NH3 and GaN.
* the equation is :math:`\Delta K_m = 0` and the flux is :math:`\nabla K_m`,
  both acting directly (linearly) on the network output.
* the temperature :math:`T_m = K_m^{-1}(K_m)` is recovered by six Newton steps
  started from the wall law. It enters the temperature jump of the loss and
  every evaluation, plot and comparison with the finite volumes.
* the loss is the equation residual in each material plus, on each
  interface, the temperature jump and the flux jump (soft interfaces).

The optimizer is the energy natural gradient (ENG) of ScimBa.
"""

import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm

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
    ParamVecFunction,
)
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import InteriorResidual

jax.config.update("jax_enable_x64", True)

SAVE_DIR = Path("output_inc_nh3_two_gan_parametric_minimal_K_unknown")
FVM_FOLDER = "2D Inconel + NH3 + 2 GaN K(T) extended"

# Geometry [m] (CryS-MAS Autoclave.pcs)
R_OUT, H_OUT = 0.035, 0.370
# NH3 cavity: a cylinder closed below by a spherical cap of the same radius.
R_NH3, Z_NH3_CENTER, Z_NH3_TOP = 0.012, 0.035, 0.335
# GaN pieces on the axis, each (radius, z_min, z_max). The upper one leaves a
# 2 mm NH3 gap to the cavity wall.
GAN_PARTS = [(0.005, 0.040, 0.120), (0.010, 0.175, 0.325)]

# Conductivities k(T) = sum_i c_i T^i [W/(m K)], T in K (Autoclave.mat)
K_COEFFS = {
    "inconel": (6.9995, 0.0151),
    "nh3": (0.3039497375, -0.00019097, 0.0000001887),
    "gan": (130.0,),
}
MATERIALS = tuple(K_COEFFS)

# Parameter box mu = (T0 [K], T1 [K/m]) and the case the figures are drawn for
MU_BOUNDS = [(611.15, 811.15), (-300.0, 300.0)]
PLOT_CASE = (611.15, -300.0)

# Training
N_EPOCHS = 300
HIDDEN = [24, 24, 24]
N_COLLOC = {"inconel": 8000, "nh3": 8000, "wall": 1200, "gan_surface": 1200}
N_COLLOC_GAN = 2000  # per GaN piece
ENG_OPTIONS = {"linesearch": "strong-wolfe", "adaptive_matrix_regularization": True}


# Materials
def conductivity(name, T):
    """k(T) of material ``name``."""
    total, power_of_T = 0.0, 1.0
    for c in K_COEFFS[name]:
        total = total + c * power_of_T
        power_of_T = power_of_T * T
    return total


def kirchhoff(name, T):
    """K(T) = integral of k, so that div(k grad T) = Laplace(K(T))."""
    total, power_of_T = 0.0, T
    for i, c in enumerate(K_COEFFS[name]):
        total = total + c / (i + 1) * power_of_T
        power_of_T = power_of_T * T
    return total


T_MID = 0.5 * (MU_BOUNDS[0][0] + MU_BOUNDS[0][1])
K_REF = {name: float(conductivity(name, T_MID)) for name in MATERIALS}


# Geometry as level sets: positive inside, zero on the surface, with a unit
# gradient on it. R-functions combine half-spaces into bodies with sharp edges.
def r_and(a, b):
    return a + b - jnp.sqrt(a * a + b * b + 1.0e-30)


def r_or(a, b):
    return a + b + jnp.sqrt(a * a + b * b + 1.0e-30)


def cylinder(r, z, radius, z_min, z_max):
    side = (radius * radius - r * r) / (2.0 * radius)
    return r_and(r_and(side, z - z_min), z_max - z)


def level_cavity(x):
    """NH3 cavity, GaN included."""
    r, z = x[..., 0], x[..., 1]
    cap = (R_NH3**2 - r * r - (z - Z_NH3_CENTER) ** 2) / (2.0 * R_NH3)
    return r_or(cylinder(r, z, R_NH3, Z_NH3_CENTER, Z_NH3_TOP), cap)


def level_gan(x):
    r, z = x[..., 0], x[..., 1]
    level = cylinder(r, z, *GAN_PARTS[0])
    for part in GAN_PARTS[1:]:
        level = r_or(level, cylinder(r, z, *part))
    return level


def material_masks(x):
    cavity, gan = level_cavity(x) > 0.0, level_gan(x) > 0.0
    return {"inconel": ~cavity, "nh3": cavity & ~gan, "gan": gan}


def polyline(points):
    """Curve t in [0, 1] -> point, every segment getting the same share of t."""
    points = jnp.asarray(points, dtype=jnp.float64)
    n = points.shape[0] - 1

    def curve(t):
        s = jnp.clip(t * n, 0.0, n - 1.0e-9)
        i = jnp.floor(s).astype(int)
        return points[i] + (s - i)[..., None] * (points[i + 1] - points[i])

    return curve


def cavity_curve(t):
    """Cap, side and roof of the cavity wall, a third of t each."""
    s = 3.0 * t
    angle = 0.5 * jnp.pi * jnp.clip(s, 0.0, 1.0)
    cap = jnp.stack(
        [R_NH3 * jnp.sin(angle), Z_NH3_CENTER - R_NH3 * jnp.cos(angle)], axis=-1
    )
    side_and_roof = polyline(
        [(R_NH3, Z_NH3_CENTER), (R_NH3, Z_NH3_TOP), (0.0, Z_NH3_TOP)]
    )(jnp.clip(0.5 * (s - 1.0), 0.0, 1.0))
    return jnp.where((s < 1.0)[..., None], cap, side_and_roof)


def gan_curve(t):
    """Bottom, side and top of every GaN piece."""
    n = len(GAN_PARTS)
    s = jnp.clip(t * n, 0.0, n - 1.0e-9)
    i = jnp.floor(s).astype(int)
    curves = [
        polyline([(0.0, z_min), (radius, z_min), (radius, z_max), (0.0, z_max)])(s - i)
        for radius, z_min, z_max in GAN_PARTS
    ]
    return jnp.stack(curves)[i]


# The two material interfaces: (label, curve, level set, inside, outside).
INTERFACES = [
    ("wall", cavity_curve, level_cavity, "nh3", "inconel"),
    ("gan_surface", gan_curve, level_gan, "gan", "nh3"),
]

# Scaling
DIMS = {"x": 2, "mu": 2}
MU_CENTER = jnp.array([0.5 * (lo + hi) for lo, hi in MU_BOUNDS])
MU_HALF_WIDTH = jnp.array([0.5 * (hi - lo) for lo, hi in MU_BOUNDS])


def T_wall(x, mu):
    """Prescribed outer-wall temperature T0 + T1 z, extended to the whole domain."""
    return mu[..., 0:1] + mu[..., 1:2] * x[..., 1:2]


def swing(mu):
    """Signed temperature difference along the wall [K]."""
    return mu[..., 1:2] * H_OUT


def residual_scale(mu):
    """|swing| floored at 1 K: every residual is relative to it."""
    return jnp.sqrt(swing(mu) ** 2 + 1.0)


def wall_factor(x):
    """Zero on the outer wall r = R_OUT, z = 0 and z = H_OUT, at most 1 inside."""
    r, z = x[..., 0:1], x[..., 1:2]
    return (1.0 - (r / R_OUT) ** 2) * 4.0 * (z / H_OUT) * (1.0 - z / H_OUT)


# The autoclave is ten times longer than wide: z is mapped to a wider interval
# than the other inputs (which lie in [-1, 1]) so that the networks resolve
# the ends of the cavity and of the GaN.
Z_INPUT_RANGE = 2.0


def network_inputs(x, mu, radius, level_sets):
    """Scaled (r/radius)^2, z and mu, plus the given level sets.

    The field is even in r, hence r^2. Inside its own material a level set is
    a smooth coordinate measuring the distance to that surface.
    """
    r, z = x[..., 0:1], x[..., 1:2]
    inputs = [(r / radius) ** 2, Z_INPUT_RANGE * (2.0 * z / H_OUT - 1.0)]
    inputs += [jnp.tanh(level(x)[..., None] / R_NH3) for level in level_sets]
    inputs += [(mu - MU_CENTER) / MU_HALF_WIDTH]
    return jnp.concatenate(inputs, axis=-1)


def inputs_inconel(x, mu):
    return network_inputs(x, mu, R_OUT, (level_cavity,))


def inputs_nh3(x, mu):
    return network_inputs(x, mu, R_NH3, (level_cavity, level_gan))


def inputs_gan(x, mu):
    return network_inputs(x, mu, R_NH3, ())


# Sampling domains: one box per material; the residuals mask what lies
# outside the material.
main_domain = Square2D([(0.0, 1.0), (0.0, 1.0)], is_main_domain=True, label_str="main")
main_domain.set_mapping(
    DomainMapping.rectangle(
        origin=jnp.array([0.0, 0.0]),
        x_dir=jnp.array([R_OUT, 0.0]),
        y_dir=jnp.array([0.0, H_OUT]),
    ),
    bounds_postmap=[(0.0, R_OUT), (0.0, H_OUT)],
)
BOXES = {}


def box(label, r_max=R_OUT, z_min=0.0, z_max=H_OUT):
    domain = Square2D(
        [(0.0, r_max / R_OUT), (z_min / H_OUT, z_max / H_OUT)],
        is_main_domain=False,
        label_str=label,
    )
    main_domain.add_subdomain(domain)
    BOXES[label] = (r_max, z_min, z_max)
    return domain


class Conduction(InteriorResidual):
    """r Laplace(K) = d_r(r d_r K) + r d_zz K = 0 inside one material.

    ``length`` and ``k_ref`` make the residual dimensionless.
    """

    def __init__(self, domain, name, length, k_ref):
        super().__init__(domain=domain, size=1, model_type="x_mu")
        self.name, self.length, self.k_ref = name, length, k_ref

    def construct_residual(self, *vars):
        K = potential_field(self.name, vars)
        K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
        K_rr, K_zz = K_r.partial_derivative("x", 0), K_z.partial_derivative("x", 1)

        def residual(space, x, mu):
            r = x[..., 0:1]
            value = r * (K_rr(space, x, mu) + K_zz(space, x, mu)) + K_r(space, x, mu)
            value = value * self.length / (self.k_ref * residual_scale(mu))
            return jnp.where(material_masks(x)[self.name][..., None], value, 0.0)

        return ParamScalarFunction(DIMS, residual, f_type="x_mu")


def cavity_residuals():
    """Conduction residuals of the Inconel and of the NH3."""
    return {
        "inconel": Conduction(box("inconel"), "inconel", R_OUT, K_REF["inconel"]),
        "nh3": Conduction(
            box("nh3", R_NH3, Z_NH3_CENTER - R_NH3, Z_NH3_TOP),
            "nh3",
            R_NH3,
            K_REF["nh3"],
        ),
    }


def gan_residuals():
    """One conduction residual per GaN piece.

    The GaN exchanges heat only with the NH3 around it, so its equation is
    measured in NH3 units.
    """
    residuals = {}
    for i, (radius, z_min, z_max) in enumerate(GAN_PARTS):
        residuals[f"gan_{i}"] = Conduction(
            box(f"gan_{i}", radius, z_min, z_max), "gan", radius, K_REF["nh3"]
        )
        N_COLLOC[f"gan_{i}"] = N_COLLOC_GAN
    return residuals


def interface_jumps(curve, level_set, inside, outside, variables):
    """Temperature jump [K], normal-flux jump and inside normal flux [W/m2] at t.

    The normal is the gradient of the level set of the enclosed body.
    """
    T_in, T_out = temperature_field(inside, variables), temperature_field(outside, variables)

    def flux(name):
        K = potential_field(name, variables)
        K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
        return lambda s, x, mu: jnp.concatenate([K_r(s, x, mu), K_z(s, x, mu)], -1)

    flux_in, flux_out = flux(inside), flux(outside)

    def jumps(space, t, mu):
        point = curve(t)
        normal = jax.grad(lambda p: jnp.sum(level_set(p)))(point)
        normal = normal / jnp.sqrt(jnp.sum(normal * normal) + 1.0e-30)
        temperature_jump = T_in(space, point, mu) - T_out(space, point, mu)
        normal_flux_in = jnp.sum(flux_in(space, point, mu) * normal, axis=-1, keepdims=True)
        normal_flux_out = jnp.sum(flux_out(space, point, mu) * normal, axis=-1, keepdims=True)
        return temperature_jump, normal_flux_in - normal_flux_out, normal_flux_in

    return jumps


# Trial functions in the potential: K_m = K_m(T_wall) + sigma_m * D_m * N_m, the
# Inconel one times the wall factor D so that the Dirichlet condition holds
# exactly (K = K(T_out) on the outer wall, hence T = T_out since K is
# increasing). sigma_m is the signed swing times k_m at the mean wall
# temperature T_c, so that N_m = 1 is about one swing in kelvin, the same size
# of correction as in the temperature version (for the GaN, whose k is
# constant, the scale is identical). It is fixed by this argument, not tuned.
def potential_scale(name, mu):
    """sigma_m(mu) = T1 H k_m(T_c), T_c = T0 + T1 H / 2 [W/m]."""
    return swing(mu) * conductivity(name, mu[..., 0:1] + 0.5 * swing(mu))


def potential_inconel(raw, x, mu):
    return kirchhoff("inconel", T_wall(x, mu)) + (
        potential_scale("inconel", mu) * wall_factor(x) * raw
    )


def potential_nh3(raw, x, mu):
    return kirchhoff("nh3", T_wall(x, mu)) + potential_scale("nh3", mu) * raw


def potential_gan(raw, x, mu):
    return kirchhoff("gan", T_wall(x, mu)) + potential_scale("gan", mu) * raw


def build_space(seed=42):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    networks = [
        MLP(in_size=in_size, out_size=1, hidden_sizes=HIDDEN, key=key)
        for in_size, key in zip((5, 6, 4), keys)
    ]
    return ApproximationSpace(
        dict(DIMS),
        [(network, "scalar", None) for network in networks],
        model_type="x_mu",
        pre_processing=[inputs_inconel, inputs_nh3, inputs_gan],
        post_processing=[potential_inconel, potential_nh3, potential_gan],
    )


def potential_field(name, variables):
    """Kirchhoff potential K_m of material ``name``, from the variables of the space."""
    return variables[MATERIALS.index(name)]


def temperature_field(name, variables):
    """Temperature: the potential inverted by Newton steps from the wall law."""
    K = potential_field(name, variables)

    def T(space, x, mu):
        target, value = K(space, x, mu), T_wall(x, mu)
        for _ in range(6):
            value = value - (kirchhoff(name, value) - target) / conductivity(name, value)
        return value

    return ParamScalarFunction(DIMS, T, f_type="x_mu")


class Interface(InteriorResidual):
    """Temperature jump and normal-flux jump across an interface, both zero.

    The interface is a curve, sampled through a dummy box: the second
    coordinate of a sample, divided by H_OUT, is the curve parameter. The
    normal is the gradient of the level set of the enclosed body.
    """

    def __init__(self, domain, curve, level_set, inside, outside, length, k_ref):
        super().__init__(domain=domain, size=2, model_type="x_mu")
        self.curve, self.level_set = curve, level_set
        self.inside, self.outside = inside, outside
        self.length, self.k_ref = length, k_ref

    def construct_residual(self, *vars):
        jumps = interface_jumps(self.curve, self.level_set, self.inside, self.outside, vars)

        def residual(space, x, mu):
            temperature_jump, flux_jump, _ = jumps(space, x[..., 1] / H_OUT, mu)
            return jnp.concatenate(
                [temperature_jump, flux_jump * self.length / self.k_ref], axis=-1
            ) / residual_scale(mu)

        return ParamVecFunction(2, DIMS, residual, f_type="x_mu")


R_GAN = min(part[0] for part in GAN_PARTS)


class AutoclaveConduction(AbstractPhysicalModel):
    def __init__(self):
        super().__init__(main_domain=main_domain)
        self.physical_residuals = cavity_residuals()
        self.physical_residuals["wall"] = Interface(
            box("wall"), cavity_curve, level_cavity, "nh3", "inconel", R_NH3, K_REF["inconel"]
        )
        # Like the GaN equation, the GaN surface flux is measured in NH3 units.
        self.physical_residuals["gan_surface"] = Interface(
            box("gan_surface"), gan_curve, level_gan, "gan", "nh3", R_GAN, K_REF["nh3"]
        )
        self.physical_residuals.update(gan_residuals())
        self.data_residuals = {}


model = AutoclaveConduction()

# Every collocation point draws its own case mu, uniformly in MU_BOUNDS.
sampler = TensorizedSampler(
    [
        DomainSampler(main_domain, main_domain_fully_covered=True),
        UniformParametricSampler(MU_BOUNDS),
    ],
    model_type="x_mu",
    bc=False,
    ic=False,
)


# Names used by the plotting and diagnostics block below
# The case the diagnostics and plots are drawn for (NOT read by the model).
T0_K, T1 = PLOT_CASE


def set_case(t0, t1):
    """Select the (T0, T1) case the diagnostics and plots evaluate."""
    global T0_K, T1
    T0_K, T1 = float(t0), float(t1)


def case_mu():
    """mu of the current diagnostic case, shape (2,)."""
    return jnp.array([T0_K, T1], dtype=jnp.float64)


N_RODS = len(GAN_PARTS)
ROD_R = [part[0] for part in GAN_PARTS]
ROD_Z_MIN = [part[1] for part in GAN_PARTS]
ROD_Z_MAX = [part[2] for part in GAN_PARTS]

EPS_AXIS = 1.0e-7
Z_NH3_MIN = Z_NH3_CENTER - R_NH3
Z_NH3_MAX_LOWER = Z_NH3_CENTER
# The spherical cap as a curve, for drawing the lower NH3/Inconel boundary.
NH3_LOWER_R = R_NH3 * np.sin(np.linspace(0.0, 0.5 * np.pi, 91))
NH3_LOWER_Z = Z_NH3_CENTER - R_NH3 * np.cos(np.linspace(0.0, 0.5 * np.pi, 91))

MATERIAL_OF = {"inc": "inconel", "nh3": "nh3", "gan": "gan"}


def _temperature_eval(material):
    def evaluate(space, x, mu):
        return temperature_field(MATERIAL_OF[material], space.create_variables())(space, x, mu)

    return evaluate


T_inc_eval, T_nh3_eval, T_gan_eval = (_temperature_eval(m) for m in ("inc", "nh3", "gan"))


def k_inc_value(temperature_K):
    return conductivity("inconel", temperature_K)


def k_nh3_value(temperature_K):
    return conductivity("nh3", temperature_K)


def k_gan_value(temperature_K):
    return conductivity("gan", temperature_K)


# Scaling of the residual maps: the one of the loss terms (for the GaN, with
# the smallest piece radius).
PDE_REF_MATERIAL = {"inc": "inc", "nh3": "nh3", "gan": "gan"}
_PDE_REF = {
    "inc": K_REF["inconel"] / R_OUT,
    "nh3": K_REF["nh3"] / R_NH3,
    "gan": K_REF["nh3"] / min(ROD_R),
}


def potential_pde_ref(material, mu):
    return _PDE_REF[material] * residual_scale(mu)


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

    r_clipped = np.clip(r, 0.0, R_NH3)

    return Z_NH3_CENTER - np.sqrt(R_NH3**2 - r_clipped**2)


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

    if material not in MATERIAL_OF:
        raise ValueError(f"Unknown material '{material}'. Use 'inc', 'nh3' or 'gan'.")

    mu = jnp.broadcast_to(jnp.asarray(mu, dtype=jnp.float64), (x.shape[0], 2))

    field = temperature_field(MATERIAL_OF[material], space.create_variables())

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
            "GaN pieces" if N_RODS > 1 else "GaN block",
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

        values = np.asarray(history_dict[key], dtype=np.float64)
        values = values.reshape(values.shape[0], -1).sum(axis=1)

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

    # Loss groups: each network's own interior PDE terms, and the interface
    # conditions when they are part of the loss.
    inconel_pde_loss = summed_history(["inconel"])

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

    nh3_pde_loss = summed_history(["nh3"])

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

    interface_keys = [key for key in ("wall", "gan_surface") if key in history_dict]

    interface_curves = []

    if interface_keys:
        interface_curves = [
            {
                "values": summed_history(interface_keys),
                "label": "Interface conditions",
                "color": "tab:green",
            }
        ]

    save_loss_plot(
        title="Global PINN training-loss history",
        output_filename=output_name,
        curves=interface_curves
        + [
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


# Interface conditions


def check_interface_conditions(space, n_points=400, verbose=True, mu=None):
    """Temperature and flux jumps across the cavity wall and the GaN surface.

    Points closer than 2 % of a curve segment to a corner are left out: the
    normal is not defined there.
    """
    if mu is None:
        mu = case_mu()

    variables = space.create_variables()

    results = {}

    for label, curve, level_set, inside, outside in INTERFACES:
        n_segments = 3 * (N_RODS if label == "gan_surface" else 1)

        t = (np.arange(n_points) + 0.5) / n_points
        position = (t * n_segments) % 1.0
        t = jnp.asarray(t[(position > 0.02) & (position < 0.98)])

        jumps = interface_jumps(curve, level_set, inside, outside, variables)

        t_jump, q_jump, q_inside = jax.vmap(lambda t_one: jumps(space, t_one, mu))(t)

        t_jump = float(jnp.max(jnp.abs(t_jump)))
        q_jump = float(jnp.max(jnp.abs(q_jump)))
        q_scale = float(jnp.mean(jnp.abs(q_inside)))

        results[label] = (t_jump, q_jump)

        if verbose:
            print(
                f"Interface condition ({inside}/{outside}): max|T jump|={t_jump:.3e} K, "
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
            "max_relative_error_percent": float(100.0 * np.max(np.abs(error) / t_fvm)),
            "start_rmse_K": float(np.sqrt(np.mean((t0 + t1 * points[:, 1] - t_fvm) ** 2))),
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


def plot_profiles(space, df, radii=(0.0, 0.008, 0.011, 0.020)):
    """T - T_wall along z at a few radii: what the networks had to learn.

    The wall law itself is the zero line.
    """
    figure, axes = plt.subplots(1, len(radii), figsize=(3.6 * len(radii), 4.2), sharey=True)

    for ax, radius in zip(axes, radii):
        line = df[np.isclose(df["x"], radius)].sort_values("y")
        z = line["y"].to_numpy(dtype=np.float64)
        wall = T0_K + T1 * z
        t_pinn, _ = evaluate_pinn_on_points(space, np.column_stack([np.full_like(z, radius), z]))

        ax.plot(z * 1.0e3, line["Temperature"] - wall, "ko", ms=2.5, label="FVM")
        ax.plot(z * 1.0e3, t_pinn - wall, color="tab:red", lw=1.5, label="PINN")
        ax.axhline(0.0, color="0.4", ls="--", lw=1.0, label="wall law")
        ax.set_title(f"r = {radius * 1.0e3:.0f} mm")
        ax.set_xlabel("z [mm]")
        ax.grid(True, alpha=0.30)

    axes[0].set_ylabel("T - T_wall [K]")
    axes[0].legend(fontsize=8)

    figure.tight_layout()

    output_path = SAVE_DIR / "fvm_vs_pinn_profiles.png"

    figure.savefig(output_path, dpi=250, bbox_inches="tight")

    plt.close(figure)

    print(f"Saved wall-law deviation profiles in: {output_path.resolve()}")


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    space = build_space(seed=42)

    print("Number of neural models:", len(space.models))

    pinn = Projector(model, space, sampler, **ENG_OPTIONS)

    key = jax.random.PRNGKey(42)

    # The initial-loss report draws its OWN sample, so training starts from
    # PRNGKey(42) itself.
    _, samples = sampler.sample(jax.random.PRNGKey(0), N_COLLOC, {})

    print("Sample keys:", samples.keys())

    initial_losses = pinn.evaluate_losses(space, samples)

    print("Initial losses:", initial_losses)

    start = timeit.default_timer()

    key, pinn = pinn.project(key, space, N_EPOCHS, N_COLLOC, {}, tqdm_desc="two GaN pieces")

    elapsed = timeit.default_timer() - start

    plot_loss_history(pinn.losses, output_name="loss_history.png")

    print("Best loss:", pinn.best_loss)
    print(f"Training time: {elapsed:.2f} s")

    pinn.save("inconel_nh3_two_gan", postfix="parametric_minimal_K_unknown")

    print(
        "Saved trained projector under scriptname 'inconel_nh3_two_gan', "
        "'parametric_minimal_K_unknown'."
    )

    # FVM validation. FV reference with T_outer(z) imposed on the sidewall as
    # well; the matching (T0, T1) case is selected by read_fvm_table.
    PROJECT_DIR = Path(__file__).resolve().parents[1] / "data"
    FVM_FILE_2D = PROJECT_DIR / FVM_FOLDER / "2D_Heat_data.csv"

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
            f"  worst relative error {rows['max_relative_error_percent'].max():.3f} %"
        )

    # Full diagnostics for one case: the validated one by default; call
    # set_case(T0, T1) before this block for any other.
    set_case(*PLOT_CASE)

    plot_all_interior_residuals(pinn.space)

    check_interface_conditions(pinn.space)

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

    # 1D FVM/PINN line comparison. r = 4 mm runs through the GaN.
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

    plot_profiles(pinn.space, fvm_df_all_radii)
