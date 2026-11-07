r"""Transient heat conduction in an Inconel autoclave filled with NH3 (time-discrete PINN, Alexander scheme, fixed collocation points, potential as unknown).

This is ``../autoclave_inconel_nh3_transient/2D_heat_Inconel_+_Ammonia_transient_discrete_K_unknown_alexander.py``
with ``--no-resample`` as the default, on the full CrysMAS run of 0 to 7200 s
(``Autoclave_stable/Autoclave_stable.csv``): t_end = 7200 s, the end of the
heating ramps, so the 15 steps are 480 s long. Everything else is unchanged.

This is ``2D_heat_Inconel_+_Ammonia_transient_discrete_K_unknown.py`` with the
third-order Alexander scheme in place of implicit Euler. Alexander (1977) is a
3-stage, L-stable SDIRK method with error O(dt^3), so fewer time steps reach
a given time error, and fewer trainings accumulate their optimization errors:
15 steps of 480 s, 300 epochs for the initial condition and 100 epochs per
Runge-Kutta stage (3 stages, so 300 per step). Everything below the next
paragraph describes the formulation, which is unchanged.

The collocation points are drawn once per training (the initial fit and each
stage) instead of at every epoch; the option ``--resample`` draws them at
every epoch.

The base script is ``2D_heat_Inconel_+_Ammonia_transient_discrete.py`` with the Kirchhoff
potential in place of the temperature as the unknown, as
``2D_heat_Inconel_+_Ammonia_transient_global_K_unknown.py`` does for the
global PINN; everything else is the same.

Same problem and same formulation as ``2D_heat_Inconel_+_Ammonia_transient_global_K_unknown.py``:
axisymmetric (r, z) conduction with temperature-dependent material data,

.. math::

    \rho_m(T) c_{p,m}(T) \, \partial_t T = \nabla \cdot (k_m(T) \nabla T)
    \quad \text{in each material } m, \quad 0 < t < t_{end},

temperature and normal heat flux continuous across the cavity wall, symmetry
on the axis, the outer wall heated by the ramps of the stable case (rates
423 K / 7200 s at the bottom and 523 K / 7200 s at the top, side wall linear in
z) and the uniform initial field T = 300 K. The window ends with the ramps
and the CrysMAS run, at t_end = 7200 s. Here the time is discretized with the
``DiscretePINN`` of ScimBa-JAX: Alexander steps in the scaled time
tau = t / t_end, one PINN training per Runge-Kutta stage, each starting from
the networks of the previous one.

Taken from the global script:

* one MLP per material, with the inputs (r/R_m)^2, scaled z and the cavity
  level set;
* the networks build the Kirchhoff potential :math:`K_m(T) = \int k_m \, dT`
  of each material, around the potential of the side-wall law
  :math:`K_m(T_{lift}(\tau, z))`, with the correction scale
  :math:`\sigma_m = \Delta T \, k_m(T_c)`; the Inconel network is multiplied by
  a factor that vanishes on the outer wall, so the Dirichlet conditions hold
  exactly (K = K(T_Side), hence T = T_Side, since K is increasing);
* the temperature :math:`T_m = K_m^{-1}(K_m)` is recovered by six Newton
  steps started from the side-wall law; it enters the storage coefficient, the
  temperature jump and every evaluation;
* each material's equation applied inside that material only, and on the
  cavity wall the temperature jump and the normal-flux jump (soft interface),
  scaled as in the global script;
* the evaluation: comparison with CrysMAS at every node, error maps, line
  comparisons along every radial line of the CrysMAS nodes, probe histories.

What the time discretization changes:

* the networks take (r, z) only. The unknown of a step is the scaled
  departure of the potential from that of the lift at the step's time,
  :math:`w_m = (K_m - K_m(T_{lift}(\tau, z))) / \sigma_m`, which obeys

  .. math::

      \partial_\tau w_m + R_m(w) = f_m, \qquad
      R_m = -\frac{t_{end} \, k_m(T_m)}{\rho_m c_{p,m}(T_m) \, \sigma_m} \Delta K_m,
      \qquad f_m = -\frac{k_m(T_{lift}) \, \partial_\tau T_{lift}}{\sigma_m};

  the initial condition w = 0 is fitted before the first step;
* the DiscretePINN discretizes in time one variable and the residual of the
  main domain. The two networks therefore form one vector variable
  (w_inconel, w_nh3) and the main residual has one component per material.
  Each component is zero outside its material, in the variable as well as in
  R and f, so that a network is free outside its material, as with the masked
  residuals of the global script;
* the storage term of the solver is w itself, so the residual is the equation
  multiplied by k / (rho c_p); the axisymmetric Laplacian is written with K_r / r,
  replaced by K_rr on the axis;
* each network is zero outside its material, so the wall values of each side
  are taken a distance EPS_WALL from the cavity wall, on that side;
* the PINN exists at the step times; CrysMAS is interpolated linearly in time
  to them.
"""

import argparse
import contextlib
import json
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from scimba_jax.domains.domain_mapping import DomainMapping
from scimba_jax.domains.meshless_domains.domains_2d import Square2D
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
from scimba_jax.nonlinear_approximation.networks.scimba_module import ScimbaModule
from scimba_jax.nonlinear_approximation.numerical_solvers import (
    discrete_pinns as discrete_pinns_module,
)
from scimba_jax.nonlinear_approximation.numerical_solvers.discrete_pinns import (
    DiscretePINN,
)
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import InteriorResidual
from scimba_jax.time_discrete import (
    abstract_time_discrete_nonlinear_solver as time_discrete_solver_module,
)
from scimba_jax.time_discrete.butcher_tableau import build_alexander_tableau

jax.config.update("jax_enable_x64", True)


# Collocation points drawn once per training. ``Projector.project`` and
# ``project_scan`` take ``resample``, but ``DiscretePINN`` builds its projectors
# itself and does not pass it. Without changing the library, the solver is
# given, for the time of the training only, a projector that always uses
# ``resample=False``.
class FixedPointsProjector(Projector):
    """Projector that draws its collocation points once per training."""

    def project(self, *args, **kwargs):
        kwargs["resample"] = False
        return super().project(*args, **kwargs)

    def project_scan(self, *args, **kwargs):
        kwargs["resample"] = False
        return super().project_scan(*args, **kwargs)


@contextlib.contextmanager
def collocation_points(resample):
    """With ``resample=False``, the solver's trainings draw their points once."""
    modules = (discrete_pinns_module, time_discrete_solver_module)
    saved = [module.Projector for module in modules]
    if not resample:
        for module in modules:
            module.Projector = FixedPointsProjector
    try:
        yield
    finally:
        for module, projector in zip(modules, saved):
            module.Projector = projector


# Heating ramps of the stable case [K, K/s] and the simulated time window [s]:
# the end of the ramps and of the CrysMAS run.
T_START = 300.0
RATE_BOT = (723.0 - T_START) / 7200.0
RATE_TOP = (823.0 - T_START) / 7200.0
T_END = 7200.0
# Largest rise of the wall temperature in the window [K]: the scale of the case.
DELTA_T = max(RATE_BOT, RATE_TOP) * T_END

# CrysMAS transient run, saved every 20 s from t = 20 s to t = 7200 s.
PROJECT_DIR = Path(__file__).resolve().parents[1] / "data"
CRYSMAS_FILE = PROJECT_DIR / "Autoclave_stable" / "Autoclave_stable.csv"

# Geometry [m] (CryS-MAS Autoclave.pcs)
R_OUT, H_OUT = 0.035, 0.370
# NH3 cavity: a cylinder closed below by a spherical cap of the same radius.
R_NH3, Z_NH3_CENTER, Z_NH3_TOP = 0.012, 0.035, 0.335

# Material data as polynomials sum_i c_i T^i, T in K (Autoclave.mat)
K_COEFFS = {  # thermal conductivity [W/(m K)]
    "inconel": (6.9995, 0.0151),
    "nh3": (0.3039497375, -0.00019097, 0.0000001887),
}
RHO_COEFFS = {  # density [kg/m3]
    "inconel": (8260.0,),
    "nh3": (409.5645667,),
}
CP_COEFFS = {  # heat capacity [J/(kg K)]
    "inconel": (533.0,),
    "nh3": (8070.52380694, -26.30052033, 0.04090610, -0.00002052),
}
MATERIALS = tuple(K_COEFFS)

# Time stepping and training: Alexander steps of t_end / 15 = 480 s. With
# implicit Euler (first order) the start-up error stayed at 1 to 4 K for any
# affordable step; the third-order scheme lowers the time error with fewer
# steps, so fewer trainings add up their optimization errors. In the 1D
# radial check over 0 to 7200 s (axis lag at mid-height, against implicit
# Euler with dt = 1 s), 15 steps are 0.97 K off at the first step, 0.11 K
# at the second and within 0.02 K after that.
NT = 15
N_EPOCHS_INIT = 300  # fit of the initial condition w = 0
N_EPOCHS = 100  # per Runge-Kutta stage; 3 stages, so 300 per step
HIDDEN = [24, 24, 24]
# The main residual covers the whole box; about 28 % of its points fall in the
# NH3 cavity.
N_COLLOC = {"main": 16000, "wall": 1500}
ENG_OPTIONS = {"linesearch": "strong-wolfe", "adaptive_matrix_regularization": True}

# Scaled time tau = t / t_end
TIME_DOMAIN = (0.0, 1.0)
DIMS = {"x": 2}
# Distance [m] from the cavity wall at which each side's values are taken
EPS_WALL = 1.0e-7


# Materials
def polynomial(coeffs, T):
    """sum_i c_i T^i, without powers."""
    total, power_of_T = 0.0, 1.0
    for c in coeffs:
        total = total + c * power_of_T
        power_of_T = power_of_T * T
    return total


def conductivity(name, T):
    return polynomial(K_COEFFS[name], T)


def heat_capacity_per_volume(name, T):
    """rho(T) c_p(T) [J/(m3 K)]."""
    return polynomial(RHO_COEFFS[name], T) * polynomial(CP_COEFFS[name], T)


def kirchhoff(name, T):
    """K(T) = integral of k, so that div(k grad T) = Laplace(K(T))."""
    total, power_of_T = 0.0, T
    for i, c in enumerate(K_COEFFS[name]):
        total = total + c / (i + 1) * power_of_T
        power_of_T = power_of_T * T
    return total


# Reference values for the scaling of the interface residual, at 600 K.
K_REF = {name: float(conductivity(name, 600.0)) for name in MATERIALS}


# Boundary data
def temperature_bottom(tau):
    return T_START + RATE_BOT * T_END * tau


def temperature_top(tau):
    return T_START + RATE_TOP * T_END * tau


def temperature_lift(tau, x):
    """Side-wall law T_Side(t, z), extended to the whole domain."""
    z = x[..., 1:2]
    return temperature_bottom(tau) + z / H_OUT * (
        temperature_top(tau) - temperature_bottom(tau)
    )


def lift_rate(x):
    """d T_lift / d tau [K]: constant in time for linear ramps."""
    z = x[..., 1:2]
    return T_END * (RATE_BOT + z / H_OUT * (RATE_TOP - RATE_BOT))


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
    """NH3 cavity."""
    r, z = x[..., 0], x[..., 1]
    cap = (R_NH3**2 - r * r - (z - Z_NH3_CENTER) ** 2) / (2.0 * R_NH3)
    return r_or(cylinder(r, z, R_NH3, Z_NH3_CENTER, Z_NH3_TOP), cap)


def material_masks(x):
    cavity = level_cavity(x) > 0.0
    return {"inconel": ~cavity, "nh3": cavity}


def polyline(points):
    """Curve s in [0, 1] -> point, every segment getting the same share of s."""
    points = jnp.asarray(points, dtype=jnp.float64)
    n = points.shape[0] - 1

    def curve(s):
        s = jnp.clip(s * n, 0.0, n - 1.0e-9)
        i = jnp.floor(s).astype(int)
        return points[i] + (s - i)[..., None] * (points[i + 1] - points[i])

    return curve


def cavity_curve(s):
    """Cap, side and roof of the cavity wall, a third of s each."""
    s = 3.0 * s
    angle = 0.5 * jnp.pi * jnp.clip(s, 0.0, 1.0)
    cap = jnp.stack(
        [R_NH3 * jnp.sin(angle), Z_NH3_CENTER - R_NH3 * jnp.cos(angle)], axis=-1
    )
    side_and_roof = polyline(
        [(R_NH3, Z_NH3_CENTER), (R_NH3, Z_NH3_TOP), (0.0, Z_NH3_TOP)]
    )(jnp.clip(0.5 * (s - 1.0), 0.0, 1.0))
    return jnp.where((s < 1.0)[..., None], cap, side_and_roof)


def cavity_floor_z(r):
    """Height [m] of the cavity floor (spherical cap) at radius r <= R_NH3."""
    return Z_NH3_CENTER - np.sqrt(np.maximum(R_NH3**2 - r**2, 0.0))


def wall_factor(x):
    """Zero on the outer wall r = R_OUT, z = 0 and z = H_OUT, at most 1 inside."""
    r, z = x[..., 0:1], x[..., 1:2]
    return (1.0 - (r / R_OUT) ** 2) * 4.0 * (z / H_OUT) * (1.0 - z / H_OUT)


# Network inputs. The autoclave is ten times longer than wide: z is mapped to
# a wider interval than the other inputs so that the networks resolve the
# ends of the cavity.
Z_INPUT_RANGE = 2.0
N_INPUTS = 3


def material_inputs(x, radius):
    """(r/radius)^2, scaled z and the cavity level set."""
    r, z = x[..., 0:1], x[..., 1:2]
    return [
        (r / radius) ** 2,
        Z_INPUT_RANGE * (2.0 * z / H_OUT - 1.0),
        jnp.tanh(level_cavity(x)[..., None] / R_NH3),
    ]


def network_inputs(x):
    """Inputs of the Inconel network followed by those of the NH3 network."""
    return jnp.concatenate(
        material_inputs(x, R_OUT) + material_inputs(x, R_NH3), axis=-1
    )


class MaterialNetworks(ScimbaModule):
    """One MLP per material, each on its own inputs; output (N_inconel, N_nh3).

    The DiscretePINN applies its time discretization to the first variable of
    the space only, so the two networks of the global script are gathered
    here into one model, that is one vector variable.
    """

    def __init__(self, key):
        keys = jax.random.split(key, len(MATERIALS))
        self.networks = tuple(
            MLP(in_size=N_INPUTS, out_size=1, hidden_sizes=HIDDEN, key=k) for k in keys
        )

    def __call__(self, features):
        return jnp.concatenate(
            [
                network(features[..., N_INPUTS * i : N_INPUTS * (i + 1)])
                for i, network in enumerate(self.networks)
            ],
            axis=-1,
        )


# Trial functions in the potential: K_m = K_m(T_lift) + sigma_m * w_m, with
# w_inconel = wall factor * N_inconel so that the Dirichlet conditions hold
# exactly, and w_nh3 = N_nh3. sigma_m = delta_T * k_m(T_c) makes w_m = 1 about
# one delta_T, the same size of correction as in the temperature version. It
# is fixed by this argument, not tuned. Each component is zero outside its
# material.
T_C = T_START + 0.5 * DELTA_T  # middle of the temperature range of the window [K]


def potential_scale(name):
    """sigma_m = delta_T * k_m(T_c) [W/m]."""
    return DELTA_T * conductivity(name, T_C)


def invert_kirchhoff(name, potential, start):
    """T with K_m(T) = potential: six Newton steps from ``start``."""
    value = start
    for _ in range(6):
        value = value - (kirchhoff(name, value) - potential) / conductivity(name, value)
    return value


def departure(raw, x):
    masks = material_masks(x)
    w_inconel = jnp.where(
        masks["inconel"][..., None], wall_factor(x) * raw[..., 0:1], 0.0
    )
    w_nh3 = jnp.where(masks["nh3"][..., None], raw[..., 1:2], 0.0)
    return jnp.concatenate([w_inconel, w_nh3], axis=-1)


def build_space(seed=42):
    return ApproximationSpace(
        dict(DIMS),
        [(MaterialNetworks(jax.random.PRNGKey(seed)), "vec", len(MATERIALS))],
        model_type="x",
        pre_processing=[network_inputs],
        post_processing=[departure],
    )


def departure_field(name, variables):
    """Scaled potential departure w_m of material ``name`` (zero outside the material)."""
    v, i = variables[0], MATERIALS.index(name)
    return ParamScalarFunction(
        DIMS, lambda space, x: v(space, x)[..., i : i + 1], f_type="x"
    )


def potential_field(name, variables, tau):
    """Kirchhoff potential K_m = K_m(T_lift) + sigma_m w_m of material ``name`` at scaled time tau."""
    w = departure_field(name, variables)
    return ParamScalarFunction(
        DIMS,
        lambda space, x: (
            kirchhoff(name, temperature_lift(tau, x))
            + potential_scale(name) * w(space, x)
        ),
        f_type="x",
    )


def temperature_field(name, variables, tau):
    """Temperature: the potential inverted by Newton steps from the side-wall law."""
    K = potential_field(name, variables, tau)
    return ParamScalarFunction(
        DIMS,
        lambda space, x: invert_kirchhoff(name, K(space, x), temperature_lift(tau, x)),
        f_type="x",
    )


# Sampling domains: the stage equation on the whole box ("main"), the
# interface conditions on the cavity wall. The wall is a curve, sampled through
# a box whose second coordinate divided by H_OUT is the curve parameter. A
# subdomain is a hole of the main sampling, so this box is a thin strip along
# the outer wall.
main_domain = Square2D([(0.0, 1.0), (0.0, 1.0)], is_main_domain=True, label_str="main")
main_domain.set_mapping(
    DomainMapping.rectangle(
        origin=jnp.array([0.0, 0.0]),
        x_dir=jnp.array([R_OUT, 0.0]),
        y_dir=jnp.array([0.0, H_OUT]),
    ),
    bounds_postmap=[(0.0, R_OUT), (0.0, H_OUT)],
)
wall_box = Square2D([(0.999, 1.0), (0.0, 1.0)], is_main_domain=False, label_str="wall")
main_domain.add_subdomain(wall_box)
MAIN = main_domain.get_label()


def source(tau, x):
    """f_m = -d_tau K_m(T_lift) / sigma_m, in the component of the material of x."""
    masks = material_masks(x)
    T_lift = temperature_lift(tau, x)
    return jnp.concatenate(
        [
            jnp.where(
                masks[name][..., None],
                -conductivity(name, T_lift) * lift_rate(x) / potential_scale(name),
                0.0,
            )
            for name in MATERIALS
        ],
        axis=-1,
    )


class StageConduction(InteriorResidual):
    """R_m = -t_end k_m / (rho_m c_p,m sigma_m) * Laplace K_m in material m, zero outside.

    Axisymmetric Laplacian K_rr + K_r / r + K_zz, with K_r / r -> K_rr on the
    axis. The DiscretePINN turns it into the stage equation of stage i,
    w_i + a_ii dtau R(w_i) = w_previous - dtau sum_j a_ij (R(w_j) - f) + a_ii dtau f.
    """

    def __init__(self, domain, tau):
        super().__init__(
            domain=domain, size=len(MATERIALS), model_type="x", f_rhs=source
        )
        self.tau = tau

    def construct_residual(self, *vars):
        terms = []
        for name in MATERIALS:
            T = temperature_field(name, vars, self.tau)
            K = potential_field(name, vars, self.tau)
            K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
            K_rr, K_zz = K_r.partial_derivative("x", 0), K_z.partial_derivative("x", 1)
            terms.append((name, T, K_r, K_rr, K_zz))

        def residual(space, x):
            r = x[..., 0:1]
            masks = material_masks(x)
            values = []
            for name, T, K_r, K_rr, K_zz in terms:
                radial = jnp.where(
                    r > 0.0,
                    K_r(space, x) / jnp.where(r > 0.0, r, 1.0),
                    K_rr(space, x),
                )
                laplacian = K_rr(space, x) + radial + K_zz(space, x)
                T_x = T(space, x)
                diffusivity = conductivity(name, T_x) / heat_capacity_per_volume(
                    name, T_x
                )
                value = -T_END * diffusivity * laplacian / potential_scale(name)
                values.append(jnp.where(masks[name][..., None], value, 0.0))
            return jnp.concatenate(values, axis=-1)

        return ParamVecFunction(len(MATERIALS), DIMS, residual, f_type="x")


def interface_jumps(variables, tau, inside="nh3", outside="inconel"):
    """Temperature jump [K] and normal-flux jump [W/m2] on the cavity wall.

    The normal is the gradient of the level set of the cavity. Each side is
    evaluated EPS_WALL from the wall, on that side.
    """
    T_in, T_out = (
        temperature_field(inside, variables, tau),
        temperature_field(outside, variables, tau),
    )

    def flux(name):
        K = potential_field(name, variables, tau)
        K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
        return lambda sp, x: jnp.concatenate([K_r(sp, x), K_z(sp, x)], -1)

    flux_in, flux_out = flux(inside), flux(outside)

    def jumps(space, s):
        point = cavity_curve(s)
        normal = jax.grad(lambda p: jnp.sum(level_cavity(p)))(point)
        normal = normal / jnp.sqrt(jnp.sum(normal * normal) + 1.0e-30)
        point_in, point_out = point + EPS_WALL * normal, point - EPS_WALL * normal
        temperature_jump = T_in(space, point_in) - T_out(space, point_out)
        flux_jump = jnp.sum(
            (flux_in(space, point_in) - flux_out(space, point_out)) * normal,
            axis=-1,
            keepdims=True,
        )
        return temperature_jump, flux_jump

    return jumps


class Interface(InteriorResidual):
    """Temperature jump and normal-flux jump across the cavity wall, both zero.

    Not discretized in time by the solver: imposed on the networks of the
    step, at the step's time.
    """

    def __init__(self, domain, tau):
        super().__init__(domain=domain, size=2, model_type="x")
        self.tau = tau

    def construct_residual(self, *vars):
        jumps = interface_jumps(vars, self.tau)

        def residual(space, x):
            temperature_jump, flux_jump = jumps(space, x[..., 1] / H_OUT)
            flux_jump = flux_jump * R_NH3 / K_REF["inconel"]
            return jnp.concatenate([temperature_jump, flux_jump], axis=-1) / DELTA_T

        return ParamVecFunction(2, DIMS, residual, f_type="x")


class AutoclaveAtTime(AbstractPhysicalModel):
    def __init__(self, tau):
        super().__init__(main_domain=main_domain)
        self.physical_residuals = {
            MAIN: StageConduction(main_domain, tau),
            "wall": Interface(wall_box, tau),
        }
        self.data_residuals = {}


def pde_factory(tau):
    """The physical model at the (possibly traced) scaled time tau of a step."""
    return AutoclaveAtTime(tau)


def initial_departure(x):
    """Initial condition T = T_lift(0) = 300 K, that is w = 0."""
    return jnp.zeros(x.shape[:-1] + (len(MATERIALS),))


def build_solver(nt):
    # Boundary conditions are built into the trial functions: only interior
    # points are drawn.
    sampler = TensorizedSampler(
        [DomainSampler(main_domain)], model_type="x", bc=False, ic=False
    )
    return DiscretePINN(
        main_domain,
        TIME_DOMAIN,
        sampler,
        len(MATERIALS),
        nt,
        build_alexander_tableau(),
        implicit_pde=pde_factory,
        model_type="x",
    )


# Evaluation
def temperature_at(space, tau, points):
    """PINN temperature at points (n, 2) and scaled time tau, each point from its material's network."""
    points = jnp.asarray(points, dtype=jnp.float64)
    w = space.create_variables()[0].vmap_on_physical_variables()(space, points)
    T_lift = temperature_lift(tau, points)
    masks = material_masks(points)
    temperature = T_lift
    for i, name in enumerate(MATERIALS):
        potential = kirchhoff(name, T_lift) + potential_scale(name) * w[:, i : i + 1]
        temperature = jnp.where(
            masks[name][:, None],
            invert_kirchhoff(name, potential, T_lift),
            temperature,
        )
    return np.asarray(temperature[:, 0])


def space_at(spaces, tau):
    """Space of the step whose end time is nearest to tau (spaces[0] is tau = 0)."""
    n_steps = len(spaces) - 1
    return spaces[int(round(float(tau) * n_steps))]


def draw_geometry(ax):
    angle = np.linspace(0.0, 0.5 * np.pi, 91)
    ax.plot(
        R_NH3 * np.sin(angle) * 1e3,
        (Z_NH3_CENTER - R_NH3 * np.cos(angle)) * 1e3,
        "k",
        lw=0.8,
    )
    ax.plot(
        [R_NH3 * 1e3, R_NH3 * 1e3, 0.0],
        [Z_NH3_CENTER * 1e3, Z_NH3_TOP * 1e3, Z_NH3_TOP * 1e3],
        "k",
        lw=0.8,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("r [mm]")


def plot_snapshots(
    spaces, save_dir, taus=(0.0, 0.25, 0.5, 0.75, 1.0), n_r=120, n_z=1000
):
    """Temperature and deviation from the side-wall law at the steps nearest to taus."""
    r = np.linspace(0.0, R_OUT, n_r)
    z = np.linspace(0.0, H_OUT, n_z)
    rr, zz = np.meshgrid(r, z)
    points = np.column_stack([rr.ravel(), zz.ravel()])
    n_steps = len(spaces) - 1

    figure, axes = plt.subplots(2, len(taus), figsize=(2.6 * len(taus), 12.0))
    for j, tau in enumerate(taus):
        tau = round(tau * n_steps) / n_steps
        temperature = temperature_at(space_at(spaces, tau), tau, points).reshape(
            rr.shape
        )
        deviation = temperature - np.asarray(temperature_lift(tau, points)).reshape(
            rr.shape
        )
        for ax, field, label in (
            (axes[0, j], temperature, "T [K]"),
            (axes[1, j], deviation, "T - T_Side [K]"),
        ):
            # A uniform field (t = 0) gets a fixed 1 K range instead of rounding noise.
            levels = (
                50
                if np.ptp(field) > 1.0e-6
                else np.linspace(-0.5, 0.5, 3) + field.mean()
            )
            contour = ax.contourf(
                rr * 1e3, zz * 1e3, field, levels=levels, cmap="turbo"
            )
            figure.colorbar(contour, ax=ax, label=label, shrink=0.8)
            draw_geometry(ax)
        axes[0, j].set_title(f"t = {tau * T_END:.0f} s")
    axes[0, 0].set_ylabel("z [mm]")
    axes[1, 0].set_ylabel("z [mm]")
    figure.suptitle("Stable heating case, time-discrete PINN")
    figure.tight_layout()
    path = save_dir / "snapshots.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path.resolve()}")


PROBES = {  # (r, z) [m] at mid-height of the cavity
    "NH3, axis": (0.0, 0.5 * (Z_NH3_CENTER + Z_NH3_TOP)),
    "cavity wall": (R_NH3, 0.5 * (Z_NH3_CENTER + Z_NH3_TOP)),
    "Inconel, mid-wall": (
        0.5 * (R_NH3 + R_OUT),
        0.5 * (Z_NH3_CENTER + Z_NH3_TOP),
    ),
}


def plot_probe_histories(spaces, save_dir, reference=None):
    """Temperature against time at the probes, and the lag behind the outer wall.

    The PINN values are those of the steps (markers on the line). With a
    CrysMAS ``reference`` (see read_crysmas), its values interpolated to the
    probes are drawn as markers of the same colour.
    """
    taus = np.arange(len(spaces)) / (len(spaces) - 1)
    figure, (ax_T, ax_lag) = plt.subplots(1, 2, figsize=(12.0, 4.5))
    if reference is not None:
        times, points, T_ref = reference
        probe_points = np.array(list(PROBES.values()))
        T_ref_probes = interpolate_crysmas(points, T_ref, probe_points)
    table = {"t_s": taus * T_END}
    results = {}
    for k, (label, point) in enumerate(PROBES.items()):
        point = np.array([point])
        history = np.array(
            [temperature_at(space, tau, point)[0] for space, tau in zip(spaces, taus)]
        )
        wall = np.array([float(temperature_lift(tau, point)[0, 0]) for tau in taus])
        table[f"T {label}"], table[f"lag {label}"] = history, history - wall
        results[label] = {
            "T_end_K": float(history[-1]),
            "lag_end_K": float(history[-1] - wall[-1]),
        }
        color = f"C{k}"
        style = {"color": color, "marker": ".", "ms": 4}
        ax_T.plot(taus * T_END, history, label=f"{label}, PINN", **style)
        ax_lag.plot(taus * T_END, history - wall, label=f"{label}, PINN", **style)
        print(
            f"{label:>18}: T(t_end) = {history[-1]:.2f} K, "
            f"T - T_Side = {history[-1] - wall[-1]:+.2f} K"
        )
        if reference is not None:
            wall_ref = np.asarray(temperature_lift(times[:, None] / T_END, point))[:, 0]
            style = {
                "color": color,
                "ls": "none",
                "marker": "o",
                "ms": 3,
                "markevery": 4,
                "mfc": "none",
            }
            ax_T.plot(times, T_ref_probes[k], label=f"{label}, CrysMAS", **style)
            ax_lag.plot(
                times, T_ref_probes[k] - wall_ref, label=f"{label}, CrysMAS", **style
            )
            results[label] |= {
                "crysmas_end_K": float(T_ref_probes[k, -1]),
                "difference_end_K": float(history[-1] - T_ref_probes[k, -1]),
            }
    pd.DataFrame(table).to_csv(save_dir / "probe_histories.csv", index=False)
    ax_T.plot(
        taus * T_END,
        [float(temperature_lift(tau, point)[0, 0]) for tau in taus],
        "k--",
        label="outer wall, same z",
    )
    ax_T.set_ylabel("T [K]")
    ax_lag.set_ylabel("T - T_Side [K]")
    for ax in (ax_T, ax_lag):
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7)
    figure.tight_layout()
    path = save_dir / "probe_histories.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path.resolve()}")
    return results


# Comparison with CrysMAS
def read_crysmas(path):
    """CrysMAS run: times (n_t,) [s], nodes (n_nodes, 2) as (r, z) [m], T (n_t, n_nodes) [K].

    The csv file (columns x = r, y = z, Temperature, time) stacks one block of
    nodes per saved time, with the same nodes in the same order in every block.
    """
    data = pd.read_csv(path).sort_values("time", kind="stable")
    times = np.unique(data["time"].to_numpy(dtype=np.float64))
    n_nodes = len(data) // len(times)
    nodes = data[["x", "y"]].to_numpy(dtype=np.float64).reshape(len(times), n_nodes, 2)
    if len(data) != len(times) * n_nodes or not np.allclose(nodes, nodes[0]):
        raise ValueError(f"The nodes of {path} differ between the saved times.")
    temperature = (
        data["Temperature"].to_numpy(dtype=np.float64).reshape(len(times), n_nodes)
    )
    return times, nodes[0], temperature


def interpolate_crysmas(nodes, temperature, points):
    """CrysMAS temperature at arbitrary points (n, 2), linear in the node triangulation.

    Returns shape (n, n_t). A point outside the triangulation (rounding on the
    border of the domain) takes the value of the nearest node.
    """
    values = LinearNDInterpolator(nodes, temperature.T)(points)
    outside = np.isnan(values[:, 0])
    if np.any(outside):
        values[outside] = NearestNDInterpolator(nodes, temperature.T)(points[outside])
    return values


def crysmas_at_times(reference, new_times):
    """The CrysMAS reference at new_times, linear in time between its saved times."""
    times, nodes, T_ref = reference
    i = np.clip(np.searchsorted(times, new_times) - 1, 0, len(times) - 2)
    w = np.clip((new_times - times[i]) / (times[i + 1] - times[i]), 0.0, 1.0)
    return new_times, nodes, (1.0 - w)[:, None] * T_ref[i] + w[:, None] * T_ref[i + 1]


def compare_with_crysmas(T_pinn, save_dir, reference):
    """PINN minus CrysMAS at every node and step time; error statistics per time and material.

    ``reference`` holds CrysMAS at the step times (see crysmas_at_times). Every
    node counts once (the nodes are not equally spaced, so this is a node
    average and not a volume average). The material of a node follows the
    level set of the PINN geometry.
    """
    times, nodes, T_ref = reference
    masks = {name: np.asarray(mask) for name, mask in material_masks(nodes).items()}
    error = T_pinn - T_ref

    def rms(values):
        return float(np.sqrt(np.mean(values**2)))

    rows = []
    for i, t in enumerate(times):
        row = {
            "time_s": t,
            "rmse_K": rms(error[i]),
            "max_abs_error_K": float(np.max(np.abs(error[i]))),
            "mean_error_K": float(np.mean(error[i])),
            "max_relative_error_percent": float(
                np.max(np.abs(error[i]) / T_ref[i]) * 100.0
            ),
        }
        for name, mask in masks.items():
            row[f"rmse_{name}_K"] = rms(error[i, mask])
            row[f"max_abs_error_{name}_K"] = float(np.max(np.abs(error[i, mask])))
        rows.append(row)
    summary = pd.DataFrame(rows)
    path = save_dir / "crysmas_comparison_per_time.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path.resolve()}")

    i_worst, j_worst = np.unravel_index(np.argmax(np.abs(error)), error.shape)
    print(
        f"CrysMAS comparison over {len(times)} step times "
        f"(t = {times[0]:.1f} to {times[-1]:.1f} s), {len(nodes)} nodes:"
    )
    print(f"  RMSE over all nodes and times: {rms(error):.4f} K")
    for name, mask in masks.items():
        print(
            f"  RMSE {name:>7}: {rms(error[:, mask]):.4f} K, "
            f"max |error| {np.max(np.abs(error[:, mask])):.4f} K"
        )
    print(
        f"  max |error| {abs(error[i_worst, j_worst]):.4f} K at t = {times[i_worst]:.1f} s, "
        f"r = {nodes[j_worst, 0] * 1e3:.1f} mm, z = {nodes[j_worst, 1] * 1e3:.1f} mm"
    )
    print(f"  at t_end = {times[-1]:.1f} s: RMSE {summary['rmse_K'].iloc[-1]:.4f} K")
    metrics = {
        "rmse_K": rms(error),
        "max_abs_error_K": float(abs(error[i_worst, j_worst])),
        "max_abs_error_time_s": float(times[i_worst]),
        "max_abs_error_r_mm": float(nodes[j_worst, 0] * 1e3),
        "max_abs_error_z_mm": float(nodes[j_worst, 1] * 1e3),
        "rmse_last_time_K": float(summary["rmse_K"].iloc[-1]),
        "max_relative_error_percent": float(
            summary["max_relative_error_percent"].max()
        ),
    }
    for name, mask in masks.items():
        metrics[f"rmse_{name}_K"] = rms(error[:, mask])
        metrics[f"max_abs_error_{name}_K"] = float(np.max(np.abs(error[:, mask])))
    return summary, metrics


def plot_crysmas_error_history(summary, save_dir):
    """RMSE and largest error against CrysMAS at every step time."""
    figure, (ax_rmse, ax_max) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    t = summary["time_s"]
    for column, label in (("", "all nodes"), ("_inconel", "Inconel"), ("_nh3", "NH3")):
        ax_rmse.plot(t, summary[f"rmse{column}_K"], ".-", label=label)
        ax_max.plot(t, summary[f"max_abs_error{column}_K"], ".-", label=label)
    ax_rmse.set_ylabel("RMSE, PINN vs CrysMAS [K]")
    ax_max.set_ylabel("max |PINN - CrysMAS| [K]")
    for ax in (ax_rmse, ax_max):
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.4)
        ax.legend()
    figure.tight_layout()
    path = save_dir / "crysmas_error_history.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path.resolve()}")


def plot_crysmas_maps(
    save_dir, reference, T_pinn, target_times=(1800.0, 3600.0, np.inf)
):
    """CrysMAS field, PINN field and their difference at the step times nearest to target_times."""
    times, nodes, T_ref = reference
    triangulation = Triangulation(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3)
    for target in target_times:
        i = int(np.argmin(np.abs(times - min(target, times[-1]))))
        error = T_pinn[i] - T_ref[i]
        bound = max(float(np.max(np.abs(error))), 1.0e-6)
        # One colour scale for both temperature fields, so that they can be compared by eye.
        T_levels = np.linspace(
            min(T_ref[i].min(), T_pinn[i].min()),
            max(T_ref[i].max(), T_pinn[i].max()),
            51,
        )
        figure, axes = plt.subplots(1, 3, figsize=(8.5, 11.0))
        for ax, field, title, options in (
            (axes[0], T_ref[i], "CrysMAS", {"levels": T_levels, "cmap": "turbo"}),
            (axes[1], T_pinn[i], "PINN", {"levels": T_levels, "cmap": "turbo"}),
            (
                axes[2],
                error,
                "PINN - CrysMAS",
                {"levels": np.linspace(-bound, bound, 41), "cmap": "RdBu_r"},
            ),
        ):
            contour = ax.tricontourf(triangulation, field, **options)
            figure.colorbar(contour, ax=ax, label="K", shrink=0.7)
            draw_geometry(ax)
            ax.set_title(title)
        axes[0].set_ylabel("z [mm]")
        figure.suptitle(
            f"Stable heating case, time-discrete PINN, t = {times[i]:.1f} s"
        )
        figure.tight_layout()
        path = save_dir / f"crysmas_comparison_t{times[i]:.0f}s.png"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {path.resolve()}")


def plot_line_comparisons(
    save_dir, reference, T_pinn, target_times=(1800.0, 3600.0, np.inf), label="PINN"
):
    """PINN and CrysMAS along every radial line of the CrysMAS nodes.

    For each step time nearest to target_times, two images with one panel per
    radius r = 0 to 35 mm: the temperature against z, and the relative error
    (PINN - CrysMAS) / CrysMAS [%]. The cavity floor and roof are marked on the
    lines that cross the cavity. The errors per radius and time go to
    line_comparison_per_radius.csv.
    """
    times, nodes, T_ref = reference
    radii = np.unique(np.round(nodes[:, 0], 6))
    n_cols = 6
    n_rows = int(np.ceil(len(radii) / n_cols))
    rows = []
    for target in target_times:
        i = int(np.argmin(np.abs(times - min(target, times[-1]))))
        for kind in ("temperature", "relative_error"):
            figure, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(3.0 * n_cols, 2.6 * n_rows),
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes).ravel()
            for k, r in enumerate(radii):
                line = np.flatnonzero(np.isclose(nodes[:, 0], r))
                line = line[np.argsort(nodes[line, 1])]
                z_mm = nodes[line, 1] * 1e3
                ref, pinn = T_ref[i, line], T_pinn[i, line]
                error = pinn - ref
                relative = 100.0 * error / ref
                ax = axes[k]
                if kind == "temperature":
                    rmse = float(np.sqrt(np.mean(error**2)))
                    ax.plot(z_mm, ref, "o", ms=2.5, color="black", label="CrysMAS")
                    ax.plot(z_mm, pinn, "-", lw=1.5, color="tab:red", label=label)
                    ax.set_title(
                        f"r = {r * 1e3:.0f} mm, RMSE = {rmse:.3f} K", fontsize=8
                    )
                    rows.append(
                        {
                            "time_s": times[i],
                            "r_mm": r * 1e3,
                            "n_points": len(line),
                            "rmse_K": rmse,
                            "max_abs_error_K": float(np.max(np.abs(error))),
                            "max_relative_error_percent": float(
                                np.max(np.abs(relative))
                            ),
                        }
                    )
                else:
                    ax.plot(z_mm, relative, "-", lw=1.2, color="tab:blue")
                    ax.axhline(0.0, color="0.5", lw=0.6)
                    ax.set_title(
                        f"r = {r * 1e3:.0f} mm, max {np.max(np.abs(relative)):.3f} %",
                        fontsize=8,
                    )
                if r <= R_NH3:
                    for z_interface in (cavity_floor_z(r), Z_NH3_TOP):
                        ax.axvline(
                            z_interface * 1e3, color="0.5", lw=0.8, ls="--", alpha=0.7
                        )
                ax.tick_params(labelsize=7)
            for ax in axes[len(radii) :]:
                ax.axis("off")
            if kind == "temperature":
                axes[0].legend(fontsize=7, loc="best")
                title, name = "temperature [K] against z [mm]", "line_comparison"
            else:
                title = "relative error (PINN - CrysMAS) / CrysMAS [%] against z [mm]"
                name = "line_relative_error"
            figure.suptitle(
                f"{label} and CrysMAS along all {len(radii)} radial lines, t = {times[i]:.1f} s: {title}",
                fontsize=12,
            )
            path = save_dir / f"{name}_t{times[i]:.0f}s.png"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {path.resolve()}")
    pd.DataFrame(rows).to_csv(save_dir / "line_comparison_per_radius.csv", index=False)


def plot_loss_history(losses, n_epochs_init, n_steps, save_dir):
    """Total loss over all epochs (initial fit, then every step), and the best loss of each step."""
    history = np.asarray(losses.losses_history["total"]).ravel()
    per_step = history[n_epochs_init:].reshape(n_steps, -1)
    step_times = T_END * (np.arange(n_steps) + 1) / n_steps
    pd.DataFrame({"epoch": np.arange(len(history)), "total": history}).to_csv(
        save_dir / "loss_history.csv", index=False
    )
    figure, (ax_all, ax_step) = plt.subplots(1, 2, figsize=(12.0, 4.0))
    ax_all.semilogy(history)
    ax_all.axvline(n_epochs_init, color="0.5", lw=0.8, ls="--", label="first step")
    ax_all.set_xlabel("Epoch (initial fit, then the steps one after the other)")
    ax_all.set_ylabel("Total loss")
    ax_all.legend()
    ax_step.semilogy(step_times, per_step.min(axis=1), "o-", ms=3)
    ax_step.set_xlabel("t at the end of the step [s]")
    ax_step.set_ylabel("Best total loss of the step")
    for ax in (ax_all, ax_step):
        ax.grid(True, which="both", alpha=0.4)
    figure.tight_layout()
    path = save_dir / "loss_history.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nt", type=int, default=NT, help="number of time steps")
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS, help="ENG epochs per Runge-Kutta stage"
    )
    parser.add_argument(
        "--epochs-init",
        type=int,
        default=N_EPOCHS_INIT,
        help="ENG epochs of the initial-condition fit",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resample",
        action="store_true",
        help="draw the collocation points at every epoch instead of once per training",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="subfolder of output_inc_nh3_transient_7200s "
        "(default discrete_K_unknown_alexander[_fixed_points])",
    )
    args = parser.parse_args()
    resample = args.resample
    if args.out is None:
        args.out = "discrete_K_unknown_alexander" + (
            "" if resample else "_fixed_points"
        )

    save_dir = Path("output_inc_nh3_transient_7200s") / args.out
    save_dir.mkdir(parents=True, exist_ok=True)

    discrete_pinn = build_solver(args.nt)
    key = jax.random.PRNGKey(args.seed)

    start = timeit.default_timer()
    print("Initializing the discrete PINN...")
    with collocation_points(resample):
        key, discrete_pinn = discrete_pinn.initialize(
            key,
            build_space(args.seed),
            initial_departure,
            args.epochs_init,
            N_COLLOC,
            **ENG_OPTIONS,
        )
    space_init = discrete_pinn.space
    print("Initializing the discrete PINN... Done\n")

    # The progress bar (jax-tqdm) ticks once per time step.
    print(
        f"\nSolving with the Alexander method ({args.nt} steps, 3 stages, "
        f"{args.epochs} epochs per stage, resampling: {resample})..."
    )
    with collocation_points(resample):
        key, discrete_pinn = discrete_pinn.solve(
            key,
            space_init,
            args.epochs,
            N_COLLOC,
            keep_space_history=True,
            **ENG_OPTIONS,
        )
        # JAX runs asynchronously: wait for the last step before reading the clock.
        jax.block_until_ready(discrete_pinn.space_history)
    training_time = timeit.default_timer() - start
    print("\nSolving with the Alexander method... Done\n")
    print(f"Training time: {training_time:.1f} s")

    # The space at tau = 0 and after every step
    spaces = [space_init] + [
        jax.tree_util.tree_map(lambda leaf, n=n: leaf[n], discrete_pinn.space_history)
        for n in range(args.nt)
    ]
    taus = np.arange(args.nt + 1) / args.nt

    plot_loss_history(discrete_pinn.losses, args.epochs_init, args.nt, save_dir)
    plot_snapshots(spaces, save_dir)

    run_summary = {
        "case": "stable",
        "unknown": "Kirchhoff potential",
        "t_end_s": T_END,
        "time_scheme": "Alexander (3rd order, 3-stage SDIRK)",
        "nt": args.nt,
        "dt_s": T_END / args.nt,
        "seed": args.seed,
        "resample_every_epoch": resample,
        "epochs_init": args.epochs_init,
        "epochs_per_stage": args.epochs,
        "epochs_per_step": 3 * args.epochs,
        "n_parameters": int(
            sum(leaf.size for leaf in jax.tree_util.tree_leaves(space_init.models))
        ),
        "training_time_s": training_time,
    }
    reference = None
    if not CRYSMAS_FILE.exists():
        print(f"CrysMAS file not found: {CRYSMAS_FILE}: comparison skipped.")
    else:
        reference = read_crysmas(CRYSMAS_FILE)
        reference_steps = crysmas_at_times(reference, taus[1:] * T_END)
        nodes = reference_steps[1]
        T_pinn = np.stack(
            [temperature_at(spaces[n], taus[n], nodes) for n in range(1, args.nt + 1)]
        )
        summary, run_summary["crysmas"] = compare_with_crysmas(
            T_pinn, save_dir, reference_steps
        )
        plot_crysmas_error_history(summary, save_dir)
        plot_crysmas_maps(save_dir, reference_steps, T_pinn)
        plot_line_comparisons(save_dir, reference_steps, T_pinn)
        # The field on the CrysMAS nodes at the step times, to compare runs later.
        np.savez_compressed(
            save_dir / "field_on_reference_grid.npz",
            times=reference_steps[0],
            nodes=nodes,
            T=T_pinn,
        )
    run_summary["probes"] = plot_probe_histories(spaces, save_dir, reference)

    (save_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(f"Saved {(save_dir / 'run_summary.json').resolve()}")
