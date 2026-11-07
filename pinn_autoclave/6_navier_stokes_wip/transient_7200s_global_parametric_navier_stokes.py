r"""Transient heating of the Inconel/NH3 autoclave with buoyancy-driven NH3 flow, parametric global PINN, potential as unknown.

This is ``2D_heat_Inconel_+_Ammonia_transient_7200s_global_parametric_K_unknown.py``
with the Boussinesq Navier-Stokes equations in the NH3 cavity, in the
axisymmetric stream-function/vorticity form of the formulation note
``nh3_flow_formulation_v2`` (Sections 6.2, 6.4, 6.5). The conduction model,
the Inconel, the interface conditions, the sampling, the optimizer and the
evaluation are unchanged.

Flow model. With s = r^2, the Stokes stream function psi = r^2 psi~ and the
azimuthal vorticity omega = r omega~ (omega = d_z u_r - d_r u_z):

.. math::

    u_r = -r \, \partial_z \tilde\psi, \qquad
    u_z = 2 \tilde\psi + r \, \partial_r \tilde\psi, \qquad
    \Delta_5 f = f_{rr} + \frac{3}{r} f_r + f_{zz},

    R_\psi = \Delta_5 \tilde\psi + \tilde\omega, \qquad
    R_\omega = \partial_t \tilde\omega + \mathbf{u} \cdot \nabla \tilde\omega
    - \nu \Delta_5 \tilde\omega + g_{eff} \beta \, \frac{\partial_r T}{r},

and the NH3 energy equation gets the advection term,
:math:`\frac{\rho c_p}{k}(\partial_t K + \mathbf{u} \cdot \nabla K) = \Delta K`
(:math:`\nabla K = k \nabla T`). Continuity holds exactly by construction;
f_r / r is replaced by f_rr on the axis, where f_r = 0. These are the note's
Eqs. (6.10) and (6.11) in (r, z) and in dimensional form.

Hard constraints (note, Eq. (6.15)):
:math:`\tilde\psi = S_\psi(\mu) \, \tau \, D^2 N_\psi` and
:math:`\tilde\omega = S_\omega(\mu) \, \tau \, N_\omega`, with D twice the
cavity level set over R_NH3 (zero on the cavity wall, about 1 inside), so
that u = 0 on the wall (no slip, no penetration) and u = 0 at t = 0 (fluid at
rest in the uniform initial field). The networks see (r/R_NH3)^2, so psi~ and
omega~ are even in r and u_r = 0 on the axis.

Properties and Rayleigh number. Conduction keeps the CrysMAS data of the
heat-conduction scripts, so that the comparison with CrysMAS stays the same.
The flow needs nu and beta, which the CrysMAS material file does not give:
the values of Schimmel et al. (2023), Table 4, as in the note. The physical
Rayleigh number is far above the steady laminar range (note, P2), so gravity
is reduced as in the note's "reduced-gravity reading": g_eff beta is chosen so
that Ra = g_eff beta dT R^3 / (nu kappa) = RA with the note's scales
(dT = 238.02 K, kappa = 1.645e-7 m2/s, R = 12 mm). RA = 2e4 is the note's
axisymmetric value; ``--ra`` changes it, and ``--ra 0`` switches buoyancy off
(no flow, the conduction model).

Scales: the driving temperature difference is the radial lag of the cavity
behind the wall, LAG_FRACTION * s(mu) (13.1 K for the stable case at t_end in
CrysMAS); U_s = g_eff beta dT_lag R^2 / (16 nu) is a Stokes estimate,
S_psi = U_s and S_omega = U_s / R^2. They fix the size of the network outputs
and of the residuals and are not tuned.

The CrysMAS runs are heat conduction only. The difference between this model
and CrysMAS is therefore the effect of the flow plus the PINN error; the
conduction-only parametric run gives the PINN error alone (0.03 to 0.09 K).

Original description of the conduction model:

Same problem and formulation as ``2D_heat_Inconel_+_Ammonia_transient_global.py``
(the "single-case script"), with the end temperatures of the heating ramps as
parameters:

.. math::

    \mu = (T_{Bot,e}, T_{Top,e}) \in [600, 1000]^2 \ \mathrm{K}, \qquad
    T_{Bot}(t) = 300 + (T_{Bot,e} - 300) \, \frac{t}{t_{ramp}}, \qquad
    T_{Top}(t) = 300 + (T_{Top,e} - 300) \, \frac{t}{t_{ramp}},

with the start temperature (300 K) and the ramp duration of the formulation
(t_ramp = 7200 s) fixed. The stable case of the formulation is
mu = (723, 823) K, the unstable case mu = (950, 620) K. The model covers the
time window of the CrysMAS runs, 0 <= t <= t_end = 7200 s. One training covers
every case of the box: each collocation point draws its own mu.

* the networks take (tau, r, z, mu) as input, tau = t / t_end and mu scaled to
  [-1, 1];
* the networks build the Kirchhoff potential :math:`K_m(T) = \int k_m \, dT`
  of each material. This is ``2D_heat_Inconel_+_Ammonia_transient_global_parametric.py``
  with the potential in place of the temperature as the unknown, as
  ``2D_heat_Inconel_+_Ammonia_transient_global_K_unknown.py`` does for the
  single case; network inputs, sampling, loss terms, weights, optimizer and
  evaluation are the same;
* trial functions
  :math:`K_m = K_m(T_{lift}(\tau, z; \mu)) + \sigma_m(\mu) \, \tau \, N_m`,
  with the Inconel network times the outer-wall factor (K = K(T_Side), hence
  T = T_Side, on the outer wall). :math:`s(\mu)` is the root mean square of
  the two temperature rises at t_end, a smooth scale of the case, and
  :math:`\sigma_m = s(\mu) \, k_m(T_c)` with :math:`T_c = 300 + s(\mu)/2`,
  so that N_m = 1 is about s(mu) kelvin, as in the temperature version;
* the temperature :math:`T_m = K_m^{-1}(K_m)` is recovered by six Newton
  steps started from the side-wall law; it enters the storage term, the
  temperature jump and every evaluation;
* the equation reads
  :math:`\frac{\rho_m c_{p,m}}{k_m}(T_m) \, \partial_t K_m = \Delta K_m`,
  since :math:`\partial_t K_m = k_m(T) \, \partial_t T`; residuals and
  interface conditions are scaled with :math:`s(\mu)` as in the temperature
  version.

The script is standalone: geometry, material laws, level sets, domains and the
CrysMAS readers are written out here, identical to the single-case script.

After training, the model is evaluated for the stable and unstable cases
against the CrysMAS runs (fields, probes and every radial line of the CrysMAS nodes), and
over the whole parameter box through the lag of the NH3 on the axis at t_end.

Usage (from the repository root)::

    python ".../2D_heat_Inconel_+_Ammonia_transient_7200s_global_parametric_K_unknown_navier_stokes.py" --seed 42 --out parametric_K_unknown_navier_stokes_seed42
"""

import argparse
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
from scimba_jax.nonlinear_approximation.integration.monte_carlo_parameters import (
    UniformParametricSampler,
)
from scimba_jax.nonlinear_approximation.integration.monte_carlo_time import (
    UniformTimeSampler,
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

# Heating ramps: fixed start temperature and ramp duration, end temperatures as
# parameters. Simulated time window [s]: the end of the ramps and of the
# CrysMAS runs.
T_START, T_RAMP = 300.0, 7200.0
T_END = 7200.0
MU_BOUNDS = [(600.0, 1000.0), (600.0, 1000.0)]  # (T_Bot,e, T_Top,e) [K]
STABLE_MU = (723.0, 823.0)
UNSTABLE_MU = (950.0, 620.0)

# CrysMAS transient runs of the two cases, saved every 20 s from t = 20 s to
# t_end: case name -> (mu, csv file).
PROJECT_DIR = Path(__file__).resolve().parents[1] / "data"
CASES = {
    "stable": (STABLE_MU, PROJECT_DIR / "Autoclave_stable" / "Autoclave_stable.csv"),
    "unstable": (
        UNSTABLE_MU,
        PROJECT_DIR / "Autoclave_unstable" / "Autoclave_unstable.csv",
    ),
}

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

# NH3 flow (Boussinesq). nu and beta: Schimmel et al. (2023), Table 4.
NU_NH3 = 1.570e-7  # kinematic viscosity [m2/s]
BETA_NH3 = 2.631e-3  # thermal expansion coefficient [1/K]
# Reduced gravity: Ra = g_eff beta dT R^3 / (nu kappa) with the scales of the
# formulation note.
RA = 2.0e4
KAPPA_NOTE, DT_NOTE = 1.645e-7, 238.02  # [m2/s], [K]


def buoyancy_coefficient(ra):
    """g_eff beta [m/(s2 K)] giving the Rayleigh number ra."""
    return ra * NU_NH3 * KAPPA_NOTE / (DT_NOTE * R_NH3**3)


G_BETA = buoyancy_coefficient(RA)  # set from --ra in __main__
# The scales stay those of RA = 2e4, so that --ra 0 keeps them finite.
G_BETA_SCALE = buoyancy_coefficient(2.0e4)
# Radial lag of the cavity behind the wall over s(mu): CrysMAS, stable case,
# axis at mid-height, t_end: 13.1 K / 475.6 K.
LAG_FRACTION = 0.0275

# Training: the budget of the single-case script.
N_EPOCHS = 300
HIDDEN = [24, 24, 24]
N_COLLOC = {"inconel": 8000, "nh3": 8000, "wall": 1500}
ENG_OPTIONS = {"linesearch": "strong-wolfe", "adaptive_matrix_regularization": True}

TIME_DOMAIN = (0.0, 1.0)  # scaled time tau = t / t_end
DIMS = {"x": 2, "t": 1, "mu": 2}
MU_CENTER = jnp.array([0.5 * (lo + hi) for lo, hi in MU_BOUNDS])
MU_HALF_WIDTH = jnp.array([0.5 * (hi - lo) for lo, hi in MU_BOUNDS])
OUTPUT_ROOT = Path("output_inc_nh3_transient_7200s")


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


# Reference values for the scaling of the residuals, at 600 K.
K_REF = {name: float(conductivity(name, 600.0)) for name in MATERIALS}


# Boundary data of case mu
def temperature_bottom(tau, mu):
    return T_START + (mu[..., 0:1] - T_START) * (T_END / T_RAMP) * tau


def temperature_top(tau, mu):
    return T_START + (mu[..., 1:2] - T_START) * (T_END / T_RAMP) * tau


def temperature_lift(tau, x, mu):
    """Side-wall law T_Side(t, z) of case mu, extended to the whole domain."""
    z = x[..., 1:2]
    return temperature_bottom(tau, mu) + z / H_OUT * (
        temperature_top(tau, mu) - temperature_bottom(tau, mu)
    )


def scale(mu):
    """Root mean square of the two temperature rises at t_end [K]."""
    rise_bottom = (mu[..., 0:1] - T_START) * T_END / T_RAMP
    rise_top = (mu[..., 1:2] - T_START) * T_END / T_RAMP
    return jnp.sqrt(0.5 * (rise_bottom * rise_bottom + rise_top * rise_top))


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


# Network inputs: scaled tau, (r/radius)^2, z stretched to [-2, 2] (the autoclave
# is ten times longer than wide), the cavity level set, and mu scaled to [-1, 1].
Z_INPUT_RANGE = 2.0


def network_inputs(tau, x, mu, radius):
    r, z = x[..., 0:1], x[..., 1:2]
    inputs = [
        2.0 * tau - 1.0,
        (r / radius) ** 2,
        Z_INPUT_RANGE * (2.0 * z / H_OUT - 1.0),
        jnp.tanh(level_cavity(x)[..., None] / R_NH3),
        (mu - MU_CENTER) / MU_HALF_WIDTH,
    ]
    return jnp.concatenate(inputs, axis=-1)


def inputs_inconel(tau, x, mu):
    return network_inputs(tau, x, mu, R_OUT)


def inputs_nh3(tau, x, mu):
    return network_inputs(tau, x, mu, R_NH3)


N_INPUTS = 6


# Trial functions in the potential: K_m = K_m(T_lift) + sigma_m(mu) * tau * N_m,
# the Inconel one times the wall factor so that the Dirichlet conditions hold
# exactly (K = K(T_Side) on the outer wall, hence T = T_Side since K is
# increasing). sigma_m = s(mu) k_m(T_c), T_c = T_START + s(mu) / 2, makes
# N_m = 1 about s(mu) kelvin, the same size of correction as in the
# temperature version. It is fixed by this argument, not tuned.
def potential_scale(name, mu):
    """sigma_m(mu) = s(mu) k_m(T_c) [W/m]."""
    return scale(mu) * conductivity(name, T_START + 0.5 * scale(mu))


def potential_inconel(raw, tau, x, mu):
    return kirchhoff("inconel", temperature_lift(tau, x, mu)) + (
        potential_scale("inconel", mu) * tau * wall_factor(x) * raw
    )


def potential_nh3(raw, tau, x, mu):
    return kirchhoff("nh3", temperature_lift(tau, x, mu)) + (
        potential_scale("nh3", mu) * tau * raw
    )


def velocity_scale(mu):
    """Stokes estimate U_s = g_eff beta dT_lag R^2 / (16 nu) [m/s] of case mu."""
    return G_BETA_SCALE * LAG_FRACTION * scale(mu) * R_NH3**2 / (16.0 * NU_NH3)


def cavity_factor(x):
    """Twice the cavity level set over R_NH3: zero on the cavity wall, about 1 inside."""
    return 2.0 * level_cavity(x)[..., None] / R_NH3


def stream_nh3(raw, tau, x, mu):
    """psi~ = S_psi tau D^2 N_psi [m/s]: no slip on the wall, rest at t = 0."""
    return velocity_scale(mu) * tau * cavity_factor(x) ** 2 * raw


def vorticity_nh3(raw, tau, x, mu):
    """omega~ = omega / r = S_omega tau N_omega [1/(m s)]."""
    return velocity_scale(mu) / R_NH3**2 * tau * raw


# Variables of the space, in order: the potentials of the materials, then
# psi~ and omega~ of the NH3.
FIELDS = MATERIALS + ("stream", "vorticity")


def build_space(seed=42):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(FIELDS))
    networks = [
        MLP(in_size=N_INPUTS, out_size=1, hidden_sizes=HIDDEN, key=key) for key in keys
    ]
    return ApproximationSpace(
        dict(DIMS),
        [(network, "scalar", None) for network in networks],
        model_type="t_x_mu",
        pre_processing=[inputs_inconel, inputs_nh3, inputs_nh3, inputs_nh3],
        post_processing=[potential_inconel, potential_nh3, stream_nh3, vorticity_nh3],
    )


def potential_field(name, variables):
    """Kirchhoff potential K_m of material ``name``, from the variables of the space."""
    return variables[MATERIALS.index(name)]


def temperature_field(name, variables):
    """Temperature: the potential inverted by Newton steps from the side-wall law."""
    K = potential_field(name, variables)

    def temperature(space, tau, x, mu):
        target, value = K(space, tau, x, mu), temperature_lift(tau, x, mu)
        for _ in range(6):
            value = value - (kirchhoff(name, value) - target) / conductivity(
                name, value
            )
        return value

    return ParamScalarFunction(DIMS, temperature, f_type="t_x_mu")


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


def box(label, r_max=R_OUT, z_min=0.0, z_max=H_OUT):
    domain = Square2D(
        [(0.0, r_max / R_OUT), (z_min / H_OUT, z_max / H_OUT)],
        is_main_domain=False,
        label_str=label,
    )
    main_domain.add_subdomain(domain)
    return domain


class TransientConduction(InteriorResidual):
    """r (rho c_p / (k t_end)) d_tau K - [d_r(r d_r K) + r d_zz K] = 0 in one material, case mu.

    d_tau K = k(T) d_tau T, so this is the equation of the temperature version.
    """

    def __init__(self, domain, name, length, k_ref):
        super().__init__(
            domain=domain, size=1, model_type="t_x_mu", time_domain=TIME_DOMAIN
        )
        self.name, self.length, self.k_ref = name, length, k_ref

    def construct_residual(self, *vars):
        T = temperature_field(self.name, vars)
        K = potential_field(self.name, vars)
        K_tau = K.d_t()
        K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
        K_rr, K_zz = K_r.partial_derivative("x", 0), K_z.partial_derivative("x", 1)

        def residual(space, tau, x, mu):
            r = x[..., 0:1]
            T_x = T(space, tau, x, mu)
            storage = heat_capacity_per_volume(self.name, T_x) / (
                conductivity(self.name, T_x) * T_END
            )
            conduction = r * (K_rr(space, tau, x, mu) + K_zz(space, tau, x, mu)) + K_r(
                space, tau, x, mu
            )
            value = r * storage * K_tau(space, tau, x, mu) - conduction
            value = value * self.length / (self.k_ref * scale(mu))
            return jnp.where(material_masks(x)[self.name][..., None], value, 0.0)

        return ParamScalarFunction(DIMS, residual, f_type="t_x_mu")


def over_r(r, f_r, f_rr):
    """f_r / r, replaced by its limit f_rr on the axis (f_r = 0 there)."""
    return jnp.where(r > 0.0, f_r / jnp.where(r > 0.0, r, 1.0), f_rr)


def flow_derivatives(vars):
    """psi~, omega~ and the derivatives the residuals and the velocity need."""
    psi, omega = vars[FIELDS.index("stream")], vars[FIELDS.index("vorticity")]
    psi_r, psi_z = psi.partial_derivative("x", 0), psi.partial_derivative("x", 1)
    omega_r, omega_z = (
        omega.partial_derivative("x", 0),
        omega.partial_derivative("x", 1),
    )
    return {
        "psi": psi,
        "psi_r": psi_r,
        "psi_z": psi_z,
        "psi_rr": psi_r.partial_derivative("x", 0),
        "psi_zz": psi_z.partial_derivative("x", 1),
        "omega": omega,
        "omega_tau": omega.d_t(),
        "omega_r": omega_r,
        "omega_z": omega_z,
        "omega_rr": omega_r.partial_derivative("x", 0),
        "omega_zz": omega_z.partial_derivative("x", 1),
    }


def velocity(r, psi, psi_r, psi_z):
    """(u_r, u_z) [m/s] from psi~ and its derivatives at radius r."""
    return -r * psi_z, 2.0 * psi + r * psi_r


class FlowAndHeatNH3(InteriorResidual):
    """Energy, stream-function and vorticity equations in the NH3, case mu.

    Energy: the conduction residual of TransientConduction with the advection
    term r (rho c_p / k) u . grad K, scaled the same way. Stream function:
    R_psi scaled by R^2 / U_s. Vorticity: R_omega scaled by
    R^2 / (g beta dT_lag), the size of the buoyancy term.
    """

    def __init__(self, domain):
        super().__init__(
            domain=domain, size=3, model_type="t_x_mu", time_domain=TIME_DOMAIN
        )

    def construct_residual(self, *vars):
        T = temperature_field("nh3", vars)
        K = potential_field("nh3", vars)
        K_tau = K.d_t()
        K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
        K_rr, K_zz = K_r.partial_derivative("x", 0), K_z.partial_derivative("x", 1)
        flow = flow_derivatives(vars)

        def residual(space, tau, x, mu):
            r = x[..., 0:1]
            f = {name: field(space, tau, x, mu) for name, field in flow.items()}
            k_r, k_z = K_r(space, tau, x, mu), K_z(space, tau, x, mu)
            k_rr, k_zz = K_rr(space, tau, x, mu), K_zz(space, tau, x, mu)
            T_x = T(space, tau, x, mu)
            k = conductivity("nh3", T_x)
            rho_cp_over_k = heat_capacity_per_volume("nh3", T_x) / k
            u_r, u_z = velocity(r, f["psi"], f["psi_r"], f["psi_z"])

            energy = (
                r * rho_cp_over_k * (K_tau(space, tau, x, mu) / T_END)
                + r * rho_cp_over_k * (u_r * k_r + u_z * k_z)
                - (r * (k_rr + k_zz) + k_r)
            )
            energy = energy * R_NH3 / (K_REF["nh3"] * scale(mu))

            u_s = velocity_scale(mu)
            stream = (
                f["psi_rr"]
                + 3.0 * over_r(r, f["psi_r"], f["psi_rr"])
                + f["psi_zz"]
                + f["omega"]
            ) * (R_NH3**2 / u_s)

            # d_r T / r = (d_r K / k) / r; on the axis d_rr K / k.
            buoyancy = G_BETA * over_r(r, k_r, k_rr) / k
            vorticity = (
                f["omega_tau"] / T_END
                + u_r * f["omega_r"]
                + u_z * f["omega_z"]
                - NU_NH3
                * (
                    f["omega_rr"]
                    + 3.0 * over_r(r, f["omega_r"], f["omega_rr"])
                    + f["omega_zz"]
                )
                + buoyancy
            ) * (R_NH3**2 / (G_BETA_SCALE * LAG_FRACTION * scale(mu)))

            value = jnp.concatenate([energy, stream, vorticity], axis=-1)
            return jnp.where(material_masks(x)["nh3"][..., None], value, 0.0)

        return ParamVecFunction(3, DIMS, residual, f_type="t_x_mu")


class Interface(InteriorResidual):
    """Temperature jump and normal-flux jump across the cavity wall, case mu.

    The wall is a curve, sampled through a dummy box: the second coordinate of
    a sample, divided by H_OUT, is the curve parameter.
    """

    def __init__(self, domain):
        super().__init__(
            domain=domain, size=2, model_type="t_x_mu", time_domain=TIME_DOMAIN
        )

    def construct_residual(self, *vars):
        T_in, T_out = temperature_field("nh3", vars), temperature_field("inconel", vars)

        def flux(name):
            K = potential_field(name, vars)
            K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
            return lambda sp, tau, x, mu: jnp.concatenate(
                [K_r(sp, tau, x, mu), K_z(sp, tau, x, mu)], -1
            )

        flux_in, flux_out = flux("nh3"), flux("inconel")

        def residual(space, tau, x, mu):
            point = cavity_curve(x[..., 1] / H_OUT)
            normal = jax.grad(lambda p: jnp.sum(level_cavity(p)))(point)
            normal = normal / jnp.sqrt(jnp.sum(normal * normal) + 1.0e-30)
            temperature_jump = T_in(space, tau, point, mu) - T_out(
                space, tau, point, mu
            )
            flux_jump = jnp.sum(
                (flux_in(space, tau, point, mu) - flux_out(space, tau, point, mu))
                * normal,
                axis=-1,
                keepdims=True,
            )
            flux_jump = flux_jump * R_NH3 / K_REF["inconel"]
            return jnp.concatenate([temperature_jump, flux_jump], axis=-1) / scale(mu)

        return ParamVecFunction(2, DIMS, residual, f_type="t_x_mu")


class AutoclaveTransientConductionParametric(AbstractPhysicalModel):
    def __init__(self):
        super().__init__(main_domain=main_domain, time_domain=TIME_DOMAIN)
        self.physical_residuals = {
            "inconel": TransientConduction(
                box("inconel"), "inconel", R_OUT, K_REF["inconel"]
            ),
            "nh3": FlowAndHeatNH3(box("nh3", R_NH3, Z_NH3_CENTER - R_NH3, Z_NH3_TOP)),
            "wall": Interface(box("wall")),
        }
        self.data_residuals = {}


def build_model_and_sampler():
    model = AutoclaveTransientConductionParametric()
    # Every collocation point gets its own time and its own case mu.
    sampler = TensorizedSampler(
        [
            UniformTimeSampler(TIME_DOMAIN),
            DomainSampler(main_domain, main_domain_fully_covered=True),
            UniformParametricSampler(MU_BOUNDS),
        ],
        model_type="t_x_mu",
        bc=False,
        ic=False,
    )
    return model, sampler


# Evaluation
def temperature_at(space, tau, points, mu):
    """PINN temperature at points (n, 2), scaled time tau and case mu ((2,) or (n, 2))."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    tau = jnp.full((n, 1), float(tau))
    mu = jnp.broadcast_to(jnp.asarray(mu, dtype=jnp.float64), (n, 2))
    masks = material_masks(points)
    variables = space.create_variables()
    temperature = np.empty(n)
    for name in MATERIALS:
        mask = np.asarray(masks[name])
        if np.any(mask):
            field = temperature_field(name, variables)
            temperature[mask] = np.asarray(
                field(space, tau[mask], jnp.asarray(points[mask]), mu[mask])
            ).ravel()
    return temperature


def flow_at(space, tau, points, mu):
    """Stokes stream function psi = r^2 psi~ [m3/s] and velocity (u_r, u_z) [m/s] at points (n, 2) of the cavity."""
    points = jnp.asarray(points, dtype=jnp.float64)
    n = points.shape[0]
    tau = jnp.full((n, 1), float(tau))
    mu = jnp.broadcast_to(jnp.asarray(mu, dtype=jnp.float64), (n, 2))
    flow = flow_derivatives(space.create_variables())
    # Derivatives are evaluated point by point (vmap).
    values = {
        name: np.asarray(
            flow[name].vmap_on_physical_variables()(space, tau, points, mu)
        ).reshape(n)
        for name in ("psi", "psi_r", "psi_z")
    }
    r = np.asarray(points[:, 0])
    u_r, u_z = velocity(r, values["psi"], values["psi_r"], values["psi_z"])
    return r**2 * values["psi"], u_r, u_z


def plot_flow(space, save_dir, mu, case, target_times=(1800.0, 3600.0, T_END)):
    """Speed and streamlines in the cavity at target_times, and the largest speed against time."""
    r = np.linspace(0.0, R_NH3, 25)
    z = np.linspace(Z_NH3_CENTER - R_NH3, Z_NH3_TOP, 400)
    rr, zz = np.meshgrid(r, z)
    points = np.column_stack([rr.ravel(), zz.ravel()])
    inside = np.asarray(level_cavity(points)) > 0.0

    figure, axes = plt.subplots(
        1, len(target_times), figsize=(2.4 * len(target_times), 11.0)
    )
    for ax, t in zip(np.atleast_1d(axes), target_times):
        psi, u_r, u_z = flow_at(space, t / T_END, points, mu)
        speed = np.where(inside, np.hypot(u_r, u_z), np.nan).reshape(rr.shape)
        psi = np.where(inside, psi, np.nan).reshape(rr.shape)
        contour = ax.contourf(
            rr * 1e3, zz * 1e3, speed * 1e3, levels=30, cmap="viridis"
        )
        figure.colorbar(contour, ax=ax, label="|u| [mm/s]", shrink=0.8)
        if np.nanmax(np.abs(psi)) > 0.0:
            ax.contour(rr * 1e3, zz * 1e3, psi, levels=15, colors="w", linewidths=0.6)
        draw_geometry(ax)
        ax.set_xlim(0.0, R_NH3 * 1e3)
        ax.set_title(f"t = {t:.0f} s")
    np.atleast_1d(axes)[0].set_ylabel("z [mm]")
    figure.suptitle(
        f"{case.capitalize()} case, NH3 flow (Ra = {RA:g}), white: streamlines"
    )
    figure.tight_layout()
    figure.savefig(save_dir / f"{case}_flow.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    times = np.linspace(0.0, T_END, 25)
    max_speed = np.array(
        [
            np.max(np.hypot(*flow_at(space, t / T_END, points[inside], mu)[1:]))
            for t in times
        ]
    )
    pd.DataFrame({"t_s": times, "max_speed_m_per_s": max_speed}).to_csv(
        save_dir / f"{case}_max_speed.csv", index=False
    )
    figure, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(times, max_speed * 1e3, "o-", ms=3)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("max |u| in the cavity [mm/s]")
    ax.grid(True, alpha=0.4)
    figure.tight_layout()
    figure.savefig(save_dir / f"{case}_max_speed.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    # Peclet number of the largest speed, with the NH3 diffusivity at 600 K
    kappa = float(K_REF["nh3"] / heat_capacity_per_volume("nh3", 600.0))
    return {
        "max_speed_t_end_mm_s": float(max_speed[-1] * 1e3),
        "max_speed_mm_s": float(max_speed.max() * 1e3),
        "peclet_max": float(max_speed.max() * R_NH3 / kappa),
    }


def field_on_grid(space, mu, times, nodes):
    return np.stack([temperature_at(space, t / T_END, nodes, mu) for t in times])


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


def error_metrics(T_pinn, T_ref, nodes, times):
    """RMSE and largest error of T_pinn - T_ref (n_t, n_nodes), overall and per material."""
    error = T_pinn - T_ref
    masks = {name: np.asarray(mask) for name, mask in material_masks(nodes).items()}
    i, j = np.unravel_index(np.argmax(np.abs(error)), error.shape)
    metrics = {
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_K": float(abs(error[i, j])),
        "max_abs_error_time_s": float(times[i]),
        "max_abs_error_r_mm": float(nodes[j, 0] * 1e3),
        "max_abs_error_z_mm": float(nodes[j, 1] * 1e3),
        "rmse_last_time_K": float(np.sqrt(np.mean(error[-1] ** 2))),
        "max_relative_error_percent": float(np.max(np.abs(error) / T_ref) * 100.0),
    }
    for name, mask in masks.items():
        metrics[f"rmse_{name}_K"] = float(np.sqrt(np.mean(error[:, mask] ** 2)))
        metrics[f"max_abs_error_{name}_K"] = float(np.max(np.abs(error[:, mask])))
    per_time = pd.DataFrame(
        {
            "time_s": times,
            "rmse_K": np.sqrt(np.mean(error**2, axis=1)),
            "max_abs_error_K": np.max(np.abs(error), axis=1),
            **{
                f"rmse_{name}_K": np.sqrt(np.mean(error[:, mask] ** 2, axis=1))
                for name, mask in masks.items()
            },
        }
    )
    return metrics, per_time


def print_metrics(title, metrics):
    print(f"{title}:")
    print(
        f"  RMSE {metrics['rmse_K']:.4f} K (Inconel {metrics['rmse_inconel_K']:.4f}, "
        f"NH3 {metrics['rmse_nh3_K']:.4f})"
    )
    print(
        f"  max |difference| {metrics['max_abs_error_K']:.4f} K at t = "
        f"{metrics['max_abs_error_time_s']:.1f} s, r = {metrics['max_abs_error_r_mm']:.1f} mm, "
        f"z = {metrics['max_abs_error_z_mm']:.1f} mm"
    )


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


def plot_maps(
    save_dir,
    times,
    nodes,
    T_ref,
    T_pinn,
    target_times=(1800.0, 3600.0, np.inf),
    case="stable",
):
    """CrysMAS field, PINN field and difference at the saved times nearest to target_times."""
    triangulation = Triangulation(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3)
    for target in target_times:
        i = int(np.argmin(np.abs(times - min(target, times[-1]))))
        difference = T_pinn[i] - T_ref[i]
        bound = max(float(np.max(np.abs(difference))), 1.0e-6)
        T_levels = np.linspace(
            min(T_ref[i].min(), T_pinn[i].min()),
            max(T_ref[i].max(), T_pinn[i].max()),
            51,
        )
        figure, axes = plt.subplots(1, 3, figsize=(8.5, 11.0))
        for ax, field, title, options in (
            (axes[0], T_ref[i], "CrysMAS", {"levels": T_levels, "cmap": "turbo"}),
            (
                axes[1],
                T_pinn[i],
                "parametric PINN",
                {"levels": T_levels, "cmap": "turbo"},
            ),
            (
                axes[2],
                difference,
                "parametric - CrysMAS",
                {"levels": np.linspace(-bound, bound, 41), "cmap": "RdBu_r"},
            ),
        ):
            contour = ax.tricontourf(triangulation, field, **options)
            figure.colorbar(contour, ax=ax, label="K", shrink=0.7)
            draw_geometry(ax)
            ax.set_title(title)
        axes[0].set_ylabel("z [mm]")
        figure.suptitle(f"{case.capitalize()} heating case, t = {times[i]:.1f} s")
        figure.tight_layout()
        figure.savefig(
            save_dir / f"{case}_maps_t{times[i]:.0f}s.png", dpi=200, bbox_inches="tight"
        )
        plt.close(figure)


def plot_error_history(save_dir, per_time, case="stable"):
    figure, ax = plt.subplots(figsize=(7.0, 4.0))
    for column, label in (
        ("rmse_K", "all nodes"),
        ("rmse_inconel_K", "Inconel"),
        ("rmse_nh3_K", "NH3"),
    ):
        ax.plot(per_time["time_s"], per_time[column], label=label)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("RMSE, parametric PINN vs CrysMAS [K]")
    ax.grid(True, alpha=0.4)
    ax.legend()
    figure.tight_layout()
    figure.savefig(
        save_dir / f"{case}_vs_crysmas_error_history.png", dpi=200, bbox_inches="tight"
    )
    plt.close(figure)


def plot_line_comparisons(
    save_dir,
    reference,
    T_pinn,
    target_times=(1800.0, 3600.0, np.inf),
    label="parametric PINN",
    case="stable",
):
    """PINN and CrysMAS along every radial line of the CrysMAS nodes.

    For each saved time nearest to target_times, two images with one panel per
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
            path = save_dir / f"{case}_{name}_t{times[i]:.0f}s.png"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {path.resolve()}")
    pd.DataFrame(rows).to_csv(
        save_dir / f"{case}_line_comparison_per_radius.csv", index=False
    )


PROBES = {  # (r, z) [m] at mid-height of the cavity
    "NH3, axis": (0.0, 0.5 * (Z_NH3_CENTER + Z_NH3_TOP)),
    "cavity wall": (R_NH3, 0.5 * (Z_NH3_CENTER + Z_NH3_TOP)),
    "Inconel, mid-wall": (
        0.5 * (R_NH3 + R_OUT),
        0.5 * (Z_NH3_CENTER + Z_NH3_TOP),
    ),
}


def probe_histories(space, save_dir, mu, reference=None, n_t=101, case="stable"):
    """Probe temperatures and lags of case mu; CrysMAS (times, values (n_probes, n_t)) as markers."""
    taus = np.linspace(0.0, 1.0, n_t)
    mu_array = np.asarray(mu, dtype=np.float64)[None, :]
    table, results = {"t_s": taus * T_END}, {}
    figure, (ax_T, ax_lag) = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for k, (label, point) in enumerate(PROBES.items()):
        point = np.array([point])
        history = np.array([temperature_at(space, tau, point, mu)[0] for tau in taus])
        wall = np.array(
            [float(temperature_lift(tau, point, mu_array)[0, 0]) for tau in taus]
        )
        table[f"T {label}"], table[f"lag {label}"] = history, history - wall
        results[label] = {
            "T_end_K": float(history[-1]),
            "lag_end_K": float(history[-1] - wall[-1]),
        }
        ax_T.plot(taus * T_END, history, color=f"C{k}", label=f"{label}, parametric")
        ax_lag.plot(
            taus * T_END, history - wall, color=f"C{k}", label=f"{label}, parametric"
        )
        if reference is not None:
            times, values = reference
            wall_ref = np.asarray(
                temperature_lift(times[:, None] / T_END, point, mu_array)
            )[:, 0]
            style = {
                "color": f"C{k}",
                "ls": "none",
                "marker": "o",
                "ms": 3,
                "markevery": 4,
            }
            ax_T.plot(times, values[k], label=f"{label}, CrysMAS", **style)
            ax_lag.plot(times, values[k] - wall_ref, label=f"{label}, CrysMAS", **style)
            results[label] |= {
                "crysmas_end_K": float(values[k, -1]),
                "difference_end_K": float(history[-1] - values[k, -1]),
            }
    pd.DataFrame(table).to_csv(save_dir / f"{case}_probe_histories.csv", index=False)
    ax_T.set_ylabel("T [K]")
    ax_lag.set_ylabel("T - T_Side [K]")
    for ax in (ax_T, ax_lag):
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7)
    figure.suptitle(
        f"{case.capitalize()} heating case, mu = ({mu[0]:.0f} K, {mu[1]:.0f} K)"
    )
    figure.tight_layout()
    figure.savefig(
        save_dir / f"{case}_probe_histories.png", dpi=200, bbox_inches="tight"
    )
    plt.close(figure)
    return results


def lag_map(space, save_dir, n=21):
    """Lag of the NH3 on the axis at mid-height behind the outer wall at t_end, over the box."""
    t_bot = np.linspace(*MU_BOUNDS[0], n)
    t_top = np.linspace(*MU_BOUNDS[1], n)
    grid_bot, grid_top = np.meshgrid(t_bot, t_top, indexing="ij")
    mu = np.column_stack([grid_bot.ravel(), grid_top.ravel()])
    point = np.repeat(np.array([PROBES["NH3, axis"]]), len(mu), axis=0)
    lag = (
        temperature_at(space, 1.0, point, mu)
        - np.asarray(temperature_lift(1.0, point, mu))[:, 0]
    )
    lag = lag.reshape(n, n)
    pd.DataFrame(
        {
            "T_bot_end_K": grid_bot.ravel(),
            "T_top_end_K": grid_top.ravel(),
            "lag_axis_K": lag.ravel(),
        }
    ).to_csv(save_dir / "lag_map_axis_t_end.csv", index=False)
    figure, ax = plt.subplots(figsize=(6.0, 5.0))
    contour = ax.contourf(grid_bot, grid_top, lag, levels=30, cmap="viridis")
    figure.colorbar(
        contour, ax=ax, label="T - T_Side, NH3 on the axis at mid-height [K]"
    )
    for case, (mu_case, _) in CASES.items():
        ax.plot(*mu_case, "w*", ms=12, mec="k")
        ax.annotate(case, mu_case, textcoords="offset points", xytext=(8, 8), color="w")
    ax.set_xlabel("T_Bot,e [K]")
    ax.set_ylabel("T_Top,e [K]")
    ax.set_title(f"Lag at t_end = {T_END:.1f} s")
    figure.tight_layout()
    figure.savefig(save_dir / "lag_map_axis_t_end.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return {"lag_axis_min_K": float(lag.min()), "lag_axis_max_K": float(lag.max())}


def plot_loss_history(losses, save_dir):
    figure, ax = plt.subplots(figsize=(7.0, 4.0))
    # Shape (n_epochs, n_components): the interface term has two components.
    component_names = {
        "wall": ("wall, temperature jump", "wall, flux jump"),
        "nh3": ("nh3, energy", "nh3, stream function", "nh3, vorticity"),
    }
    table = {}
    for name, history in losses.losses_history.items():
        history = np.asarray(history).reshape(len(history), -1)
        labels = component_names.get(name, (name,) * history.shape[1])
        for j in range(history.shape[1]):
            ax.semilogy(history[:, j], label=labels[j])
            table[labels[j]] = history[:, j]
    pd.DataFrame(table).to_csv(save_dir / "loss_history.csv", index_label="epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    figure.tight_layout()
    figure.savefig(save_dir / "loss_history.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="parametric_K_unknown_navier_stokes",
        help="subfolder of output_inc_nh3_transient_7200s",
    )
    parser.add_argument(
        "--ra",
        type=float,
        default=RA,
        help="Rayleigh number of the reduced gravity (0: no buoyancy)",
    )
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=HIDDEN, help="hidden layer sizes"
    )
    parser.add_argument(
        "--points-scale", type=float, default=1.0, help="factor on the points per epoch"
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="draw the collocation points once instead of at every epoch",
    )
    args = parser.parse_args()
    RA = args.ra
    G_BETA = buoyancy_coefficient(RA)
    resample = not args.no_resample
    variant = (
        ""
        if (list(args.hidden) == HIDDEN and args.points_scale == 1.0)
        else (f"_h{'-'.join(map(str, args.hidden))}_p{args.points_scale:g}")
    ) + ("" if resample else "_noresample")
    HIDDEN = list(args.hidden)
    N_COLLOC = {
        label: int(round(n * args.points_scale)) for label, n in N_COLLOC.items()
    }
    print(f"Hidden layers {HIDDEN}, points per epoch {N_COLLOC}")

    save_dir = OUTPUT_ROOT / args.out
    save_dir.mkdir(parents=True, exist_ok=True)

    model, sampler = build_model_and_sampler()
    space = build_space(seed=args.seed)
    pinn = Projector(model, space, sampler, **ENG_OPTIONS)

    _, samples = sampler.sample(jax.random.PRNGKey(0), N_COLLOC)
    print("Initial losses:", pinn.evaluate_losses(space, samples))
    start = timeit.default_timer()
    _, pinn = pinn.project(
        jax.random.PRNGKey(args.seed),
        space,
        args.epochs,
        N_COLLOC,
        tqdm_desc="parametric, potential as unknown",
        resample=resample,
    )
    training_time = timeit.default_timer() - start
    print("Best loss:", pinn.best_loss)
    print(f"Training time: {training_time:.1f} s")
    pinn.save(
        "inconel_nh3_transient",
        postfix=f"global_parametric_K_unknown_navier_stokes_7200s_ra{RA:g}_seed{args.seed}_{args.epochs}ep{variant}",
    )

    plot_loss_history(pinn.losses, save_dir)
    summary = {
        "unknown": "Kirchhoff potential; psi~, omega~ in the NH3",
        "rayleigh_number": RA,
        "g_eff_beta_m_per_s2_K": G_BETA,
        "nu_m2_per_s": NU_NH3,
        "beta_per_K": BETA_NH3,
        "seed": args.seed,
        "epochs": args.epochs,
        "resample_every_epoch": resample,
        "t_end_s": T_END,
        "hidden": HIDDEN,
        "points_per_epoch": N_COLLOC,
        "mu_bounds": MU_BOUNDS,
        "n_parameters": int(
            sum(leaf.size for leaf in jax.tree_util.tree_leaves(space.models))
        ),
        "training_time_s": training_time,
        "best_loss": {
            name: np.asarray(value).ravel().tolist()
            for name, value in pinn.best_loss.items()
        },
    }

    # Stable and unstable cases against CrysMAS, at the CrysMAS nodes and times.
    for case, (mu, path) in CASES.items():
        times, nodes, T_crysmas = read_crysmas(path)
        field = field_on_grid(pinn.space, mu, times, nodes)
        np.savez_compressed(
            save_dir / f"{case}_field_on_reference_grid.npz",
            times=times,
            nodes=nodes,
            T=field,
        )
        metrics, per_time = error_metrics(field, T_crysmas, nodes, times)
        per_time.to_csv(save_dir / f"{case}_vs_crysmas_per_time.csv", index=False)
        print_metrics(
            f"{case.capitalize()} case, parametric PINN against CrysMAS", metrics
        )
        plot_error_history(save_dir, per_time, case)
        plot_maps(save_dir, times, nodes, T_crysmas, field, case=case)
        plot_line_comparisons(save_dir, (times, nodes, T_crysmas), field, case=case)
        probes_ref = (
            times,
            interpolate_crysmas(nodes, T_crysmas, np.array(list(PROBES.values()))),
        )
        summary[case] = {
            "mu": list(mu),
            "crysmas": metrics,
            "probes": probe_histories(pinn.space, save_dir, mu, probes_ref, case=case),
            "flow": plot_flow(pinn.space, save_dir, mu, case),
        }
        print(f"  NH3 flow: {summary[case]['flow']}")

    summary["lag_map"] = lag_map(pinn.space, save_dir)
    (save_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {(save_dir / 'run_summary.json').resolve()}")
