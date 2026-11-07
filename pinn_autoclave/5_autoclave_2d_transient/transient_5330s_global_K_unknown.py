r"""Transient heat conduction in an Inconel autoclave filled with NH3 (global PINN, potential as unknown).

Axisymmetric (r, z) problem with temperature-dependent material data

.. math::

    \rho_m(T) c_{p,m}(T) \, \partial_t T = \nabla \cdot (k_m(T) \nabla T)
    \quad \text{in each material } m, \quad 0 < t < t_{end},

with temperature and normal heat flux continuous across the cavity wall
(Inconel/NH3) and symmetry on the axis r = 0. The outer wall is heated by the
linear ramps of the stable case of the formulation:

.. math::

    T_{Bot}(t) = 300 + a_{Bot} \, t, \qquad T_{Top}(t) = 300 + a_{Top} \, t, \qquad
    T_{Side}(t, z) = T_{Bot}(t) + \frac{z}{z_{Top}} (T_{Top}(t) - T_{Bot}(t)),

with the rates a_Bot = 423 K / 7200 s and a_Top = 523 K / 7200 s (the ramps of
the formulation, 300 to 723 K and 300 to 823 K). The CrysMAS reference run of
this case ends at t = 5330.6 s, so the model is trained and evaluated on
0 <= t <= t_end = 5330.6 s. The initial condition is the uniform field
T = 300 K.

Global space-time PINN whose networks build the Kirchhoff potential
:math:`K_m(T) = \int k_m \, dT` of each material. It is the script
``2D_heat_Inconel_+_Ammonia_transient_global.py`` with the potential in place
of the temperature as the unknown; network inputs, sampling, loss terms,
weights, optimizer and evaluation are the same.

* the networks take (t, r, z) as input, with the time scaled to
  tau = t / t_end in [0, 1];
* one MLP per material,
  :math:`K_m = K_m(T_{lift}(\tau, z)) + \sigma_m \, \tau \, N_m`, where
  :math:`T_{lift}` is the side-wall law extended to the whole domain and
  :math:`\sigma_m = \Delta T \, k_m(T_c)`, with :math:`T_c` the middle of the
  temperature range of the window, so that N_m = 1 is about one Delta T, as
  in the temperature version. The factor tau makes the initial condition
  exact (K = K(300 K), hence T = 300 K);
* the Inconel network is also multiplied by a factor that vanishes on the
  outer wall: K = K(T_Side) there, hence T = T_Side, since K is increasing;
* the temperature :math:`T_m = K_m^{-1}(K_m)` is recovered by six Newton
  steps started from the side-wall law. It enters the storage term, the
  temperature jump and every evaluation and plot;
* with :math:`\partial_t K_m = k_m(T) \, \partial_t T` the equation reads
  :math:`\frac{\rho_m c_{p,m}}{k_m}(T_m) \, \partial_t K_m = \Delta K_m`: the
  conduction term and the flux :math:`\nabla K_m` are linear in the network
  output;
* the loss is the equation residual in each material plus, on the cavity
  wall, the temperature jump and the flux jump (soft interface).

The PINN is compared with the CrysMAS run at every saved time and node, and
along every radial line of the CrysMAS nodes.
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

# Heating ramps of the stable case [K, K/s] and the simulated time window [s]:
# the end of the CrysMAS run.
T_START = 300.0
RATE_BOT = (723.0 - T_START) / 7200.0
RATE_TOP = (823.0 - T_START) / 7200.0
T_END = 5330.6
# Largest rise of the wall temperature in the window [K]: the scale of the case.
DELTA_T = max(RATE_BOT, RATE_TOP) * T_END

# CrysMAS transient run, saved every ~50 s from t = 1 s to t = 5330.6 s.
PROJECT_DIR = Path(__file__).resolve().parents[1] / "data"
CRYSMAS_FILE = PROJECT_DIR / "Transient CrysMAS data" / "Autoclave_stable.csv"

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

# Training
N_EPOCHS = 300
HIDDEN = [24, 24, 24]
N_COLLOC = {"inconel": 8000, "nh3": 8000, "wall": 1500}
ENG_OPTIONS = {"linesearch": "strong-wolfe", "adaptive_matrix_regularization": True}

# Scaled time tau = t / t_end
TIME_DOMAIN = (0.0, 1.0)
DIMS = {"x": 2, "t": 1}


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


def network_inputs(tau, x, radius):
    """Scaled tau, (r/radius)^2, z and the cavity level set."""
    r, z = x[..., 0:1], x[..., 1:2]
    inputs = [
        2.0 * tau - 1.0,
        (r / radius) ** 2,
        Z_INPUT_RANGE * (2.0 * z / H_OUT - 1.0),
        jnp.tanh(level_cavity(x)[..., None] / R_NH3),
    ]
    return jnp.concatenate(inputs, axis=-1)


def inputs_inconel(tau, x):
    return network_inputs(tau, x, R_OUT)


def inputs_nh3(tau, x):
    return network_inputs(tau, x, R_NH3)


N_INPUTS = 4


# Trial functions in the potential: K_m = K_m(T_lift) + sigma_m * tau * N_m, the
# Inconel one times the wall factor so that the Dirichlet conditions hold
# exactly (K = K(T_Side) on the outer wall, hence T = T_Side since K is
# increasing). sigma_m = delta_T * k_m(T_c) makes N_m = 1 about one delta_T, the
# same size of correction as in the temperature version. It is fixed by this
# argument, not tuned.
T_C = T_START + 0.5 * DELTA_T  # middle of the temperature range of the window [K]


def potential_scale(name):
    """sigma_m = delta_T * k_m(T_c) [W/m]."""
    return DELTA_T * conductivity(name, T_C)


def potential_inconel(raw, tau, x):
    return kirchhoff("inconel", temperature_lift(tau, x)) + (
        potential_scale("inconel") * tau * wall_factor(x) * raw
    )


def potential_nh3(raw, tau, x):
    return kirchhoff("nh3", temperature_lift(tau, x)) + (
        potential_scale("nh3") * tau * raw
    )


def build_space(seed=42):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(MATERIALS))
    networks = [
        MLP(in_size=N_INPUTS, out_size=1, hidden_sizes=HIDDEN, key=key) for key in keys
    ]
    return ApproximationSpace(
        dict(DIMS),
        [(network, "scalar", None) for network in networks],
        model_type="t_x",
        pre_processing=[inputs_inconel, inputs_nh3],
        post_processing=[potential_inconel, potential_nh3],
    )


def potential_field(name, variables):
    """Kirchhoff potential K_m of material ``name``, from the variables of the space."""
    return variables[MATERIALS.index(name)]


def temperature_field(name, variables):
    """Temperature: the potential inverted by Newton steps from the side-wall law."""
    K = potential_field(name, variables)

    def temperature(space, tau, x):
        target, value = K(space, tau, x), temperature_lift(tau, x)
        for _ in range(6):
            value = value - (kirchhoff(name, value) - target) / conductivity(
                name, value
            )
        return value

    return ParamScalarFunction(DIMS, temperature, f_type="t_x")


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
    """r (rho c_p / (k t_end)) d_tau K - [d_r(r d_r K) + r d_zz K] = 0 in one material.

    d_tau K = k(T) d_tau T, so this is the equation of the temperature version.
    ``length`` and ``k_ref`` make the residual dimensionless, relative to the
    temperature rise DELTA_T of the heating case.
    """

    def __init__(self, domain, name, length, k_ref):
        super().__init__(
            domain=domain, size=1, model_type="t_x", time_domain=TIME_DOMAIN
        )
        self.name, self.length, self.k_ref = name, length, k_ref

    def construct_residual(self, *vars):
        T = temperature_field(self.name, vars)
        K = potential_field(self.name, vars)
        K_tau = K.d_t()
        K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
        K_rr, K_zz = K_r.partial_derivative("x", 0), K_z.partial_derivative("x", 1)

        def residual(space, tau, x):
            r = x[..., 0:1]
            T_x = T(space, tau, x)
            storage = heat_capacity_per_volume(self.name, T_x) / (
                conductivity(self.name, T_x) * T_END
            )
            conduction = r * (K_rr(space, tau, x) + K_zz(space, tau, x)) + K_r(
                space, tau, x
            )
            value = r * storage * K_tau(space, tau, x) - conduction
            value = value * self.length / (self.k_ref * DELTA_T)
            return jnp.where(material_masks(x)[self.name][..., None], value, 0.0)

        return ParamScalarFunction(DIMS, residual, f_type="t_x")


def interface_jumps(variables, inside="nh3", outside="inconel"):
    """Temperature jump [K] and normal-flux jump [W/m2] on the cavity wall.

    The normal is the gradient of the level set of the cavity.
    """
    T_in, T_out = (
        temperature_field(inside, variables),
        temperature_field(outside, variables),
    )

    def flux(name):
        K = potential_field(name, variables)
        K_r, K_z = K.partial_derivative("x", 0), K.partial_derivative("x", 1)
        return lambda sp, tau, x: jnp.concatenate(
            [K_r(sp, tau, x), K_z(sp, tau, x)], -1
        )

    flux_in, flux_out = flux(inside), flux(outside)

    def jumps(space, tau, s):
        point = cavity_curve(s)
        normal = jax.grad(lambda p: jnp.sum(level_cavity(p)))(point)
        normal = normal / jnp.sqrt(jnp.sum(normal * normal) + 1.0e-30)
        temperature_jump = T_in(space, tau, point) - T_out(space, tau, point)
        flux_jump = jnp.sum(
            (flux_in(space, tau, point) - flux_out(space, tau, point)) * normal,
            axis=-1,
            keepdims=True,
        )
        return temperature_jump, flux_jump

    return jumps


class Interface(InteriorResidual):
    """Temperature jump and normal-flux jump across the cavity wall, both zero.

    The wall is a curve, sampled through a dummy box: the second coordinate of
    a sample, divided by H_OUT, is the curve parameter.
    """

    def __init__(self, domain):
        super().__init__(
            domain=domain, size=2, model_type="t_x", time_domain=TIME_DOMAIN
        )

    def construct_residual(self, *vars):
        jumps = interface_jumps(vars)

        def residual(space, tau, x):
            temperature_jump, flux_jump = jumps(space, tau, x[..., 1] / H_OUT)
            flux_jump = flux_jump * R_NH3 / K_REF["inconel"]
            return jnp.concatenate([temperature_jump, flux_jump], axis=-1) / DELTA_T

        return ParamVecFunction(2, DIMS, residual, f_type="t_x")


class AutoclaveTransientConduction(AbstractPhysicalModel):
    def __init__(self):
        super().__init__(main_domain=main_domain, time_domain=TIME_DOMAIN)
        self.physical_residuals = {
            "inconel": TransientConduction(
                box("inconel"), "inconel", R_OUT, K_REF["inconel"]
            ),
            "nh3": TransientConduction(
                box("nh3", R_NH3, Z_NH3_CENTER - R_NH3, Z_NH3_TOP),
                "nh3",
                R_NH3,
                K_REF["nh3"],
            ),
            "wall": Interface(box("wall")),
        }
        self.data_residuals = {}


def build_model_and_sampler():
    model = AutoclaveTransientConduction()
    # Boundary and initial conditions are built into the trial functions:
    # only interior points (tau, r, z) are drawn.
    sampler = TensorizedSampler(
        [
            UniformTimeSampler(TIME_DOMAIN),
            DomainSampler(main_domain, main_domain_fully_covered=True),
        ],
        model_type="t_x",
        bc=False,
        ic=False,
    )
    return model, sampler


# Evaluation
def temperature_at(space, tau, points):
    """PINN temperature at points (n, 2) and scaled time tau, from each material's field."""
    points = np.asarray(points, dtype=np.float64)
    tau = jnp.full((points.shape[0], 1), float(tau))
    masks = material_masks(points)
    variables = space.create_variables()
    temperature = np.empty(points.shape[0])
    for name in MATERIALS:
        mask = np.asarray(masks[name])
        if np.any(mask):
            field = temperature_field(name, variables)
            temperature[mask] = np.asarray(
                field(space, tau[mask], jnp.asarray(points[mask]))
            ).ravel()
    return temperature


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
    space, save_dir, taus=(0.0, 0.25, 0.5, 0.75, 1.0), n_r=120, n_z=1000
):
    """Temperature and deviation from the side-wall law at several times."""
    r = np.linspace(0.0, R_OUT, n_r)
    z = np.linspace(0.0, H_OUT, n_z)
    rr, zz = np.meshgrid(r, z)
    points = np.column_stack([rr.ravel(), zz.ravel()])

    figure, axes = plt.subplots(2, len(taus), figsize=(2.6 * len(taus), 12.0))
    for j, tau in enumerate(taus):
        temperature = temperature_at(space, tau, points).reshape(rr.shape)
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
    figure.suptitle("Stable heating case")
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


def plot_probe_histories(space, save_dir, reference=None, n_t=101):
    """Temperature against time at the probes, and the lag behind the outer wall.

    With a CrysMAS ``reference`` (see read_crysmas), its values interpolated to
    the probes are drawn as markers of the same colour.
    """
    taus = np.linspace(0.0, 1.0, n_t)
    figure, (ax_T, ax_lag) = plt.subplots(1, 2, figsize=(12.0, 4.5))
    if reference is not None:
        times, points, T_ref = reference
        probe_points = np.array(list(PROBES.values()))
        T_ref_probes = interpolate_crysmas(points, T_ref, probe_points)
    table = {"t_s": taus * T_END}
    results = {}
    for k, (label, point) in enumerate(PROBES.items()):
        point = np.array([point])
        history = np.array([temperature_at(space, tau, point)[0] for tau in taus])
        wall = np.array([float(temperature_lift(tau, point)[0, 0]) for tau in taus])
        table[f"T {label}"], table[f"lag {label}"] = history, history - wall
        results[label] = {
            "T_end_K": float(history[-1]),
            "lag_end_K": float(history[-1] - wall[-1]),
        }
        color = f"C{k}"
        ax_T.plot(taus * T_END, history, color=color, label=f"{label}, PINN")
        ax_lag.plot(taus * T_END, history - wall, color=color, label=f"{label}, PINN")
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


def compare_with_crysmas(space, save_dir, reference):
    """PINN minus CrysMAS at every node and saved time; error statistics per time and material.

    Every node counts once (the nodes are not equally spaced, so this is a
    node average and not a volume average). The material of a node follows the
    level set of the PINN geometry.
    """
    times, nodes, T_ref = reference
    masks = {name: np.asarray(mask) for name, mask in material_masks(nodes).items()}
    T_pinn = np.stack([temperature_at(space, t / T_END, nodes) for t in times])
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
        f"CrysMAS comparison over {len(times)} times "
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
    return T_pinn, summary, metrics


def plot_crysmas_error_history(summary, save_dir):
    """RMSE and largest error against CrysMAS as functions of time."""
    figure, (ax_rmse, ax_max) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    t = summary["time_s"]
    for column, label in (("", "all nodes"), ("_inconel", "Inconel"), ("_nh3", "NH3")):
        ax_rmse.plot(t, summary[f"rmse{column}_K"], label=label)
        ax_max.plot(t, summary[f"max_abs_error{column}_K"], label=label)
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
    """CrysMAS field, PINN field and their difference at the saved times nearest to target_times."""
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
        figure.suptitle(f"Stable heating case, t = {times[i]:.1f} s")
        figure.tight_layout()
        path = save_dir / f"crysmas_comparison_t{times[i]:.0f}s.png"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {path.resolve()}")


def plot_line_comparisons(
    save_dir, reference, T_pinn, target_times=(1800.0, 3600.0, np.inf), label="PINN"
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
            path = save_dir / f"{name}_t{times[i]:.0f}s.png"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {path.resolve()}")
    pd.DataFrame(rows).to_csv(save_dir / "line_comparison_per_radius.csv", index=False)


def plot_loss_history(losses, save_dir):
    figure, ax = plt.subplots(figsize=(7.0, 4.0))
    # Shape (n_epochs, n_components): the interface term has two components.
    component_names = {"wall": ("wall, temperature jump", "wall, flux jump")}
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
    path = save_dir / "loss_history.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="stable_K_unknown",
        help="subfolder of output_inc_nh3_transient_global",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="draw the collocation points once instead of at every epoch",
    )
    args = parser.parse_args()
    resample = not args.no_resample
    variant = "" if resample else "_noresample"

    save_dir = Path("output_inc_nh3_transient_global") / args.out
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
        tqdm_desc="transient, stable case, potential as unknown",
        resample=resample,
    )
    training_time = timeit.default_timer() - start
    print("Best loss:", pinn.best_loss)
    print(f"Training time: {training_time:.1f} s")
    pinn.save(
        "inconel_nh3_transient",
        postfix=f"global_K_unknown_stable_seed{args.seed}{variant}",
    )

    plot_loss_history(pinn.losses, save_dir)
    plot_snapshots(pinn.space, save_dir)

    run_summary = {
        "case": "stable",
        "unknown": "Kirchhoff potential",
        "t_end_s": T_END,
        "seed": args.seed,
        "epochs": args.epochs,
        "resample_every_epoch": resample,
        "n_parameters": int(
            sum(leaf.size for leaf in jax.tree_util.tree_leaves(space.models))
        ),
        "training_time_s": training_time,
        "best_loss": {
            name: np.asarray(value).ravel().tolist()
            for name, value in pinn.best_loss.items()
        },
    }
    reference = None
    if not CRYSMAS_FILE.exists():
        print(f"CrysMAS file not found: {CRYSMAS_FILE}: comparison skipped.")
    else:
        reference = read_crysmas(CRYSMAS_FILE)
        T_pinn, summary, run_summary["crysmas"] = compare_with_crysmas(
            pinn.space, save_dir, reference
        )
        plot_crysmas_error_history(summary, save_dir)
        plot_crysmas_maps(save_dir, reference, T_pinn)
        plot_line_comparisons(save_dir, reference, T_pinn)
        # The field on the CrysMAS nodes and times, to compare runs with each other later.
        np.savez_compressed(
            save_dir / "field_on_reference_grid.npz",
            times=reference[0],
            nodes=reference[1],
            T=T_pinn,
        )
    run_summary["probes"] = plot_probe_histories(pinn.space, save_dir, reference)

    (save_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(f"Saved {(save_dir / 'run_summary.json').resolve()}")
