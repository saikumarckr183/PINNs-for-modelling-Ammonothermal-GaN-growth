import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scimba_jax.domains.meshless_domains.domains_1d import Segment1D
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

# Constants

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "Mixed_Materials_Data.csv"
SAVE_DIR = Path("stage2_outputs_scimba_interpolation_piecewise_smoothed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

xL = 0.0
x1 = 0.10
x2 = 0.22
xR = 0.32

TL_MIN = 300.0
TL_MAX = 1500.0
TR_MIN = 300.0
TR_MAX = 1500.0

PARAM_DOMAIN = [(TL_MIN, TL_MAX), (TR_MIN, TR_MAX)]

N_COLLOC = 4000

N_EPOCHS_ADAM = 000
N_EPOCHS_SSBFGS = 000
N_EPOCHS_ENG = 1000

LEARNING_RATE = 1e-4
BETA_SMOOTH = 100.0

T_REF = 0.5 * ((TL_MIN + TL_MAX) / 2.0 + (TR_MIN + TR_MAX) / 2.0)
DELTA_T_REF = max(abs(TR_MAX - TL_MIN), abs(TR_MIN - TL_MAX))

K_INCONEL_REF = 6.9995 + 0.0151 * T_REF
K_NH3_REF = (
    70.63350679
    - 0.19209961 * T_REF
    + 0.00020083 * T_REF**2
    - 0.00000007 * T_REF**3
)
K_REF = max(K_INCONEL_REF, K_NH3_REF)

TEST_CASES = [
    (420.0, 900.0),
]

# Reference data

df = pd.read_csv(CSV_PATH).copy()
df = df.sort_values(["TL", "TR", "y"])

# Material laws

def k_inconel_num(temp):
    return 6.9995 + 0.0151 * temp


def k_nh3_num(temp):
    return (
        70.63350679
        - 0.19209961 * temp
        + 0.00020083 * temp**2
        - 0.00000007 * temp**3
    )


def smooth_heaviside_np(z, beta=BETA_SMOOTH):
    return 0.5 * (1.0 + np.tanh(beta * z))


def smooth_heaviside_jax(z, beta=BETA_SMOOTH):
    return 0.5 * (1.0 + jnp.tanh(beta * z))


def k_piecewise_smooth_num(x, temp, beta=BETA_SMOOTH):
    x = np.asarray(x)

    k_inc = k_inconel_num(temp)
    k_nh3 = k_nh3_num(temp)

    s1 = smooth_heaviside_np(x - x1, beta)
    s2 = smooth_heaviside_np(x - x2, beta)

    w_left = 1.0 - s1
    w_mid = s1 * (1.0 - s2)
    w_right = s2

    return w_left * k_inc + w_mid * k_nh3 + w_right * k_inc

# Post-processing

def boundary_profile(xx, tl, tr):
    return ((xR - xx) / (xR - xL)) * tl + ((xx - xL) / (xR - xL)) * tr


def post_processing_parametric(approx, x, mu):
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)

    if x.ndim == 0:
        xx = x[None]
    elif x.ndim == 1:
        xx = x
    else:
        xx = x[..., 0:1]

    if mu.ndim == 1:
        tl = mu[0:1]
        tr = mu[1:2]
    else:
        tl = mu[..., 0:1]
        tr = mu[..., 1:2]

    return boundary_profile(xx, tl, tr) + T_REF * approx * (xx - xL) * (xR - xx)

# Residual

class MixedMaterialsInterpolationResidual(InteriorResidual):
    def __init__(self, domain, beta=100.0):
        super().__init__(domain=domain, size=1, model_type="x_mu")
        self.beta = beta

    def construct_residual(self, T, precomputed=None):
        dT_dx = T.partial_derivative("x", 0)

        def x_coord(space, x, mu):
            x = jnp.asarray(x)
            if x.ndim == 0:
                return x[None]
            if x.ndim == 1:
                return x
            return x[..., 0]

        def k_smooth(space, x, mu):
            xx = x_coord(space, x, mu)
            temp = T(space, x, mu)

            k_inc = 6.9995 + 0.0151 * temp
            k_nh3 = (
                70.63350679
                - 0.19209961 * temp
                + 0.00020083 * temp**2
                - 0.00000007 * temp**3
            )

            s1 = smooth_heaviside_jax(xx - x1, self.beta)
            s2 = smooth_heaviside_jax(xx - x2, self.beta)

            w_left = 1.0 - s1
            w_mid = s1 * (1.0 - s2)
            w_right = s2

            return w_left * k_inc + w_mid * k_nh3 + w_right * k_inc

        k_eff = ParamScalarFunction(
            T.dims,
            k_smooth,
            f_type=T.f_type,
        )

        flux = k_eff * dT_dx
        raw_residual = flux.partial_derivative("x", 0)

        residual_scale = K_REF * max(DELTA_T_REF, 1.0) / ((xR - xL) ** 2)
        return raw_residual / residual_scale

# Physical model

class MixedMaterialsInterpolationModel(AbstractPhysicalModel):
    def __init__(self, main_domain, beta=100.0):
        super().__init__(main_domain=main_domain)
        self.physical_residuals = {
            "interior": MixedMaterialsInterpolationResidual(main_domain, beta=beta),
        }
        self.data_residuals = {}

# Domain / sampler / model

domain_x = Segment1D((xL, xR), is_main_domain=True)

domain_sampler = DomainSampler(domain_x)
param_sampler = UniformParametricSampler(PARAM_DOMAIN)

sampler = TensorizedSampler(
    [domain_sampler, param_sampler],
    model_type="x_mu",
    bc=False,
    ic=False,
)

model = MixedMaterialsInterpolationModel(domain_x, beta=BETA_SMOOTH)

# Cases

def get_case_df(tl, tr):
    case_df = df[(df["TL"] == tl) & (df["TR"] == tr)].copy()
    case_df = case_df.sort_values("y")
    assert len(case_df) > 0, f"No rows found for TL={tl}, TR={tr}"
    return case_df


def make_mu_eval(n, tl, tr):
    return jnp.column_stack(
        [
            jnp.full((n,), tl, dtype=jnp.float64),
            jnp.full((n,), tr, dtype=jnp.float64),
        ]
    )

# Evaluation

def evaluate_and_plot(pinn, optimizer_name, tl, tr):
    case_df = get_case_df(tl, tr)

    x_fvm = case_df["y"].values.astype(np.float64).reshape(-1, 1)
    T_fvm = case_df["Temperature"].values.astype(np.float64).reshape(-1, 1)

    x_eval = jnp.array(x_fvm)
    mu_eval = make_mu_eval(len(x_fvm), tl, tr)

    T_pred = pinn.evaluate(x_eval, mu_eval)
    T_pred = np.array(T_pred).reshape(-1, 1)

    abs_err = np.abs(T_pred - T_fvm)
    rmse = float(np.sqrt(np.mean((T_pred - T_fvm) ** 2)))
    mae = float(np.mean(abs_err))
    rel_l2 = float(np.linalg.norm(T_pred - T_fvm) / np.linalg.norm(T_fvm))

    print("\nInterpolation test metrics:")
    print(f"TL     = {tl:.1f} K")
    print(f"TR     = {tr:.1f} K")
    print(f"RMSE   = {rmse:.6f}")
    print(f"MAE    = {mae:.6f}")
    print(f"Rel L2 = {rel_l2:.6e}")

    x_vis = np.linspace(xL, xR, 1000).reshape(-1, 1)
    mu_vis = make_mu_eval(len(x_vis), tl, tr)
    T_vis = np.array(pinn.evaluate(jnp.array(x_vis), mu_vis)).reshape(-1, 1)
    k_vis = k_piecewise_smooth_num(x_vis, T_vis, beta=BETA_SMOOTH)

    case_tag = f"{optimizer_name.lower().replace('-', '_')}_TL_{int(tl)}_TR_{int(tr)}"

    plt.figure(figsize=(7, 5))
    plt.plot(x_fvm, T_fvm, "k-", lw=2, label="FVM")
    plt.plot(x_fvm, T_pred, "r--", lw=2, label=f"ScimBa PINN ({optimizer_name})")
    plt.axvline(x1, color="gray", ls="--", lw=1.2, label="interface")
    plt.axvline(x2, color="gray", ls="--", lw=1.2)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title(f"{optimizer_name}: interpolation PINN vs FVM (TL={tl:.0f}, TR={tr:.0f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{case_tag}_pinn_vs_fvm.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    rel_err_percent = 100.0 * abs_err / np.maximum(np.abs(T_fvm), 1e-12)

    plt.figure(figsize=(7, 4))
    plt.plot(x_fvm, rel_err_percent, "g-", lw=2)
    plt.axvline(x1, color="gray", ls="--", lw=1.2)
    plt.axvline(x2, color="gray", ls="--", lw=1.2)
    plt.xlabel("x [m]")
    plt.ylabel("Relative error [%]")
    plt.title(f"{optimizer_name}: Relative error 100*|T_PINN - T_FVM|/|T_FVM|")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        SAVE_DIR / f"{case_tag}_relative_error.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x_vis, k_vis, "m-", lw=2, label="k_smooth(x, T_PINN)")
    plt.axvline(x1, color="gray", ls="--", lw=1.2, label="interface")
    plt.axvline(x2, color="gray", ls="--", lw=1.2)
    plt.xlabel("x [m]")
    plt.ylabel("Effective conductivity [W/mK]")
    plt.title(f"{optimizer_name}: Smoothed conductivity profile (TL={tl:.0f}, TR={tr:.0f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{case_tag}_smoothed_conductivity.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 4))
    for label, hist in pinn.losses.losses_history.items():
        hist_np = np.array(hist)
        if hist_np.ndim == 1:
            plt.semilogy(hist_np, label=label)
        else:
            for i in range(hist_np.shape[1]):
                plt.semilogy(hist_np[:, i], label=f"{label}_{i}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{optimizer_name}: Training history (TL={tl:.0f}, TR={tr:.0f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{case_tag}_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return {
        "optimizer": optimizer_name,
        "TL": tl,
        "TR": tr,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
    }

# Training

results = []

# Adam
print("\n\n")
print("@@@@@@@@@@@@@@@ train with Adam @@@@@@@@@@@@@@@@@@@@@@")

key = jax.random.PRNGKey(42)
nn1 = MLP(in_size=3, out_size=1, hidden_sizes=[64, 64], key=key)

space1 = ApproximationSpace(
    {"x": 1, "mu": 2},
    [(nn1, "scalar", None)],
    model_type="x_mu",
    post_processing=post_processing_parametric,
)

pinn1 = Projector(
    model,
    space1,
    sampler,
    optimizer="Adam",
    learning_rate=LEARNING_RATE,
)

key, sample_dict = sampler.sample(key, N_COLLOC)
loss0 = pinn1.evaluate_loss(space1, sample_dict)
print("initial loss:", loss0)

start = timeit.default_timer()
key, pinn1 = pinn1.project(
    key,
    space1,
    N_EPOCHS_ADAM,
    N_COLLOC,
    tqdm_desc="Mixed-material interpolation [Adam]",
)
end = timeit.default_timer()

print("best loss:", pinn1.best_loss)
print("time for %d epochs:" % N_EPOCHS_ADAM, end - start)

for tl, tr in TEST_CASES:
    results.append(evaluate_and_plot(pinn1, "Adam", tl, tr))

# SS-BFGS
print("\n\n")
print("@@@@@@@@@@@@@@@ train with SS-BFGS @@@@@@@@@@@@@@@@@@@@@@")

key = jax.random.PRNGKey(43)
nn2 = MLP(in_size=3, out_size=1, hidden_sizes=[16, 16], key=key)

space2 = ApproximationSpace(
    {"x": 1, "mu": 2},
    [(nn2, "scalar", None)],
    model_type="x_mu",
    post_processing=post_processing_parametric,
)

pinn2 = Projector(
    model,
    space2,
    sampler,
    optimizer="SS-BFGS",
)

key, sample_dict = sampler.sample(key, N_COLLOC)
loss0 = pinn2.evaluate_loss(space2, sample_dict)
print("initial loss:", loss0)

start = timeit.default_timer()
key, pinn2 = pinn2.project(
    key,
    space2,
    N_EPOCHS_SSBFGS,
    N_COLLOC,
    tqdm_desc="Mixed-material interpolation [SS-BFGS]",
)
end = timeit.default_timer()

print("best loss:", pinn2.best_loss)
print("time for %d epochs:" % N_EPOCHS_SSBFGS, end - start)

for tl, tr in TEST_CASES:
    results.append(evaluate_and_plot(pinn2, "SS-BFGS", tl, tr))

# ENG
print("\n\n")
print("@@@@@@@@@@@@@@@ train with ENG @@@@@@@@@@@@@@@@@@@@@@")

key = jax.random.PRNGKey(44)
nn3 = MLP(in_size=3, out_size=1, hidden_sizes=[16, 16], key=key)

space3 = ApproximationSpace(
    {"x": 1, "mu": 2},
    [(nn3, "scalar", None)],
    model_type="x_mu",
    post_processing=post_processing_parametric,
)

pinn3 = Projector(
    model,
    space3,
    sampler,
    optimizer="ENG",
)

key, sample_dict = sampler.sample(key, N_COLLOC)
loss0 = pinn3.evaluate_loss(space3, sample_dict)
print("initial loss:", loss0)

start = timeit.default_timer()
key, pinn3 = pinn3.project(
    key,
    space3,
    N_EPOCHS_ENG,
    N_COLLOC,
    tqdm_desc="Mixed-material interpolation [ENG]",
)
end = timeit.default_timer()

print("best loss:", pinn3.best_loss)
print("time for %d epochs:" % N_EPOCHS_ENG, end - start)

for tl, tr in TEST_CASES:
    results.append(evaluate_and_plot(pinn3, "ENG", tl, tr))

# Summary

print("\nSummary of interpolation runs:")
for res in results:
    print(
        f"{res['optimizer']:>8s} | "
        f"TL = {res['TL']:.1f} | "
        f"TR = {res['TR']:.1f} | "
        f"RMSE = {res['rmse']:.6f} | "
        f"MAE = {res['mae']:.6f} | "
        f"Rel L2 = {res['rel_l2']:.6e}"
    )

print(f"\nPlots saved in: {SAVE_DIR.resolve()}")
