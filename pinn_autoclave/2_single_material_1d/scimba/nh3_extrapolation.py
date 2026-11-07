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
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import InteriorResidual


jax.config.update("jax_enable_x64", True)


# Constants

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "NH3_Data.csv"
SAVE_DIR = Path("stage3_outputs_scimba_nh3_extrap")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

L = 0.32

PHYS_MIN = 300.0
PHYS_MAX = 1500.0

# Inner training box for extrapolation
TL_TRAIN_MIN = 500.0
TL_TRAIN_MAX = 900.0
TR_TRAIN_MIN = 500.0
TR_TRAIN_MAX = 900.0

N_COLLOC = 8000
N_EPOCHS = 1000
LEARNING_RATE = 1e-4

T_REF = 0.5 * (TL_TRAIN_MIN + TL_TRAIN_MAX)
DELTA_T_REF = max(TL_TRAIN_MAX - TL_TRAIN_MIN, TR_TRAIN_MAX - TR_TRAIN_MIN)
K_REF = float(
    70.63350679
    - 0.19209961 * T_REF
    + 0.00020083 * T_REF * T_REF
    - 0.00000007 * T_REF * T_REF * T_REF
)


# Reference data

df = pd.read_csv(CSV_PATH).copy()
df = df.sort_values(["TL", "TR", "y"])


# Physical definitions

def conductivity(T: jnp.ndarray) -> jnp.ndarray:
    return (
        70.63350679
        - 0.19209961 * T
        + 0.00020083 * T * T
        - 0.00000007 * T * T * T
    )


def boundary_profile(x: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    y = x[..., 0:1]
    TL = mu[..., 0:1]
    TR = mu[..., 1:2]
    return ((L - y) / L) * TL + (y / L) * TR


def post_processing(
    approx: jnp.ndarray, x: jnp.ndarray, mu: jnp.ndarray
) -> jnp.ndarray:
    y = x[..., 0:1]
    return boundary_profile(x, mu) + T_REF * approx * y * (L - y)


K_REF = float(
    70.63350679
    - 0.19209961 * T_REF
    + 0.00020083 * T_REF * T_REF
    - 0.00000007 * T_REF * T_REF * T_REF
)


# Residual

class NH3ExtrapolationResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, T, precomputed=None):
        dT_dy = T.partial_derivative("x", 0)
        flux = conductivity(T) * dT_dy

        raw_residual = flux.partial_derivative("x", 0)
        residual_scale = K_REF * DELTA_T_REF / (L**2)
        return raw_residual/residual_scale


# Physical model

class NH3ExtrapolationModel(AbstractPhysicalModel):
    def __init__(self, main_domain):
        super().__init__(main_domain=main_domain)
        self.physical_residuals = {
            "interior": NH3ExtrapolationResidual(main_domain),
        }
        self.data_residuals = {}


# Domains and samplers

domain_x = [(0.0, L)]
domain_mu = [
    (TL_TRAIN_MIN, TL_TRAIN_MAX),
    (TR_TRAIN_MIN, TR_TRAIN_MAX),
]

dx = Segment1D(domain_x[0], is_main_domain=True)

sampler = TensorizedSampler(
    [
        DomainSampler(dx),
        UniformParametricSampler(domain_mu),
    ],
    model_type="x_mu",
    bc=False,
    ic=False,
)


# Model, space, projector

model = NH3ExtrapolationModel(dx)

key = jax.random.PRNGKey(0)
nn = MLP(in_size=3, out_size=1, hidden_sizes=[16, 16], key=key)

space = ApproximationSpace(
    {"x": 1, "mu": 2},
    [(nn, "scalar", None)],
    model_type="x_mu",
    post_processing=post_processing,
)

print("\n\n")
print("@@@@@@@@@@@@@@@ train with Adam @@@@@@@@@@@@@@@@@@@@@@")

pinn = Projector(
    model,
    space,
    sampler,
    # optimizer="Adam",
)

key, sample_dict = sampler.sample(key, N_COLLOC)
loss0 = pinn.evaluate_loss(space, sample_dict)

print("initial loss:", loss0)

start = timeit.default_timer()
key, pinn = pinn.project(
    key,
    space,
    N_EPOCHS,
    N_COLLOC,
    tqdm_desc="NH3 extrapolation scaled residual",
)
end = timeit.default_timer()

print("best loss:", pinn.best_loss)
print("time for %d epochs:" % N_EPOCHS, end - start)


# Evaluation on one extrapolation pair

TL_TEST = 300.0
TR_TEST = 1200.0

assert PHYS_MIN <= TL_TEST <= PHYS_MAX
assert PHYS_MIN <= TR_TEST <= PHYS_MAX
assert not (
    TL_TRAIN_MIN <= TL_TEST <= TL_TRAIN_MAX
    and TR_TRAIN_MIN <= TR_TEST <= TR_TRAIN_MAX
), "Chosen test pair is inside the training box, so it is not extrapolation."

case_df = df[(df["TL"] == TL_TEST) & (df["TR"] == TR_TEST)].copy()
case_df = case_df.sort_values("y")
assert len(case_df) > 0, f"No rows found for TL={TL_TEST}, TR={TR_TEST}"

y_fvm = case_df["y"].values.astype(np.float64).reshape(-1, 1)
T_fvm = case_df["Temperature"].values.astype(np.float64).reshape(-1, 1)

x_eval = jnp.array(y_fvm)
mu_eval = jnp.array(
    np.hstack(
        [
            np.full_like(y_fvm, TL_TEST),
            np.full_like(y_fvm, TR_TEST),
        ]
    )
)

T_pred = pinn.evaluate(x_eval, mu_eval)
T_pred = np.array(T_pred).reshape(-1, 1)

abs_err = np.abs(T_pred - T_fvm)
rmse = float(np.sqrt(np.mean((T_pred - T_fvm) ** 2)))
mae = float(np.mean(abs_err))
rel_l2 = float(np.linalg.norm(T_pred - T_fvm) / np.linalg.norm(T_fvm))

print("\nExtrapolation test metrics:")
print(f"TL     = {TL_TEST:.1f} K")
print(f"TR     = {TR_TEST:.1f} K")
print(f"RMSE   = {rmse:.6f}")
print(f"MAE    = {mae:.6f}")
print(f"Rel L2 = {rel_l2:.6e}")


# Plots

plt.figure(figsize=(7, 5))
plt.plot(y_fvm, T_fvm, "k-", lw=2, label="FVM")
plt.plot(y_fvm, T_pred, "r--", lw=2, label="ScimBa PINN")
plt.xlabel("y [m]")
plt.ylabel("Temperature [K]")
plt.title(f"Extrapolation: PINN vs FVM (TL={TL_TEST:.0f} K, TR={TR_TEST:.0f} K)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "extrap_pinn_vs_fvm.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(y_fvm, abs_err, "b-", lw=2)
plt.xlabel("y [m]")
plt.ylabel("Absolute error [K]")
plt.title("Absolute error: |T_PINN - T_FVM|")
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "extrap_absolute_error.png", dpi=300, bbox_inches="tight")
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
plt.title("Training history")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "extrap_training_history.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"\nPlots saved in: {SAVE_DIR.resolve()}")
