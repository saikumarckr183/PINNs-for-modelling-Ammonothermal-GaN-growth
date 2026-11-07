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
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import InteriorResidual


jax.config.update("jax_enable_x64", True)


# Constants

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "NH3_Data.csv"
SAVE_DIR = Path("stage1_outputs_scimba_nh3_fixed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

L = 0.32

TL = 300.0
TR = 1200.0

N_COLLOC = 4000
N_EPOCHS = 1000
LEARNING_RATE = 1e-4

T_REF = 0.5 * (TL + TR)
DELTA_T_REF = abs(TR - TL)
K_REF = float(
    70.63350679
    - 0.19209961 * T_REF
    + 0.00020083 * T_REF * T_REF
    - 0.00000007 * T_REF * T_REF * T_REF
)


# Reference data

df = pd.read_csv(CSV_PATH).copy()
df = df.sort_values(["TL", "TR", "y"])

case_df = df[(df["TL"] == TL) & (df["TR"] == TR)].copy()
case_df = case_df.sort_values("y")
assert len(case_df) > 0, f"No rows found for TL={TL}, TR={TR}"


# Physical definitions

def conductivity(T: jnp.ndarray) -> jnp.ndarray:
    return (
        70.63350679
        - 0.19209961 * T
        + 0.00020083 * T * T
        - 0.00000007 * T * T * T
    )


def boundary_profile(x: jnp.ndarray) -> jnp.ndarray:
    y = x[..., 0:1]
    return ((L - y) / L) * TL + (y / L) * TR


def post_processing(approx: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    y = x[..., 0:1]
    return boundary_profile(x) + T_REF * approx * y * (L - y)


K_REF = float(
    70.63350679
    - 0.19209961 * T_REF
    + 0.00020083 * T_REF * T_REF
    - 0.00000007 * T_REF * T_REF * T_REF
)


# Residual

class NH3FixedResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x")

    def construct_residual(self, T, precomputed=None):
        dT_dy = T.partial_derivative("x", 0)
        flux = conductivity(T) * dT_dy

        raw_residual = flux.partial_derivative("x", 0)
        residual_scale = K_REF * max(DELTA_T_REF, 1.0) / (L**2)
        return raw_residual/residual_scale


# Physical model

class NH3FixedModel(AbstractPhysicalModel):
    def __init__(self, main_domain):
        super().__init__(main_domain=main_domain)
        self.physical_residuals = {
            "interior": NH3FixedResidual(main_domain),
        }
        self.data_residuals = {}


# Domain and sampler

domain_x = [(0.0, L)]

dx = Segment1D(domain_x[0], is_main_domain=True)

sampler = TensorizedSampler(
    [DomainSampler(dx)],
    model_type="x",
    bc=False,
    ic=False,
)


# Model, space, projector

model = NH3FixedModel(dx)

key = jax.random.PRNGKey(0)
nn = MLP(in_size=1, out_size=1, hidden_sizes=[16, 16], key=key)

space = ApproximationSpace(
    {"x": 1},
    [(nn, "scalar", None)],
    model_type="x",
    post_processing=post_processing,
)

print("\n\n")
print("@@@@@@@@@@@@@@@ train with ENG @@@@@@@@@@@@@@@@@@@@@@")

pinn = Projector(
    model,
    space,
    sampler,
    #  optimizer="Adam",
    learning_rate=LEARNING_RATE,
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
    tqdm_desc="NH3 fixed case scaled residual",
)
end = timeit.default_timer()

print("best loss:", pinn.best_loss)
print("time for %d epochs:" % N_EPOCHS, end - start)


# Evaluation on the fixed case

y_fvm = case_df["y"].values.astype(np.float64).reshape(-1, 1)
T_fvm = case_df["Temperature"].values.astype(np.float64).reshape(-1, 1)

x_eval = jnp.array(y_fvm)

T_pred = pinn.evaluate(x_eval)
T_pred = np.array(T_pred).reshape(-1, 1)

abs_err = np.abs(T_pred - T_fvm)
rmse = float(np.sqrt(np.mean((T_pred - T_fvm) ** 2)))
mae = float(np.mean(abs_err))
rel_l2 = float(np.linalg.norm(T_pred - T_fvm) / np.linalg.norm(T_fvm))

print("\nFixed-case test metrics:")
print(f"TL     = {TL:.1f} K")
print(f"TR     = {TR:.1f} K")
print(f"RMSE   = {rmse:.6f}")
print(f"MAE    = {mae:.6f}")
print(f"Rel L2 = {rel_l2:.6e}")


# Plots

plt.figure(figsize=(7, 5))
plt.plot(y_fvm, T_fvm, "k-", lw=2, label="FVM")
plt.plot(y_fvm, T_pred, "r--", lw=2, label="ScimBa PINN")
plt.xlabel("y [m]")
plt.ylabel("Temperature [K]")
plt.title(f"Fixed case: PINN vs FVM (TL={TL:.0f} K, TR={TR:.0f} K)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "fixed_pinn_vs_fvm.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(y_fvm, abs_err, "b-", lw=2)
plt.xlabel("y [m]")
plt.ylabel("Absolute error [K]")
plt.title("Absolute error: |T_PINN - T_FVM|")
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "fixed_absolute_error.png", dpi=300, bbox_inches="tight")
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
plt.savefig(SAVE_DIR / "fixed_training_history.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"\nPlots saved in: {SAVE_DIR.resolve()}")
