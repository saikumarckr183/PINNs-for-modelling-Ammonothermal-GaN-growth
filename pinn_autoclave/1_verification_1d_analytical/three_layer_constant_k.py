import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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

SAVE_DIR = Path("outputs_dd_analytical_constants")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Geometry
xL = 0.0
x1 = 0.10
x2 = 0.22
xR = 0.32

dx1 = x1 - xL
dx2 = x2 - x1
dx3 = xR - x2

# Directly chosen constant conductivities for the analytical benchmark
K1 = 6.9995
K2 = 70.63350679
K3 = 6.9995

# Fixed boundary-condition case
TL_FIXED = 300.0
TR_FIXED = 1500.0
FIXED_CASES = [(TL_FIXED, TR_FIXED)]
PARAM_DOMAIN_FIXED = [(TL_FIXED, TL_FIXED), (TR_FIXED, TR_FIXED)]

# Sampling / training
N_COLLOC = {
    "left": 1000,
    "mid": 1000,
    "right": 1000,
    "if_x1": 1000,
    "if_x2": 1000,
}

N_EPOCHS_ADAM = 1000
N_EPOCHS_SSBFGS = 100
N_EPOCHS_ENG = 100
LEARNING_RATE_ADAM = 1e-4

W_COMMON = {
    "left": (1.0,),
    "mid": (1.0,),
    "right": (1.0,),
    "if_x1": (10.0,),
    "if_x2": (10.0,),
}

DELTA_T_REF = max(abs(TR_FIXED - TL_FIXED), 1.0)
K_REF = max(K1, K2, K3)
FLUX_REF = K_REF * DELTA_T_REF / (xR - xL)

# Domains
domain_main = Segment1D((xL, xR), is_main_domain=True, label_str="main")
domain_left = Segment1D((xL, x1), is_main_domain=False, label_str="left")
domain_mid = Segment1D((x1, x2), is_main_domain=False, label_str= "mid")
domain_right = Segment1D((x2, xR), is_main_domain=False, label_str="right")

EPS_IF = 1e-6
domain_if_x1 = Segment1D((x1 - EPS_IF, x1 + EPS_IF), is_main_domain=False, label_str="if_x1")
domain_if_x2 = Segment1D((x2 - EPS_IF, x2 + EPS_IF), is_main_domain=False, label_str="if_x2")

domain_main.add_subdomain(domain_left)
domain_main.add_subdomain(domain_mid)
domain_main.add_subdomain(domain_right)
domain_main.add_subdomain(domain_if_x1)
domain_main.add_subdomain(domain_if_x2)

#  combines the spatial input and parameter input into one vector that the neural network can consume
def _field_input(x, mu):
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)

    if x.ndim == 2 and mu.ndim == 2:
        return jnp.concatenate([x, mu], axis=-1), "batched"
    if x.ndim == 1 and mu.ndim == 1:
        return jnp.concatenate([x, mu], axis=0), "single"
    if x.ndim == 2 and x.shape[0] == 1 and mu.ndim == 1:
        return jnp.concatenate([x[0], mu], axis=0), "single"
    if x.ndim == 1 and mu.ndim == 2 and mu.shape[0] == 1:
        return jnp.concatenate([x, mu[0]], axis=0), "single"
    raise ValueError(f"Unsupported x/mu shapes: x={x.shape}, mu={mu.shape}")


# Exact interface temperatures from Eq. (1.9)
def exact_interface_temperatures(mu):
    mu = jnp.asarray(mu, dtype=jnp.float64)

    if mu.ndim == 1:
        tl = mu[0:1]
        tr = mu[1:2]
    elif mu.ndim == 2:
        tl = mu[:, 0:1]
        tr = mu[:, 1:2]
    else:
        raise ValueError(f"Unsupported mu shape: {mu.shape}")

    denom = dx1 * K2 * K3 + dx2 * K1 * K3 + dx3 * K1 * K2
    t1 = (dx1 * K2 * K3 * tr + dx2 * K1 * K3 * tl + dx3 * K1 * K2 * tl) / denom
    t2 = (dx1 * K2 * K3 * tr + dx2 * K1 * K3 * tr + dx3 * K1 * K2 * tl) / denom
    return t1, t2


# Strong output transforms: T1 and T2 are enforced analytically
def T_left_eval(space, x, mu):
    inp, mode = _field_input(x, mu)
    raw = jax.vmap(space.models[0])(inp) if mode == "batched" else space.models[0](inp)

    T1_exact, _ = exact_interface_temperatures(mu)

    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    xx = x[..., 0:1] if x.ndim == 2 else x[0:1]
    tl = mu[..., 0:1] if mu.ndim == 2 else mu[0:1]

    tlin = ((x1 - xx) / dx1) * tl + ((xx - xL) / dx1) * T1_exact
    bubble = (xx - xL) * (x1 - xx)
    return tlin + bubble * jnp.tanh(raw)


def T_mid_eval(space, x, mu):
    inp, mode = _field_input(x, mu)
    raw = jax.vmap(space.models[1])(inp) if mode == "batched" else space.models[1](inp)

    T1_exact, T2_exact = exact_interface_temperatures(mu)

    x = jnp.asarray(x, dtype=jnp.float64)
    xx = x[..., 0:1] if x.ndim == 2 else x[0:1]

    tlin = ((x2 - xx) / dx2) * T1_exact + ((xx - x1) / dx2) * T2_exact
    bubble = (xx - x1) * (x2 - xx)
    return tlin + bubble * jnp.tanh(raw)


def T_right_eval(space, x, mu):
    inp, mode = _field_input(x, mu)
    raw = jax.vmap(space.models[2])(inp) if mode == "batched" else space.models[2](inp)

    _, T2_exact = exact_interface_temperatures(mu)

    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    xx = x[..., 0:1] if x.ndim == 2 else x[0:1]
    tr = mu[..., 1:2] if mu.ndim == 2 else mu[1:2]

    tlin = ((xR - xx) / dx3) * T2_exact + ((xx - x2) / dx3) * tr
    bubble = (xx - x2) * (xR - xx)
    return tlin + bubble * jnp.tanh(raw)


dims = {"x": 1, "mu": 2}
T_LEFT = ParamScalarFunction(dims, T_left_eval, f_type="x_mu")
T_MID = ParamScalarFunction(dims, T_mid_eval, f_type="x_mu")
T_RIGHT = ParamScalarFunction(dims, T_right_eval, f_type="x_mu")


# Constant conductivity helpers
def k_left_fun(T):
    return ParamScalarFunction(
        T.dims,
        lambda sp, x, mu: jnp.ones_like(T(sp, x, mu)) * K1,
        f_type=T.f_type,
    )


def k_mid_fun(T):
    return ParamScalarFunction(
        T.dims,
        lambda sp, x, mu: jnp.ones_like(T(sp, x, mu)) * K2,
        f_type=T.f_type,
    )


def k_right_fun(T):
    return ParamScalarFunction(
        T.dims,
        lambda sp, x, mu: jnp.ones_like(T(sp, x, mu)) * K3,
        f_type=T.f_type,
    )


class LeftResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        T_left, T_mid, T_right = vars

        dTdx = T_left.partial_derivative("x", 0)
        flux = k_left_fun(T_left) * dTdx
        raw_res = flux.partial_derivative("x", 0)

        scale = K_REF * DELTA_T_REF / (dx1**2)
        return raw_res / scale


class MidResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        T_left, T_mid, T_right = vars

        dTdx = T_mid.partial_derivative("x", 0)
        flux = k_mid_fun(T_mid) * dTdx
        raw_res = flux.partial_derivative("x", 0)

        scale = K_REF * DELTA_T_REF / (dx2**2)
        return raw_res / scale


class RightResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        T_left, T_mid, T_right = vars

        dTdx = T_right.partial_derivative("x", 0)
        flux = k_right_fun(T_right) * dTdx
        raw_res = flux.partial_derivative("x", 0)

        scale = K_REF * DELTA_T_REF / (dx3**2)
        return raw_res / scale

# Only flux continuity is imposed at interfaces because temperature continuity is already built in
class InterfaceX1Residual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        T_left, T_mid, T_right = vars

        q_left = k_left_fun(T_left) * T_left.partial_derivative("x", 0)
        q_mid = k_mid_fun(T_mid) * T_mid.partial_derivative("x", 0)

        def residual(space, x, mu):
            x_if = jnp.zeros_like(x) + x1
            return (q_left(space, x_if, mu) - q_mid(space, x_if, mu)) / FLUX_REF

        return ParamScalarFunction(T_left.dims, residual, f_type=T_left.f_type)


class InterfaceX2Residual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        T_left, T_mid, T_right = vars

        q_mid = k_mid_fun(T_mid) * T_mid.partial_derivative("x", 0)
        q_right = k_right_fun(T_right) * T_right.partial_derivative("x", 0)

        def residual(space, x, mu):
            x_if = jnp.zeros_like(x) + x2
            return (q_mid(space, x_if, mu) - q_right(space, x_if, mu)) / FLUX_REF

        return ParamScalarFunction(T_mid.dims, residual, f_type=T_mid.f_type)


class CoupledAnalyticalModel(AbstractPhysicalModel):
    def __init__(self, main_domain):
        super().__init__(main_domain=main_domain)
        self.physical_residuals = {
            "left": LeftResidual(domain_left),
            "mid": MidResidual(domain_mid),
            "right": RightResidual(domain_right),
            "if_x1": InterfaceX1Residual(domain_if_x1),
            "if_x2": InterfaceX2Residual(domain_if_x2),
        }
        self.data_residuals = {}


model = CoupledAnalyticalModel(domain_main)

domain_sampler = DomainSampler(domain_main, main_domain_fully_covered=True)
param_sampler = UniformParametricSampler(PARAM_DOMAIN_FIXED)

sampler = TensorizedSampler(
    [domain_sampler, param_sampler],
    model_type="x_mu",
    bc=False,
    ic=False,
)


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


def build_space(seed):
    key = jax.random.PRNGKey(seed)
    k1_key, k2_key, k3_key = jax.random.split(key, 3)

    nn_left = build_mlp(3, 1, [16, 16], k1_key)
    nn_mid = build_mlp(3, 1, [32, 32, 32], k2_key)
    nn_right = build_mlp(3, 1, [16, 16], k3_key)

    return ApproximationSpace(
        {"x": 1, "mu": 2},
        [
            (nn_left, "scalar", None),
            (nn_mid, "scalar", None),
            (nn_right, "scalar", None),
        ],
        model_type="x_mu",
    )


def exact_temperature_numpy(x, tl, tr):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    denom = dx1 * K2 * K3 + dx2 * K1 * K3 + dx3 * K1 * K2
    t1 = (dx1 * K2 * K3 * tr + dx2 * K1 * K3 * tl + dx3 * K1 * K2 * tl) / denom
    t2 = (dx1 * K2 * K3 * tr + dx2 * K1 * K3 * tr + dx3 * K1 * K2 * tl) / denom

    out = np.empty_like(x)
    m1 = x <= x1
    m2 = (x > x1) & (x <= x2)
    m3 = x > x2

    out[m1] = ((t1 - tl) / dx1) * (x[m1] - xL) + tl
    out[m2] = ((t2 - t1) / dx2) * (x[m2] - x1) + t1
    out[m3] = ((tr - t2) / dx3) * (x[m3] - x2) + t2
    return out.reshape(-1, 1), t1, t2


def eval_temperature(space, x_scalar, mu_row):
    x = jnp.asarray([float(x_scalar)], dtype=jnp.float64)
    mu = jnp.asarray(mu_row, dtype=jnp.float64).reshape(-1)

    if x_scalar <= x1:
        return float(jnp.asarray(T_LEFT(space, x, mu)).reshape(-1)[0])
    if x_scalar <= x2:
        return float(jnp.asarray(T_MID(space, x, mu)).reshape(-1)[0])
    return float(jnp.asarray(T_RIGHT(space, x, mu)).reshape(-1)[0])


def evaluate_coupled(space, x_eval, mu_eval):
    x_eval = np.asarray(x_eval).reshape(-1)
    mu_eval = np.asarray(mu_eval).reshape(-1, 2)
    out = np.zeros((x_eval.shape[0], 1), dtype=np.float64)
    for i in range(x_eval.shape[0]):
        out[i, 0] = eval_temperature(space, float(x_eval[i]), mu_eval[i])
    return out


def make_mu_eval(n, tl, tr):
    return jnp.column_stack(
        [
            jnp.full((n,), tl, dtype=jnp.float64),
            jnp.full((n,), tr, dtype=jnp.float64),
        ]
    )


def diagnostic_dump(space, tl, tr):
    x_test = np.array([xL, 0.05, x1, 0.15, x2, 0.27, xR], dtype=np.float64)
    mu_row = np.array([tl, tr], dtype=np.float64)
    _, t1_ex, t2_ex = exact_temperature_numpy(np.array([xL, x1, x2, xR]), tl, tr)

    print("\n=== Diagnostic dump ===")
    print(f"Exact interface temps from Eq. (1.9): T1={t1_ex:.6f}, T2={t2_ex:.6f}")
    print("x | region | T_pred | T_exact")

    for x in x_test:
        region = "left" if x <= x1 else ("mid" if x <= x2 else "right")
        t_pred = eval_temperature(space, x, mu_row)
        t_ref = float(exact_temperature_numpy(np.array([x]), tl, tr)[0].reshape(-1)[0])
        print(f"{x:8.5f} | {region:5s} | {t_pred:12.6f} | {t_ref:12.6f}")


def train_optimizer(optimizer_name, seed, n_epochs, learning_rate=None):
    print("\n\n")
    print(f"@@@@@@@@@@@@@@@ train with {optimizer_name} @@@@@@@@@@@@@@@@@@@@@@")

    space = build_space(seed)
    kwargs = {"optimizer": optimizer_name, "weights": W_COMMON}
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate

    pinn = Projector(model, space, sampler, **kwargs)

    key = jax.random.PRNGKey(seed)
    key, sample_dict = sampler.sample(key, N_COLLOC)
    print("sample_dict keys:", sample_dict.keys())
    losses0 = pinn.evaluate_losses(space, sample_dict)
    print("initial losses:", losses0)

    start = timeit.default_timer()
    key, pinn = pinn.project(
        key,
        space,
        n_epochs,
        N_COLLOC,
        tqdm_desc=f"Training {optimizer_name}",
    )
    end = timeit.default_timer()

    print("best loss:", pinn.best_loss)
    print(f"time for {n_epochs} epochs:", end - start)
    return pinn


def evaluate_and_plot(pinn, optimizer_name, tl, tr):
    diagnostic_dump(pinn.space, tl, tr)

    x_ref = np.linspace(xL, xR, 1000).reshape(-1, 1)
    T_exact, t1_exact, t2_exact = exact_temperature_numpy(x_ref, tl, tr)

    mu_eval = make_mu_eval(len(x_ref), tl, tr)
    T_pred = evaluate_coupled(pinn.space, x_ref, mu_eval)

    abs_err = np.abs(T_pred - T_exact)
    rmse = float(np.sqrt(np.mean((T_pred - T_exact) ** 2)))
    mae = float(np.mean(abs_err))
    rel_l2 = float(np.linalg.norm(T_pred - T_exact) / np.linalg.norm(T_exact))

    print("\nAnalytical-case test metrics:")
    print(f"TL         = {tl:.1f} K")
    print(f"TR         = {tr:.1f} K")
    print(f"K1         = {K1:.6f}")
    print(f"K2         = {K2:.6f}")
    print(f"K3         = {K3:.6f}")
    print(f"T1 exact   = {t1_exact:.6f}")
    print(f"T2 exact   = {t2_exact:.6f}")
    print(f"RMSE       = {rmse:.6f}")
    print(f"MAE        = {mae:.6f}")
    print(f"Rel L2     = {rel_l2:.6e}")

    case_tag = f"{optimizer_name.lower().replace('-', '_')}_TL_{int(tl)}_TR_{int(tr)}"

    plt.figure(figsize=(7, 5))
    plt.plot(x_ref, T_exact, "k-", lw=2, label="Exact")
    plt.plot(x_ref, T_pred, "r--", lw=2, label=optimizer_name)
    plt.axvline(x1, color="gray", ls="--", lw=1.2, label="interface")
    plt.axvline(x2, color="gray", ls="--", lw=1.2)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title(f"{optimizer_name}: DD PINN vs Analytical Sol")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{case_tag}_pinn_vs_Analytical_Sol.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x_ref, 100.0 * abs_err / np.maximum(np.abs(T_exact), 1e-12), "g-", lw=2)
    plt.axvline(x1, color="gray", ls="--", lw=1.2)
    plt.axvline(x2, color="gray", ls="--", lw=1.2)
    plt.xlabel("x [m]")
    plt.ylabel("Relative error [%]")
    plt.title(f"{optimizer_name}: Relative error vs Analytical Sol")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{case_tag}_relative_error.png", dpi=300, bbox_inches="tight")
    plt.close()

    k_vis = np.where(x_ref < x1, K1, np.where(x_ref < x2, K2, K3))
    plt.figure(figsize=(7, 4))
    plt.plot(x_ref, k_vis, "m-", lw=2, label="piecewise constant k(x)")
    plt.axvline(x1, color="gray", ls="--", lw=1.2, label="interface")
    plt.axvline(x2, color="gray", ls="--", lw=1.2)
    plt.xlabel("x [m]")
    plt.ylabel("Conductivity [W/mK]")
    plt.title(f"{optimizer_name}: Analytical conductivity profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{case_tag}_conductivity.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "optimizer": optimizer_name,
        "TL": tl,
        "TR": tr,
        "K1": K1,
        "K2": K2,
        "K3": K3,
        "T1_exact": t1_exact,
        "T2_exact": t2_exact,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
    }


if __name__ == "__main__":
    results = []

    pinn_adam = train_optimizer(
        "Adam",
        seed=42,
        n_epochs=N_EPOCHS_ADAM,
        learning_rate=LEARNING_RATE_ADAM,
    )
    for tl, tr in FIXED_CASES:
        results.append(evaluate_and_plot(pinn_adam, "Adam", tl, tr))

    pinn_ssbfgs = train_optimizer(
        "SS-BFGS",
        seed=43,
        n_epochs=N_EPOCHS_SSBFGS,
        learning_rate=None,
    )
    for tl, tr in FIXED_CASES:
        results.append(evaluate_and_plot(pinn_ssbfgs, "SS-BFGS", tl, tr))

    pinn_eng = train_optimizer(
        "ENG",
        seed=44,
        n_epochs=N_EPOCHS_ENG,
        learning_rate=None,
    )
    for tl, tr in FIXED_CASES:
        results.append(evaluate_and_plot(pinn_eng, "ENG", tl, tr))

    print("\nSummary of clean DD analytical runs:")
    for res in results:
        print(
            f"{res['optimizer']:>8s} | "
            f"TL = {res['TL']:.1f} | "
            f"TR = {res['TR']:.1f} | "
            f"K1 = {res['K1']:.6f} | "
            f"K2 = {res['K2']:.6f} | "
            f"K3 = {res['K3']:.6f} | "
            f"T1_ex = {res['T1_exact']:.6f} | "
            f"T2_ex = {res['T2_exact']:.6f} | "
            f"RMSE = {res['rmse']:.6f} | "
            f"MAE = {res['mae']:.6f} | "
            f"Rel L2 = {res['rel_l2']:.6e}"
        )

    print(f"\nChosen constant conductivities: K1={K1:.6f}, K2={K2:.6f}, K3={K3:.6f}")
    print(f"Plots saved in: {SAVE_DIR.resolve()}")
