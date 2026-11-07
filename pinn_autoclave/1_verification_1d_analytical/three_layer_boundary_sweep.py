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

SAVE_DIR = Path("stage1_outputs_fixed_dd_analytical_multicase")
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
K1 = 25.0
K2 = 5.0
K3 = 25.0

# Multi-case sweep
CASES = [
    (300.0, 600.0),
    (420.0, 900.0),
    (900.0, 1200.0),
    (1200.0, 1410.0),
    (300.0, 1410.0),
]
PARAM_DOMAIN = [
    (min(tl for tl, _ in CASES), max(tl for tl, _ in CASES)),
    (min(tr for _, tr in CASES), max(tr for _, tr in CASES)),
]

# Sampling / training
N_COLLOC = {
    "left": 4000,
    "mid": 4000,
    "right": 4000,
    "if_x1": 1000,
    "if_x2": 1000,
}

N_EPOCHS_ADAM = 1000
LEARNING_RATE_ADAM = 1e-4

W_COMMON = {
    "left": (1.0,),
    "mid": (1.0,),
    "right": (1.0,),
    "if_x1": (40.0,),
    "if_x2": (40.0,),
}

ALL_T = [v for pair in CASES for v in pair]
DELTA_T_REF = max(max(ALL_T) - min(ALL_T), 1.0)
K_REF = max(K1, K2, K3)
FLUX_REF = K_REF * DELTA_T_REF / (xR - xL)

# Domains
domain_main = Segment1D((xL, xR), is_main_domain=True, label_str="main")
domain_left = Segment1D((xL, x1), is_main_domain=False, label_str="left")
domain_mid = Segment1D((x1, x2), is_main_domain=False, label_str="mid")
domain_right = Segment1D((x2, xR), is_main_domain=False, label_str="right")

EPS_IF = 1e-6
domain_if_x1 = Segment1D((x1 - EPS_IF, x1 + EPS_IF), is_main_domain=False, label_str="if_x1")
domain_if_x2 = Segment1D((x2 - EPS_IF, x2 + EPS_IF), is_main_domain=False, label_str="if_x2")

domain_main.add_subdomain(domain_left)
domain_main.add_subdomain(domain_mid)
domain_main.add_subdomain(domain_right)
domain_main.add_subdomain(domain_if_x1)
domain_main.add_subdomain(domain_if_x2)


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
        dTdx = T_LEFT.partial_derivative_x(0)
        flux = k_left_fun(T_LEFT) * dTdx
        raw_res = flux.partial_derivative_x(0)
        scale = K_REF * DELTA_T_REF / (dx1**2)
        return raw_res / scale


class MidResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        dTdx = T_MID.partial_derivative_x(0)
        flux = k_mid_fun(T_MID) * dTdx
        raw_res = flux.partial_derivative_x(0)
        scale = K_REF * DELTA_T_REF / (dx2**2)
        return raw_res / scale


class RightResidual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        dTdx = T_RIGHT.partial_derivative_x(0)
        flux = k_right_fun(T_RIGHT) * dTdx
        raw_res = flux.partial_derivative_x(0)
        scale = K_REF * DELTA_T_REF / (dx3**2)
        return raw_res / scale


class InterfaceX1Residual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        q_left = k_left_fun(T_LEFT) * T_LEFT.partial_derivative_x(0)
        q_mid = k_mid_fun(T_MID) * T_MID.partial_derivative_x(0)

        def residual(space, x, mu):
            x_if = jnp.zeros_like(x) + x1
            flux_jump = (q_left(space, x_if, mu) - q_mid(space, x_if, mu)) / FLUX_REF
            return flux_jump

        return ParamScalarFunction(T_LEFT.dims, residual, f_type=T_LEFT.f_type)


class InterfaceX2Residual(InteriorResidual):
    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x_mu")

    def construct_residual(self, *vars):
        q_mid = k_mid_fun(T_MID) * T_MID.partial_derivative_x(0)
        q_right = k_right_fun(T_RIGHT) * T_RIGHT.partial_derivative_x(0)

        def residual(space, x, mu):
            x_if = jnp.zeros_like(x) + x2
            flux_jump = (q_mid(space, x_if, mu) - q_right(space, x_if, mu)) / FLUX_REF
            return flux_jump

        return ParamScalarFunction(T_MID.dims, residual, f_type=T_MID.f_type)


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
param_sampler = UniformParametricSampler(PARAM_DOMAIN)

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


def train_adam(seed=42):
    print("\n@@@@@@@@@@@@@@@ train with Adam @@@@@@@@@@@@@@@@@@@@@@")
    space = build_space(seed)
    pinn = Projector(
        model,
        space,
        sampler,
        optimizer="Adam",
        learning_rate=LEARNING_RATE_ADAM,
        weights=W_COMMON,
    )

    key = jax.random.PRNGKey(seed)
    key, sample_dict = sampler.sample(key, N_COLLOC)
    print("sample_dict keys:", sample_dict.keys())
    losses0 = pinn.evaluate_losses(space, sample_dict)
    print("initial losses:", losses0)

    start = timeit.default_timer()
    key, pinn = pinn.project(
        key,
        space,
        N_EPOCHS_ADAM,
        N_COLLOC,
        tqdmdesc="Training Adam multicase",
    )
    end = timeit.default_timer()

    print("best loss:", pinn.best_loss)
    print(f"time for {N_EPOCHS_ADAM} epochs:", end - start)
    return pinn


def evaluate_case(pinn, tl, tr):
    x_ref = np.linspace(xL, xR, 1000).reshape(-1, 1)
    T_exact, t1_exact, t2_exact = exact_temperature_numpy(x_ref, tl, tr)
    mu_eval = make_mu_eval(len(x_ref), tl, tr)
    T_pred = evaluate_coupled(pinn.space, x_ref, mu_eval)

    abs_err = np.abs(T_pred - T_exact)
    rmse = float(np.sqrt(np.mean((T_pred - T_exact) ** 2)))
    mae = float(np.mean(abs_err))
    rel_l2 = float(np.linalg.norm(T_pred - T_exact) / np.linalg.norm(T_exact))

    return x_ref, T_exact, T_pred, {
        "TL": tl,
        "TR": tr,
        "T1_exact": t1_exact,
        "T2_exact": t2_exact,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
    }


def plot_family(pinn):
    plt.figure(figsize=(8, 5))
    for tl, tr in CASES:
        x_ref = np.linspace(xL, xR, 500).reshape(-1, 1)
        T_exact, _, _ = exact_temperature_numpy(x_ref, tl, tr)
        plt.plot(x_ref, T_exact, lw=2, label=f"Exact TL={int(tl)}, TR={int(tr)}")
    plt.axvline(x1, color="gray", ls="--", lw=1.0)
    plt.axvline(x2, color="gray", ls="--", lw=1.0)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title("Analytical solution family across boundary-condition cases")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "exact_solution_family.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for tl, tr in CASES:
        x_ref = np.linspace(xL, xR, 500).reshape(-1, 1)
        mu_eval = make_mu_eval(len(x_ref), tl, tr)
        T_pred = evaluate_coupled(pinn.space, x_ref, mu_eval)
        plt.plot(x_ref, T_pred, lw=2, label=f"PINN TL={int(tl)}, TR={int(tr)}")
    plt.axvline(x1, color="gray", ls="--", lw=1.0)
    plt.axvline(x2, color="gray", ls="--", lw=1.0)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title("PINN solution family across boundary-condition cases")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "pinn_solution_family.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_case_comparisons(pinn, results):
    for res in results:
        tl, tr = res["TL"], res["TR"]
        x_ref, T_exact, T_pred, _ = evaluate_case(pinn, tl, tr)
        case_tag = f"TL_{int(tl)}_TR_{int(tr)}"

        plt.figure(figsize=(7, 5))
        plt.plot(x_ref, T_exact, "k-", lw=2, label="Exact")
        plt.plot(x_ref, T_pred, "r--", lw=2, label="PINN")
        plt.axvline(x1, color="gray", ls="--", lw=1.2, label="interface")
        plt.axvline(x2, color="gray", ls="--", lw=1.2)
        plt.xlabel("x [m]")
        plt.ylabel("Temperature [K]")
        plt.title(f"Case {case_tag}: PINN vs Exact")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"{case_tag}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_metrics_table(results):
    out_path = SAVE_DIR / "multicase_metrics.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("TL,TR,T1_exact,T2_exact,rmse,mae,rel_l2\n")
        for r in results:
            f.write(
                f"{r['TL']},{r['TR']},{r['T1_exact']},{r['T2_exact']},{r['rmse']},{r['mae']},{r['rel_l2']}\n"
            )


if __name__ == "__main__":
    pinn = train_adam(seed=42)

    results = []
    for tl, tr in CASES:
        _, _, _, metrics = evaluate_case(pinn, tl, tr)
        results.append(metrics)
        print(
            f"TL={tl:.1f}, TR={tr:.1f} | "
            f"T1_ex={metrics['T1_exact']:.6f}, T2_ex={metrics['T2_exact']:.6f} | "
            f"RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, RelL2={metrics['rel_l2']:.6e}"
        )

    plot_family(pinn)
    plot_case_comparisons(pinn, results)
    save_metrics_table(results)

    print("\nCases used:")
    for tl, tr in CASES:
        print(f"  TL={tl:.1f}, TR={tr:.1f}")

    print(f"\nChosen constant conductivities: K1={K1:.6f}, K2={K2:.6f}, K3={K3:.6f}")
    print(f"Outputs saved in: {SAVE_DIR.resolve()}")