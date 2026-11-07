from pathlib import Path
import time

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt


# 1. Problem setup

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "NH3_data.csv"

L = 0.32
TL = 420.0
TR = 900.0

LR = 1e-3
EPOCHS = 5000
N_COLLOC = 256

# Option C: nondimensionalization scales

T_REF = 0.5 * (TL + TR)
DELTA_T = abs(TR - TL)
assert DELTA_T > 0.0, "TL and TR must be different for this scaling."

# Output directory
SAVE_DIR = Path("nh3_stage1_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Ammonia conductivity polynomial:
# k(T) = 70.63350679 - 0.19209961*T + 0.00020083*T^2 - 0.00000007*T^3
def conductivity_dimensional(T):
    return (
        70.63350679
        - 0.19209961 * T
        + 0.00020083 * T**2
        - 0.00000007 * T**3
    )

K_REF = conductivity_dimensional(T_REF)

LAMBDA_PDE = 1.0
LAMBDA_DATA = 10.0
LAMBDA_REG = 1e-6


# 2. Load one single FVM case

df = pd.read_csv(CSV_PATH)
case_df = df[(df["TL"] == TL) & (df["TR"] == TR)].copy().sort_values("y")

assert len(case_df) > 0, f"No rows found for TL={TL}, TR={TR}"

y_fvm = case_df["y"].values.astype(np.float32).reshape(-1, 1)
T_fvm = case_df["Temperature"].values.astype(np.float32).reshape(-1, 1)

y_hat_fvm = (y_fvm / L).astype(np.float32)

def temperature_to_theta(T):
    return (T - T_REF) / DELTA_T

def theta_to_temperature(theta):
    return T_REF + DELTA_T * theta

theta_fvm = temperature_to_theta(T_fvm).astype(np.float32)

x_data = jnp.array(y_hat_fvm)
theta_data = jnp.array(theta_fvm)
T_data = jnp.array(T_fvm)


# 3. MLP utilities

def glorot_init(key, in_dim, out_dim):
    lim = np.sqrt(6.0 / (in_dim + out_dim))
    W = jax.random.uniform(key, (in_dim, out_dim), minval=-lim, maxval=lim)
    b = jnp.zeros((out_dim,))
    return {"W": W, "b": b}

def init_mlp(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for k, din, dout in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        params.append(glorot_init(k, din, dout))
    return params

def mlp_forward(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


# 4. Hard-BC PINN ansatz in theta-space

theta_L = float(temperature_to_theta(np.array([[TL]], dtype=np.float32))[0, 0])
theta_R = float(temperature_to_theta(np.array([[TR]], dtype=np.float32))[0, 0])

def model_theta(params, y_hat):
    y_hat_arr = jnp.array([[y_hat]], dtype=jnp.float32)
    raw = mlp_forward(params, y_hat_arr)[0, 0]

    theta_bc = (1.0 - y_hat) * theta_L + y_hat * theta_R
    theta = theta_bc + y_hat * (1.0 - y_hat) * raw
    return theta

def model_temperature(params, y_hat):
    theta = model_theta(params, y_hat)
    return theta_to_temperature(theta)

v_model_theta = jax.vmap(model_theta, in_axes=(None, 0))
v_model_temperature = jax.vmap(model_temperature, in_axes=(None, 0))


# 5. Dimensionless conductivity and PDE residual

def conductivity_from_theta(theta):
    T = theta_to_temperature(theta)
    return conductivity_dimensional(T)

def residual_single(params, y_hat):
    def theta_of_yhat(yh):
        return model_theta(params, yh)

    def flux_star_of_yhat(yh):
        theta = theta_of_yhat(yh)
        k_star = conductivity_from_theta(theta) / K_REF
        dtheta_dyhat = jax.grad(theta_of_yhat)(yh)
        return k_star * dtheta_dyhat

    dflux_star_dyhat = jax.grad(flux_star_of_yhat)(y_hat)
    return dflux_star_dyhat

v_residual = jax.vmap(residual_single, in_axes=(None, 0))


# 6. Losses

def data_loss(params, yhat_data, theta_true):
    theta_pred = v_model_theta(params, yhat_data[:, 0]).reshape(-1, 1)
    return jnp.mean((theta_pred - theta_true) ** 2)

def pde_loss(params, yhat_colloc):
    r = v_residual(params, yhat_colloc[:, 0]).reshape(-1, 1)
    return jnp.mean(r ** 2)

def reg_loss(params):
    leaves = jax.tree_util.tree_leaves(params)
    return sum(jnp.sum(p ** 2) for p in leaves)

def total_loss(params, yhat_data, theta_true, yhat_colloc):
    ld = data_loss(params, yhat_data, theta_true)
    lp = pde_loss(params, yhat_colloc)
    lr = reg_loss(params)
    loss = LAMBDA_DATA * ld + LAMBDA_PDE * lp + LAMBDA_REG * lr
    return loss, (ld, lp, lr)


# 7. Sampling

def sample_collocation(key, n_points):
    yhat = jax.random.uniform(key, (n_points, 1), minval=0.0, maxval=1.0)
    return yhat


# 8. Train step

optimizer = optax.adam(LR)

@jax.jit
def train_step(params, opt_state, yhat_data, theta_true, yhat_colloc):
    (loss_value, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
        params, yhat_data, theta_true, yhat_colloc
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value, aux


# 9. Evaluation

def evaluate(params, yhat_data, T_true):
    T_pred = v_model_temperature(params, yhat_data[:, 0]).reshape(-1, 1)
    rmse = jnp.sqrt(jnp.mean((T_pred - T_true) ** 2))
    mae = jnp.mean(jnp.abs(T_pred - T_true))
    rel_l2 = jnp.linalg.norm(T_pred - T_true) / jnp.linalg.norm(T_true)
    return T_pred, {
        "rmse": float(rmse),
        "mae": float(mae),
        "rel_l2": float(rel_l2),
    }


# 10. Main training loop

key = jax.random.PRNGKey(0)
params = init_mlp([1, 64, 64, 64, 1], key)
opt_state = optimizer.init(params)

loss_history = []
data_history = []
pde_history = []
epoch_history_eval = []
rmse_history_eval = []
mae_history_eval = []
rel_l2_history_eval = []

t0 = time.time()

for epoch in range(EPOCHS):
    key, kf = jax.random.split(key)
    yhat_colloc = sample_collocation(kf, N_COLLOC)

    params, opt_state, loss_value, (ld, lp, lr) = train_step(
        params, opt_state, x_data, theta_data, yhat_colloc
    )

    loss_history.append(float(loss_value))
    data_history.append(float(ld))
    pde_history.append(float(lp))

    if epoch % 100 == 0:
        _, metrics = evaluate(params, x_data, T_data)
        epoch_history_eval.append(epoch)
        rmse_history_eval.append(metrics["rmse"])
        mae_history_eval.append(metrics["mae"])
        rel_l2_history_eval.append(metrics["rel_l2"])

        print(
            f"Epoch {epoch:5d} | "
            f"loss={float(loss_value):.4e} | "
            f"data={float(ld):.4e} | "
            f"pde={float(lp):.4e} | "
            f"rmse={metrics['rmse']:.4f} | "
            f"rel_l2={metrics['rel_l2']:.4e}"
        )

train_time = time.time() - t0


# 11. Final prediction and metrics

T_pred, metrics = evaluate(params, x_data, T_data)
T_pred_np = np.array(T_pred)
abs_err = np.abs(T_pred_np - T_fvm)

print("\nFinal metrics:")
print(metrics)
print(f"Training time: {train_time:.2f} s")

metrics_df = pd.DataFrame([{
    "TL": TL,
    "TR": TR,
    "rmse": metrics["rmse"],
    "mae": metrics["mae"],
    "rel_l2": metrics["rel_l2"],
    "training_time_s": train_time,
    "epochs": EPOCHS,
    "n_colloc": N_COLLOC,
    "lambda_data": LAMBDA_DATA,
    "lambda_pde": LAMBDA_PDE,
    "lambda_reg": LAMBDA_REG,
    "T_ref": T_REF,
    "delta_T": DELTA_T,
    "K_ref": float(K_REF),
}])
metrics_df.to_csv(SAVE_DIR / "nh3_final_metrics.csv", index=False)

pred_df = pd.DataFrame({
    "y": y_fvm.flatten(),
    "T_fvm": T_fvm.flatten(),
    "T_pred": T_pred_np.flatten(),
    "abs_err": abs_err.flatten(),
    "y_hat": y_hat_fvm.flatten(),
    "theta_true": theta_fvm.flatten(),
    "theta_pred": np.array(v_model_theta(params, x_data[:, 0])).reshape(-1),
})
pred_df.to_csv(SAVE_DIR / "nh3_predictions.csv", index=False)

history_df = pd.DataFrame({
    "epoch": np.arange(EPOCHS),
    "total_loss": loss_history,
    "data_loss": data_history,
    "pde_loss": pde_history,
})
history_df.to_csv(SAVE_DIR / "nh3_loss_history.csv", index=False)

eval_df = pd.DataFrame({
    "epoch": epoch_history_eval,
    "rmse": rmse_history_eval,
    "mae": mae_history_eval,
    "rel_l2": rel_l2_history_eval,
})
eval_df.to_csv(SAVE_DIR / "nh3_eval_history.csv", index=False)


# 12. Saved plots

plt.figure(figsize=(7, 5))
plt.plot(y_fvm, T_fvm, "k-", lw=2, label="FVM")
plt.plot(y_fvm, T_pred_np, "r--", lw=2, label="PINN")
plt.xlabel("y [m]")
plt.ylabel("Temperature [K]")
plt.title(f"Single-case PINN vs FVM (ammonia, TL={TL} K, TR={TR} K)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "nh3_temperature_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(y_fvm, abs_err, "b-", lw=2)
plt.xlabel("y [m]")
plt.ylabel("Absolute error [K]")
plt.title("Absolute error: |T_PINN - T_FVM| (ammonia)")
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "nh3_absolute_error.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(7, 4))
plt.semilogy(loss_history, label="total")
plt.semilogy(data_history, label="data")
plt.semilogy(pde_history, label="pde")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training history (ammonia)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "nh3_training_history.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(epoch_history_eval, rmse_history_eval, "o-", lw=2)
plt.xlabel("Epoch")
plt.ylabel("RMSE [K]")
plt.title("Evaluation RMSE vs epoch (ammonia)")
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "nh3_rmse_history.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"\nAll outputs saved in: {SAVE_DIR.resolve()}")