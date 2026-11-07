from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax


# 1. USER SETTINGS

FILE_PATH = Path(__file__).resolve().parents[2] / "data" / "NH3_Data.csv"
SHEET_NAME = 0
L = 0.32

FULL_TMIN = 300.0
FULL_TMAX = 1500.0

TRAIN_TMIN = 500.0
TRAIN_TMAX = 1200.0

SEED = 0

LR = 1e-3
EPOCHS = 5000
BATCH_DATA = 2048
BATCH_COLLOC = 2048

LAMBDA_PDE = 1.0
LAMBDA_DATA = 10.0
LAMBDA_REG = 1e-6

LAYER_SIZES = [3, 128, 128, 128, 1]

N_PLOT_CASES = 6
PLOT_SPECIFIC_CASES = [(300.0, 1500.0)]

PLOT_DIR = Path("nh3_stage3_outputs")
PLOT_DIR.mkdir(exist_ok=True)


# 2. CONDUCTIVITY + SCALES

def conductivity_dimensional(T):
    return (
        70.63350679
        - 0.19209961 * T
        + 0.00020083 * T**2
        - 0.00000007 * T**3
    )

T_REF = 0.5 * (TRAIN_TMIN + TRAIN_TMAX)
DELTA_T = TRAIN_TMAX - TRAIN_TMIN
assert DELTA_T > 0.0, "TRAIN_TMAX must be greater than TRAIN_TMIN."

K_REF = conductivity_dimensional(T_REF)


# 3. DATA LOADING

def load_table(file_path, sheet_name=0):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError("Supported file types are .csv, .xlsx, .xls")
    return df


def standardize_columns(df):
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()

        if cl in ["tl", "left temperature", "t_left", "temp_left"]:
            rename_map[c] = "TL"
        elif cl in ["tr", "right temperature", "t_right", "temp_right"]:
            rename_map[c] = "TR"
        elif cl in ["y", "x", "coord", "coordinate", "position"]:
            rename_map[c] = "y"
        elif cl in ["temperature", "temperature(k)", "temp", "t"]:
            rename_map[c] = "Temperature"

    df = df.rename(columns=rename_map)

    required = ["TL", "TR", "y", "Temperature"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


df = load_table(FILE_PATH, SHEET_NAME)
df = standardize_columns(df)

df = df[(df["TL"] >= FULL_TMIN) & (df["TL"] <= FULL_TMAX) &
        (df["TR"] >= FULL_TMIN) & (df["TR"] <= FULL_TMAX)].copy()

if len(df) == 0:
    raise ValueError("No rows remain after filtering to [FULL_TMIN, FULL_TMAX].")

df = df.sort_values(["TL", "TR", "y"]).reset_index(drop=True)

pairs_df = df[["TL", "TR"]].drop_duplicates().reset_index(drop=True)
pairs = pairs_df.values

if len(pairs) < 2:
    raise ValueError("Need multiple unique (TL,TR) pairs in the dataset.")


# 4. TRAIN / EXTRAPOLATION SPLIT BY PAIRS

train_mask_pairs = (
    (pairs_df["TL"] >= TRAIN_TMIN) & (pairs_df["TL"] <= TRAIN_TMAX) &
    (pairs_df["TR"] >= TRAIN_TMIN) & (pairs_df["TR"] <= TRAIN_TMAX)
)

test_mask_pairs = ~train_mask_pairs

train_pairs = pairs_df[train_mask_pairs].values
test_pairs = pairs_df[test_mask_pairs].values

if len(train_pairs) == 0:
    raise ValueError("No training pairs inside the chosen training box.")
if len(test_pairs) == 0:
    raise ValueError("No extrapolation test pairs outside the chosen training box.")

train_pair_set = set(map(tuple, train_pairs))
test_pair_set = set(map(tuple, test_pairs))

pair_series = df[["TL", "TR"]].apply(tuple, axis=1)
train_df = df[pair_series.isin(train_pair_set)].copy()
test_df = df[pair_series.isin(test_pair_set)].copy()


# 5. NORMALIZATION

def normalize_y(y):
    return y / L

def normalize_Tbc_trainbox(T):
    return 2.0 * (T - TRAIN_TMIN) / (TRAIN_TMAX - TRAIN_TMIN) - 1.0

def denormalize_Tbc_trainbox(Tn):
    return 0.5 * (Tn + 1.0) * (TRAIN_TMAX - TRAIN_TMIN) + TRAIN_TMIN

def temperature_to_theta(T):
    return (T - T_REF) / DELTA_T

def theta_to_temperature(theta):
    return T_REF + DELTA_T * theta


train_df["yhat"] = normalize_y(train_df["y"].values.astype(np.float32))
train_df["TLn"] = normalize_Tbc_trainbox(train_df["TL"].values.astype(np.float32))
train_df["TRn"] = normalize_Tbc_trainbox(train_df["TR"].values.astype(np.float32))
train_df["theta"] = temperature_to_theta(train_df["Temperature"].values.astype(np.float32))

test_df["yhat"] = normalize_y(test_df["y"].values.astype(np.float32))
test_df["TLn"] = normalize_Tbc_trainbox(test_df["TL"].values.astype(np.float32))
test_df["TRn"] = normalize_Tbc_trainbox(test_df["TR"].values.astype(np.float32))
test_df["theta"] = temperature_to_theta(test_df["Temperature"].values.astype(np.float32))


def build_arrays(dataframe):
    X = dataframe[["yhat", "TLn", "TRn"]].values.astype(np.float32)
    theta = dataframe["theta"].values.astype(np.float32).reshape(-1, 1)
    T = dataframe["Temperature"].values.astype(np.float32).reshape(-1, 1)
    return jnp.array(X), jnp.array(theta), jnp.array(T)


x_train, theta_train, T_train = build_arrays(train_df)
x_test, theta_test, T_test = build_arrays(test_df)


# 6. MLP

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


# 7. PARAMETRIC HARD-BC PINN

def model_theta(params, x):
    yhat = x[0]
    TLn = x[1]
    TRn = x[2]

    TL = denormalize_Tbc_trainbox(TLn)
    TR = denormalize_Tbc_trainbox(TRn)

    theta_L = temperature_to_theta(TL)
    theta_R = temperature_to_theta(TR)

    x_arr = jnp.array([[yhat, TLn, TRn]], dtype=jnp.float32)
    raw = mlp_forward(params, x_arr)[0, 0]

    theta_bc = (1.0 - yhat) * theta_L + yhat * theta_R
    theta = theta_bc + yhat * (1.0 - yhat) * raw
    return theta

def model_temperature(params, x):
    theta = model_theta(params, x)
    return theta_to_temperature(theta)

v_model_theta = jax.vmap(model_theta, in_axes=(None, 0))
v_model_temperature = jax.vmap(model_temperature, in_axes=(None, 0))


# 8. PHYSICS (AMMONIA CONDUCTIVITY)

def conductivity(T):
    return (
        70.63350679
        - 0.19209961 * T
        + 0.00020083 * T**2
        - 0.00000007 * T**3
    )

def conductivity_from_theta(theta):
    T = theta_to_temperature(theta)
    return conductivity(T)

def residual_single(params, x):
    yhat = x[0]
    TLn = x[1]
    TRn = x[2]

    def theta_of_yhat(yh):
        x_local = jnp.array([yh, TLn, TRn], dtype=jnp.float32)
        return model_theta(params, x_local)

    def flux_star_of_yhat(yh):
        theta = theta_of_yhat(yh)
        k_star = conductivity_from_theta(theta) / K_REF
        dtheta_dyhat = jax.grad(theta_of_yhat)(yh)
        return k_star * dtheta_dyhat

    dflux_star_dyhat = jax.grad(flux_star_of_yhat)(yhat)
    return dflux_star_dyhat

v_residual = jax.vmap(residual_single, in_axes=(None, 0))


# 9. LOSSES

def data_loss(params, x_data, theta_true):
    theta_pred = v_model_theta(params, x_data).reshape(-1, 1)
    return jnp.mean((theta_pred - theta_true) ** 2)

def pde_loss(params, x_colloc):
    r = v_residual(params, x_colloc).reshape(-1, 1)
    return jnp.mean(r ** 2)

def reg_loss(params):
    leaves = jax.tree_util.tree_leaves(params)
    return sum(jnp.sum(p ** 2) for p in leaves)

def total_loss(params, x_data, theta_true, x_colloc):
    ld = data_loss(params, x_data, theta_true)
    lp = pde_loss(params, x_colloc)
    lr = reg_loss(params)
    loss = LAMBDA_DATA * ld + LAMBDA_PDE * lp + LAMBDA_REG * lr
    return loss, (ld, lp, lr)


# 10. SAMPLING

def sample_data_batch(key, x_all, theta_all, batch_size):
    n = x_all.shape[0]
    idx = jax.random.randint(key, (batch_size,), 0, n)
    return x_all[idx], theta_all[idx]

def sample_collocation(key, n_points):
    k1, k2, k3 = jax.random.split(key, 3)
    yhat = jax.random.uniform(k1, (n_points, 1), minval=0.0, maxval=1.0)

    # collocation only inside training parameter box
    TLn = jax.random.uniform(k2, (n_points, 1), minval=-1.0, maxval=1.0)
    TRn = jax.random.uniform(k3, (n_points, 1), minval=-1.0, maxval=1.0)

    return jnp.hstack([yhat, TLn, TRn])


# 11. TRAIN STEP

optimizer = optax.adam(LR)

@jax.jit
def train_step(params, opt_state, x_data_batch, theta_batch, x_colloc):
    (loss_value, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
        params, x_data_batch, theta_batch, x_colloc
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value, aux


# 12. EVALUATION

def evaluate(params, x_data, T_true):
    T_pred = v_model_temperature(params, x_data).reshape(-1, 1)
    rmse = jnp.sqrt(jnp.mean((T_pred - T_true) ** 2))
    mae = jnp.mean(jnp.abs(T_pred - T_true))
    rel_l2 = jnp.linalg.norm(T_pred - T_true) / jnp.linalg.norm(T_true)
    return T_pred, {
        "rmse": float(rmse),
        "mae": float(mae),
        "rel_l2": float(rel_l2),
    }


# 13. TRAINING LOOP

key = jax.random.PRNGKey(SEED)
params = init_mlp(LAYER_SIZES, key)
opt_state = optimizer.init(params)

loss_history = []
data_history = []
pde_history = []
reg_history = []
test_rmse_history = []
test_mae_history = []
test_rel_l2_history = []
test_epoch_history = []

t0 = time.time()

for epoch in range(EPOCHS):
    key, k1, k2 = jax.random.split(key, 3)

    x_batch, theta_batch = sample_data_batch(k1, x_train, theta_train, BATCH_DATA)
    x_colloc = sample_collocation(k2, BATCH_COLLOC)

    params, opt_state, loss_value, (ld, lp, lr) = train_step(
        params, opt_state, x_batch, theta_batch, x_colloc
    )

    loss_history.append(float(loss_value))
    data_history.append(float(ld))
    pde_history.append(float(lp))
    reg_history.append(float(lr))

    if epoch % 100 == 0:
        _, test_metrics = evaluate(params, x_test, T_test)
        test_epoch_history.append(epoch)
        test_rmse_history.append(test_metrics["rmse"])
        test_mae_history.append(test_metrics["mae"])
        test_rel_l2_history.append(test_metrics["rel_l2"])
        print(
            f"Epoch {epoch:5d} | "
            f"loss={float(loss_value):.4e} | "
            f"data={float(ld):.4e} | "
            f"pde={float(lp):.4e} | "
            f"reg={float(lr):.4e} | "
            f"extra_rmse={test_metrics['rmse']:.4f} | "
            f"extra_rel_l2={test_metrics['rel_l2']:.4e}"
        )

train_time = time.time() - t0


# 14. FINAL METRICS

T_pred_test, test_metrics = evaluate(params, x_test, T_test)

print("\nFinal extrapolation test metrics (ammonia):")
print(test_metrics)
print(f"Training time: {train_time:.2f} s")
print(f"Train rows: {len(train_df)} | Extrapolation test rows: {len(test_df)}")
print(f"Train pairs: {len(train_pairs)} | Extrapolation test pairs: {len(test_pairs)}")
print(f"Training box: TL,TR in [{TRAIN_TMIN}, {TRAIN_TMAX}]")
print(f"Full data box: TL,TR in [{FULL_TMIN}, {FULL_TMAX}]")

metrics_df = pd.DataFrame([{
    "rmse": test_metrics["rmse"],
    "mae": test_metrics["mae"],
    "rel_l2": test_metrics["rel_l2"],
    "training_time_s": train_time,
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "train_pairs": len(train_pairs),
    "test_pairs": len(test_pairs),
    "FULL_TMIN": FULL_TMIN,
    "FULL_TMAX": FULL_TMAX,
    "TRAIN_TMIN": TRAIN_TMIN,
    "TRAIN_TMAX": TRAIN_TMAX,
    "T_ref": T_REF,
    "delta_T": DELTA_T,
    "K_ref": float(K_REF),
}])
metrics_df.to_csv(PLOT_DIR / "stage3_ammonia_final_metrics.csv", index=False)


# 15. SAVE TRAINING HISTORY

history_df = pd.DataFrame({
    "epoch": np.arange(EPOCHS),
    "total_loss": loss_history,
    "data_loss": data_history,
    "pde_loss": pde_history,
    "reg_loss": reg_history,
})
history_df.to_csv(PLOT_DIR / "stage3_ammonia_loss_history.csv", index=False)

test_hist_df = pd.DataFrame({
    "epoch": test_epoch_history,
    "test_rmse": test_rmse_history,
    "test_mae": test_mae_history,
    "test_rel_l2": test_rel_l2_history,
})
test_hist_df.to_csv(PLOT_DIR / "stage3_ammonia_test_metrics_history.csv", index=False)


# 16. TRAINING HISTORY PLOTS

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(loss_history, label="total")
ax.semilogy(data_history, label="data")
ax.semilogy(pde_history, label="pde")
ax.semilogy(reg_history, label="reg")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Stage 3 extrapolation PINN training history (ammonia)")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(PLOT_DIR / "stage3_ammonia_training_history.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(test_epoch_history, test_rmse_history, "o-", lw=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Extrapolation test RMSE [K]")
ax.set_title("Extrapolation RMSE vs epoch (ammonia)")
ax.grid(True)
fig.tight_layout()
fig.savefig(PLOT_DIR / "stage3_ammonia_extrapolation_rmse.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


# 17. CASE SELECTION

test_pairs_unique = test_df[["TL", "TR"]].drop_duplicates().values.tolist()

if len(PLOT_SPECIFIC_CASES) > 0:
    cases_to_plot = []
    test_pair_set_exact = set((float(a), float(b)) for a, b in test_pairs_unique)
    for pair in PLOT_SPECIFIC_CASES:
        pair = (float(pair[0]), float(pair[1]))
        if pair in test_pair_set_exact:
            cases_to_plot.append(pair)
        else:
            print(f"Warning: requested pair {pair} is not in extrapolation test set, skipping.")
else:
    def dist_from_train_box(pair):
        tl, tr = pair
        d_tl = 0.0 if TRAIN_TMIN <= tl <= TRAIN_TMAX else min(abs(tl - TRAIN_TMIN), abs(tl - TRAIN_TMAX))
        d_tr = 0.0 if TRAIN_TMIN <= tr <= TRAIN_TMAX else min(abs(tr - TRAIN_TMIN), abs(tr - TRAIN_TMAX))
        return d_tl + d_tr

    test_pairs_sorted = sorted(test_pairs_unique, key=dist_from_train_box, reverse=True)
    cases_to_plot = [tuple(x) for x in test_pairs_sorted[:N_PLOT_CASES]]

if len(cases_to_plot) == 0:
    print("No extrapolation test pairs available for plotting.")
else:
    print("\nPlotting extrapolation test pairs (ammonia):")
    for pair in cases_to_plot:
        print(pair)


# 18. PREDICTION HELPER

def predict_case(params, TL_case, TR_case, y_values):
    y_values = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
    yhat = normalize_y(y_values).astype(np.float32)

    TLn_val = normalize_Tbc_trainbox(np.array([[TL_case]], dtype=np.float32))[0, 0]
    TRn_val = normalize_Tbc_trainbox(np.array([[TR_case]], dtype=np.float32))[0, 0]

    TLn = np.full_like(yhat, TLn_val, dtype=np.float32)
    TRn = np.full_like(yhat, TRn_val, dtype=np.float32)

    X_case = jnp.array(np.hstack([yhat, TLn, TRn]).astype(np.float32))
    T_pred = np.array(v_model_temperature(params, X_case)).reshape(-1, 1)
    theta_pred = np.array(v_model_theta(params, X_case)).reshape(-1, 1)
    return T_pred, theta_pred


# 19. EXTRAPOLATION COMPARISON PLOTS

case_metrics_rows = []
all_case_predictions = []

for TL_case, TR_case in cases_to_plot:
    case_df = test_df[(test_df["TL"] == TL_case) & (test_df["TR"] == TR_case)].copy()
    case_df = case_df.sort_values("y")

    y_case = case_df["y"].values.astype(np.float32).reshape(-1, 1)
    T_fvm = case_df["Temperature"].values.astype(np.float32).reshape(-1, 1)
    theta_true = case_df["theta"].values.astype(np.float32).reshape(-1, 1)

    T_pred, theta_pred = predict_case(params, TL_case, TR_case, y_case)

    abs_err = np.abs(T_pred - T_fvm)
    rmse_case = float(np.sqrt(np.mean((T_pred - T_fvm) ** 2)))
    mae_case = float(np.mean(np.abs(T_pred - T_fvm)))
    rel_l2_case = float(np.linalg.norm(T_pred - T_fvm) / np.linalg.norm(T_fvm))

    case_metrics_rows.append({
        "TL": TL_case,
        "TR": TR_case,
        "rmse": rmse_case,
        "mae": mae_case,
        "rel_l2": rel_l2_case,
    })

    case_pred_df = pd.DataFrame({
        "TL": np.full(len(y_case), TL_case),
        "TR": np.full(len(y_case), TR_case),
        "y": y_case.flatten(),
        "yhat": normalize_y(y_case).flatten(),
        "T_fvm": T_fvm.flatten(),
        "T_pred": T_pred.flatten(),
        "abs_err": abs_err.flatten(),
        "theta_true": theta_true.flatten(),
        "theta_pred": theta_pred.flatten(),
    })
    all_case_predictions.append(case_pred_df)

    print(f"\nExtrapolation case TL={TL_case:.1f}, TR={TR_case:.1f}")
    print(f"  RMSE   = {rmse_case:.6f} K")
    print(f"  MAE    = {mae_case:.6f} K")
    print(f"  relL2  = {rel_l2_case:.6e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(y_case, T_fvm, "k-", lw=2, label="FVM")
    ax.plot(y_case, T_pred, "r--", lw=2, label="Parametric PINN")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"Stage 3 ammonia extrapolation: TL={TL_case:.0f} K, TR={TR_case:.0f} K")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / f"stage3_ammonia_TL{int(TL_case)}_TR{int(TR_case)}_profile.png",
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y_case, abs_err, "b-", lw=2)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("Absolute error [K]")
    ax.set_title(
        f"Absolute error |PINN-FVM| (ammonia)\n"
        f"TL={TL_case:.0f} K, TR={TR_case:.0f} K, "
        f"RMSE={rmse_case:.3f}, relL2={rel_l2_case:.3e}"
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / f"stage3_ammonia_TL{int(TL_case)}_TR{int(TR_case)}_error.png",
        dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)


# 20. SAVE CASE TABLES

if len(case_metrics_rows) > 0:
    case_metrics_df = pd.DataFrame(case_metrics_rows)
    case_metrics_df.to_csv(PLOT_DIR / "stage3_ammonia_case_metrics.csv", index=False)

if len(all_case_predictions) > 0:
    all_case_predictions_df = pd.concat(all_case_predictions, ignore_index=True)
    all_case_predictions_df.to_csv(PLOT_DIR / "stage3_ammonia_case_predictions.csv", index=False)

print(f"\nAll outputs saved in: {PLOT_DIR.resolve()}")