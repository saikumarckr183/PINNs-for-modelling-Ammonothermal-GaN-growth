import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from scimba_jax.domains.domain_mapping import DomainMapping
from scimba_jax.domains.meshless_domains.domains_2d import (
    Segment2D,
    Square2D,
)
from scimba_jax.nonlinear_approximation.approximation_spaces.approximation_spaces import (
    ApproximationSpace,
)
from scimba_jax.nonlinear_approximation.integration.monte_carlo import (
    DomainSampler,
    TensorizedSampler,
)
from scimba_jax.nonlinear_approximation.model_class.funcparam_vectorial import (
    ParamScalarFunction,
)
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import (
    Projector,
)
from scimba_jax.physical_models.abstract_physical_model import (
    AbstractPhysicalModel,
)
from scimba_jax.physical_models.abstract_residuals import (
    InteriorResidual,
)


jax.config.update(
    "jax_enable_x64",
    True,
)


# Output directory

SAVE_DIR = Path(
    "output_inconel_only"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Geometry and physical properties

R_OUT = 0.035
H_OUT = 0.370

T0_K = 611.15
T1 = -300.0


def k_inc_value(
    temperature_K,
):
    """
    Temperature-dependent Inconel conductivity [W/(m K)]:

        k_inc(T) = 6.9995 + 0.0151 * T,  T in K.

    Matches the correlation used in the Inconel + NH3 model.
    """

    return (
        6.9995
        + 0.0151
        * temperature_K
    )


def T_outer(
    x,
):
    """
    Prescribed outer temperature field:

        T_outer(z) = T0_K + T1 * z.
    """

    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    z = x[..., 1:2]

    return (
        T0_K
        + T1
        * z
    )


# Training parameters

N_COLLOC = {
    "inconel": 6000,
}

N_EPOCHS = 1000

LEARNING_RATE = 1.0e-4

HIDDEN = [16, 16, 16]


# Reference scales

DELTA_T_REF = (
    abs(T1)
    * H_OUT
)

L_REF = R_OUT

# Fixed conductivity reference for residual normalization.
#
# K_REF is only a numerical scale for the r-weighted PDE residual.
# It does not replace k_inc(T) in the physics and is kept fixed
# throughout training.
T_K_MIN_FOR_SCALING = 495.0
T_K_MAX_FOR_SCALING = 615.0

T_K_SCALE_GRID = jnp.linspace(
    T_K_MIN_FOR_SCALING,
    T_K_MAX_FOR_SCALING,
    2001,
)

K_INC_SCALE_VALUES = k_inc_value(
    T_K_SCALE_GRID,
)

if bool(
    jnp.any(
        K_INC_SCALE_VALUES <= 0.0
    )
):
    raise ValueError(
        "k_inc_value(T) is non-positive in the anticipated "
        "temperature range."
    )

K_INC_MIN_REF = float(
    jnp.min(
        K_INC_SCALE_VALUES,
    )
)

K_INC_MAX_REF = float(
    jnp.max(
        K_INC_SCALE_VALUES,
    )
)

K_REF = K_INC_MAX_REF

FLUX_REF = (
    K_REF
    * DELTA_T_REF
    / L_REF
)

PDE_R_WEIGHTED_REF = FLUX_REF


# Main mapped cylinder rectangle
# Reference domain:
#     0 <= xi <= 1
#     0 <= eta <= 1
# Physical domain:
#     0 <= r <= R_OUT
#     0 <= z <= H_OUT

main_mapping = DomainMapping.rectangle(
    origin=jnp.array(
        [
            0.0,
            0.0,
        ],
        dtype=jnp.float64,
    ),
    x_dir=jnp.array(
        [
            R_OUT,
            0.0,
        ],
        dtype=jnp.float64,
    ),
    y_dir=jnp.array(
        [
            0.0,
            H_OUT,
        ],
        dtype=jnp.float64,
    ),
)

main_domain = Square2D(
    [
        (
            0.0,
            1.0,
        ),
        (
            0.0,
            1.0,
        ),
    ],
    is_main_domain=True,
    label_str="main",
)

main_domain.set_mapping(
    main_mapping,
    bounds_postmap=[
        (
            0.0,
            R_OUT,
        ),
        (
            0.0,
            H_OUT,
        ),
    ],
)


# Full Inconel interior domain
inconel_domain = Square2D(
    [
        (
            0.0,
            1.0,
        ),
        (
            0.0,
            1.0,
        ),
    ],
    is_main_domain=False,
    label_str="inconel",
)

main_domain.add_subdomain(
    inconel_domain,
)

# Boundary segments in mapped reference coordinates

domain_outer_side = Segment2D(
    pt1=(
        1.0,
        0.0,
    ),
    pt2=(
        1.0,
        1.0,
    ),
    label_str="outer_side",
)

domain_outer_bottom = Segment2D(
    pt1=(
        0.0,
        0.0,
    ),
    pt2=(
        1.0,
        0.0,
    ),
    label_str="outer_bottom",
)

domain_outer_top = Segment2D(
    pt1=(
        0.0,
        1.0,
    ),
    pt2=(
        1.0,
        1.0,
    ),
    label_str="outer_top",
)

domain_axis = Segment2D(
    pt1=(
        0.0,
        0.0,
    ),
    pt2=(
        0.0,
        1.0,
    ),
    label_str="axis",
)


for boundary in (
    domain_outer_side,
    domain_outer_bottom,
    domain_outer_top,
    domain_axis,
):
    main_domain.add_bc_domain(
        boundary,
    )


BOUNDARY_GROUPS = {
    # Outer temperature is imposed strongly in T_inc_eval.
    "outer_side": [
        domain_outer_side.get_label(),
    ],

    "outer_bottom": [
        domain_outer_bottom.get_label(),
    ],

    "outer_top": [
        domain_outer_top.get_label(),
    ],

    # Explicit symmetry condition.
    "axis": [
        domain_axis.get_label(),
    ],
}

main_domain.set_boundaries_dict(
    BOUNDARY_GROUPS,
)


N_BC_COLLOC = {
    group_name: {
        boundary_label: 500
        for boundary_label in boundary_labels
    }
    for group_name, boundary_labels in BOUNDARY_GROUPS.items()
}


# Neural coordinates and strong trial function

def scaled_coordinates(
    x,
):
    """
    Use squared radial coordinate rho = (r/R_OUT)^2.

    This makes the neural correction naturally even in r and supports
    axisymmetric regularity at r = 0.
    """

    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    r_scaled = (
        x[..., 0:1]
        / R_OUT
    )

    z_scaled = (
        x[..., 1:2]
        / H_OUT
    )

    rho = r_scaled**2
    eta = z_scaled

    return jnp.concatenate(
        [
            rho,
            eta,
        ],
        axis=-1,
    )


def outer_distance_factor(
    x,
):
    """
    Strong outer-Dirichlet factor.

    It vanishes on:
        r = R_OUT,
        z = 0,
        z = H_OUT.

    Thus T_inc = T_outer on the complete exterior boundary.
    """

    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    r = x[..., 0:1]
    z = x[..., 1:2]

    d_side = (
        1.0
        - (
            r / R_OUT
        )**2
    )

    d_bottom = (
        z / H_OUT
    )

    d_top = (
        H_OUT
        - z
    ) / H_OUT

    return (
        d_side
        * d_bottom
        * d_top
    )


# post_processing: takes the RAW scalar network output (already
# evaluated on pre_processed coordinates by ApproximationSpace) and
# the ORIGINAL (unscaled) x, and returns the physical temperature.
#
# T_inc = T_outer(z)
#         + DELTA_T_REF * D_outer(r,z) * theta_inc(rho, eta).
#
# This strongly enforces T_inc = T_outer on the outer side, bottom,
# and top boundaries.

def inconel_post_processing(
    theta_inc,
    x,
):
    return (
        T_outer(
            x,
        )
        + DELTA_T_REF
        * outer_distance_factor(
            x,
        )
        * theta_inc
    )


def evaluate_temperature(
    space,
    index,
    x,
):
    x = jnp.asarray(
        x,
        dtype=jnp.float64,
    )

    temperature = space.create_variables()[index]

    if x.ndim == 2:
        return jax.vmap(
            lambda x_one: temperature(
                space,
                x_one,
            )
        )(x)

    return temperature(
        space,
        x,
    )


def T_inc_eval(
    space,
    x,
):
    return evaluate_temperature(
        space,
        0,
        x,
    )


T_INC = ParamScalarFunction(
    {"x": 2},
    T_inc_eval,
    f_type="x",
)

# Temperature-dependent conductivity and axisymmetric PDE

def k_inc(
    T,
):
    """
    Temperature-dependent Inconel thermal conductivity as
    ParamScalarFunction.
    """

    return ParamScalarFunction(
        T.dims,
        lambda space, x: k_inc_value(
            T(
                space,
                x,
            )
        ),
        f_type=T.f_type,
    )


def axisymmetric_conduction(
    T,
    conductivity,
    space,
    x,
):
    """
    Nondimensional r-weighted axisymmetric steady-conduction residual:

        r*d/dr(k*dT/dr)
        + k*dT/dr
        + r*d/dz(k*dT/dz)
        = 0.

    For constant k this is equivalent to:

        k * [r*T_rr + T_r + r*T_zz] = 0.
    """

    dT_dr = T.partial_derivative(
        "x",
        0,
    )

    dT_dz = T.partial_derivative(
        "x",
        1,
    )

    q_r = (
        conductivity(
            T,
        )
        * dT_dr
    )

    q_z = (
        conductivity(
            T,
        )
        * dT_dz
    )

    r = x[..., 0:1]

    raw_residual = (
        r
        * q_r.partial_derivative(
            "x",
            0,
        )(
            space,
            x,
        )
        + q_r(
            space,
            x,
        )
        + r
        * q_z.partial_derivative(
            "x",
            1,
        )(
            space,
            x,
        )
    )

    return (
        raw_residual
        / PDE_R_WEIGHTED_REF
    )


# Residual classes

class InconelInteriorResidual(
    InteriorResidual,
):
    """
    Axisymmetric steady conduction in the full Inconel cylinder.
    """

    def __init__(
        self,
    ):
        super().__init__(
            domain=inconel_domain,
            size=1,
            model_type="x",
        )

    def construct_residual(
        self,
        *vars,
    ):
        T_inc = vars[0]

        def residual(
            space,
            x,
        ):
            return axisymmetric_conduction(
                T_inc,
                k_inc,
                space,
                x,
            )

        return ParamScalarFunction(
            T_inc.dims,
            residual,
            f_type=T_inc.f_type,
        )


# Physical model

class InconelOnlyModel(
    AbstractPhysicalModel,
):
    """
    Homogeneous axisymmetric steady Inconel conduction model.
    """

    def __init__(
        self,
    ):
        super().__init__(
            main_domain=main_domain,
        )

        self.physical_residuals = {
            "inconel": InconelInteriorResidual(),
        }

        self.data_residuals = {}


model = InconelOnlyModel()


# Sampler

domain_sampler = DomainSampler(
    main_domain,
    main_domain_fully_covered=True,
)

sampler = TensorizedSampler(
    [
        domain_sampler,
    ],
    model_type="x",
    bc=True,
    ic=False,
)


# Neural network and approximation space

def build_mlp(
    in_size,
    out_size,
    hidden_sizes,
    key,
):
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


def build_space(
    seed=42,
):
    """
    One neural model only:

        space.models[0] -> Inconel temperature field.
    """

    key = jax.random.PRNGKey(
        seed,
    )

    nn_inc = build_mlp(
        in_size=2,
        out_size=1,
        hidden_sizes=HIDDEN,
        key=key,
    )

    return ApproximationSpace(
        {"x": 2},
        [
            (
                nn_inc,
                "scalar",
                None,
            ),
        ],
        model_type="x",
        pre_processing=scaled_coordinates,
        post_processing=inconel_post_processing,
    )


# Loss weights

W_COMMON = {
    "inconel": (1.0,),
}


# Inconel-only plotting and validation functions

def evaluate_inconel_field(
    space,
    points,
):
    """
    Evaluate the physical Inconel PINN temperature at physical
    coordinates points[:, 0] = r and points[:, 1] = z.
    """

    points = jnp.asarray(
        points,
        dtype=jnp.float64,
    )

    return np.asarray(
        T_INC(
            space,
            points,
        )
    ).reshape(
        -1
    )


def save_figure(
    figure,
    output_name,
    dpi=250,
):
    """
    Save one Matplotlib figure safely in SAVE_DIR.
    """

    output_path = (
        SAVE_DIR
        / output_name
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        str(output_path),
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure,
    )

    print(
        f"Saved plot in:\n{output_path}"
    )

    return output_path


def plot_inconel_temperature_field(
    space,
    nr=201,
    nz=601,
):
    """
    Plot the composite physical PINN temperature field.
    """

    r = np.linspace(
        0.0,
        R_OUT,
        nr,
    )

    z = np.linspace(
        0.0,
        H_OUT,
        nz,
    )

    rr, zz = np.meshgrid(
        r,
        z,
        indexing="xy",
    )

    points = np.column_stack(
        [
            rr.ravel(),
            zz.ravel(),
        ]
    )

    temperature = evaluate_inconel_field(
        space,
        points,
    ).reshape(
        rr.shape
    )

    figure, axis = plt.subplots(
        figsize=(
            6.2,
            9.0,
        ),
    )

    contour = axis.contourf(
        rr * 1.0e3,
        zz * 1.0e3,
        temperature,
        levels=60,
        cmap="turbo",
    )

    colorbar = figure.colorbar(
        contour,
        ax=axis,
    )

    colorbar.set_label(
        "PINN temperature [K]"
    )

    axis.set_aspect(
        "equal"
    )

    axis.set_xlabel(
        "r [mm]"
    )

    axis.set_ylabel(
        "z [mm]"
    )

    axis.set_title(
        "Inconel-only axisymmetric PINN temperature"
    )

    figure.tight_layout()

    save_figure(
        figure,
        "inconel_only_temperature_field.png",
    )


def radial_derivative_at_points(
    space,
    points,
):
    """
    Evaluate dT_inc/dr at physical points using JAX automatic
    differentiation.
    """

    points = jnp.asarray(
        points,
        dtype=jnp.float64,
    )

    def temperature_at_one_point(
        x_one,
    ):
        return jnp.ravel(
            T_INC(
                space,
                x_one,
            )
        )[0]

    derivative_function = jax.grad(
        temperature_at_one_point,
    )

    return np.asarray(
        jax.vmap(
            lambda x_one: derivative_function(
                x_one,
            )[0]
        )(
            points
        )
    )


def plot_axis_derivative_diagnostic(
    space,
    nz=501,
):
    """
    Plot dT/dr close to the axis and report its maximum magnitude.
    """

    z = np.linspace(
        0.0,
        H_OUT,
        nz,
    )

    r_probe = 1.0e-6

    points = np.column_stack(
        [
            np.full(
                nz,
                r_probe,
            ),
            z,
        ]
    )

    dT_dr = radial_derivative_at_points(
        space,
        points,
    )

    print("\n" + "=" * 72)
    print("INCONEL-ONLY AXIS DIAGNOSTIC")
    print("=" * 72)

    print(
        f"Probe radius [m]: {r_probe:.3e}"
    )

    print(
        "Maximum |dT/dr| [K/m]: "
        f"{np.max(np.abs(dT_dr)):.8e}"
    )

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            4.8,
        ),
    )

    axis.plot(
        z * 1.0e3,
        dT_dr,
        color="tab:blue",
        lw=1.8,
    )

    axis.axhline(
        0.0,
        color="black",
        lw=1.0,
        ls="--",
    )

    axis.set_xlabel(
        "z [mm]"
    )

    axis.set_ylabel(
        r"$\partial T/\partial r$ [K/m]"
    )

    axis.set_title(
        "Near-axis radial derivative: Inconel-only PINN"
    )

    axis.grid(
        True,
        alpha=0.30,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "inconel_only_axis_derivative.png",
    )


def plot_inconel_loss_history(
    losses,
):
    """
    Plot the interior PDE and total weighted loss histories.
    """

    history = losses.losses_history

    total_loss = np.asarray(
        history[
            "total"
        ],
        dtype=np.float64,
    ).reshape(
        -1
    )

    interior_loss = np.asarray(
        history[
            "inconel"
        ],
        dtype=np.float64,
    ).reshape(
        -1
    )

    epochs = np.arange(
        1,
        total_loss.size + 1,
    )

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            5.5,
        ),
    )

    positive_total = (
        np.isfinite(
            total_loss
        )
        & (total_loss > 0.0)
    )

    positive_interior = (
        np.isfinite(
            interior_loss
        )
        & (interior_loss > 0.0)
    )

    if np.any(
        positive_interior
    ):
        axis.semilogy(
            epochs[
                positive_interior
            ],
            interior_loss[
                positive_interior
            ],
            color="tab:blue",
            lw=1.8,
            label="Inconel interior PDE",
        )

    if np.any(
        positive_total
    ):
        axis.semilogy(
            epochs[
                positive_total
            ],
            total_loss[
                positive_total
            ],
            color="tab:red",
            lw=2.2,
            label="Total weighted loss",
        )

    axis.set_xlabel(
        "Epoch"
    )

    axis.set_ylabel(
        "Weighted loss"
    )

    axis.set_title(
        "Inconel-only PINN loss history"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.30,
    )

    axis.legend(
        loc="best"
    )

    figure.tight_layout()

    save_figure(
        figure,
        "inconel_only_loss_history.png",
    )


# FVM validation (CryS-MAS reference)

def load_inconel_fvm_data(
    csv_path,
):
    """
    Load the Inconel-only CryS-MAS reference data and keep only the
    rows matching this script's (T0_K, T1) case.

    The CSV holds a parametric sweep over several (T0, T1) pairs, so
    the matching rows must be selected before any comparison.
    """

    csv_path = Path(
        csv_path
    ).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            "Inconel-only FVM data file does not exist:\n"
            f"{csv_path}"
        )

    df = pd.read_csv(
        csv_path,
    )

    required_columns = {
        "x",
        "y",
        "Temperature",
        "T0",
        "T1",
    }

    missing_columns = required_columns.difference(
        df.columns,
    )

    if missing_columns:
        raise ValueError(
            "The FVM data file is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[
            "x",
            "y",
            "Temperature",
            "T0",
            "T1",
        ],
    ).copy()

    df = df[
        np.isclose(
            df["T0"].to_numpy(
                dtype=np.float64,
            ),
            T0_K,
        )
        & np.isclose(
            df["T1"].to_numpy(
                dtype=np.float64,
            ),
            T1,
        )
    ].copy()

    if len(df) == 0:
        raise ValueError(
            "No FVM rows found matching "
            f"T0={T0_K}, T1={T1}."
        )

    return df


def extract_inconel_fvm_vertical_line(
    csv_path,
    target_r=0.004,
    output_name="inconel_only_line_data_extracted.csv",
):
    """
    Extract FVM values on the available radius closest to target_r.

    The returned CSV contains one constant-r vertical probe line and
    is compatible with plot_fvm_vs_pinn_inconel().
    """

    full_df = load_inconel_fvm_data(
        csv_path,
    )

    radial_values = np.unique(
        full_df["x"].to_numpy(
            dtype=np.float64,
        )
    )

    radial_values = np.sort(
        radial_values,
    )

    selected_r = radial_values[
        np.argmin(
            np.abs(
                radial_values
                - target_r
            )
        )
    ]

    line_df = full_df[
        np.isclose(
            full_df["x"].to_numpy(
                dtype=np.float64,
            ),
            selected_r,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    ].copy()

    line_df = line_df.sort_values(
        by="y",
    )

    if len(line_df) < 2:
        raise ValueError(
            "The extracted FVM line has fewer than two nodes. "
            f"Selected r = {selected_r:.10f} m."
        )

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    line_file = (
        SAVE_DIR
        / output_name
    ).resolve()

    line_df.to_csv(
        str(line_file),
        index=False,
    )

    print("\n" + "=" * 72)
    print("EXTRACTED INCONEL-ONLY FVM LINE")
    print("=" * 72)
    print(
        f"Requested radius: {target_r * 1.0e3:.3f} mm"
    )
    print(
        f"Selected FVM radius: {selected_r * 1.0e3:.6f} mm"
    )
    print(
        f"Extracted FVM nodes: {len(line_df)}"
    )
    print(
        f"Saved extracted FVM line:\n{line_file}"
    )

    return line_file


def plot_fvm_vs_pinn_inconel(
    space,
    csv_path,
    output_name="fvm_vs_pinn_inconel_only_line_comparison.png",
):
    """
    Compare FVM temperatures with the Inconel-only PINN prediction
    along a single vertical FVM line.

    The Excel/CSV input must contain one vertical line: all x values
    must have the same radial coordinate.
    """

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        SAVE_DIR
        / output_name
    ).resolve()

    comparison_csv = (
        SAVE_DIR
        / "fvm_vs_pinn_inconel_only_line_comparison.csv"
    ).resolve()

    df = pd.read_csv(
        csv_path,
    )

    r_fvm = df["x"].to_numpy(
        dtype=np.float64,
    )

    z_fvm = df["y"].to_numpy(
        dtype=np.float64,
    )

    T_fvm = df["Temperature"].to_numpy(
        dtype=np.float64,
    )

    # This must be a line at one radius, not the full 2D data.
    if not np.allclose(
        r_fvm,
        r_fvm[0],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        unique_r = np.unique(
            r_fvm,
        )

        raise ValueError(
            "plot_fvm_vs_pinn_inconel() expects one vertical FVM "
            "line with a constant x/r coordinate.\n"
            f"Found {len(unique_r)} radial values.\n"
            "Extract one line first."
        )

    fvm_points = np.column_stack(
        [
            r_fvm,
            z_fvm,
        ]
    )

    T_pinn = evaluate_inconel_field(
        space,
        fvm_points,
    )

    error_K = (
        T_pinn
        - T_fvm
    )

    abs_error_K = np.abs(
        error_K,
    )

    rmse_K = np.sqrt(
        np.mean(
            error_K**2,
        )
    )

    mae_K = np.mean(
        abs_error_K,
    )

    max_abs_error_K = np.max(
        abs_error_K,
    )

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "pinn_minus_fvm_K": error_K,
            "absolute_error_K": abs_error_K,
        }
    )

    comparison_df.to_csv(
        str(comparison_csv),
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM VS PINN LINE COMPARISON: INCONEL ONLY")
    print("=" * 72)

    print(
        f"Number of FVM points: {len(T_fvm)}"
    )

    print(
        f"Probe radius [m]: {r_fvm[0]:.8f}"
    )

    print(
        "Axial-coordinate range [m]: "
        f"{z_fvm.min():.8f} to "
        f"{z_fvm.max():.8f}"
    )

    print(
        f"RMSE [K]: {rmse_K:.8e}"
    )

    print(
        f"MAE [K]: {mae_K:.8e}"
    )

    print(
        "Maximum absolute error [K]: "
        f"{max_abs_error_K:.8e}"
    )

    order = np.argsort(
        z_fvm,
    )

    z_sorted = z_fvm[order]
    T_fvm_sorted = T_fvm[order]
    T_pinn_sorted = T_pinn[order]
    error_sorted = error_K[order]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                3,
                1,
            ],
        },
    )

    axes[0].plot(
        z_sorted * 1.0e3,
        T_fvm_sorted,
        "o",
        ms=4.0,
        color="black",
        label="FVM",
    )

    axes[0].plot(
        z_sorted * 1.0e3,
        T_pinn_sorted,
        "-",
        lw=2.0,
        color="tab:red",
        label="PINN",
    )

    axes[0].set_ylabel(
        "Temperature [K]",
    )

    axes[0].set_title(
        "FVM vs PINN temperature: "
        f"r = {r_fvm[0] * 1.0e3:.3f} mm"
    )

    axes[0].grid(
        True,
        alpha=0.30,
    )

    axes[0].legend()

    axes[0].text(
        0.02,
        0.03,
        (
            f"RMSE = {rmse_K:.3e} K\n"
            f"MAE = {mae_K:.3e} K\n"
            f"Max |error| = "
            f"{max_abs_error_K:.3e} K"
        ),
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.90,
        },
    )

    axes[1].axhline(
        0.0,
        color="black",
        lw=1.0,
    )

    axes[1].plot(
        z_sorted * 1.0e3,
        error_sorted,
        "-o",
        ms=3.0,
        lw=1.2,
        color="tab:blue",
    )

    axes[1].set_xlabel(
        "z [mm]",
    )

    axes[1].set_ylabel(
        "PINN - FVM [K]",
    )

    axes[1].grid(
        True,
        alpha=0.30,
    )

    fig.tight_layout()

    fig.savefig(
        str(figure_path),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig,
    )

    print(
        f"Saved FVM/PINN line plot in:\n{figure_path}"
    )

    print(
        f"Saved FVM/PINN line data in:\n{comparison_csv}"
    )

    return comparison_df


def plot_fvm_vs_pinn_inconel_relative_error(
    space,
    csv_path,
    output_name="fvm_vs_pinn_inconel_only_line_relative_error.png",
):
    """
    Plot the relative error (PINN - FVM) / FVM [%] along a single
    vertical FVM line, for the Inconel-only case.
    """

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        SAVE_DIR
        / output_name
    ).resolve()

    comparison_csv = (
        SAVE_DIR
        / "fvm_vs_pinn_inconel_only_line_relative_error.csv"
    ).resolve()

    df = pd.read_csv(
        csv_path,
    )

    r_fvm = df["x"].to_numpy(
        dtype=np.float64,
    )

    z_fvm = df["y"].to_numpy(
        dtype=np.float64,
    )

    T_fvm = df["Temperature"].to_numpy(
        dtype=np.float64,
    )

    if not np.allclose(
        r_fvm,
        r_fvm[0],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        unique_r = np.unique(
            r_fvm,
        )

        raise ValueError(
            "plot_fvm_vs_pinn_inconel_relative_error() expects one "
            "vertical FVM line with a constant x/r coordinate.\n"
            f"Found {len(unique_r)} radial values.\n"
            "Extract one line first."
        )

    fvm_points = np.column_stack(
        [
            r_fvm,
            z_fvm,
        ]
    )

    T_pinn = evaluate_inconel_field(
        space,
        fvm_points,
    )

    relative_error_pct = (
        (T_pinn - T_fvm)
        / T_fvm
        * 100.0
    )

    absolute_relative_error_pct = np.abs(
        relative_error_pct,
    )

    rmse_pct = np.sqrt(
        np.mean(
            relative_error_pct**2,
        )
    )

    mae_pct = np.mean(
        absolute_relative_error_pct,
    )

    max_abs_relative_error_pct = np.max(
        absolute_relative_error_pct,
    )

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "relative_error_pct": relative_error_pct,
            "absolute_relative_error_pct": absolute_relative_error_pct,
        }
    )

    comparison_df.to_csv(
        str(comparison_csv),
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM VS PINN LINE RELATIVE ERROR: INCONEL ONLY")
    print("=" * 72)

    print(
        f"Number of FVM points: {len(T_fvm)}"
    )

    print(
        f"Probe radius [m]: {r_fvm[0]:.8f}"
    )

    print(
        f"Relative RMSE [%]: {rmse_pct:.8e}"
    )

    print(
        f"Relative MAE [%]: {mae_pct:.8e}"
    )

    print(
        "Maximum absolute relative error [%]: "
        f"{max_abs_relative_error_pct:.8e}"
    )

    order = np.argsort(
        z_fvm,
    )

    z_sorted = z_fvm[order]
    relative_error_sorted = relative_error_pct[order]

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0),
    )

    ax.axhline(
        0.0,
        color="black",
        lw=1.0,
    )

    ax.plot(
        z_sorted * 1.0e3,
        relative_error_sorted,
        "-o",
        ms=3.0,
        lw=1.2,
        color="tab:blue",
    )

    ax.set_xlabel(
        "z [mm]",
    )

    ax.set_ylabel(
        "(PINN - FVM) / FVM [%]",
    )

    ax.set_title(
        "FVM vs PINN relative error: "
        f"r = {r_fvm[0] * 1.0e3:.3f} mm"
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.text(
        0.02,
        0.03,
        (
            f"Relative RMSE = {rmse_pct:.3e} %\n"
            f"Relative MAE = {mae_pct:.3e} %\n"
            f"Max |relative error| = "
            f"{max_abs_relative_error_pct:.3e} %"
        ),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.90,
        },
    )

    fig.tight_layout()

    fig.savefig(
        str(figure_path),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig,
    )

    print(
        f"Saved FVM/PINN line relative-error plot in:\n{figure_path}"
    )

    print(
        f"Saved FVM/PINN line relative-error data in:\n{comparison_csv}"
    )

    return comparison_df


def plot_fvm_vs_pinn_2d_error_inconel(
    space,
    csv_path,
    output_name="fvm_vs_pinn_inconel_only_2d_error.png",
):
    """
    Plot the signed and absolute nodal error over the full 2D
    Inconel-only FVM node set.
    """

    df = load_inconel_fvm_data(
        csv_path,
    )

    r_fvm = df["x"].to_numpy(
        dtype=np.float64,
    )

    z_fvm = df["y"].to_numpy(
        dtype=np.float64,
    )

    T_fvm = df["Temperature"].to_numpy(
        dtype=np.float64,
    )

    fvm_points = np.column_stack(
        [
            r_fvm,
            z_fvm,
        ]
    )

    T_pinn = evaluate_inconel_field(
        space,
        fvm_points,
    )

    signed_error_K = T_pinn - T_fvm

    absolute_error_K = np.abs(
        signed_error_K,
    )

    rmse_K = np.sqrt(
        np.mean(
            signed_error_K**2,
        )
    )

    mae_K = np.mean(
        absolute_error_K,
    )

    max_abs_error_K = np.max(
        absolute_error_K,
    )

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "pinn_minus_fvm_K": signed_error_K,
            "absolute_error_K": absolute_error_K,
        }
    )

    print("\n" + "=" * 72)
    print("FVM VS PINN 2D ERROR SUMMARY: INCONEL ONLY")
    print("=" * 72)
    print(
        f"Valid FVM rows compared: {len(df)}"
    )
    print(
        f"Unique radial coordinates: {np.unique(r_fvm).size}"
    )
    print(
        f"Unique axial coordinates: {np.unique(z_fvm).size}"
    )
    print(
        f"Global RMSE [K]: {rmse_K:.8e}"
    )
    print(
        f"Global MAE [K]: {mae_K:.8e}"
    )
    print(
        "Global maximum absolute error [K]: "
        f"{max_abs_error_K:.8e}"
    )

    signed_limit = np.max(
        np.abs(
            signed_error_K,
        )
    )

    signed_norm = TwoSlopeNorm(
        vmin=-signed_limit,
        vcenter=0.0,
        vmax=signed_limit,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 8.0),
        constrained_layout=True,
    )

    signed_plot = axes[0].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=signed_error_K,
        s=12,
        marker="s",
        cmap="coolwarm",
        norm=signed_norm,
        linewidths=0.0,
        edgecolors="none",
    )

    axes[0].set_title(
        r"Signed nodal error: "
        r"$T_{\mathrm{PINN}}-T_{\mathrm{FVM}}$"
    )

    axes[0].set_xlabel(
        "r [mm]",
    )

    axes[0].set_ylabel(
        "z [mm]",
    )

    axes[0].set_aspect(
        "equal",
    )

    axes[0].grid(
        True,
        alpha=0.20,
    )

    signed_colorbar = figure.colorbar(
        signed_plot,
        ax=axes[0],
    )

    signed_colorbar.set_label(
        "PINN - FVM [K]",
    )

    absolute_plot = axes[1].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=absolute_error_K,
        s=12,
        marker="s",
        cmap="magma",
        vmin=0.0,
        vmax=max_abs_error_K,
        linewidths=0.0,
        edgecolors="none",
    )

    axes[1].set_title(
        r"Absolute nodal error: "
        r"$|T_{\mathrm{PINN}}-T_{\mathrm{FVM}}|$"
    )

    axes[1].set_xlabel(
        "r [mm]",
    )

    axes[1].set_ylabel(
        "z [mm]",
    )

    axes[1].set_aspect(
        "equal",
    )

    axes[1].grid(
        True,
        alpha=0.20,
    )

    absolute_colorbar = figure.colorbar(
        absolute_plot,
        ax=axes[1],
    )

    absolute_colorbar.set_label(
        "Absolute error [K]",
    )

    figure.suptitle(
        (
            "Inconel-only FVM--PINN 2D Nodal Temperature Error\n"
            f"Nodes = {len(T_fvm)}, "
            f"RMSE = {rmse_K:.3f} K, "
            f"MAE = {mae_K:.3f} K, "
            f"Max |error| = {max_abs_error_K:.3f} K"
        ),
        fontsize=14,
    )

    output_path = SAVE_DIR / output_name

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure,
    )

    csv_out_path = (
        SAVE_DIR
        / "fvm_vs_pinn_inconel_only_2d_error_data.csv"
    )

    comparison_df.to_csv(
        csv_out_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM--PINN 2D ERROR FILES: INCONEL ONLY")
    print("=" * 72)
    print(
        f"Saved direct-node error plot in:\n"
        f"{output_path.resolve()}"
    )
    print(
        f"Saved pointwise comparison table in:\n"
        f"{csv_out_path.resolve()}"
    )
    print(
        f"Rows written to comparison CSV: "
        f"{len(comparison_df)}"
    )

    return comparison_df


def plot_fvm_vs_pinn_2d_relative_error_inconel(
    space,
    csv_path,
    output_name="fvm_vs_pinn_inconel_only_2d_relative_error.png",
):
    """
    Plot the signed and absolute relative error
    (PINN - FVM) / FVM [%] over the full 2D Inconel-only FVM node set.
    """

    df = load_inconel_fvm_data(
        csv_path,
    )

    r_fvm = df["x"].to_numpy(
        dtype=np.float64,
    )

    z_fvm = df["y"].to_numpy(
        dtype=np.float64,
    )

    T_fvm = df["Temperature"].to_numpy(
        dtype=np.float64,
    )

    fvm_points = np.column_stack(
        [
            r_fvm,
            z_fvm,
        ]
    )

    T_pinn = evaluate_inconel_field(
        space,
        fvm_points,
    )

    relative_error_pct = (
        (T_pinn - T_fvm)
        / T_fvm
        * 100.0
    )

    absolute_relative_error_pct = np.abs(
        relative_error_pct,
    )

    rmse_pct = np.sqrt(
        np.mean(
            relative_error_pct**2,
        )
    )

    mae_pct = np.mean(
        absolute_relative_error_pct,
    )

    max_abs_relative_error_pct = np.max(
        absolute_relative_error_pct,
    )

    comparison_df = pd.DataFrame(
        {
            "r_m": r_fvm,
            "z_m": z_fvm,
            "T_fvm_K": T_fvm,
            "T_pinn_K": T_pinn,
            "relative_error_pct": relative_error_pct,
            "absolute_relative_error_pct": absolute_relative_error_pct,
        }
    )

    print("\n" + "=" * 72)
    print("FVM VS PINN 2D RELATIVE ERROR SUMMARY: INCONEL ONLY")
    print("=" * 72)
    print(
        f"Valid FVM rows compared: {len(df)}"
    )
    print(
        f"Global relative RMSE [%]: {rmse_pct:.8e}"
    )
    print(
        f"Global relative MAE [%]: {mae_pct:.8e}"
    )
    print(
        "Global maximum absolute relative error [%]: "
        f"{max_abs_relative_error_pct:.8e}"
    )

    signed_limit = np.max(
        np.abs(
            relative_error_pct,
        )
    )

    signed_norm = TwoSlopeNorm(
        vmin=-signed_limit,
        vcenter=0.0,
        vmax=signed_limit,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 8.0),
        constrained_layout=True,
    )

    signed_plot = axes[0].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=relative_error_pct,
        s=12,
        marker="s",
        cmap="coolwarm",
        norm=signed_norm,
        linewidths=0.0,
        edgecolors="none",
    )

    axes[0].set_title(
        r"Signed nodal relative error: "
        r"$(T_{\mathrm{PINN}}-T_{\mathrm{FVM}})/T_{\mathrm{FVM}}$"
    )

    axes[0].set_xlabel(
        "r [mm]",
    )

    axes[0].set_ylabel(
        "z [mm]",
    )

    axes[0].set_aspect(
        "equal",
    )

    axes[0].grid(
        True,
        alpha=0.20,
    )

    signed_colorbar = figure.colorbar(
        signed_plot,
        ax=axes[0],
    )

    signed_colorbar.set_label(
        "(PINN - FVM) / FVM [%]",
    )

    absolute_plot = axes[1].scatter(
        r_fvm * 1.0e3,
        z_fvm * 1.0e3,
        c=absolute_relative_error_pct,
        s=12,
        marker="s",
        cmap="magma",
        vmin=0.0,
        vmax=max_abs_relative_error_pct,
        linewidths=0.0,
        edgecolors="none",
    )

    axes[1].set_title(
        r"Absolute nodal relative error: "
        r"$|T_{\mathrm{PINN}}-T_{\mathrm{FVM}}|/T_{\mathrm{FVM}}$"
    )

    axes[1].set_xlabel(
        "r [mm]",
    )

    axes[1].set_ylabel(
        "z [mm]",
    )

    axes[1].set_aspect(
        "equal",
    )

    axes[1].grid(
        True,
        alpha=0.20,
    )

    absolute_colorbar = figure.colorbar(
        absolute_plot,
        ax=axes[1],
    )

    absolute_colorbar.set_label(
        "|(PINN - FVM) / FVM| [%]",
    )

    figure.suptitle(
        (
            "Inconel-only FVM--PINN 2D Nodal Relative Temperature Error\n"
            f"Nodes = {len(T_fvm)}, "
            f"Relative RMSE = {rmse_pct:.3f} %, "
            f"Relative MAE = {mae_pct:.3f} %, "
            f"Max |relative error| = {max_abs_relative_error_pct:.3f} %"
        ),
        fontsize=14,
    )

    output_path = SAVE_DIR / output_name

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure,
    )

    csv_out_path = (
        SAVE_DIR
        / "fvm_vs_pinn_inconel_only_2d_relative_error_data.csv"
    )

    comparison_df.to_csv(
        csv_out_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FVM--PINN 2D RELATIVE ERROR FILES: INCONEL ONLY")
    print("=" * 72)
    print(
        f"Saved relative-error plot in:\n"
        f"{output_path.resolve()}"
    )
    print(
        f"Saved pointwise comparison table in:\n"
        f"{csv_out_path.resolve()}"
    )
    print(
        f"Rows written to comparison CSV: "
        f"{len(comparison_df)}"
    )

    return comparison_df


# Run

if __name__ == "__main__":

    space = build_space(
        seed=42,
    )

    print(
        "Number of neural models:",
        len(
            space.models,
        ),
    )

    pinn = Projector(
        model,
        space,
        sampler,
        #optimizer="Adam",
        weights=W_COMMON,
        learning_rate=LEARNING_RATE,
    )

    key = jax.random.PRNGKey(
        42,
    )

    key, samples = sampler.sample(
        key,
        N_COLLOC,
        N_BC_COLLOC,
    )

    print("\n" + "=" * 72)
    print("SAMPLE KEYS")
    print("=" * 72)

    print(
        samples.keys(),
    )

    print("\n" + "=" * 72)
    print("INITIAL LOSSES")
    print("=" * 72)

    initial_losses = pinn.evaluate_losses(
        space,
        samples,
    )

    print(
        initial_losses,
    )

    start = timeit.default_timer()

    key, pinn = pinn.project(
        key,
        space,
        N_EPOCHS,
        N_COLLOC,
        N_BC_COLLOC,
        tqdmdesc="Inconel-only axisymmetric conduction",
    )

    elapsed = (
        timeit.default_timer()
        - start
    )

    print("\n" + "=" * 72)
    print("FINAL LOSSES ON FRESH SAMPLES")
    print("=" * 72)

    key, final_samples = sampler.sample(
        key,
        N_COLLOC,
        N_BC_COLLOC,
    )

    final_losses = pinn.evaluate_losses(
        pinn.space,
        final_samples,
    )

    print(
        final_losses,
    )

    plot_inconel_temperature_field(
        pinn.space,
    )

    plot_axis_derivative_diagnostic(
        pinn.space,
    )

    plot_inconel_loss_history(
        pinn.losses,
    )

    print("\n" + "=" * 72)
    print("BEST LOSS")
    print("=" * 72)

    print(
        pinn.best_loss,
    )

    print(
        f"\nTraining time: {elapsed:.2f} s"
    )

    # FVM validation (CryS-MAS reference)

    PROJECT_DIR = Path(
        __file__
    ).resolve().parents[5]

    FVM_FILE_INCONEL = (
        PROJECT_DIR
        / "2D_Heat_Inconel_data.csv"
    )

    print("\n" + "=" * 72)
    print("FVM VALIDATION FILE CHECK: INCONEL ONLY")
    print("=" * 72)

    print(
        f"Expected Inconel-only FVM file:\n"
        f"{FVM_FILE_INCONEL.resolve()}"
    )

    print(
        "File exists:",
        FVM_FILE_INCONEL.exists(),
    )

    if not FVM_FILE_INCONEL.exists():
        raise FileNotFoundError(
            "The Inconel-only FVM data file was not found:\n"
            f"{FVM_FILE_INCONEL.resolve()}\n\n"
            "Copy 2D_Heat_Inconel_data.csv into the "
            "scimba_dev_og folder, or replace FVM_FILE_INCONEL "
            "with the correct absolute path."
        )

    # Full 2D direct-node FVM/PINN error comparison.

    print("\n" + "=" * 72)
    print("RUNNING FVM VS PINN 2D NODAL COMPARISON")
    print("=" * 72)

    comparison_2d_df = plot_fvm_vs_pinn_2d_error_inconel(
        pinn.space,
        csv_path=FVM_FILE_INCONEL,
    )

    comparison_2d_relative_df = plot_fvm_vs_pinn_2d_relative_error_inconel(
        pinn.space,
        csv_path=FVM_FILE_INCONEL,
    )

    # 1D FVM/PINN line comparison.

    TARGET_LINE_RADIUS = 0.004

    FVM_FILE_LINE = extract_inconel_fvm_vertical_line(
        csv_path=FVM_FILE_INCONEL,
        target_r=TARGET_LINE_RADIUS,
    )

    print("\n" + "=" * 72)
    print("RUNNING FVM VS PINN 1D LINE COMPARISON")
    print("=" * 72)

    comparison_line_df = plot_fvm_vs_pinn_inconel(
        pinn.space,
        csv_path=FVM_FILE_LINE,
    )

    comparison_line_relative_df = plot_fvm_vs_pinn_inconel_relative_error(
        pinn.space,
        csv_path=FVM_FILE_LINE,
    )

    print("\n" + "=" * 72)
    print("INCONEL-ONLY FVM VALIDATION COMPLETE")
    print("=" * 72)

    print("2D FVM/PINN error plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_only_2d_error.png").resolve())

    print("\n2D FVM/PINN relative-error plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_only_2d_relative_error.png").resolve())

    print("\nExtracted 1D FVM line data:")
    print(FVM_FILE_LINE.resolve())

    print("\n1D FVM/PINN line comparison plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_only_line_comparison.png").resolve())

    print("\n1D FVM/PINN line relative-error plot:")
    print((SAVE_DIR / "fvm_vs_pinn_inconel_only_line_relative_error.png").resolve())
