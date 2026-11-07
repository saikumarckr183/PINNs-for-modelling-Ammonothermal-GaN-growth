from __future__ import annotations


import os
import timeit


import jax


jax.config.update(
    "jax_enable_x64",
    True,
)


import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import scimba_jax


from scimba_jax.domains.domain_mapping import (
    DomainMapping,
)
from scimba_jax.domains.meshless_domains.base import (
    DEFAULT_LABEL_STR_SURFACIC,
    DEFAULT_LABEL_STR_VOLUMETRIC,
    SurfacicDomain,
    VolumetricDomain,
)
from scimba_jax.domains.meshless_domains.domains_nd import (
    HypercubeND,
)
from scimba_jax.domains.sdf import (
    SignedDistance,
)
from scimba_jax.nonlinear_approximation.approximation_spaces.approximation_spaces import (
    ApproximationSpace,
)
from scimba_jax.nonlinear_approximation.integration.monte_carlo import (
    DomainSampler,
    TensorizedSampler,
)
from scimba_jax.nonlinear_approximation.model_class.funcparam_vectorial import (
    ParamFieldFunction,
    ParamVecFunction,
)
from scimba_jax.nonlinear_approximation.networks.mlp import (
    MLP,
)
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import (
    Projector,
)
from scimba_jax.physical_models.abstract_physical_model import (
    AbstractPhysicalModel,
)
from scimba_jax.physical_models.abstract_residuals import (
    BoundaryResidual,
    InteriorResidual,
    PARAM_FUNC_TYPE,
    static,
)
from scimba_jax.physical_models.elliptic_pde.navier_stokes_stationary import (
    _extract_u_p_t,
)
from scimba_jax.plots.plots_nd import plot_abstract_approx_space



# General configuration
MODE = "vec"


OUTPUT_DIR = "output_axisymmetric_nh3_steady"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# NH3 cavity geometry (PHYSICAL units, meters) — used only to build the
# nondimensionalization block below and for final physical-unit conversion.
# All domain/network code downstream operates on nondimensional r_hat, z_hat.
# ---------------------------------------------------------------------------
R_NH3 = 0.012
Z_NH3_TOP = 0.335


NH3_LOWER_R = jnp.array(
    [
        0.0000,
        0.0031,
        0.0060,
        0.0085,
        0.0104,
        0.0116,
        0.0120,
    ],
    dtype=jnp.float64,
)


NH3_LOWER_Z = jnp.array(
    [
        0.0230,
        0.0234,
        0.0246,
        0.0265,
        0.0290,
        0.0319,
        0.0350,
    ],
    dtype=jnp.float64,
)


Z_NH3_AXIS_BOTTOM = float(
    NH3_LOWER_Z[
        0
    ]
)


Z_NH3_WALL_BOTTOM = float(
    NH3_LOWER_Z[
        -1
    ]
)



# Prescribed wall-temperature profile (PHYSICAL units, Kelvin)
# T_BC(z) = T_INTERCEPT + T_SLOPE * z


T_INTERCEPT = 800.65
T_SLOPE = -400.0
T_REF = 700.05


# NH3 properties (PHYSICAL units — updated to Schimmel et al. 2023 values)


RHO_NH3 = 233.950
MU_NH3 = 3.673e-5
CP_NH3 = 4216.5
K_NH3 = 0.16224
ALPHA_NH3 = 2.631e-3


NU_NH3 = MU_NH3 / RHO_NH3
KAPPA_NH3 = K_NH3 / (
    RHO_NH3
    * CP_NH3
)



# gravity term (PHYSICAL units, m/s^2)
GRAVITY_Z = -9.81


# ---------------------------------------------------------------------------
# Nondimensionalization block
# ---------------------------------------------------------------------------

# Reference scales
L_REF = Z_NH3_TOP - Z_NH3_AXIS_BOTTOM          # vertical length scale [m]
R_REF = R_NH3                                   # radial length scale [m]
DELTA_T_REF = abs(T_SLOPE) * L_REF              # wall-to-wall temperature drop [K]

# Buoyancy-consistent velocity scale: U = sqrt(alpha * g * dT * L)
U_REF = (ALPHA_NH3 * abs(GRAVITY_Z) * DELTA_T_REF * L_REF) ** 0.5
P_REF = RHO_NH3 * U_REF**2

# Dimensionless viscosity / diffusivity (residual coefficients)
NU_HAT = NU_NH3 / (U_REF * L_REF)
KAPPA_HAT = KAPPA_NH3 / (U_REF * L_REF)

# Dimensionless groups, for diagnostics/logging only
PRANDTL_NUMBER = NU_NH3 / KAPPA_NH3
RAYLEIGH_NUMBER = (
    ALPHA_NH3 * abs(GRAVITY_Z) * DELTA_T_REF * L_REF**3
) / (NU_NH3 * KAPPA_NH3)

print("\n" + "=" * 72)
print("NONDIMENSIONALIZATION SUMMARY")
print("=" * 72)
print(f"L_REF        = {L_REF:.6f} m")
print(f"R_REF        = {R_REF:.6f} m")
print(f"DELTA_T_REF  = {DELTA_T_REF:.4f} K")
print(f"U_REF        = {U_REF:.6e} m/s")
print(f"P_REF        = {P_REF:.6e} Pa")
print(f"NU_HAT       = {NU_HAT:.6e}")
print(f"KAPPA_HAT    = {KAPPA_HAT:.6e}")
print(f"Prandtl (Pr) = {PRANDTL_NUMBER:.6f}")
print(f"Rayleigh(Ra) = {RAYLEIGH_NUMBER:.6e}")
print(
    "Sanity check vs. paper-reported growth-stage velocity (~0.02 m/s): "
    f"U_REF / 0.02 = {U_REF / 0.02:.2f}x "
    "(laminar Boussinesq predicts a faster scale than the paper's "
    "turbulence-model result at this Ra; large deviation is expected, not a bug)."
)

# Nondimensional coordinates, each scaled by its own physical extent:
#   r_hat = r / R_REF          in [0, 1]
#   z_hat = (z - Z_NH3_AXIS_BOTTOM) / L_REF   in [0, 1]
# All geometry classes below are built directly in (r_hat, z_hat).

R_HAT_MAX = 1.0
Z_HAT_AXIS_BOTTOM = 0.0
Z_HAT_TOP = 1.0
Z_HAT_WALL_BOTTOM = (Z_NH3_WALL_BOTTOM - Z_NH3_AXIS_BOTTOM) / L_REF

NH3_LOWER_R_HAT = NH3_LOWER_R / R_REF
NH3_LOWER_Z_HAT = (NH3_LOWER_Z - Z_NH3_AXIS_BOTTOM) / L_REF

# Nondimensional temperature boundary profile:
#   T_hat_BC(z_hat) = (T_BC(z) - T_REF) / DELTA_T_REF
# with z = Z_NH3_AXIS_BOTTOM + z_hat * L_REF, precomputed as a linear function:
#   T_hat_BC(z_hat) = A_HAT + B_HAT * z_hat
T_INTERCEPT_AT_AXIS_BOTTOM = T_INTERCEPT + T_SLOPE * Z_NH3_AXIS_BOTTOM
A_HAT = (T_INTERCEPT_AT_AXIS_BOTTOM - T_REF) / DELTA_T_REF
B_HAT = (T_SLOPE * L_REF) / DELTA_T_REF

print(f"A_HAT (T_hat_BC intercept) = {A_HAT:.6f}")
print(f"B_HAT (T_hat_BC slope)     = {B_HAT:.6f}")


# Training settings


N_COLLOC = 4000
N_BC_COLLOC = 1200
N_EPOCHS = 100


HIDDEN_SIZES = [
    16,
    16,
    16,
    16,
]


WEIGHTS = {
    # Interior residual ordering:
    # [radial momentum, axial momentum, continuity, energy]
    "interior": [
        5.0,
        5.0,
        1.0,
        1.0,
    ],


    # Axis residual ordering:
    # [u_r, d(u_z)/dr, dT/dr]
    "axis": [
        20.0,
        10.0,
        10.0,
    ],


    # Physical-wall residual ordering:
    # [u_r, u_z, T - T_BC(z)]
    "outer_side": [
        20.0,
        20.0,
        20.0,
    ],


    "outer_top": [
        20.0,
        20.0,
        20.0,
    ],


    "lower_polyline": [
        20.0,
        20.0,
        20.0,
    ],
}



# Coordinate helper functions (now operate on nondimensional x = [r_hat, z_hat])


def r_coord(
    x,
):
    """
    Return r_hat from x = [r_hat, z_hat].
    """


    return x[
        0
    ]



def z_coord(
    x,
):
    """
    Return z_hat from x = [r_hat, z_hat].
    """


    return x[
        1
    ]



def r2_coord(
    x,
):
    """
    Return r_hat^2 from x = [r_hat, z_hat].
    """


    return x[
        0
    ] ** 2



def temperature_boundary_profile(
    x,
):
    """
    Prescribed nondimensional temperature on every material boundary.


    T_hat_BC(z_hat) = A_HAT + B_HAT * z_hat.
    """


    return (
        A_HAT
        + B_HAT
        * z_coord(
            x
        )
    )



# JAX and NumPy lower polyline profiles (nondimensional)


def nh3_lower_profile_jax(
    r,
):
    """
    JAX-compatible piecewise-linear lower NH3/Inconel boundary.


    Input:
        r, scalar or array in [0, 1] (nondimensional r_hat).


    Output:
        z_hat_lower(r_hat).
    """


    return jnp.interp(
        r,
        NH3_LOWER_R_HAT,
        NH3_LOWER_Z_HAT,
    )



def nh3_lower_profile_numpy(
    r,
):
    """
    NumPy version for plotting masks and diagnostics (nondimensional).
    """


    return np.interp(
        np.asarray(
            r,
            dtype=np.float64,
        ),
        np.asarray(
            NH3_LOWER_R_HAT,
            dtype=np.float64,
        ),
        np.asarray(
            NH3_LOWER_Z_HAT,
            dtype=np.float64,
        ),
    )



# Signed-distance function for ammonia (nondimensional domain)


class NH3CavitySDF(
    SignedDistance,
):
    r"""
    Approximate SDF for:


        0 <= r_hat <= 1
        z_hat_lower(r_hat) <= z_hat <= 1


    Sign convention:
        sdf < 0  inside NH3
        sdf = 0  boundary
        sdf > 0  outside NH3


    The returned field is a practical implicit-domain function
    suitable for SCIMBA meshless sampling. It is not an exact
    Euclidean distance near polyline vertices.
    """


    def __init__(
        self,
        radius: float,
        z_top: float,
    ):
        super().__init__(
            dim=2,
            threshold=0.0,
        )


        self.radius = radius
        self.z_top = z_top


    def _sdf_pointwise(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Return one signed implicit value for one point x = [r_hat, z_hat].
        """


        r = x[
            0
        ]


        z = x[
            1
        ]


        z_lower = nh3_lower_profile_jax(
            r,
        )


        implicit_value = jnp.maximum(
            jnp.maximum(
                -r,
                r - self.radius,
            ),
            jnp.maximum(
                z_lower - z,
                z - self.z_top,
            ),
        )


        return implicit_value[
            None
        ]



# Boundary-domain classes (all in nondimensional r_hat, z_hat)


class AxisSegmentNH3(
    SurfacicDomain,
):
    """
    Axis boundary:
        r_hat = 0,
        Z_HAT_AXIS_BOTTOM <= z_hat <= Z_HAT_TOP.
    """


    def __init__(
        self,
        z_min: float,
        z_max: float,
        label_str: str = DEFAULT_LABEL_STR_SURFACIC,
        label_idx: int = 0,
    ):
        parametric_domain = HypercubeND(
            [
                (
                    z_min,
                    z_max,
                ),
            ]
        )


        def map_fn(
            t,
        ):
            z = t[
                0:1
            ]


            return jnp.concatenate(
                [
                    jnp.zeros(
                        (
                            1,
                        ),
                        dtype=t.dtype,
                    ),
                    z,
                ],
                axis=-1,
            )


        def jac_fn(
            t,
        ):
            del t


            return jnp.array(
                [
                    [0.0],
                    [1.0],
                ],
                dtype=scimba_jax.get_default_dtype(),
            )


        mapping = DomainMapping(
            1,
            2,
            map_fn,
            jac_fn,
            None,
            None,
        )


        super().__init__(
            domain_type="AxisSegmentNH3",
            parametric_domain=parametric_domain,
            surface=mapping,
            label_str=label_str,
            label_idx=label_idx,
        )



class OuterSideSegmentNH3(
    SurfacicDomain,
):
    """
    Outer vertical ammonia wall:
        r_hat = 1,
        Z_HAT_WALL_BOTTOM <= z_hat <= Z_HAT_TOP.
    """


    def __init__(
        self,
        radius: float,
        z_min: float,
        z_max: float,
        label_str: str = DEFAULT_LABEL_STR_SURFACIC,
        label_idx: int = 0,
    ):
        parametric_domain = HypercubeND(
            [
                (
                    z_min,
                    z_max,
                ),
            ]
        )


        def map_fn(
            t,
        ):
            z = t[
                0:1
            ]


            return jnp.concatenate(
                [
                    jnp.array(
                        [
                            radius,
                        ],
                        dtype=t.dtype,
                    ),
                    z,
                ],
                axis=-1,
            )


        def jac_fn(
            t,
        ):
            del t


            return jnp.array(
                [
                    [0.0],
                    [1.0],
                ],
                dtype=scimba_jax.get_default_dtype(),
            )


        mapping = DomainMapping(
            1,
            2,
            map_fn,
            jac_fn,
            None,
            None,
        )


        super().__init__(
            domain_type="OuterSideSegmentNH3",
            parametric_domain=parametric_domain,
            surface=mapping,
            label_str=label_str,
            label_idx=label_idx,
        )



class TopSegmentNH3(
    SurfacicDomain,
):
    """
    Top boundary:
        0 <= r_hat <= 1,
        z_hat = Z_HAT_TOP.
    """


    def __init__(
        self,
        radius: float,
        z_top: float,
        label_str: str = DEFAULT_LABEL_STR_SURFACIC,
        label_idx: int = 0,
    ):
        parametric_domain = HypercubeND(
            [
                (
                    0.0,
                    radius,
                ),
            ]
        )


        def map_fn(
            t,
        ):
            r = t[
                0:1
            ]


            return jnp.concatenate(
                [
                    r,
                    jnp.array(
                        [
                            z_top,
                        ],
                        dtype=t.dtype,
                    ),
                ],
                axis=-1,
            )


        def jac_fn(
            t,
        ):
            del t


            return jnp.array(
                [
                    [1.0],
                    [0.0],
                ],
                dtype=scimba_jax.get_default_dtype(),
            )


        mapping = DomainMapping(
            1,
            2,
            map_fn,
            jac_fn,
            None,
            None,
        )


        super().__init__(
            domain_type="TopSegmentNH3",
            parametric_domain=parametric_domain,
            surface=mapping,
            label_str=label_str,
            label_idx=label_idx,
        )



class LowerPolylineNH3(
    SurfacicDomain,
):
    """
    Piecewise-linear NH3/Inconel interface (nondimensional r_hat, z_hat).


    Parameter:
        s in [0, N_SEGMENTS]


    For a parameter s:
        segment i = floor(s)
        local coordinate = s - i


    The final endpoint is handled by clipping s below N_SEGMENTS.
    """


    def __init__(
        self,
        r_points: jnp.ndarray,
        z_points: jnp.ndarray,
        label_str: str = DEFAULT_LABEL_STR_SURFACIC,
        label_idx: int = 0,
    ):
        self.r_points = r_points
        self.z_points = z_points
        self.n_segments = int(
            r_points.shape[
                0
            ]
            - 1
        )


        parametric_domain = HypercubeND(
            [
                (
                    0.0,
                    float(
                        self.n_segments
                    ),
                ),
            ]
        )


        def polyline_position(
            s,
        ):
            s_safe = jnp.clip(
                s,
                0.0,
                float(
                    self.n_segments
                )
                - 1.0e-7,
            )


            index = jnp.floor(
                s_safe
            ).astype(
                jnp.int32
            )


            local_coordinate = (
                s_safe
                - index.astype(
                    s_safe.dtype
                )
            )


            r0 = r_points[
                index
            ]


            r1 = r_points[
                index
                + 1
            ]


            z0 = z_points[
                index
            ]


            z1 = z_points[
                index
                + 1
            ]


            r = (
                1.0
                - local_coordinate
            ) * r0 + local_coordinate * r1


            z = (
                1.0
                - local_coordinate
            ) * z0 + local_coordinate * z1


            return r, z, index


        def map_fn(
            t,
        ):
            s = t[
                0
            ]


            r, z, _ = polyline_position(
                s,
            )


            return jnp.stack(
                [
                    r,
                    z,
                ]
            )


        def jac_fn(
            t,
        ):
            s = t[
                0
            ]


            _, _, index = polyline_position(
                s,
            )


            dr_ds = (
                r_points[
                    index
                    + 1
                ]
                - r_points[
                    index
                ]
            )


            dz_ds = (
                z_points[
                    index
                    + 1
                ]
                - z_points[
                    index
                ]
            )


            return jnp.stack(
                [
                    dr_ds,
                    dz_ds,
                ]
            ).reshape(
                2,
                1,
            )


        mapping = DomainMapping(
            1,
            2,
            map_fn,
            jac_fn,
            None,
            None,
        )


        super().__init__(
            domain_type="LowerPolylineNH3",
            parametric_domain=parametric_domain,
            surface=mapping,
            label_str=label_str,
            label_idx=label_idx,
        )



# Main ammonia domain (nondimensional)


class NH3CavityDomain(
    VolumetricDomain,
):
    """
    Axisymmetric ammonia cavity in the nondimensional (r_hat, z_hat) meridional plane.
    """


    def __init__(
        self,
        radius: float,
        z_top: float,
        is_main_domain: bool = False,
        label_str: str = DEFAULT_LABEL_STR_VOLUMETRIC,
        label_idx: int = 0,
    ):
        bounds = jnp.array(
            [
                (
                    0.0,
                    radius,
                ),
                (
                    Z_HAT_AXIS_BOTTOM,
                    z_top,
                ),
            ],
            dtype=scimba_jax.get_default_dtype(),
        )


        sdf = NH3CavitySDF(
            radius,
            z_top,
        )


        super().__init__(
            domain_type="NH3CavityDomain",
            dim=2,
            sdf=sdf,
            bounds=bounds,
            is_main_domain=is_main_domain,
            label_str=label_str,
            label_idx=label_idx,
        )


        self.radius = radius
        self.z_top = z_top


    def full_bc_domain(
        self,
    ):
        label_bc = "bc"


        if not self.is_main_domain:
            label_bc += (
                " "
                + self.get_label()
            )


        boundaries = [
            AxisSegmentNH3(
                Z_HAT_AXIS_BOTTOM,
                self.z_top,
                label_str=label_bc + " axis",
            ),
            OuterSideSegmentNH3(
                self.radius,
                Z_HAT_WALL_BOTTOM,
                self.z_top,
                label_str=label_bc + " outer_side",
            ),
            TopSegmentNH3(
                self.radius,
                self.z_top,
                label_str=label_bc + " outer_top",
            ),
            LowerPolylineNH3(
                NH3_LOWER_R_HAT,
                NH3_LOWER_Z_HAT,
                label_str=label_bc + " lower_polyline",
            ),
        ]


        if self.is_mapped:
            for boundary in boundaries:
                boundary._set_mapping(
                    self.mapping,
                )


        return boundaries



# Interior axisymmetric Boussinesq residual (nondimensional)


class AxisymmetricBoussinesqResidual(
    InteriorResidual,
):
    r"""
    Steady axisymmetric Boussinesq equations, nondimensionalized.


    x = [r_hat, z_hat]
    output = [u_r_hat, u_z_hat, p_hat, T_hat]


    Residual order:
        0: radial momentum
        1: axial momentum
        2: continuity
        3: energy


    r-weighted residuals are used to avoid explicit 1/r_hat and
    1/r_hat^2 factors at r_hat = 0.


    nu -> NU_HAT = sqrt(Pr/Ra), kappa -> KAPPA_HAT = 1/sqrt(Ra*Pr),
    alpha -> 1.0, T_ref -> 0.0, gravity_z -> -1.0 (direction only;
    magnitude absorbed into U_REF during nondimensionalization).
    """


    nu: float = static(1.0)
    kappa: float = static(1.0)
    alpha: float = static(0.0)
    T_ref: float = static(0.0)
    gravity_z: float = static(0.0)
    mode: str = static("vec")


    def __init__(
        self,
        domain,
        nu: float,
        kappa: float,
        alpha: float,
        T_ref: float,
        gravity_z: float,
        mode: str = "vec",
    ):
        super().__init__(
            domain=domain,
            size=4,
            model_type="x",
            f_rhs=None,
        )


        self.nu = nu
        self.kappa = kappa
        self.alpha = alpha
        self.T_ref = T_ref
        self.gravity_z = gravity_z
        self.mode = mode


    def construct_residual(
        self,
        *vars: PARAM_FUNC_TYPE,
    ):
        """
        Return symbolic residual fields.
        """


        u, pressure, temperature = _extract_u_p_t(
            vars,
            self.mode,
            dim=2,
        )


        ur = u.component(
            0,
        )


        uz = u.component(
            1,
        )


        ur_r = ur.partial_derivative(
            "x",
            0,
        )


        ur_z = ur.partial_derivative(
            "x",
            1,
        )


        uz_r = uz.partial_derivative(
            "x",
            0,
        )


        uz_z = uz.partial_derivative(
            "x",
            1,
        )


        ur_rr = ur_r.partial_derivative(
            "x",
            0,
        )


        ur_zz = ur_z.partial_derivative(
            "x",
            1,
        )


        uz_rr = uz_r.partial_derivative(
            "x",
            0,
        )


        uz_zz = uz_z.partial_derivative(
            "x",
            1,
        )


        temperature_r = temperature.partial_derivative(
            "x",
            0,
        )


        temperature_z = temperature.partial_derivative(
            "x",
            1,
        )


        temperature_rr = temperature_r.partial_derivative(
            "x",
            0,
        )


        temperature_zz = temperature_z.partial_derivative(
            "x",
            1,
        )


        pressure_r = pressure.partial_derivative(
            "x",
            0,
        )


        pressure_z = pressure.partial_derivative(
            "x",
            1,
        )


        radial_momentum_base = (
            ur
            * ur_r
            + uz
            * ur_z
            + pressure_r
            - self.nu
            * (
                ur_rr
                + ur_zz
            )
        )


        radial_momentum = (
            radial_momentum_base
            * r2_coord
            - (
                ur_r
                * r_coord
            )
            * self.nu
            + self.nu
            * ur
        )


        axial_momentum_base = (
            ur
            * uz_r
            + uz
            * uz_z
            + pressure_z
            - self.alpha
            * (
                temperature
                - self.T_ref
            )
            * self.gravity_z
            - self.nu
            * (
                uz_rr
                + uz_zz
            )
        )


        axial_momentum = (
            axial_momentum_base
            * r_coord
            - self.nu
            * uz_r
        )


        continuity = (
            ur_r
            * r_coord
            + ur
            + uz_z
            * r_coord
        )


        energy_base = (
            ur
            * temperature_r
            + uz
            * temperature_z
            - self.kappa
            * (
                temperature_rr
                + temperature_zz
            )
        )


        energy = (
            energy_base
            * r_coord
            - self.kappa
            * temperature_r
        )


        return ParamVecFunction.cat(
            [
                radial_momentum,
                axial_momentum,
                continuity,
                energy,
            ]
        )


# Boundary residuals


class AxisRegularityBC(
    BoundaryResidual,
):
    r"""
    Axis regularity conditions on r_hat = 0.


    Residual ordering:
        [
            u_r,
            d(u_z)/dr,
            dT/dr,
        ]


    Conditions:
        u_r = 0
        du_z/dr = 0
        dT/dr = 0
    """


    _mode: str = static("vec")


    def __init__(
        self,
        domain,
        mode: str = "vec",
    ):
        super().__init__(
            domain=domain,
            size=3,
            model_type="x",
            f_rhs=None,
        )


        self._mode = mode


    def construct_residual(
        self,
        *vars: PARAM_FUNC_TYPE,
    ):
        u, _, temperature = _extract_u_p_t(
            vars,
            self._mode,
            dim=2,
        )


        ur = u.component(
            0,
        )


        uz = u.component(
            1,
        )


        uz_r = uz.partial_derivative(
            "x",
            0,
        )


        temperature_r = temperature.partial_derivative(
            "x",
            0,
        )


        return ParamVecFunction.cat(
            [
                ur,
                uz_r,
                temperature_r,
            ]
        )



class NoSlipTemperatureBC(
    BoundaryResidual,
):
    r"""
    Physical wall residual (nondimensional).


    Residual ordering:
        [
            u_r,
            u_z,
            T_hat - T_hat_BC(z_hat),
        ]


    Applied to:
        outer side wall,
        top wall,
        lower NH3/Inconel polyline.
    """


    _mode: str = static("vec")


    def __init__(
        self,
        domain,
        mode: str = "vec",
    ):
        super().__init__(
            domain=domain,
            size=3,
            model_type="x",
            f_rhs=None,
        )


        self._mode = mode


    def construct_residual(
        self,
        *vars: PARAM_FUNC_TYPE,
    ):
        u, _, temperature = _extract_u_p_t(
            vars,
            self._mode,
            dim=2,
        )


        ur = u.component(
            0,
        )


        uz = u.component(
            1,
        )


        zero_field = temperature * 0.0


        z_field = (
            zero_field
            + 1.0
        ) * z_coord


        prescribed_temperature = (
            z_field
            * B_HAT
            + A_HAT
        )


        return ParamVecFunction.cat(
            [
                ur,
                uz,
                temperature
                - prescribed_temperature,
            ]
        )



# physical model


class AxisymmetricBoussinesqModel(
    AbstractPhysicalModel,
):
    """
    High-level steady axisymmetric no-swirl Boussinesq model (nondimensional).
    """


    def __init__(
        self,
        main_domain,
        nu: float,
        kappa: float,
        alpha: float,
        T_ref: float,
        gravity_z: float,
        mode: str,
        bc_residuals,
    ):
        super().__init__(
            main_domain=main_domain,
        )


        self.physical_residuals = {
            main_domain.get_label(): AxisymmetricBoussinesqResidual(
                domain=main_domain,
                nu=nu,
                kappa=kappa,
                alpha=alpha,
                T_ref=T_ref,
                gravity_z=gravity_z,
                mode=mode,
            ),
        }


        self.physical_residuals.update(
            bc_residuals,
        )



# ── Patch v2: stronger symmetry-breaking seed + reduced continuity weight ──
# Two changes from before:
#   1. INIT_PERTURBATION_AMPLITUDE raised from 1e-2 to 0.5 (comparable to the
#      O(1) target scale set by U_REF), since 1e-2 was likely optimized away
#      within the first few epochs given how strongly the hydrostatic
#      solution attracts at ANY Ra (confirmed by your stage-1 result).

INIT_PERTURBATION_AMPLITUDE = 0.5  # nondimensional, O(1) target scale


def post_processing_axisymmetric(output, x):
    r = x[0:1]
    z = x[1:2]

    ur_raw = output[0:1]
    uz = output[1:2]
    pressure = output[2:3]
    temperature_correction = output[3:4]

    # Symmetry-breaking seed: single convective cell, exactly satisfying
    # no-slip at z_hat=0,1 (via sin(pi*z_hat)) and at r_hat=0,1 (via the
    # r*(1-r) envelope on ur and sin(pi*r_hat) envelope on uz), so it does
    # not violate the boundary residuals it will be trained against.
    seed_ur = jnp.sin(jnp.pi * z) * r * (1.0 - r)
    seed_uz = jnp.cos(jnp.pi * z) * jnp.sin(jnp.pi * r)

    ur = r * ur_raw + INIT_PERTURBATION_AMPLITUDE * seed_ur
    uz = uz + INIT_PERTURBATION_AMPLITUDE * seed_uz

    temperature_base = A_HAT + B_HAT * z
    temperature = temperature_base + temperature_correction

    return jnp.concatenate([ur, uz, pressure, temperature], axis=-1)


# ---------------------------------------------------------------------------
# Physical-unit conversion helpers (for plotting / finite-volume comparison)
# ---------------------------------------------------------------------------


def to_physical_units(
    r_hat,
    z_hat,
    ur_hat,
    uz_hat,
    p_hat,
    T_hat,
):
    """
    Map nondimensional PINN outputs back to physical units.


    r = r_hat * R_REF
    z = Z_NH3_AXIS_BOTTOM + z_hat * L_REF
    u_r = u_r_hat * U_REF
    u_z = u_z_hat * U_REF
    p = p_hat * P_REF
    T = T_REF + T_hat * DELTA_T_REF
    """


    r = r_hat * R_REF
    z = Z_NH3_AXIS_BOTTOM + z_hat * L_REF
    u_r = ur_hat * U_REF
    u_z = uz_hat * U_REF
    p = p_hat * P_REF
    temperature = T_REF + T_hat * DELTA_T_REF


    return r, z, u_r, u_z, p, temperature

# ── plotting helpers ────────────────────────────────────────────────────────
# Paste this block BEFORE plotting_block_v3 (before Figure 1). These are
# unchanged from your original script, since draw_nh3_outline already uses
# physical-unit constants (R_NH3, Z_NH3_TOP, NH3_LOWER_R, NH3_LOWER_Z), which
# matches v3's RR/ZZ grids (also converted to physical units via
# to_physical_units before plotting). No unit mismatch.

def draw_nh3_outline(axis):
    """
    Draw the exact meridional NH3 cavity boundary, in physical units (mm),
    matching the *1.0e3 scaling used on RR/ZZ in the contour plots.
    """
    lower_r = np.asarray(NH3_LOWER_R, dtype=np.float64)
    lower_z = np.asarray(NH3_LOWER_Z, dtype=np.float64)

    axis.plot(
        [0.0, 0.0],
        [Z_NH3_AXIS_BOTTOM * 1.0e3, Z_NH3_TOP * 1.0e3],
        color="black",
        lw=1.2,
    )

    axis.plot(
        [R_NH3 * 1.0e3, R_NH3 * 1.0e3],
        [Z_NH3_WALL_BOTTOM * 1.0e3, Z_NH3_TOP * 1.0e3],
        color="black",
        lw=1.2,
    )

    axis.plot(
        [0.0, R_NH3 * 1.0e3],
        [Z_NH3_TOP * 1.0e3, Z_NH3_TOP * 1.0e3],
        color="black",
        lw=1.2,
    )

    axis.plot(
        lower_r * 1.0e3,
        lower_z * 1.0e3,
        color="black",
        lw=1.5,
    )


def symmetric_limits(values):
    """
    Return symmetric color limits around zero.
    """
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return (-1.0, 1.0)

    maximum = np.max(np.abs(valid))

    if maximum <= 0.0:
        maximum = 1.0

    return (-maximum, maximum)


def save_figure(figure, filename):
    """
    Save and close one Matplotlib figure.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)

    figure.savefig(filepath, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved plot:\n{os.path.abspath(filepath)}")



# Domain and boundaries (nondimensional: radius=1.0, z_top=1.0)


dx = NH3CavityDomain(
    R_HAT_MAX,
    Z_HAT_TOP,
    is_main_domain=True,
)


dx.set_boundaries_dict(
    {
        "axis": [
            "axis",
        ],
        "outer_side": [
            "outer_side",
        ],
        "outer_top": [
            "outer_top",
        ],
        "lower_polyline": [
            "lower_polyline",
        ],
    }
)


boundaries = dx.get_all_boundaries()


print("\n" + "=" * 72)
print("DOMAIN BOUNDARY LABELS")
print("=" * 72)
print(
    boundaries.keys(),
)



# Boundary residual assignment


bc_residuals = {
    "axis": AxisRegularityBC(
        boundaries[
            "axis"
        ],
        mode=MODE,
    ),
    "outer_side": NoSlipTemperatureBC(
        boundaries[
            "outer_side"
        ],
        mode=MODE,
    ),
    "outer_top": NoSlipTemperatureBC(
        boundaries[
            "outer_top"
        ],
        mode=MODE,
    ),
    "lower_polyline": NoSlipTemperatureBC(
        boundaries[
            "lower_polyline"
        ],
        mode=MODE,
    ),
}


# Physics model (nondimensional coefficients: NU_HAT, KAPPA_HAT, alpha=1.0,
# T_ref=0.0, gravity_z=-1.0 — direction only, magnitude absorbed into U_REF)


model = AxisymmetricBoussinesqModel(
    main_domain=dx,
    nu=NU_HAT,
    kappa=KAPPA_HAT,
    alpha=1.0,
    T_ref=0.0,
    gravity_z=-1.0,
    mode=MODE,
    bc_residuals=bc_residuals,
)



# Approximation space


key = jax.random.PRNGKey(
    0,
)


nn = MLP(
    in_size=2,
    out_size=4,
    hidden_sizes=HIDDEN_SIZES,
    key=key,
)


space = ApproximationSpace(
    {
        "x": 2,
    },
    [
        (
            nn,
            "vec",
            4,
        ),
    ],
    model_type="x",
    post_processing=post_processing_axisymmetric,
)



# Sampler and PINN projector


sampler = TensorizedSampler(
    [
        DomainSampler(
            dx,
        ),
    ],
    bc=True,
)


pinn = Projector(
    model,
    space,
    sampler,
    weights=WEIGHTS,
    one_loss_by_equation=True,
)



# Training


# ── Ra-continuation v2: also ramp boundary weights, not just Ra ───────────
# Replaces the previous ra_continuation_block. Key change: WEIGHTS is now
# rebuilt PER STAGE, with boundary (no-slip/wall) weights starting low and
# increasing to their target values only in the final stage. This prevents
# the wall no-slip terms from dominating the loss before the interior
# momentum terms have a chance to establish nonzero circulation.
#
# Interior weights stay favoring momentum over continuity throughout
# (as in the previous fix): [5.0, 5.0, 1.0, 1.0].

RA_CONTINUATION_FACTORS = [1.0e4, 1.0e3, 1.0e2, 1.0e1, 1.0]

# Boundary weight schedule: (axis_uz_T, wall_weight) per stage, ramping up.
# Final stage matches your original target weights (20.0 everywhere on walls,
# axis = [20.0, 10.0, 10.0]).
BOUNDARY_WEIGHT_SCHEDULE = [1.0, 2.0, 5.0, 10.0, 20.0]

N_EPOCHS_PER_STAGE = 300  # raised from 100/200, since the network needs more
                          # steps to develop circulation before boundary
                          # weights fully clamp it

INIT_PERTURBATION_AMPLITUDE = 0.5

key = jax.random.PRNGKey(0)
nn = MLP(in_size=2, out_size=4, hidden_sizes=HIDDEN_SIZES, key=key)
space = ApproximationSpace(
    {"x": 2},
    [(nn, "vec", 4)],
    model_type="x",
    post_processing=post_processing_axisymmetric,
)

stage_histories = []

for stage_idx, factor in enumerate(RA_CONTINUATION_FACTORS):
    nu_hat_stage = NU_HAT * factor
    kappa_hat_stage = KAPPA_HAT * factor
    ra_effective = 1.0 / (nu_hat_stage * kappa_hat_stage)
    wall_w = BOUNDARY_WEIGHT_SCHEDULE[stage_idx]

    stage_weights = {
        "interior": [5.0, 5.0, 1.0, 1.0],
        "axis": [wall_w, wall_w / 2.0, wall_w / 2.0],
        "outer_side": [wall_w, wall_w, wall_w],
        "outer_top": [wall_w, wall_w, wall_w],
        "lower_polyline": [wall_w, wall_w, wall_w],
    }

    print("\n" + "=" * 72)
    print(f"CONTINUATION STAGE {stage_idx + 1}/{len(RA_CONTINUATION_FACTORS)}")
    print(f"factor = {factor:.3e}, Ra_eff ~ {ra_effective:.3e}, "
          f"wall_weight = {wall_w:.2f}")
    print("=" * 72)

    stage_bc_residuals = {
        "axis": AxisRegularityBC(boundaries["axis"], mode=MODE),
        "outer_side": NoSlipTemperatureBC(boundaries["outer_side"], mode=MODE),
        "outer_top": NoSlipTemperatureBC(boundaries["outer_top"], mode=MODE),
        "lower_polyline": NoSlipTemperatureBC(boundaries["lower_polyline"], mode=MODE),
    }

    stage_model = AxisymmetricBoussinesqModel(
        main_domain=dx,
        nu=nu_hat_stage,
        kappa=kappa_hat_stage,
        alpha=1.0,
        T_ref=0.0,
        gravity_z=-1.0,
        mode=MODE,
        bc_residuals=stage_bc_residuals,
    )

    pinn = Projector(
        stage_model,
        space,
        sampler,
        weights=stage_weights,
        one_loss_by_equation=True,
    )

    key, initial_samples = sampler.sample(key, N_COLLOC, N_BC_COLLOC)
    initial_loss = pinn.evaluate_loss(space, initial_samples)
    print(f"Initial loss at this stage:\n{initial_loss}")

    start_time = timeit.default_timer()
    key, pinn = pinn.project(key, space, N_EPOCHS_PER_STAGE, N_COLLOC, N_BC_COLLOC)
    stage_time = timeit.default_timer() - start_time

    print(f"Best loss at this stage:\n{pinn.best_loss}")
    print(f"Stage training time: {stage_time:.2f} s")

    stage_histories.append(
        {
            "stage": stage_idx,
            "factor": factor,
            "ra_effective": ra_effective,
            "wall_weight": wall_w,
            "best_loss": pinn.best_loss,
            "training_time": stage_time,
        }
    )

print("\n" + "=" * 72)
print("CONTINUATION SUMMARY")
print("=" * 72)
for record in stage_histories:
    print(record)

# ── Diagnostic: check actual field magnitudes after Ra-continuation ───────
# Run this immediately after the continuation loop finishes (after the
# final `pinn = ...` from the last stage). This settles definitively
# whether the "near-zero residual" outcome corresponds to a trivial
# u=0 solution or a genuine nonzero converged flow.

diag_r_hat = np.linspace(1.0e-3, 1.0, 50)
diag_z_hat = np.linspace(0.0, 1.0, 50)
diag_rr, diag_zz = np.meshgrid(diag_r_hat, diag_z_hat, indexing="xy")
diag_points = np.column_stack([diag_rr.ravel(), diag_zz.ravel()]).astype(np.float64)

diag_predictions = np.asarray(
    pinn.evaluate(jnp.asarray(diag_points, dtype=scimba_jax.get_default_dtype()))
)

diag_ur_hat = diag_predictions[:, 0]
diag_uz_hat = diag_predictions[:, 1]
diag_p_hat = diag_predictions[:, 2]
diag_T_hat = diag_predictions[:, 3]

print("=" * 72)
print("DIRECT FIELD MAGNITUDE CHECK (nondimensional, post-training)")
print("=" * 72)
print(f"u_r_hat:  min={diag_ur_hat.min():.6e}  max={diag_ur_hat.max():.6e}  "
      f"mean_abs={np.abs(diag_ur_hat).mean():.6e}")
print(f"u_z_hat:  min={diag_uz_hat.min():.6e}  max={diag_uz_hat.max():.6e}  "
      f"mean_abs={np.abs(diag_uz_hat).mean():.6e}")
print(f"p_hat:    min={diag_p_hat.min():.6e}  max={diag_p_hat.max():.6e}")
print(f"T_hat:    min={diag_T_hat.min():.6e}  max={diag_T_hat.max():.6e}")
print()
print(f"Target O(1) scale set by U_REF: expected |u_hat| ~ 1e-1 to 1e0 for real convection")
print(f"Observed max|u_r_hat| = {np.abs(diag_ur_hat).max():.3e}, "
      f"max|u_z_hat| = {np.abs(diag_uz_hat).max():.3e}")

if np.abs(diag_uz_hat).max() < 1.0e-3:
    print("\n>>> CONCLUSION: velocity field is essentially ZERO. Trivial solution confirmed directly.")
else:
    print("\n>>> CONCLUSION: velocity field is NONZERO at O(1e-3) or larger -- residuals near")
    print("    zero may correspond to a genuine (if weak) converged flow, not the trivial solution.")

# Also check: does the seed pattern survive in the trained network, or has
# training learned to cancel it via ur_raw/uz outputs specifically?
print("\n" + "=" * 72)
print("RAW NETWORK OUTPUT CHECK (before any hard-coded seed/transformation)")
print("=" * 72)
raw_out = jax.vmap(lambda xx: nn(xx))(jnp.asarray(diag_points, dtype=scimba_jax.get_default_dtype()))
raw_out_np = np.asarray(raw_out)
print(f"raw ur_raw: min={raw_out_np[:,0].min():.6e} max={raw_out_np[:,0].max():.6e}")
print(f"raw uz:     min={raw_out_np[:,1].min():.6e} max={raw_out_np[:,1].max():.6e}")

training_time = (
    timeit.default_timer()
    - start_time
)


print("\n" + "=" * 72)
print("BEST LOSS")
print("=" * 72)
print(
    pinn.best_loss,
)


print(
    f"\nTraining time: {training_time:.2f} s"
)


print(
    "\nReminder: network outputs are nondimensional. Use "
    "to_physical_units(r_hat, z_hat, ur_hat, uz_hat, p_hat, T_hat) "
    "to convert to SI units (m, K, Pa, m/s) before plotting or comparing "
    "against finite-volume reference data."
)

# ── visualisation (v3: 4 separate figures, nondimensional-aware) ───────────
# Reuses your existing helpers: draw_nh3_outline, symmetric_limits,
# save_figure, nh3_lower_profile_numpy. Grid and evaluation are now done in
# NONDIMENSIONAL (r_hat, z_hat) coordinates, then converted to physical
# units (mm, K, m/s, Pa) via to_physical_units() before plotting, so the
# figures look the same as your old dimensional version.

PLOT_NR = 161
PLOT_NZ = 501


def cavity_mask_hat_numpy(rr_hat, zz_hat):
    lower_z_hat = nh3_lower_profile_numpy(rr_hat)
    return (
        (rr_hat >= 0.0)
        & (rr_hat <= 1.0)
        & (zz_hat >= lower_z_hat)
        & (zz_hat <= 1.0)
    )


def create_plot_grid_hat():
    """Rectangular r_hat-z_hat grid (both in [0, 1]) and cavity mask."""
    r_hat_values = np.linspace(0.0, 1.0, PLOT_NR)
    z_hat_values = np.linspace(0.0, 1.0, PLOT_NZ)

    rr_hat, zz_hat = np.meshgrid(r_hat_values, z_hat_values, indexing="xy")
    mask = cavity_mask_hat_numpy(rr_hat, zz_hat)

    points_hat = np.column_stack([rr_hat.ravel(), zz_hat.ravel()]).astype(np.float64)

    return rr_hat, zz_hat, points_hat, mask


def evaluate_pinn_on_grid_physical():
    """Evaluate PINN on the nondimensional grid, return physical-unit fields."""
    rr_hat, zz_hat, points_hat, mask = create_plot_grid_hat()

    predictions = np.asarray(
        pinn.evaluate(jnp.asarray(points_hat, dtype=scimba_jax.get_default_dtype()))
    ).copy()

    ur_hat = predictions[:, 0].reshape(PLOT_NZ, PLOT_NR).copy()
    uz_hat = predictions[:, 1].reshape(PLOT_NZ, PLOT_NR).copy()
    p_hat = predictions[:, 2].reshape(PLOT_NZ, PLOT_NR).copy()
    T_hat = predictions[:, 3].reshape(PLOT_NZ, PLOT_NR).copy()

    rr_phys, zz_phys, ur_phys, uz_phys, p_phys, T_phys = to_physical_units(
        rr_hat, zz_hat, ur_hat, uz_hat, p_hat, T_hat
    )

    outside_domain = ~mask
    ur_phys[outside_domain] = np.nan
    uz_phys[outside_domain] = np.nan
    p_phys[outside_domain] = np.nan
    T_phys[outside_domain] = np.nan

    return {
        "rr_hat": rr_hat,
        "zz_hat": zz_hat,
        "rr": rr_phys,
        "zz": zz_phys,
        "mask": mask,
        "ur": ur_phys,
        "uz": uz_phys,
        "pressure": p_phys,
        "temperature": T_phys,
    }


field_data = evaluate_pinn_on_grid_physical()

RR = field_data["rr"]
ZZ = field_data["zz"]
UR = field_data["ur"]
UZ = field_data["uz"]
PRESSURE = field_data["pressure"]
TEMPERATURE = field_data["temperature"]
SPEED = np.sqrt(UR**2 + UZ**2)


# ---------------------------------------------------------------------------
# Figure 1: Loss history
# ---------------------------------------------------------------------------

figure, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)

try:
    pinn.losses.plot(
        axis,
        loss_groups=["nh3", "axis", "outer_side", "outer_top", "lower_polyline"],
    )
except Exception as error:
    print("\nLoss-group plot failed; using default SCIMBA loss plot.")
    print(f"Reason: {error}")
    pinn.losses.plot(axis)

axis.set_xlabel("Epoch")
axis.set_ylabel("Loss")
axis.set_title("Steady axisymmetric NH$_3$ PINN loss history (nondimensional)")
axis.grid(True, which="both", alpha=0.30)

handles, labels = axis.get_legend_handles_labels()
axis.legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=8,
    ncol=1,
    frameon=True,
    borderaxespad=0.0,
)

save_figure(figure, "01_loss_history.png")


# ---------------------------------------------------------------------------
# Figure 2: Fields + residuals (library combined plot, no inline loss legend)
# ---------------------------------------------------------------------------


plot_abstract_approx_space(
    pinn.space,
    dx,
    components=[0, 1, 2, 3],   # u_r_hat, u_z_hat, p_hat, T_hat
    residual=pinn.model,
    draw_contours=True,
    n_drawn_contours=20,
    title="Nondimensional NH3 Boussinesq PINN — fields and residuals",
)
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_fields_residuals.png"),
    dpi=250,
    bbox_inches="tight",
)
plt.show()


# ---------------------------------------------------------------------------
# Figure 3: Divergence diagnostic (physical units, finite-difference,
# using your original numerical approach but on physical UR/UZ/RR/ZZ)
# ---------------------------------------------------------------------------

r_values = RR[0, :]
z_values = ZZ[:, 0]

dr_plot = float(r_values[1] - r_values[0])
dz_plot = float(z_values[1] - z_values[0])

ur_filled = np.nan_to_num(UR, nan=0.0)
uz_filled = np.nan_to_num(UZ, nan=0.0)

dur_dr = np.gradient(ur_filled, dr_plot, axis=1)
duz_dz = np.gradient(uz_filled, dz_plot, axis=0)

r_safe = np.maximum(RR, 1.0e-8)

divergence = dur_dr + UR / r_safe + duz_dz
divergence[~field_data["mask"]] = np.nan
divergence[:, 0] = np.nan

div_min, div_max = symmetric_limits(divergence)

figure, axis = plt.subplots(figsize=(6.0, 10.0), constrained_layout=True)

contour = axis.contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    divergence,
    levels=50,
    cmap="RdBu_r",
    vmin=div_min,
    vmax=div_max,
)

colorbar = figure.colorbar(contour, ax=axis)
colorbar.set_label(r"$\nabla_{\mathrm{axi}}\cdot u$ [1/s]")

draw_nh3_outline(axis)

axis.set_xlabel("r [mm]")
axis.set_ylabel("z [mm]")
axis.set_aspect("equal", adjustable="box")
axis.set_title("Axisymmetric divergence diagnostic (physical units)")

save_figure(figure, "03_divergence.png")


# ---------------------------------------------------------------------------
# Figure 4: Pressure (physical units, relative to grid mean)
# ---------------------------------------------------------------------------

pressure_mean = np.nanmean(PRESSURE)
pressure_display = PRESSURE - pressure_mean

pressure_min, pressure_max = symmetric_limits(pressure_display)

figure, axis = plt.subplots(figsize=(6.0, 10.0), constrained_layout=True)

contour = axis.contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    pressure_display,
    levels=50,
    cmap="coolwarm",
    vmin=pressure_min,
    vmax=pressure_max,
)

colorbar = figure.colorbar(contour, ax=axis)
colorbar.set_label("Pressure minus grid mean [Pa]")

draw_nh3_outline(axis)

axis.set_xlabel("r [mm]")
axis.set_ylabel("z [mm]")
axis.set_aspect("equal", adjustable="box")
axis.set_title("Pressure field relative to grid mean (physical units)")

save_figure(figure, "04_pressure_relative.png")


print(
    "\nAll 4 plots saved in:\n" + os.path.abspath(OUTPUT_DIR)
)
