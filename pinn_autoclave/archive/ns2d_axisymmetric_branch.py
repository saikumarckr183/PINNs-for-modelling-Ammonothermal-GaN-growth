r"""
Steady axisymmetric Boussinesq Navier-Stokes PINN

Coordinates:
    x[..., 0] = r  radial coordinate [m]
    x[..., 1] = z  axial/vertical coordinate [m]

Unknown vector:
    [u_r, u_z, p, T]

Geometry:
    0 <= r <= R_NH3
    z_lower(r) <= z <= Z_NH3_TOP

where z_lower(r) is the lower NH3/Inconel CryS-MAS polyline.
"""

from __future__ import annotations

import os
import timeit

import jax

# Float32 is faster and uses less memory.
# Turn on only if you later need high-precision diagnostics.
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


# General configuration
MODE = "vec"

OUTPUT_DIR = "output_axisymmetric_nh3_steady"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

# NH3 cavity geometry
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
    dtype=jnp.float32,
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
    dtype=jnp.float32,
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


# Prescribed wall-temperature profile
# T_BC(z) = T_0 + T_1 * z

T_INTERCEPT = 800.65
T_SLOPE = -400.0
T_REF = 700.05


# NH3 properties

RHO_NH3 = 600.0
MU_NH3 = 1.0e-4
CP_NH3 = 5000.0
K_NH3 = 0.50
ALPHA_NH3 = 2.0e-3

NU_NH3 = MU_NH3 / RHO_NH3
KAPPA_NH3 = K_NH3 / (
    RHO_NH3
    * CP_NH3
)


#gravity term
GRAVITY_Z = -9.81

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
        1.0,
        1.0,
        5.0,
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


# Coordinate helper functions
# SCIMBA's ParamFunction symbolic algebra can use coordinate
# extraction functions exactly as in AxisymmetricHeatResidual.

def r_coord(
    x,
):
    """
    Return r from x = [r, z].
    """

    return x[
        0
    ]


def z_coord(
    x,
):
    """
    Return z from x = [r, z].
    """

    return x[
        1
    ]


def r2_coord(
    x,
):
    """
    Return r^2 from x = [r, z].
    """

    return x[
        0
    ] ** 2


def temperature_boundary_profile(
    x,
):
    """
    Prescribed temperature on every material boundary.

    T_BC(z) = T_INTERCEPT + T_SLOPE * z.
    """

    return (
        T_INTERCEPT
        + T_SLOPE
        * z_coord(
            x
        )
    )


# JAX and NumPy lower polyline profiles

def nh3_lower_profile_jax(
    r,
):
    """
    JAX-compatible piecewise-linear lower NH3/Inconel boundary.

    Input:
        r, scalar or array in [0, R_NH3].

    Output:
        z_lower(r).
    """

    return jnp.interp(
        r,
        NH3_LOWER_R,
        NH3_LOWER_Z,
    )


def nh3_lower_profile_numpy(
    r,
):
    """
    NumPy version for plotting masks and diagnostics.
    """

    return np.interp(
        np.asarray(
            r,
            dtype=np.float64,
        ),
        np.asarray(
            NH3_LOWER_R,
            dtype=np.float64,
        ),
        np.asarray(
            NH3_LOWER_Z,
            dtype=np.float64,
        ),
    )


# Signed-distance function for ammonia

class NH3CavitySDF(
    SignedDistance,
):
    r"""
    Approximate SDF for:

        0 <= r <= R_NH3
        z_lower(r) <= z <= Z_NH3_TOP

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
        Return one signed implicit value for one point x = [r, z].
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

        # Negative inside each inequality:
        #
        # r >= 0              -> -r <= 0
        # r <= R_NH3          -> r - R_NH3 <= 0
        # z >= z_lower(r)     -> z_lower - z <= 0
        # z <= z_top          -> z - z_top <= 0
        #
        # max(...) is <= 0 only when all inequalities hold.
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


# Boundary-domain classes

class AxisSegmentNH3(
    SurfacicDomain,
):
    """
    Axis boundary:
        r = 0,
        Z_NH3_AXIS_BOTTOM <= z <= Z_NH3_TOP.
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
        r = R_NH3,
        Z_NH3_WALL_BOTTOM <= z <= Z_NH3_TOP.
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
        0 <= r <= R_NH3,
        z = Z_NH3_TOP.
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
    Piecewise-linear NH3/Inconel interface.

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


# Main ammonia domain

class NH3CavityDomain(
    VolumetricDomain,
):
    """
    Axisymmetric ammonia cavity in the (r, z) meridional plane.
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
                    Z_NH3_AXIS_BOTTOM,
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
                Z_NH3_AXIS_BOTTOM,
                self.z_top,
                label_str=label_bc + " axis",
            ),
            OuterSideSegmentNH3(
                self.radius,
                Z_NH3_WALL_BOTTOM,
                self.z_top,
                label_str=label_bc + " outer_side",
            ),
            TopSegmentNH3(
                self.radius,
                self.z_top,
                label_str=label_bc + " outer_top",
            ),
            LowerPolylineNH3(
                NH3_LOWER_R,
                NH3_LOWER_Z,
                label_str=label_bc + " lower_polyline",
            ),
        ]

        if self.is_mapped:
            for boundary in boundaries:
                boundary._set_mapping(
                    self.mapping,
                )

        return boundaries


# Interior axisymmetric Boussinesq residual

class AxisymmetricBoussinesqResidual(
    InteriorResidual,
):
    r"""
    Steady axisymmetric Boussinesq equations.

    x = [r, z]
    output = [u_r, u_z, p, T]

    Residual order:
        0: radial momentum
        1: axial momentum
        2: continuity
        3: energy

    r-weighted residuals are used to avoid explicit 1/r and
    1/r^2 factors at r = 0.

    Radial momentum, multiplied by r^2:

        r^2 * [
            ur*ur_r + uz*ur_z + p_r
            - nu*(ur_rr + ur_zz)
        ]
        - nu*r*ur_r
        + nu*ur = 0

    Axial momentum, multiplied by r:

        r * [
            ur*uz_r + uz*uz_z + p_z
            + alpha*(T - T_ref)*g_z
            - nu*(uz_rr + uz_zz)
        ]
        - nu*uz_r = 0

    Continuity, multiplied by r:

        r*ur_r + ur + r*uz_z = 0

    Energy, multiplied by r:

        r * [
            ur*T_r + uz*T_z
            - kappa*(T_rr + T_zz)
        ]
        - kappa*T_r = 0
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
    Axis regularity conditions on r = 0.

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
    Physical wall residual.

    Residual ordering:
        [
            u_r,
            u_z,
            T - T_BC(z),
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

        # Build T_BC(z) in a callable-safe symbolic order.
        # temperature * 0.0 makes a SCIMBA symbolic scalar field
        # with the same argument structure as temperature.
        # Then z_coord is multiplied on the right side of that
        # symbolic field, so SCIMBA creates a valid coordinate field.
        zero_field = temperature * 0.0

        z_field = (
            zero_field
            + 1.0
        ) * z_coord

        prescribed_temperature = (
            z_field
            * T_SLOPE
            + T_INTERCEPT
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
    High-level steady axisymmetric no-swirl Boussinesq model.
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


def post_processing_axisymmetric(
    output,
    x,
):
    """
    Convert raw neural-network outputs into physical variables.

    Raw output:
        [u_r_raw, u_z_raw, p_raw, T_correction]

    Returned output:
        [u_r, u_z, p, T]

    Strongly imposed:
        u_r(0,z) = 0

    Temperature initialization:
        T starts from T_BC(z) = T_INTERCEPT + T_SLOPE*z.
    """

    r = x[
        0:1
    ]

    z = x[
        1:2
    ]

    ur_raw = output[
        0:1
    ]

    uz = output[
        1:2
    ]

    pressure = output[
        2:3
    ]

    temperature_correction = output[
        3:4
    ]

    ur = r * ur_raw

    temperature_base = (
        T_INTERCEPT
        + T_SLOPE
        * z
    )

    temperature = (
        temperature_base
        + temperature_correction
    )

    return jnp.concatenate(
        [
            ur,
            uz,
            pressure,
            temperature,
        ],
        axis=-1,
    )

# Domain and boundaries

dx = NH3CavityDomain(
    R_NH3,
    Z_NH3_TOP,
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

# Physics model

model = AxisymmetricBoussinesqModel(
    main_domain=dx,
    nu=NU_NH3,
    kappa=KAPPA_NH3,
    alpha=ALPHA_NH3,
    T_ref=T_REF,
    gravity_z=GRAVITY_Z,
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

key, initial_samples = sampler.sample(
    key,
    N_COLLOC,
    N_BC_COLLOC,
)

initial_loss = pinn.evaluate_loss(
    space,
    initial_samples,
)

print("\n" + "=" * 72)
print("INITIAL LOSS")
print("=" * 72)
print(
    initial_loss,
)

start_time = timeit.default_timer()

key, pinn = pinn.project(
    key,
    space,
    N_EPOCHS,
    N_COLLOC,
    N_BC_COLLOC,
)

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


# Plotting helpers

PLOT_NR = 161
PLOT_NZ = 501


def lower_profile_numpy(
    r,
):
    """
    Alias for plotting readability.
    """

    return nh3_lower_profile_numpy(
        r,
    )


def cavity_mask_numpy(
    rr,
    zz,
):
    """
    True for points inside the actual NH3 cavity.
    """

    lower_z = lower_profile_numpy(
        rr,
    )

    return (
        (rr >= 0.0)
        & (
            rr <= R_NH3
        )
        & (
            zz >= lower_z
        )
        & (
            zz <= Z_NH3_TOP
        )
    )


def create_plot_grid():
    """
    Create a rectangular r-z grid and ammonia-cavity mask.
    """

    r_values = np.linspace(
        0.0,
        R_NH3,
        PLOT_NR,
    )

    z_values = np.linspace(
        Z_NH3_AXIS_BOTTOM,
        Z_NH3_TOP,
        PLOT_NZ,
    )

    rr, zz = np.meshgrid(
        r_values,
        z_values,
        indexing="xy",
    )

    mask = cavity_mask_numpy(
        rr,
        zz,
    )

    points = np.column_stack(
        [
            rr.ravel(),
            zz.ravel(),
        ]
    ).astype(
        np.float32,
    )

    return rr, zz, points, mask


def evaluate_pinn_on_grid():
    """
    Evaluate all four PINN outputs on the common r-z plotting grid.

    Output columns:
        0: u_r
        1: u_z
        2: p
        3: T
    """

    rr, zz, points, mask = create_plot_grid()

    predictions = np.asarray(
        pinn.evaluate(
            jnp.asarray(
                points,
                dtype=scimba_jax.get_default_dtype(),
            )
        )
    ).copy()

    ur = predictions[
        :,
        0
    ].reshape(
        PLOT_NZ,
        PLOT_NR,
    ).copy()

    uz = predictions[
        :,
        1
    ].reshape(
        PLOT_NZ,
        PLOT_NR,
    ).copy()

    pressure = predictions[
        :,
        2
    ].reshape(
        PLOT_NZ,
        PLOT_NR,
    ).copy()

    temperature = predictions[
        :,
        3
    ].reshape(
        PLOT_NZ,
        PLOT_NR,
    ).copy()

    outside_domain = ~mask

    ur[
        outside_domain
    ] = np.nan

    uz[
        outside_domain
    ] = np.nan

    pressure[
        outside_domain
    ] = np.nan

    temperature[
        outside_domain
    ] = np.nan

    return {
        "rr": rr,
        "zz": zz,
        "mask": mask,
        "ur": ur,
        "uz": uz,
        "pressure": pressure,
        "temperature": temperature,
    }

def draw_nh3_outline(
    axis,
):
    """
    Draw the exact meridional NH3 cavity boundary.
    """

    lower_r = np.asarray(
        NH3_LOWER_R,
        dtype=np.float64,
    )

    lower_z = np.asarray(
        NH3_LOWER_Z,
        dtype=np.float64,
    )

    axis.plot(
        [
            0.0,
            0.0,
        ],
        [
            Z_NH3_AXIS_BOTTOM,
            Z_NH3_TOP,
        ],
        color="black",
        lw=1.2,
    )

    axis.plot(
        [
            R_NH3,
            R_NH3,
        ],
        [
            Z_NH3_WALL_BOTTOM,
            Z_NH3_TOP,
        ],
        color="black",
        lw=1.2,
    )

    axis.plot(
        [
            0.0,
            R_NH3,
        ],
        [
            Z_NH3_TOP,
            Z_NH3_TOP,
        ],
        color="black",
        lw=1.2,
    )

    axis.plot(
        lower_r,
        lower_z,
        color="black",
        lw=1.5,
    )


def save_figure(
    figure,
    filename,
):
    """
    Save and close one Matplotlib figure.
    """

    filepath = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    figure.savefig(
        filepath,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure,
    )

    print(
        f"Saved plot:\n{os.path.abspath(filepath)}"
    )


def symmetric_limits(
    values,
):
    """
    Return symmetric color limits around zero.
    """

    valid = values[
        np.isfinite(
            values
        )
    ]

    if valid.size == 0:
        return (
            -1.0,
            1.0,
        )

    maximum = np.max(
        np.abs(
            valid
        )
    )

    if maximum <= 0.0:
        maximum = 1.0

    return (
        -maximum,
        maximum,
    )


# Plot 00: geometry

figure, axis = plt.subplots(
    figsize=(
        6.0,
        10.0,
    ),
    constrained_layout=True,
)

lower_r_np = np.asarray(
    NH3_LOWER_R,
    dtype=np.float64,
)

lower_z_np = np.asarray(
    NH3_LOWER_Z,
    dtype=np.float64,
)

polygon_r = np.concatenate(
    [
        np.array(
            [
                0.0,
                0.0,
                R_NH3,
            ]
        ),
        lower_r_np[
            ::-1
        ],
    ]
)

polygon_z = np.concatenate(
    [
        np.array(
            [
                Z_NH3_AXIS_BOTTOM,
                Z_NH3_TOP,
                Z_NH3_TOP,
            ]
        ),
        lower_z_np[
            ::-1
        ],
    ]
)

axis.fill(
    polygon_r,
    polygon_z,
    color="royalblue",
    alpha=0.18,
)

draw_nh3_outline(
    axis,
)

axis.text(
    0.50 * R_NH3,
    0.50
    * (
        Z_NH3_AXIS_BOTTOM
        + Z_NH3_TOP
    ),
    "NH$_3$",
    ha="center",
    va="center",
    color="navy",
    fontsize=16,
)

axis.set_xlabel(
    "r [m]"
)

axis.set_ylabel(
    "z [m]"
)

axis.set_aspect(
    "equal",
    adjustable="box",
)

axis.set_title(
    "Axisymmetric ammonia cavity\n"
    "CryS-MAS lower polyline"
)

axis.grid(
    True,
    alpha=0.25,
)

save_figure(
    figure,
    "00_nh3_geometry.png",
)

# Evaluate solution on plotting grid

field_data = evaluate_pinn_on_grid()

RR = field_data[
    "rr"
]

ZZ = field_data[
    "zz"
]

UR = field_data[
    "ur"
]

UZ = field_data[
    "uz"
]

PRESSURE = field_data[
    "pressure"
]

TEMPERATURE = field_data[
    "temperature"
]

SPEED = np.sqrt(
    UR**2
    + UZ**2
)


# Plot 01: temperature

figure, axis = plt.subplots(
    figsize=(
        6.0,
        10.0,
    ),
    constrained_layout=True,
)

contour = axis.contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    TEMPERATURE,
    levels=50,
    cmap="inferno",
)

colorbar = figure.colorbar(
    contour,
    ax=axis,
)

colorbar.set_label(
    "Temperature [K]"
)

draw_nh3_outline(
    axis,
)

axis.set_xlabel(
    "r [mm]"
)

axis.set_ylabel(
    "z [mm]"
)

axis.set_aspect(
    "equal",
    adjustable="box",
)

axis.set_title(
    "NH$_3$ temperature field"
)

save_figure(
    figure,
    "01_temperature.png",
)


# Plot 02: velocity magnitude and vectors

figure, axis = plt.subplots(
    figsize=(
        6.0,
        10.0,
    ),
    constrained_layout=True,
)

contour = axis.contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    SPEED,
    levels=50,
    cmap="viridis",
)

colorbar = figure.colorbar(
    contour,
    ax=axis,
)

colorbar.set_label(
    r"Velocity magnitude $|\mathbf{u}|$ [m/s]"
)

draw_nh3_outline(
    axis,
)

quiver_stride_r = 10
quiver_stride_z = 18

r_quiver = RR[
    ::quiver_stride_z,
    ::quiver_stride_r,
]

z_quiver = ZZ[
    ::quiver_stride_z,
    ::quiver_stride_r,
]

ur_quiver = UR[
    ::quiver_stride_z,
    ::quiver_stride_r,
]

uz_quiver = UZ[
    ::quiver_stride_z,
    ::quiver_stride_r,
]

valid_quiver = (
    np.isfinite(
        ur_quiver
    )
    & np.isfinite(
        uz_quiver
    )
)

axis.quiver(
    r_quiver[
        valid_quiver
    ]
    * 1.0e3,
    z_quiver[
        valid_quiver
    ]
    * 1.0e3,
    ur_quiver[
        valid_quiver
    ],
    uz_quiver[
        valid_quiver
    ],
    color="white",
    width=0.0025,
    scale=None,
)

axis.set_xlabel(
    "r [mm]"
)

axis.set_ylabel(
    "z [mm]"
)

axis.set_aspect(
    "equal",
    adjustable="box",
)

axis.set_title(
    "NH$_3$ velocity magnitude and meridional vectors"
)

save_figure(
    figure,
    "02_velocity_magnitude_vectors.png",
)

# Plot 03: velocity components

ur_min, ur_max = symmetric_limits(
    UR,
)

uz_min, uz_max = symmetric_limits(
    UZ,
)

figure, axes = plt.subplots(
    1,
    2,
    figsize=(
        12.0,
        9.0,
    ),
    constrained_layout=True,
)

ur_contour = axes[
    0
].contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    UR,
    levels=50,
    cmap="coolwarm",
    vmin=ur_min,
    vmax=ur_max,
)

figure.colorbar(
    ur_contour,
    ax=axes[
        0
    ],
    label=r"$u_r$ [m/s]",
)

draw_nh3_outline(
    axes[
        0
    ],
)

axes[
    0
].set_xlabel(
    "r [mm]"
)

axes[
    0
].set_ylabel(
    "z [mm]"
)

axes[
    0
].set_aspect(
    "equal",
    adjustable="box",
)

axes[
    0
].set_title(
    r"Radial velocity $u_r$"
)

uz_contour = axes[
    1
].contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    UZ,
    levels=50,
    cmap="coolwarm",
    vmin=uz_min,
    vmax=uz_max,
)

figure.colorbar(
    uz_contour,
    ax=axes[
        1
    ],
    label=r"$u_z$ [m/s]",
)

draw_nh3_outline(
    axes[
        1
    ],
)

axes[
    1
].set_xlabel(
    "r [mm]"
)

axes[
    1
].set_ylabel(
    "z [mm]"
)

axes[
    1
].set_aspect(
    "equal",
    adjustable="box",
)

axes[
    1
].set_title(
    r"Axial velocity $u_z$"
)

save_figure(
    figure,
    "03_velocity_components.png",
)


# Plot 04: pressure

pressure_mean = np.nanmean(
    PRESSURE,
)

pressure_display = (
    PRESSURE
    - pressure_mean
)

pressure_min, pressure_max = symmetric_limits(
    pressure_display,
)

figure, axis = plt.subplots(
    figsize=(
        6.0,
        10.0,
    ),
    constrained_layout=True,
)

contour = axis.contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    pressure_display,
    levels=50,
    cmap="coolwarm",
    vmin=pressure_min,
    vmax=pressure_max,
)

colorbar = figure.colorbar(
    contour,
    ax=axis,
)

colorbar.set_label(
    "Pressure minus grid mean [model units]"
)

draw_nh3_outline(
    axis,
)

axis.set_xlabel(
    "r [mm]"
)

axis.set_ylabel(
    "z [mm]"
)

axis.set_aspect(
    "equal",
    adjustable="box",
)

axis.set_title(
    "Pressure field relative to grid mean"
)

save_figure(
    figure,
    "04_pressure_relative.png",
)


# Plot 05: analytical temperature error

temperature_exact = (
    T_INTERCEPT
    + T_SLOPE
    * ZZ
)

temperature_error = (
    TEMPERATURE
    - temperature_exact
)

valid_temperature_error = temperature_error[
    np.isfinite(
        temperature_error
    )
]

temperature_rmse = np.sqrt(
    np.mean(
        valid_temperature_error**2
    )
)

temperature_mae = np.mean(
    np.abs(
        valid_temperature_error
    )
)

temperature_max_error = np.max(
    np.abs(
        valid_temperature_error
    )
)

error_limit = np.max(
    np.abs(
        valid_temperature_error
    )
)

if error_limit <= 0.0:
    error_limit = 1.0e-12

print("\n" + "=" * 72)
print("ZERO-GRAVITY ANALYTICAL TEMPERATURE CHECK")
print("=" * 72)
print(
    f"RMSE [K]: {temperature_rmse:.8e}"
)
print(
    f"MAE [K]: {temperature_mae:.8e}"
)
print(
    f"Maximum absolute error [K]: "
    f"{temperature_max_error:.8e}"
)

figure, axis = plt.subplots(
    figsize=(
        6.0,
        10.0,
    ),
    constrained_layout=True,
)

contour = axis.contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    temperature_error,
    levels=50,
    cmap="coolwarm",
    vmin=-error_limit,
    vmax=error_limit,
)

colorbar = figure.colorbar(
    contour,
    ax=axis,
)

colorbar.set_label(
    "PINN - analytical temperature [K]"
)

draw_nh3_outline(
    axis,
)

axis.set_xlabel(
    "r [mm]"
)

axis.set_ylabel(
    "z [mm]"
)

axis.set_aspect(
    "equal",
    adjustable="box",
)

axis.set_title(
    "Temperature error\n"
    f"RMSE = {temperature_rmse:.3e} K"
)

save_figure(
    figure,
    "05_temperature_error_vs_linear_solution.png",
)

# Plot 06: numerical divergence diagnostic

r_values = RR[
    0,
    :
]

z_values = ZZ[
    :,
    0
]

dr_plot = float(
    r_values[
        1
    ]
    - r_values[
        0
    ]
)

dz_plot = float(
    z_values[
        1
    ]
    - z_values[
        0
    ]
)

ur_filled = np.nan_to_num(
    UR,
    nan=0.0,
)

uz_filled = np.nan_to_num(
    UZ,
    nan=0.0,
)

dur_dr = np.gradient(
    ur_filled,
    dr_plot,
    axis=1,
)

duz_dz = np.gradient(
    uz_filled,
    dz_plot,
    axis=0,
)

r_safe = np.maximum(
    RR,
    1.0e-8,
)

divergence = (
    dur_dr
    + UR / r_safe
    + duz_dz
)

divergence[
    ~field_data[
        "mask"
    ]
] = np.nan

divergence[
    :,
    0
] = np.nan

div_min, div_max = symmetric_limits(
    divergence,
)

figure, axis = plt.subplots(
    figsize=(
        6.0,
        10.0,
    ),
    constrained_layout=True,
)

contour = axis.contourf(
    RR * 1.0e3,
    ZZ * 1.0e3,
    divergence,
    levels=50,
    cmap="RdBu_r",
    vmin=div_min,
    vmax=div_max,
)

colorbar = figure.colorbar(
    contour,
    ax=axis,
)

colorbar.set_label(
    r"$\nabla_{\mathrm{axi}}\cdot u$ [1/s]"
)

draw_nh3_outline(
    axis,
)

axis.set_xlabel(
    "r [mm]"
)

axis.set_ylabel(
    "z [mm]"
)

axis.set_aspect(
    "equal",
    adjustable="box",
)

axis.set_title(
    "Axisymmetric divergence diagnostic"
)

save_figure(
    figure,
    "06_axisymmetric_divergence.png",
)

# Plot 07: loss history

figure, axis = plt.subplots(
    figsize=(
        10.0,
        6.0,
    ),
    constrained_layout=True,
)

try:
    pinn.losses.plot(
        axis,
        loss_groups=[
            "nh3",
            "axis",
            "outer_side",
            "outer_top",
            "lower_polyline",
        ],
    )

except Exception as error:
    print(
        "\nLoss-group plot failed; "
        "using default SCIMBA loss plot."
    )
    print(
        f"Reason: {error}"
    )

    pinn.losses.plot(
        axis,
    )

axis.set_xlabel(
    "Epoch"
)

axis.set_ylabel(
    "Loss"
)

axis.set_title(
    "Steady axisymmetric NH$_3$ PINN loss history"
)

axis.grid(
    True,
    which="both",
    alpha=0.30,
)

save_figure(
    figure,
    "07_loss_history.png",
)

# Final summaries

print("\n" + "=" * 72)
print("FINAL FIELD RANGES")
print("=" * 72)

print(
    f"u_r min/max [m/s]: "
    f"{np.nanmin(UR):.8e}, "
    f"{np.nanmax(UR):.8e}"
)

print(
    f"u_z min/max [m/s]: "
    f"{np.nanmin(UZ):.8e}, "
    f"{np.nanmax(UZ):.8e}"
)

print(
    f"|u| max [m/s]: "
    f"{np.nanmax(SPEED):.8e}"
)

print(
    f"pressure min/max [model units]: "
    f"{np.nanmin(PRESSURE):.8e}, "
    f"{np.nanmax(PRESSURE):.8e}"
)

print(
    f"temperature min/max [K]: "
    f"{np.nanmin(TEMPERATURE):.8e}, "
    f"{np.nanmax(TEMPERATURE):.8e}"
)

print("\n" + "=" * 72)
print("MODEL CONFIGURATION")
print("=" * 72)

print(
    f"GRAVITY_Z = {GRAVITY_Z:.6e} m/s^2"
)

print(
    f"nu_NH3 = {NU_NH3:.6e} m^2/s"
)

print(
    f"kappa_NH3 = {KAPPA_NH3:.6e} m^2/s"
)

print(
    f"T_BC(z) = {T_INTERCEPT:.6f} "
    f"+ ({T_SLOPE:.6f}) * z"
)

print(
    "\nAll plots saved in:\n"
    + os.path.abspath(
        OUTPUT_DIR
    )
)
