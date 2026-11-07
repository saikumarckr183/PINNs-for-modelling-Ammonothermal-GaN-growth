r"""
Steady 3D Boussinesq Navier–Stokes in a cylinder with a bottom hemisphere.

"""
import os
import jax
jax.config.update("jax_enable_x64", False)

import timeit
from typing import Sequence

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import scimba_jax
from scimba_jax.domains.domain_mapping import DomainMapping
from scimba_jax.domains.meshless_domains.base import (
    DEFAULT_LABEL_STR_SURFACIC,
    DEFAULT_LABEL_STR_VOLUMETRIC,
    SurfacicDomain,
    VolumetricDomain,
)
from scimba_jax.domains.meshless_domains.domains_1d import Segment1D
from scimba_jax.domains.meshless_domains.domains_nd import HypercubeND
from scimba_jax.domains.sdf import SignedDistance

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
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_residuals import BoundaryResidual, static
from scimba_jax.physical_models.elliptic_pde.navier_stokes_stationary import (
    NavierStokesThermalND,
    _extract_u_p_t,
)

from scimba_jax.plots._utils.eval_utilities import eval_on_np_tensors
from mpl_toolkits.mplot3d import Axes3D

# configuration
MODE = "vec"

# geometry
R = 0.01
H = 0.03

# thermal BCs
T_hot = 750.05
T_cold = 650.05
T_ref = 700.05

# physical parameters
nu = 1.570e-7                           # Kinematic viscosity m2/s
kappa = 1.644e-7                         # Thermal diffusivity m2/s
alpha = 2.631e-3                        # Thermal expansion coefficient 1/K
gravity = (0.0, 0.0, -9.81)

# training
N_COLLOC = 7000
N_BC_COLLOC = 5000
N_EPOCHS = 100

WEIGHTS = {
    "interior": [1.0, 1.0, 1.0, 2.0, 1.0],   # [mx, my, mz, div, energy]
    "bc_top": [20.0, 20.0, 20.0, 20.0],      # [ux, uy, uz, dT/dn]
    "bc_bottom": [20.0, 20.0, 20.0, 20.0],   # [ux, uy, uz, dT/dn]
    "bc_side": [20.0, 20.0, 20.0, 20.0],     # [ux, uy, uz, T - T_profile(z)]
}

# Custom SDF for cylinder + bottom hemisphere
class CylinderPlusBottomHemisphereSDF(SignedDistance):
    """
    Approximate signed distance for union of:
        1) finite cylinder: x^2 + y^2 <= R^2, 0 <= z <= H
        2) lower hemisphere: x^2 + y^2 + z^2 <= R^2, z <= 0

    Sign convention from ScimBa:
        sdf < 0 inside, sdf = 0 on boundary, sdf > 0 outside
    """

    def __init__(self, radius: float, height: float):
        super().__init__(dim=3, threshold=0.0)
        self.R = radius
        self.H = height

    def _sdf_cylinder(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Exact SDF of a finite vertical cylinder of radius R and z in [0, H].
        """
        xy = x[..., :2]
        z = x[..., 2:3]

        radial = jnp.sqrt(xy[..., 0:1]**2 + xy[..., 1:2]**2) - self.R
        axial = jnp.maximum(-z, z - self.H)

        outside_radial = jnp.clip(radial, min=0.0)
        outside_axial = jnp.clip(axial, min=0.0)
        outside = jnp.sqrt(outside_radial**2 + outside_axial**2)

        inside = jnp.minimum(jnp.maximum(radial, axial), 0.0)
        return (outside + inside).squeeze(-1)

    def _sdf_lower_hemisphere(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        SDF for the lower half of the sphere x^2+y^2+z^2 <= R^2 with z <= 0.
        Realized as intersection of:
            sphere interior
            half-space z <= 0
        """
        sphere = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2 + x[..., 2]**2) - self.R
        halfspace = x[..., 2]
        return jnp.maximum(sphere, halfspace)

    def _sdf_pointwise(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Union of cylinder and lower hemisphere via min of component SDFs.
        """
        s_cyl = self._sdf_cylinder(x)
        s_hemi = self._sdf_lower_hemisphere(x)
        return jnp.minimum(s_cyl, s_hemi)[None]


# Custom boundary surfaces
class DiskSurface3D(SurfacicDomain):
    """
    Disk at constant z = z0 with radius R.
    Parametric coordinates: (r, theta) in [0,R] x [0,2pi]
    """

    def __init__(
        self,
        radius: float,
        z0: float,
        upward: bool = True,
        label_str: str = DEFAULT_LABEL_STR_SURFACIC,
        label_idx: int = 0,
    ):
        self.radius = radius
        self.z0 = z0
        self.upward = upward

        parametric_domain = HypercubeND([(0.0, radius), (0.0, 2 * jnp.pi)])

        def map_fn(t: jnp.ndarray) -> jnp.ndarray:
            r = t[0:1]
            th = t[1:2]
            return jnp.concatenate(
                [r * jnp.cos(th), r * jnp.sin(th), jnp.array([z0], dtype=t.dtype)],
                axis=-1,
            )

        def jac_fn(t: jnp.ndarray) -> jnp.ndarray:
            r = t[0:1]
            th = t[1:2]
            dr = jnp.concatenate(
                [jnp.cos(th), jnp.sin(th), jnp.array([0.0], dtype=t.dtype)], axis=-1
            )
            dth = jnp.concatenate(
                [-r * jnp.sin(th), r * jnp.cos(th), jnp.array([0.0], dtype=t.dtype)],
                axis=-1,
            )
            jac = jnp.stack([dr, dth], axis=-1)
            if upward:
                return jac
            return jnp.stack([dth, dr], axis=-1)

        surface = DomainMapping(2, 3, map_fn, jac_fn, None, None)

        super().__init__(
            domain_type="DiskSurface3D",
            parametric_domain=parametric_domain,
            surface=surface,
            label_str=label_str,
            label_idx=label_idx,
        )


class CylinderSideSurface3D(SurfacicDomain):
    """
    Lateral wall of cylinder of radius R and z in [0,H].
    Parametric coordinates: (z, theta) in [0,H] x [0,2pi]
    """

    def __init__(
        self,
        radius: float,
        height: float,
        label_str: str = DEFAULT_LABEL_STR_SURFACIC,
        label_idx: int = 0,
    ):
        self.radius = radius
        self.height = height

        parametric_domain = HypercubeND([(0.0, height), (0.0, 2 * jnp.pi)])

        def map_fn(t: jnp.ndarray) -> jnp.ndarray:
            z = t[0:1]
            th = t[1:2]
            return jnp.concatenate(
                [radius * jnp.cos(th), radius * jnp.sin(th), z], axis=-1
            )

        def jac_fn(t: jnp.ndarray) -> jnp.ndarray:
            th = t[1:2]
            dz = jnp.array([0.0, 0.0, 1.0], dtype=t.dtype)
            dth = jnp.concatenate(
                [-radius * jnp.sin(th), radius * jnp.cos(th), jnp.array([0.0], dtype=t.dtype)],
                axis=-1,
            )
            return jnp.stack([dz, dth], axis=-1)

        surface = DomainMapping(2, 3, map_fn, jac_fn, None, None)

        super().__init__(
            domain_type="CylinderSideSurface3D",
            parametric_domain=parametric_domain,
            surface=surface,
            label_str=label_str,
            label_idx=label_idx,
        )


class LowerHemisphereSurface3D(SurfacicDomain):
    """
    Lower hemisphere of radius R centered at origin.
    Parametric coordinates: (phi, theta) in [0, pi/2] x [0, 2pi]
    with
        x = R sin(phi) cos(theta)
        y = R sin(phi) sin(theta)
        z = -R cos(phi)
    """

    def __init__(
        self,
        radius: float,
        label_str: str = DEFAULT_LABEL_STR_SURFACIC,
        label_idx: int = 0,
    ):
        self.radius = radius

        parametric_domain = HypercubeND([(0.0, 0.5 * jnp.pi), (0.0, 2 * jnp.pi)])

        def map_fn(t: jnp.ndarray) -> jnp.ndarray:
            phi = t[0:1]
            th = t[1:2]
            return jnp.concatenate(
                [
                    radius * jnp.sin(phi) * jnp.cos(th),
                    radius * jnp.sin(phi) * jnp.sin(th),
                    -radius * jnp.cos(phi),
                ],
                axis=-1,
            )

        def jac_fn(t: jnp.ndarray) -> jnp.ndarray:
            phi = t[0:1]
            th = t[1:2]
            dphi = jnp.concatenate(
                [
                    radius * jnp.cos(phi) * jnp.cos(th),
                    radius * jnp.cos(phi) * jnp.sin(th),
                    radius * jnp.sin(phi),
                ],
                axis=-1,
            )
            dth = jnp.concatenate(
                [
                    -radius * jnp.sin(phi) * jnp.sin(th),
                    radius * jnp.sin(phi) * jnp.cos(th),
                    jnp.array([0.0], dtype=t.dtype),
                ],
                axis=-1,
            )
            return jnp.stack([dphi, dth], axis=-1)

        surface = DomainMapping(2, 3, map_fn, jac_fn, None, None)

        super().__init__(
            domain_type="LowerHemisphereSurface3D",
            parametric_domain=parametric_domain,
            surface=surface,
            label_str=label_str,
            label_idx=label_idx,
        )


# Custom volumetric domain
class CylinderPlusBottomHemisphere3D(VolumetricDomain):
    def __init__(
        self,
        radius: float,
        height: float,
        is_main_domain: bool = False,
        label_str: str = DEFAULT_LABEL_STR_VOLUMETRIC,
        label_idx: int = 0,
    ):
        bounds = jnp.array(
            [(-radius, radius), (-radius, radius), (-radius, height)],
            dtype=scimba_jax.get_default_dtype(),
        )
        sdf = CylinderPlusBottomHemisphereSDF(radius, height)

        super().__init__(
            domain_type="CylinderPlusBottomHemisphere3D",
            dim=3,
            sdf=sdf,
            bounds=bounds,
            is_main_domain=is_main_domain,
            label_str=label_str,
            label_idx=label_idx,
        )

        self.radius = radius
        self.height = height

    def full_bc_domain(self):
        label_bc = "bc"
        if not self.is_main_domain:
            label_bc += " " + self.get_label()

        res = [
            DiskSurface3D(self.radius, self.height, upward=True, label_str=label_bc + " top"),
            CylinderSideSurface3D(self.radius, self.height, label_str=label_bc + " side"),
            LowerHemisphereSurface3D(self.radius, label_str=label_bc + " bottom"),
        ]

        if self.is_mapped:
            for r in res:
                r._set_mapping(self.mapping)

        return res


# Boundary residuals (new conditions)

class NoSlipIsothermalBC(BoundaryResidual):
    """
    Residual: [ux, uy, uz, T - T_wall]

    - Enforces no-slip: ux = 0, uy = 0, uz = 0 (weakly).
    - Enforces Dirichlet temperature: T = T_wall (weakly).
    """

    _ns_dim: int = static(3)
    _ns_mode: str = static("vec")
    _T_wall: float = static(0.0)

    def __init__(self, domain, T_wall: float, dim: int = 3, mode: str = "vec"):
        super().__init__(domain=domain, size=dim + 1, model_type="x", f_rhs=None)
        self._ns_dim = dim
        self._ns_mode = mode
        self._T_wall = T_wall

    def construct_residual(self, *vars):
        u, _, T = _extract_u_p_t(vars, self._ns_mode, self._ns_dim)
        return ParamVecFunction.cat(
            [u.component(i) for i in range(self._ns_dim)] + [T - self._T_wall]
        )


class NoSlipAdiabaticBC(BoundaryResidual):
    """
    Residual: [ux, uy, uz, dT/dn]

    - Enforces no-slip: ux = 0, uy = 0, uz = 0 (weakly).
    - Enforces Neumann temperature: dT/dn = 0 (adiabatic boundary).
    """

    _ns_dim: int = static(3)
    _ns_mode: str = static("vec")

    def __init__(self, domain, dim: int = 3, mode: str = "vec"):
        super().__init__(domain=domain, size=dim + 1, model_type="x", f_rhs=None)
        self._ns_dim = dim
        self._ns_mode = mode

    def construct_residual(self, *vars):
        u, _, T = _extract_u_p_t(vars, self._ns_mode, self._ns_dim)

        # gradient of T in physical space
        grad_T = T.gradient_x()

        # boundary normal n is provided via the sample dict; we recover it as a field
        n_model = ParamFieldFunction(
            T.dims,
            lambda *args: args[T.argnums["n"]],  # "n" = normal entry in args
            T.f_type,
        )

        dTdn = grad_T.dot(n_model)
        return ParamVecFunction.cat(
            [u.component(i) for i in range(self._ns_dim)] + [dTdn]
        )


class NoSlipProfileBC(BoundaryResidual):
    """
    Residual: [ux, uy, uz, T - T_profile]

    - No-slip: ux = 0, uy = 0, uz = 0 (weakly).
    - Prescribed temperature profile along the side wall:

          T(z) = T_hot - (T_hot - T_cold) * (z - z_min) / (z_max - z_min)

      instead of dT/dn = 0.

    This BC uses a ParamFieldFunction T_profile that depends on the same
    arguments as T, and extracts z from the spatial coordinates internally.
    """

    _ns_dim: int = static(3)
    _ns_mode: str = static("vec")
    _z_min: float = static(0.0)
    _z_max: float = static(0.0)
    _T_hot: float = static(0.0)
    _T_cold: float = static(0.0)

    def __init__(
        self,
        domain,
        T_hot: float,
        T_cold: float,
        z_min: float,
        z_max: float,
        dim: int = 3,
        mode: str = "vec",
    ):
        super().__init__(domain=domain, size=dim + 1, model_type="x", f_rhs=None)
        self._ns_dim = dim
        self._ns_mode = mode
        self._T_hot = T_hot
        self._T_cold = T_cold
        self._z_min = z_min
        self._z_max = z_max

    def construct_residual(self, *rho, precomputed=None):
        # rho is a tuple of Param... objects (u, p, T) with normals already
        # embedded via BoundaryResidual._pre_processing.

        u, _, T = _extract_u_p_t(rho, self._ns_mode, self._ns_dim)

        # Define T_profile as a parametric field function that:
        #   - uses the same dims and argument structure as T,
        #   - extracts z from the spatial coordinates x inside its fn.
        def t_profile_fn(*args):
            """
            args are the same as T's fn: typically (space, t?, x, n, mu?...).
            We locate x via T.argnums['x'] and compute the linear profile in z.
            """
            # Find position of x in the argument list used by T
            x_index = T.argnums.get("x", None)
            if x_index is None:
                raise RuntimeError(
                    "NoSlipProfileBC: T.argnums has no 'x' entry; "
                    "cannot build T_profile(z)."
                )

            x = args[x_index]       # x = (..., 3) spatial coordinates
            z = x[..., 2:3]         # vertical coordinate

            return self._T_hot - (self._T_hot - self._T_cold) * (
                (z - self._z_min) / (self._z_max - self._z_min)
            )

        T_profile = ParamFieldFunction(
            T.dims,
            t_profile_fn,
            T.f_type,
            "x",    # field lives in physical space
        )

        # Residual vector: [ux, uy, uz, T - T_profile(z)]
        return ParamVecFunction.cat(
            [u.component(i) for i in range(self._ns_dim)] + [T - T_profile]
        )

# Assign BCs with new conditions

dx = CylinderPlusBottomHemisphere3D(R, H, is_main_domain=True)
dx.set_boundaries_dict(
    {
        "bc_top": ["top"],
        "bc_side": ["side"],
        "bc_bottom": ["bottom"],
    }
)
boundaries = dx.get_all_boundaries()

bc_residuals = {
    # Top disk: no-slip + adiabatic (dT/dn = 0) instead of T = T_cold
    "bc_top": NoSlipAdiabaticBC(boundaries["bc_top"], dim=3, mode=MODE),

    # Side wall: no-slip + prescribed T(z) profile instead of dT/dn = 0
    "bc_side": NoSlipProfileBC(
        boundaries["bc_side"],
        T_hot=T_hot,
        T_cold=T_cold,
        z_min=0.0,
        z_max=H,
        dim=3,
        mode=MODE,
    ),

    # Bottom hemisphere: no-slip + adiabatic (dT/dn = 0) instead of T = T_hot
    "bc_bottom": NoSlipAdiabaticBC(boundaries["bc_bottom"], dim=3, mode=MODE),
}


# Physics model
key = jax.random.PRNGKey(0)

model = NavierStokesThermalND(
    dx,
    dim=3,
    nu=nu,
    kappa=kappa,
    alpha=alpha,
    T_ref=T_ref,
    gravity=gravity,
    mode=MODE,
    bc_residuals=bc_residuals,
    f_rhs=None,
)

# Approximation space
if MODE == "vec":
    nn = MLP(in_size=3, out_size=5, hidden_sizes=[32, 32, 32, 32, 32], key=key)

    # No hard constraint in this first curved-geometry version.
    def post_processing_vec(output, x):
        return output

    space = ApproximationSpace(
        {"x": 3},
        [(nn, "vec", 5)],
        model_type="x",
        post_processing=post_processing_vec,
    )
else:
    raise NotImplementedError("This prototype is written for MODE='vec' only.")

# Sampler and projector
sampler = TensorizedSampler(
    [DomainSampler(dx)],
    bc=True,
)

# Training
pinn = Projector(model, space, sampler, weights=WEIGHTS, one_loss_by_equation=True)

key, sample_dict = sampler.sample(key, N_COLLOC, N_BC_COLLOC)
loss0 = pinn.evaluate_loss(space, sample_dict)
print("initial loss:", loss0)

start = timeit.default_timer()
key, pinn = pinn.project(key, space, N_EPOCHS, N_COLLOC, N_BC_COLLOC)
end = timeit.default_timer()

print(f"best loss: {pinn.best_loss}")
print(f"time for {N_EPOCHS} epochs: {end - start:.1f}s")



PLOT_DIR = "output/plots_3d_hemi_cylinder"
os.makedirs(PLOT_DIR, exist_ok=True)

# Meridional slice y = 0:
# x in [-R, R], z in [-R, H]
SLICE_NX, SLICE_NZ = 181, 241

# Helper: evaluate one component on the slice
def _eval_component_on_slice(comp, derivs=None):
    kwargs = {}
    if derivs is not None:
        kwargs["derivatives"] = derivs

    xv = jnp.linspace(-R, R, SLICE_NX)
    zv = jnp.linspace(-R, H, SLICE_NZ)
    XX, ZZ = jnp.meshgrid(xv, zv, indexing="xy")
    YY = jnp.zeros_like(XX)

    X_flat = jnp.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    X_np = np.array(X_flat, dtype=np.float64)

    inside_mask = np.array(dx.sdf(X_np)).reshape(-1) <= 0.0

    t_np = np.zeros((X_np.shape[0], 0), dtype=np.float64)
    v_np = np.zeros((X_np.shape[0], 0), dtype=np.float64)
    mu_np = np.zeros((X_np.shape[0], 0), dtype=np.float64)

    symb_dict = {
        "components": ["u0", "u1", "u2", "p", "T"],
        "space_variables": ["x", "y", "z"],
        "time_variable": ["t"],
        "phase_variables": [],
    }

    evals = eval_on_np_tensors(
        pinn.space,
        t_np,
        X_np,
        v_np,
        mu_np,
        symb_dict=symb_dict,
        component=comp,
        **kwargs,
    )

    approx = np.array(evals["approximation"]).reshape(-1)
    approx[~inside_mask] = np.nan
    approx = approx.reshape(SLICE_NZ, SLICE_NX)

    out = {"approximation": approx, "XX": np.array(XX), "ZZ": np.array(ZZ)}

    if derivs is not None:
        for d in derivs:
            arr = np.array(evals[d]).reshape(-1)
            arr[~inside_mask] = np.nan
            out[d] = arr.reshape(SLICE_NZ, SLICE_NX)

    return out

# Helper: evaluate approximation and residual for one component
def _eval_component_and_residual_on_slice(comp):
    xv = jnp.linspace(-R, R, SLICE_NX)
    zv = jnp.linspace(-R, H, SLICE_NZ)
    XX, ZZ = jnp.meshgrid(xv, zv, indexing="xy")
    YY = jnp.zeros_like(XX)

    X_flat = jnp.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    X_np = np.array(X_flat, dtype=np.float64)

    inside_mask = np.array(dx.sdf(X_np)).reshape(-1) <= 0.0

    t_np = np.zeros((X_np.shape[0], 0), dtype=np.float64)
    v_np = np.zeros((X_np.shape[0], 0), dtype=np.float64)
    mu_np = np.zeros((X_np.shape[0], 0), dtype=np.float64)

    symb_dict = {
        "components": ["u0", "u1", "u2", "p", "T"],
        "space_variables": ["x", "y", "z"],
        "time_variable": ["t"],
        "phase_variables": [],
    }

    labelsx_interior = np.array(["interior"] * X_np.shape[0], dtype=str)

    evals = eval_on_np_tensors(
        pinn.space,
        t_np,
        X_np,
        v_np,
        mu_np,
        symb_dict=symb_dict,
        component=comp,
        labelsx=labelsx_interior,
        residual=pinn.model,
    )

    approx = np.array(evals["approximation"]).reshape(-1)
    resid = np.array(evals["residual"]).reshape(-1)

    approx[~inside_mask] = np.nan
    resid[~inside_mask] = np.nan

    approx = approx.reshape(SLICE_NZ, SLICE_NX)
    resid = resid.reshape(SLICE_NZ, SLICE_NX)

    l2_norm = np.sqrt(np.nanmean(resid**2))

    return {
        "approximation": approx,
        "residual": resid,
        "l2_norm": l2_norm,
        "XX": np.array(XX),
        "ZZ": np.array(ZZ),
    }

# Helper: draw vessel outline in meridional slice
def _draw_vessel_outline(ax):
    theta = np.linspace(np.pi, 2.0 * np.pi, 400)
    arc_x = R * np.cos(theta)
    arc_z = R * np.sin(theta)

    ax.plot([-R, -R], [0.0, H], color="black", lw=1.0)
    ax.plot([ R,  R], [0.0, H], color="black", lw=1.0)
    ax.plot([-R,  R], [H, H], color="black", lw=1.0)
    ax.plot(arc_x, arc_z, color="black", lw=1.0)

# 1) Geometry and BCs
fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)

# Outer rectangle for cylinder
ax.plot([-R, R, R, -R, -R], [0, 0, H, H, 0], color="black", lw=2)
ax.fill([-R, R, R, -R], [0, 0, H, H], color="lightgray", alpha=0.15)

# Side wall line for annotation
ax.plot([R, R], [0, H], color="black", lw=2)

# Bottom hemisphere outline
yy = np.linspace(-R, R, 400)
zz = -np.sqrt(np.maximum(R**2 - yy**2, 0.0))
ax.plot(yy, zz, color="black", lw=2)

ax.text(0.55 * R, 0.55 * H, "A", fontsize=18, color="green")   # fluid region
ax.text(0.10 * R, 0.25 * H, "B", fontsize=18, color="blue")    # hemisphere

ax.text(0.02 * R, H + 0.02, r"$T=T_{\mathrm{cold}}$", fontsize=11, color="blue")
ax.text(0.02 * R, -0.18 * R, r"$T=T_{\mathrm{hot}}$", fontsize=11, color="red")
ax.text(R + 0.02, 0.5 * H, r"$\partial T/\partial n=0$", fontsize=11,
        rotation=90, va="center")
ax.text(0.25 * R, 0.78 * H, r"$u=0$", fontsize=11, color="purple")

ax.set_xlim(-1.1 * R, 1.25 * R)
ax.set_ylim(-1.15 * R, H + 0.1)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_title("Geometry and boundary conditions")

fig.savefig(os.path.join(PLOT_DIR, "00_geometry_bcs.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# 2) Temperature field
temp_data = _eval_component_on_slice(4)

fig, ax = plt.subplots(figsize=(7.5, 9), constrained_layout=True)
cf = ax.contourf(temp_data["XX"], temp_data["ZZ"],
                 temp_data["approximation"], levels=40, cmap="inferno")
fig.colorbar(cf, ax=ax, label="T", fraction=0.046, pad=0.04)
_draw_vessel_outline(ax)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Meridional slice y = 0: Temperature")

fig.savefig(os.path.join(PLOT_DIR, "01_temperature.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# 3) Velocity magnitude + vectors
ux_data = _eval_component_on_slice(0)
uy_data = _eval_component_on_slice(1)
uz_data = _eval_component_on_slice(2)

Ux = ux_data["approximation"]
Uy = uy_data["approximation"]
Uz = uz_data["approximation"]
Speed = np.sqrt(Ux**2 + Uy**2 + Uz**2)

fig, ax = plt.subplots(figsize=(7.5, 9), constrained_layout=True)
cf = ax.contourf(ux_data["XX"], ux_data["ZZ"], Speed,
                 levels=40, cmap="viridis")
fig.colorbar(cf, ax=ax, label=r"$|u|$", fraction=0.046, pad=0.04)
_draw_vessel_outline(ax)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal", adjustable="box")

step = 10
ax.quiver(
    ux_data["XX"][::step, ::step],
    ux_data["ZZ"][::step, ::step],
    Ux[::step, ::step],
    Uz[::step, ::step],
    color="white",
    scale=10,
    width=0.0025,
)
ax.set_title("Meridional slice y = 0: Velocity magnitude + vectors")

fig.savefig(os.path.join(PLOT_DIR, "02_velocity_mag_vectors.png"),
            dpi=300, bbox_inches="tight")
plt.show()


# 4) Pressure field
p_data = _eval_component_on_slice(3)

fig, ax = plt.subplots(figsize=(7.5, 9), constrained_layout=True)
cf = ax.contourf(p_data["XX"], p_data["ZZ"],
                 p_data["approximation"], levels=40, cmap="coolwarm")
fig.colorbar(cf, ax=ax, label="p", fraction=0.046, pad=0.04)
_draw_vessel_outline(ax)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Meridional slice y = 0: Pressure")

fig.savefig(os.path.join(PLOT_DIR, "03_pressure.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# 5) Divergence (approximate from Uy, Uz in slice)
xv = np.array(temp_data["XX"][0, :])
zv = np.array(temp_data["ZZ"][:, 0])
dy = float(xv[1] - xv[0])
dz = float(zv[1] - zv[0])

dUy_dy = np.gradient(np.nan_to_num(Uy, nan=0.0), dy, axis=1)
dUz_dz = np.gradient(np.nan_to_num(Uz, nan=0.0), dz, axis=0)
Div = dUy_dy + dUz_dz
Div[np.isnan(temp_data["approximation"])] = np.nan

fig, ax = plt.subplots(figsize=(7.5, 9), constrained_layout=True)
cf = ax.contourf(temp_data["XX"], temp_data["ZZ"], Div,
                 levels=40, cmap="RdBu_r")
fig.colorbar(cf, ax=ax, label=r"$\nabla \cdot u$", fraction=0.046, pad=0.04)
_draw_vessel_outline(ax)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Meridional slice y = 0: Divergence")

fig.savefig(os.path.join(PLOT_DIR, "04_divergence.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# 6) Approximation + residual panels for each component
for comp, comp_name in [
    (0, r"$u_x$"),
    (1, r"$u_y$"),
    (2, r"$u_z$"),
    (3, r"$p$"),
    (4, r"$T$"),
]:
    data = _eval_component_and_residual_on_slice(comp)

    fig, (ax_a, ax_r) = plt.subplots(
        1, 2,
        figsize=(11, 6),
        constrained_layout=True
    )

    # Approximation
    cf_a = ax_a.contourf(data["XX"], data["ZZ"],
                         data["approximation"], levels=40, cmap="turbo")
    ax_a.contour(data["XX"], data["ZZ"],
                 data["approximation"], levels=20,
                 colors="white", linewidths=0.5, alpha=0.7)
    fig.colorbar(cf_a, ax=ax_a, fraction=0.046, pad=0.04)
    _draw_vessel_outline(ax_a)
    ax_a.set_xlabel("x")
    ax_a.set_ylabel("z")
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_title(f"Approximation: {comp_name}")

    # Residual
    cf_r = ax_r.contourf(data["XX"], data["ZZ"],
                         data["residual"], levels=40, cmap="turbo")
    ax_r.contour(data["XX"], data["ZZ"],
                 data["residual"], levels=20,
                 colors="white", linewidths=0.5, alpha=0.7)
    fig.colorbar(cf_r, ax=ax_r, fraction=0.046, pad=0.04)
    _draw_vessel_outline(ax_r)
    ax_r.set_xlabel("x")
    ax_r.set_ylabel("z")
    ax_r.set_aspect("equal", adjustable="box")
    ax_r.set_title(
        rf"Residual: {comp_name}, $L^2$ norm = {data['l2_norm']:.2e}"
    )

    safe_name = (comp_name.replace("$", "")
                              .replace("{", "")
                              .replace("}", "")
                              .replace("\\", ""))
    fig.savefig(os.path.join(PLOT_DIR,
                             f"approx_residual_{safe_name}.png"),
                dpi=300, bbox_inches="tight")
    plt.show()

# 7) Loss history
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
try:
    pinn.losses.plot(ax, loss_groups=["bc", "interior"])
except Exception:
    pinn.losses.plot(ax)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Loss history")
ax.grid(True, alpha=0.3)

fig.savefig(os.path.join(PLOT_DIR, "loss_history.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# COMPLETE POST-PROCESSING BLOCK:
# Separate 3D plots for ux, uy, uz, p, and T

# 1. Create random points in the bounding box
N_PLOT_TRY = 30000
rng = np.random.default_rng(123)

xyz_all = np.column_stack(
    [
        rng.uniform(-R, R, N_PLOT_TRY),   # x values
        rng.uniform(-R, R, N_PLOT_TRY),   # y values
        rng.uniform(-R, H, N_PLOT_TRY),   # z values
    ]
).astype(np.float32)


# 2. Keep only points inside cylinder + bottom hemisphere
# SDF <= 0 means inside the fluid domain
sdf_values = jax.vmap(dx.sdf._sdf_pointwise)(jnp.asarray(xyz_all))
sdf_values = np.asarray(sdf_values).reshape(-1)

xyz = xyz_all[sdf_values <= 0.0]

x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]

print("Number of points used for 3D plotting:", len(xyz))


# 3. Evaluate trained PINN
# Output order for MODE = "vec":
# [ux, uy, uz, p, T]
prediction = np.asarray(
    pinn.evaluate(
        jnp.asarray(xyz, dtype=scimba_jax.get_default_dtype())
    )
)

ux = prediction[:, 0]
uy = prediction[:, 1]
uz = prediction[:, 2]
p_pred = prediction[:, 3]
T_pred = prediction[:, 4]

print("ux min/max:", float(ux.min()), float(ux.max()))
print("uy min/max:", float(uy.min()), float(uy.max()))
print("uz min/max:", float(uz.min()), float(uz.max()))
print("p  min/max:", float(p_pred.min()), float(p_pred.max()))
print("T  min/max:", float(T_pred.min()), float(T_pred.max()))


# 4. Helper: make ONE 3D scalar-field plot
def plot_3d_scalar(x, y, z, values, title, colorbar_label, cmap):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x,
        y,
        z,
        c=values,
        cmap=cmap,
        s=4,
        alpha=0.65,
    )

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.08)
    colorbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Geometry dimensions: x in [-R,R], y in [-R,R], z in [-R,H]
    ax.set_box_aspect((2 * R, 2 * R, H + R))

    plt.tight_layout()
    plt.show()


# 5. Create five separate 3D plots

# Plot 1: ux
plot_3d_scalar(
    x, y, z,
    ux,
    title=r"3D Field: $u_x$",
    colorbar_label=r"$u_x$",
    cmap="coolwarm",
)

# Plot 2: uy
plot_3d_scalar(
    x, y, z,
    uy,
    title=r"3D Field: $u_y$",
    colorbar_label=r"$u_y$",
    cmap="coolwarm",
)

# Plot 3: uz
plot_3d_scalar(
    x, y, z,
    uz,
    title=r"3D Field: $u_z$",
    colorbar_label=r"$u_z$",
    cmap="coolwarm",
)

# Plot 4: pressure
plot_3d_scalar(
    x, y, z,
    p_pred,
    title=r"3D Field: Pressure $p$",
    colorbar_label=r"Pressure $p$",
    cmap="viridis",
)

# Plot 5: temperature
plot_3d_scalar(
    x, y, z,
    T_pred,
    title=r"3D Field: Temperature $T$",
    colorbar_label=r"Temperature $T$",
    cmap="inferno",
)
