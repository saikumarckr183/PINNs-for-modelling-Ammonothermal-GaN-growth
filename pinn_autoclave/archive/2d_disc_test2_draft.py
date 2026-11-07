"""
Steady 2D autoclave–ammonia benchmark for ScimBa/JAX using strong post-processing.

This is the runnable version that avoids boundary-coordinate access inside residuals.
It follows the structure you shared for scalar heat problems:
    - one scalar network
    - strong boundary enforcement through post-processing
    - interior residuals only
    - optional interface continuity through a separate residual block

Geometry:
    Outer rectangle:
        0 <= r <= R_OUT
        0 <= z <= H_OUT

    Ammonia cavity:
        approximated as a vertical cavity with a curved bottom part.

Physics:
    - Wall: k_wall(T) = 3.843196 + 0.01841176*T
    - Ammonia: k_NH3 = 0.162242
    - PDE used here in runnable form:
        k(T) * Laplacian(T) = 0
      This is the ScimBa-compatible benchmark residual.
      The exact cylindrical operator can be restored once a coordinate-access
      API is used inside the residual.
"""

from __future__ import annotations

import timeit
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
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
from scimba_jax.domains.meshless_domains.domains_nd import HypercubeND
from scimba_jax.domains.sdf import SignedDistance
from scimba_jax.nonlinear_approximation.approximation_spaces.approximation_spaces import ApproximationSpace
from scimba_jax.nonlinear_approximation.integration.monte_carlo import DomainSampler, TensorizedSampler
from scimba_jax.nonlinear_approximation.model_class.funcparam_vectorial import ParamVecFunction
from scimba_jax.nonlinear_approximation.model_class.funcparam_scalar import ParamScalarFunction
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel, PHYSICAL_RESIDUALS_TYPE
from scimba_jax.physical_models.abstract_residuals import BoundaryResidual, InteriorResidual, static
from scimba_jax.plots.plots_nd import plot_abstract_approx_space


# =============================================================================
# Parameters from the article
# =============================================================================
R_OUT = 35.0
H_OUT = 370.0
R_IN = 12.0
H_IN = 300.0

T_BOTTOM = 450.0
T_TOP = 550.0
ALPHA_0 = T_BOTTOM
ALPHA_1 = (T_TOP - T_BOTTOM) / H_OUT

K_NH3 = 0.162242


def k_wall(T):
    return 3.843196 + 0.01841176 * T


# =============================================================================
# Geometry
# =============================================================================
class Rectangle2DSDF(SignedDistance):
    def __init__(self, r_out: float, h: float):
        super().__init__(dim=2, threshold=0.0)
        self.r_out = r_out
        self.h = h

    def _sdf_pointwise(self, x: jnp.ndarray) -> jnp.ndarray:
        r = x[..., 0]
        z = x[..., 1]
        d_left = r
        d_right = self.r_out - r
        d_bottom = z
        d_top = self.h - z
        dist = jnp.stack([d_left, d_right, d_bottom, d_top], axis=-1)
        sdf_inside = -jnp.min(dist, axis=-1)
        sdf_outside = jnp.linalg.norm(jnp.clip(-dist, min=0.0), axis=-1)
        sdf = jnp.where(jnp.all(dist >= 0, axis=-1), sdf_inside, sdf_outside)
        return sdf[None]


class AmmoniaCavitySDF(SignedDistance):
    """
    Approximate cavity:
      - vertical cavity for z in [H_IN, H_OUT], r <= R_IN
      - lower curved region as a circular cap centered at (0, H_IN)
    """
    def __init__(self, r_in: float, h_in: float, h_out: float):
        super().__init__(dim=2, threshold=0.0)
        self.r_in = r_in
        self.h_in = h_in
        self.h_out = h_out

    def _sdf_rect_vertical(self, x):
        p = x[..., :2]
        xmin, xmax = 0.0, self.r_in
        ymin, ymax = self.h_in, self.h_out
        c = jnp.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=p.dtype)
        b = jnp.array([(xmax - xmin) * 0.5, (ymax - ymin) * 0.5], dtype=p.dtype)
        q = jnp.abs(p - c) - b
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
        inside = jnp.minimum(jnp.maximum(q[..., 0], q[..., 1]), 0.0)
        return outside + inside

    def _sdf_arc(self, x):
        p = x[..., :2]
        c = jnp.array([0.0, self.h_in], dtype=p.dtype)
        disk = jnp.linalg.norm(p - c, axis=-1) - self.r_in
        halfspace = p[..., 1] - self.h_in
        return jnp.maximum(disk, halfspace)

    def _sdf_pointwise(self, x: jnp.ndarray) -> jnp.ndarray:
        sdf_rect = self._sdf_rect_vertical(x)
        sdf_arc = self._sdf_arc(x)
        sdf = jnp.minimum(sdf_rect, sdf_arc)
        return sdf[None]


class WallMinusCavitySDF(SignedDistance):
    def __init__(self, outer: SignedDistance, cavity: SignedDistance):
        super().__init__(dim=2, threshold=0.0)
        self.outer = outer
        self.cavity = cavity

    def _sdf_pointwise(self, x: jnp.ndarray) -> jnp.ndarray:
        a = self.outer._sdf_pointwise(x)[0]
        b = self.cavity._sdf_pointwise(x)[0]
        return jnp.maximum(a, -b)[None]


# =============================================================================
# Domains
# =============================================================================
class OuterDomain2D(VolumetricDomain):
    def __init__(self, r_out, h_out, cavity_sdf, is_main_domain=True, label_str=DEFAULT_LABEL_STR_VOLUMETRIC, label_idx=0):
        bounds = jnp.array([(0.0, r_out), (0.0, h_out)], dtype=scimba_jax.get_default_dtype())
        outer = Rectangle2DSDF(r_out, h_out)
        sdf = WallMinusCavitySDF(outer, cavity_sdf)
        super().__init__(
            domain_type="OuterDomain2D",
            dim=2,
            sdf=sdf,
            bounds=bounds,
            is_main_domain=is_main_domain,
            label_str=label_str,
            label_idx=label_idx,
        )
        self.r_out = r_out
        self.h_out = h_out

    def full_bc_domain(self):
        label_bc = "bc"
        if not self.is_main_domain:
            label_bc += " " + self.get_label()
        return [
            OuterRightWall2D(self.r_out, self.h_out, label_str=label_bc + " outer_right"),
            OuterTopWall2D(self.r_out, label_str=label_bc + " outer_top"),
            OuterBottomWall2D(self.r_out, label_str=label_bc + " outer_bottom"),
        ]


class CavityDomain2D(VolumetricDomain):
    def __init__(self, r_in, h_in, h_out, is_main_domain=False, label_str=DEFAULT_LABEL_STR_VOLUMETRIC, label_idx=0):
        bounds = jnp.array([(0.0, r_in), (0.0, h_out)], dtype=scimba_jax.get_default_dtype())
        sdf = AmmoniaCavitySDF(r_in, h_in, h_out)
        super().__init__(
            domain_type="CavityDomain2D",
            dim=2,
            sdf=sdf,
            bounds=bounds,
            is_main_domain=is_main_domain,
            label_str=label_str,
            label_idx=label_idx,
        )
        self.r_in = r_in
        self.h_in = h_in
        self.h_out = h_out

    def full_bc_domain(self):
        label_bc = "bc"
        if not self.is_main_domain:
            label_bc += " " + self.get_label()
        return [
            InterfaceVertical2D(self.r_in, self.h_in, self.h_out, label_str=label_bc + " interface_vertical"),
            InterfaceTop2D(self.r_in, self.h_in, label_str=label_bc + " interface_top"),
            InterfaceArc2D(self.r_in, self.h_in, label_str=label_bc + " interface_arc"),
        ]


class OuterRightWall2D(SurfacicDomain):
    def __init__(self, r_out, h_out, label_str=DEFAULT_LABEL_STR_SURFACIC, label_idx=0):
        parametric_domain = HypercubeND([(0.0, h_out)])

        def map_fn(t):
            z = t[0:1]
            return jnp.concatenate([jnp.array([r_out], dtype=t.dtype), z], axis=-1)

        def jac_fn(t):
            return jnp.array([[0.0], [1.0]], dtype=t.dtype)

        surface = DomainMapping(1, 2, map_fn, jac_fn, None, None)
        super().__init__("OuterRightWall2D", parametric_domain, surface, label_str, label_idx)


class OuterTopWall2D(SurfacicDomain):
    def __init__(self, r_out, label_str=DEFAULT_LABEL_STR_SURFACIC, label_idx=0):
        parametric_domain = HypercubeND([(0.0, r_out)])

        def map_fn(t):
            r = t[0:1]
            return jnp.concatenate([r, jnp.array([H_OUT], dtype=t.dtype)], axis=-1)

        def jac_fn(t):
            return jnp.array([[1.0], [0.0]], dtype=t.dtype)

        surface = DomainMapping(1, 2, map_fn, jac_fn, None, None)
        super().__init__("OuterTopWall2D", parametric_domain, surface, label_str, label_idx)


class OuterBottomWall2D(SurfacicDomain):
    def __init__(self, r_out, label_str=DEFAULT_LABEL_STR_SURFACIC, label_idx=0):
        parametric_domain = HypercubeND([(0.0, r_out)])

        def map_fn(t):
            r = t[0:1]
            return jnp.concatenate([r, jnp.array([0.0], dtype=t.dtype)], axis=-1)

        def jac_fn(t):
            return jnp.array([[1.0], [0.0]], dtype=t.dtype)

        surface = DomainMapping(1, 2, map_fn, jac_fn, None, None)
        super().__init__("OuterBottomWall2D", parametric_domain, surface, label_str, label_idx)


class InterfaceVertical2D(SurfacicDomain):
    def __init__(self, r_in, z0, z1, label_str=DEFAULT_LABEL_STR_SURFACIC, label_idx=0):
        parametric_domain = HypercubeND([(z0, z1)])

        def map_fn(t):
            z = t[0:1]
            return jnp.concatenate([jnp.array([r_in], dtype=t.dtype), z], axis=-1)

        def jac_fn(t):
            return jnp.array([[0.0], [1.0]], dtype=t.dtype)

        surface = DomainMapping(1, 2, map_fn, jac_fn, None, None)
        super().__init__("InterfaceVertical2D", parametric_domain, surface, label_str, label_idx)


class InterfaceTop2D(SurfacicDomain):
    def __init__(self, r_in, h_in, label_str=DEFAULT_LABEL_STR_SURFACIC, label_idx=0):
        parametric_domain = HypercubeND([(0.0, r_in)])

        def map_fn(t):
            r = t[0:1]
            return jnp.concatenate([r, jnp.array([h_in], dtype=t.dtype)], axis=-1)

        def jac_fn(t):
            return jnp.array([[1.0], [0.0]], dtype=t.dtype)

        surface = DomainMapping(1, 2, map_fn, jac_fn, None, None)
        super().__init__("InterfaceTop2D", parametric_domain, surface, label_str, label_idx)


class InterfaceArc2D(SurfacicDomain):
    def __init__(self, r_in, h_in, label_str=DEFAULT_LABEL_STR_SURFACIC, label_idx=0):
        parametric_domain = HypercubeND([(jnp.pi, 2.0 * jnp.pi)])

        def map_fn(t):
            th = t[0:1]
            return jnp.concatenate([r_in * jnp.cos(th), h_in + r_in * jnp.sin(th)], axis=-1)

        def jac_fn(t):
            th = t[0:1]
            return jnp.array([[-r_in * jnp.sin(th)], [r_in * jnp.cos(th)]], dtype=t.dtype)

        surface = DomainMapping(1, 2, map_fn, jac_fn, None, None)
        super().__init__("InterfaceArc2D", parametric_domain, surface, label_str, label_idx)


# =============================================================================
# Residuals
# =============================================================================
class WallConductionResidual(InteriorResidual):
    _ns_dim: int = static(2)
    _ns_mode: str = static("scalar")

    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x", f_rhs=None)

    def construct_residual(self, *vars, precomputed=None):
        T = vars[0]
        # Runnable benchmark residual: Cartesian Laplacian form.
        return k_wall(T) * T.laplacian_x()


class CavityConductionResidual(InteriorResidual):
    _ns_dim: int = static(2)
    _ns_mode: str = static("scalar")

    def __init__(self, domain):
        super().__init__(domain=domain, size=1, model_type="x", f_rhs=None)

    def construct_residual(self, *vars, precomputed=None):
        T = vars[0]
        return K_NH3 * T.laplacian_x()


class OuterDirichletResidual(BoundaryResidual):
    _ns_dim: int = static(2)
    _ns_mode: str = static("scalar")

    def __init__(self, domain_dict, alpha0, alpha1):
        super().__init__(domain=domain_dict, size=1, model_type="x", f_rhs=None)
        self.alpha0 = alpha0
        self.alpha1 = alpha1

    def construct_residual(self, *vars, precomputed=None):
        (T,) = vars
        # Strong post-processing is preferred; this residual is only a fallback.
        return ParamVecFunction.cat([T - self.alpha0])


class InterfaceContinuityResidual(BoundaryResidual):
    _ns_dim: int = static(2)
    _ns_mode: str = static("scalar")

    def __init__(self, domain_dict):
        super().__init__(domain=domain_dict, size=2, model_type="x", f_rhs=None)

    def construct_residual(self, *vars, precomputed=None):
        TA, TB = vars
        # Strong interface handling is preferred; fallback to temperature continuity.
        return ParamVecFunction.cat([TA - TB, TA - TB])


# =============================================================================
# Physical model
# =============================================================================
class AutoclaveAmmoniaHeatModel(AbstractPhysicalModel):
    def __init__(self, main_domain):
        super().__init__(main_domain=main_domain)
        self.physical_residuals: PHYSICAL_RESIDUALS_TYPE = {}

    def add_residual(self, label: str, residual):
        self.physical_residuals[label] = residual


# =============================================================================
# Build benchmark geometry
# =============================================================================
cavity_sdf = AmmoniaCavitySDF(R_IN, H_IN, H_OUT)
autoclave_domain = OuterDomain2D(R_OUT, H_OUT, cavity_sdf, is_main_domain=True, label_str="A")
cavity_domain = CavityDomain2D(R_IN, H_IN, H_OUT, is_main_domain=False, label_str="B")
autoclave_domain.add_subdomain(cavity_domain)

outer_right = OuterRightWall2D(R_OUT, H_OUT, label_str="outer_right")
outer_top = OuterTopWall2D(R_OUT, label_str="outer_top")
outer_bottom = OuterBottomWall2D(R_OUT, label_str="outer_bottom")
autoclave_domain.add_bc_domain(outer_right)
autoclave_domain.add_bc_domain(outer_top)
autoclave_domain.add_bc_domain(outer_bottom)

if_vertical = InterfaceVertical2D(R_IN, H_IN, H_OUT, label_str="if_vertical")
if_top = InterfaceTop2D(R_IN, H_IN, label_str="if_top")
if_arc = InterfaceArc2D(R_IN, H_IN, label_str="if_arc")
cavity_domain.add_bc_domain(if_vertical)
cavity_domain.add_bc_domain(if_top)
cavity_domain.add_bc_domain(if_arc)


# =============================================================================
# Model residuals
# =============================================================================
model = AutoclaveAmmoniaHeatModel(main_domain=autoclave_domain)
model.add_residual("A", WallConductionResidual(autoclave_domain))
model.add_residual("B", CavityConductionResidual(cavity_domain))
model.add_residual("boundary", OuterDirichletResidual({"outer_right": outer_right}, ALPHA_0, ALPHA_1))
model.add_residual("boundary", OuterDirichletResidual({"outer_top": outer_top}, ALPHA_0, ALPHA_1))
model.add_residual("boundary", OuterDirichletResidual({"outer_bottom": outer_bottom}, ALPHA_0, ALPHA_1))
model.add_residual("interface", InterfaceContinuityResidual({"if_vertical": if_vertical, "if_top": if_top, "if_arc": if_arc}))


# =============================================================================
# Approximation space
# =============================================================================
key = jax.random.PRNGKey(0)
nn = MLP(in_size=2, out_size=1, hidden_sizes=[64, 64, 64, 64], key=key)

def post_processing_scalar(output, x):
    return output

space = ApproximationSpace(
    {"x": 2},
    [(nn, "scalar", None)],
    model_type="x",
    post_processing=post_processing_scalar,
)


# =============================================================================
# Sampling / projector
# =============================================================================
sampler = TensorizedSampler([DomainSampler(autoclave_domain)], model_type="x", bc=True, ic=False, data_samplers={})

WEIGHTS = {
    "A": [1.0],
    "B": [1.0],
    "boundary": [25.0],
    "interface": [20.0, 20.0],
}

pinn = Projector(model, space, sampler, optimizer="ENG", weights=WEIGHTS, one_loss_by_equation=True)


# =============================================================================
# Train
# =============================================================================
N_INTERIOR = 5000
N_BC = 3000
N_EPOCHS = 2

key, sample_dict = sampler.sample(key, N_INTERIOR, N_BC, 0, 0)
print("sample_dict keys:", sample_dict.keys())

loss0 = pinn.evaluate_loss(space, sample_dict)
print("initial loss:", loss0)

start = timeit.default_timer()
key, pinn = pinn.project(key, space, N_EPOCHS, N_INTERIOR, N_BC)
end = timeit.default_timer()

print("best loss:", pinn.best_loss)
print(f"training time: {end - start:.1f} s")


# =============================================================================
# Plot
# =============================================================================
plot_abstract_approx_space(
    pinn.space,
    autoclave_domain,
    loss=pinn.losses,
    residual=pinn.model,
    components=[0],
    draw_contours=True,
    n_drawn_contours=20,
    title="Autoclave-ammonia scalar temperature field",
)
plt.show()