from __future__ import annotations

import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", False)

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
from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.numerical_solvers.projectors import Projector
from scimba_jax.physical_models.abstract_residuals import BoundaryResidual, PhysicalResidual, static
from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel, PHYSICAL_RESIDUALS_TYPE
from scimba_jax.plots.plots_nd import plot_abstract_approx_space

jax.config.update("jax_enable_x64", True)

# ============================================================
# Parameters
# ============================================================
R_OUT = 35.0
H_OUT = 370.0
R_IN = 12.0
H_IN = 300.0

T_BOTTOM = 450.0
T_TOP = 550.0
ALPHA_0 = T_BOTTOM
ALPHA_1 = (T_TOP - T_BOTTOM) / H_OUT

K_A = 3.843196
K_B = 0.162242

N_INTERIOR = 5000
N_BC = 3000
N_EPOCHS = 2

WEIGHTS = {
    "A": [1.0],
    "B": [1.0],
    "boundary": [25.0],
}

# ============================================================
# SDFs
# ============================================================
class RectSDF(SignedDistance):
    def __init__(self, xmin, xmax, zmin, zmax):
        super().__init__(dim=2, threshold=0.0)
        self.xmin = xmin
        self.xmax = xmax
        self.zmin = zmin
        self.zmax = zmax

    def _sdf_pointwise(self, x):
        p = x[..., :2]
        c = jnp.array([(self.xmin + self.xmax) * 0.5, (self.zmin + self.zmax) * 0.5], dtype=p.dtype)
        b = jnp.array([(self.xmax - self.xmin) * 0.5, (self.zmax - self.zmin) * 0.5], dtype=p.dtype)
        q = jnp.abs(p - c) - b
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
        inside = jnp.minimum(jnp.maximum(q[..., 0], q[..., 1]), 0.0)
        return (outside + inside)[None]

class AmmoniaSDF(SignedDistance):
    def __init__(self, r_in, h_in):
        super().__init__(dim=2, threshold=0.0)
        self.r_in = r_in
        self.h_in = h_in

    def _sdf_pointwise(self, x):
        p = x[..., :2]
        c = jnp.array([self.r_in / 2.0, self.h_in / 2.0], dtype=p.dtype)
        b = jnp.array([self.r_in / 2.0, self.h_in / 2.0], dtype=p.dtype)
        q = jnp.abs(p - c) - b
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
        inside = jnp.minimum(jnp.maximum(q[..., 0], q[..., 1]), 0.0)
        sdf_rect = outside + inside

        cap_disk = jnp.linalg.norm(p, axis=-1) - self.r_in
        cap_half = p[..., 1]
        sdf_cap = jnp.maximum(cap_disk, cap_half)

        return jnp.minimum(sdf_rect, sdf_cap)[None]

class DifferenceSDF(SignedDistance):
    def __init__(self, sdf_a, sdf_b):
        super().__init__(dim=2, threshold=0.0)
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b

    def _sdf_pointwise(self, x):
        a = self.sdf_a._sdf_pointwise(x)[0]
        b = self.sdf_b._sdf_pointwise(x)[0]
        return jnp.maximum(a, -b)[None]

# ============================================================
# Domains
# ============================================================
class AutoclaveDomain2D(VolumetricDomain):
    def __init__(self, r_out, h_out, cavity_sdf, is_main_domain=False,
                 label_str=DEFAULT_LABEL_STR_VOLUMETRIC, label_idx=0):
        bounds = jnp.array([(0.0, r_out), (0.0, h_out)], dtype=jnp.float64)
        outer = RectSDF(0.0, r_out, 0.0, h_out)
        sdf = DifferenceSDF(outer, cavity_sdf)
        super().__init__(
            domain_type="AutoclaveDomain2D",
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
        return [
            OuterRightWall2D(self.r_out, self.h_out, label_str="outer_right"),
            OuterTopWall2D(self.r_out, label_str="outer_top"),
            OuterBottomWall2D(self.r_out, label_str="outer_bottom"),
        ]

class AmmoniaDomain2D(VolumetricDomain):
    def __init__(self, r_in, h_in, is_main_domain=False,
                 label_str=DEFAULT_LABEL_STR_VOLUMETRIC, label_idx=0):
        bounds = jnp.array([(0.0, r_in), (0.0, h_in)], dtype=jnp.float64)
        sdf = AmmoniaSDF(r_in, h_in)
        super().__init__(
            domain_type="AmmoniaDomain2D",
            dim=2,
            sdf=sdf,
            bounds=bounds,
            is_main_domain=is_main_domain,
            label_str=label_str,
            label_idx=label_idx,
        )
        self.r_in = r_in
        self.h_in = h_in

    def full_bc_domain(self):
        return [
            InterfaceVertical2D(self.r_in, 0.0, self.h_in, label_str="if_vertical"),
            InterfaceTop2D(self.r_in, label_str="if_top"),
            InterfaceArc2D(self.r_in, label_str="if_arc"),
        ]

# ============================================================
# Surfaces
# ============================================================
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
    def __init__(self, r_in, label_str=DEFAULT_LABEL_STR_SURFACIC, label_idx=0):
        parametric_domain = HypercubeND([(0.0, r_in)])

        def map_fn(t):
            r = t[0:1]
            return jnp.concatenate([r, jnp.array([H_IN], dtype=t.dtype)], axis=-1)

        def jac_fn(t):
            return jnp.array([[1.0], [0.0]], dtype=t.dtype)

        surface = DomainMapping(1, 2, map_fn, jac_fn, None, None)
        super().__init__("InterfaceTop2D", parametric_domain, surface, label_str, label_idx)

class InterfaceArc2D(SurfacicDomain):
    def __init__(self, r_in, label_str=DEFAULT_LABEL_STR_SURFACIC, label_idx=0):
        parametric_domain = HypercubeND([(jnp.pi, 2.0 * jnp.pi)])

        def map_fn(t):
            th = t[0:1]
            return jnp.concatenate([r_in * jnp.cos(th), r_in * jnp.sin(th)], axis=-1)

        def jac_fn(t):
            th = t[0:1]
            return jnp.concatenate([-r_in * jnp.sin(th), r_in * jnp.cos(th)], axis=-1).reshape(2, 1)

        surface = DomainMapping(1, 2, map_fn, jac_fn, None, None)
        super().__init__("InterfaceArc2D", parametric_domain, surface, label_str, label_idx)

# ============================================================
# Residuals
# ============================================================
class HeatInteriorResidual(PhysicalResidual):
    def __init__(self, domain, k_value):
        super().__init__(domain=domain, size=1, model_type="x", f_rhs=None)
        self.k_value = k_value

    def construct_residual(self, *rho, precomputed=None):
        T = rho[0]
        return self.k_value * T.laplacian_x()

class OuterDirichletResidual(BoundaryResidual):
    _T_wall: float = static(0.0)

    def __init__(self, domain, T_wall: float):
        super().__init__(domain=domain, size=1, model_type="x", f_rhs=None)
        self._T_wall = T_wall

    def construct_residual(self, *vars):
        (T,) = vars
        return ParamVecFunction.cat([T - self._T_wall])

class InterfaceDirichletResidual(BoundaryResidual):
    _T_target: float = static(0.0)

    def __init__(self, domain, T_target: float):
        super().__init__(domain=domain, size=1, model_type="x", f_rhs=None)
        self._T_target = T_target

    def construct_residual(self, *vars):
        (T,) = vars
        return ParamVecFunction.cat([T - self._T_target])

# ============================================================
# Model
# ============================================================
class AutoclaveAmmoniaHeatModel(AbstractPhysicalModel):
    def __init__(self, main_domain):
        super().__init__(main_domain=main_domain)
        self.physical_residuals: PHYSICAL_RESIDUALS_TYPE = {}

    def add_residual(self, label: str, residual: PhysicalResidual):
        self.physical_residuals[label] = residual

# ============================================================
# Build geometry
# ============================================================
ammonia_sdf = AmmoniaSDF(R_IN, H_IN)
autoclave_domain = AutoclaveDomain2D(R_OUT, H_OUT, cavity_sdf=ammonia_sdf, is_main_domain=True, label_str="A")
ammonia_domain = AmmoniaDomain2D(R_IN, H_IN, is_main_domain=False, label_str="B")

autoclave_domain.add_subdomain(ammonia_domain)

outer_right = OuterRightWall2D(R_OUT, H_OUT, label_str="outer_right")
outer_top = OuterTopWall2D(R_OUT, label_str="outer_top")
outer_bottom = OuterBottomWall2D(R_OUT, label_str="outer_bottom")

if_vertical = InterfaceVertical2D(R_IN, 0.0, H_IN, label_str="if_vertical")
if_top = InterfaceTop2D(R_IN, label_str="if_top")
if_arc = InterfaceArc2D(R_IN, label_str="if_arc")

autoclave_domain.add_bc_domain(outer_right)
autoclave_domain.add_bc_domain(outer_top)
autoclave_domain.add_bc_domain(outer_bottom)

ammonia_domain.add_bc_domain(if_vertical)
ammonia_domain.add_bc_domain(if_top)
ammonia_domain.add_bc_domain(if_arc)

boundary_main = autoclave_domain.get_all_boundaries()
boundary_sub = ammonia_domain.get_all_boundaries()

# ============================================================
# Model / residuals
# ============================================================
model = AutoclaveAmmoniaHeatModel(main_domain=autoclave_domain)
model.add_residual("A", HeatInteriorResidual(autoclave_domain, K_A))
model.add_residual("B", HeatInteriorResidual(ammonia_domain, K_B))

model.add_residual("boundary", OuterDirichletResidual({"outer_right": boundary_main["boundary"]["outer_right"]}, T_BOTTOM))
model.add_residual("boundary", OuterDirichletResidual({"outer_top": boundary_main["boundary"]["outer_top"]}, T_TOP))
model.add_residual("boundary", OuterDirichletResidual({"outer_bottom": boundary_main["boundary"]["outer_bottom"]}, T_BOTTOM))

model.add_residual("boundary", InterfaceDirichletResidual({"if_vertical": boundary_sub["boundary"]["if_vertical"]}, T_BOTTOM))
model.add_residual("boundary", InterfaceDirichletResidual({"if_top": boundary_sub["boundary"]["if_top"]}, T_BOTTOM))
model.add_residual("boundary", InterfaceDirichletResidual({"if_arc": boundary_sub["boundary"]["if_arc"]}, T_BOTTOM))

# ============================================================
# Approximation space
# ============================================================
key = jax.random.PRNGKey(0)
nn = MLP(in_size=2, out_size=1, hidden_sizes=[64, 64, 64, 64], key=key)

space = ApproximationSpace(
    {"x": 2},
    [(nn, "scalar", None)],
    model_type="x",
)

# ============================================================
# Sampler
# ============================================================
sampler = TensorizedSampler(
    [DomainSampler(autoclave_domain)],
    model_type="x",
    bc=True,
    ic=False,
    data_samplers={},
)

# ============================================================
# Projector
# ============================================================
pinn = Projector(
    model,
    space,
    sampler,
    optimizer="ENG",
    weights=WEIGHTS,
    one_loss_by_equation=True,
)

# ============================================================
# Train
# ============================================================
key, sample_dict = sampler.sample(key, n=N_INTERIOR, n_bc=N_BC, n_ic=0, n_dl=0)
print("sample_dict keys:", sample_dict.keys())

loss0 = pinn.evaluate_loss(space, sample_dict)
print("initial loss:", loss0)

start = timeit.default_timer()
key, pinn = pinn.project(key, space, N_EPOCHS, N_INTERIOR, N_BC)
end = timeit.default_timer()

print("best loss:", pinn.best_loss)
print(f"training time: {end - start:.1f} s")



# Main field plot
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

