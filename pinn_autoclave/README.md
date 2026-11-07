# PINNs for heat transfer in an ammonothermal GaN autoclave

Physics-informed neural networks (PINNs) for steady heat conduction (and, as work in progress, buoyancy-driven flow)
in an autoclave for ammonothermal GaN growth. Written in Python with JAX and the
[ScimBa](https://www.scimba.org/) scientific machine-learning library.

## Physical model

- **Geometry:** axisymmetric (r, z) Inconel 718 autoclave, outer radius 35 mm, height 370 mm.
  The NH₃ cavity has a radius of 12 mm and reaches z = 335 mm. In the later cases it holds one or two GaN parts.
- **Equation:** steady heat conduction ∇·(k(T) ∇T) = 0 in every material region
- **Materials:**
  - Inconel 718: k(T) = 6.9995 + 0.0151·T W/(m·K)
  - supercritical NH₃: cubic correlation k(T)
  - GaN: k = 130 W/(m·K)
- **Boundary condition:** linear wall temperature profile T_outer(z) = T₀ + T₁·z on the outer autoclave wall
- **Interfaces:** continuity of temperature and heat flux between the materials
- **Reference:** finite-volume simulations with CrysMAS, compared along lines and as 2D error maps

## Method

- One neural network (MLP) per material region (domain decomposition), coupled through interface residuals.
  The two-GaN case uses seven networks.
- Boundary conditions are enforced strongly by post-processing the network output wherever possible
- Models are trained with ScimBa's JAX projectors and optimisers. The 1D single-material problems are also written
  directly in JAX/Optax as an independent baseline.
- Generalisation tests: each problem is trained for a **fixed case**, then as a parametric model for
  **interpolation** and **extrapolation** to boundary temperatures outside the training range

## Repository layout

| Folder | Content |
|---|---|
| [`1_verification_1d_analytical/`](1_verification_1d_analytical/) | 1D three-layer conduction with constant k against the analytical solution: with and without interface temperature, a boundary-temperature sweep, and a space-discrete PINN version |
| [`2_single_material_1d/`](2_single_material_1d/) | 1D conduction with k(T) for Inconel and NH₃: fixed case, interpolation, extrapolation. `scimba/` holds the ScimBa versions, `plain_jax/` the hand-written JAX/Optax PINNs, and `conductivity_correlations.py` the k(T) correlations |
| [`3_multi_material_1d/`](3_multi_material_1d/) | 1D multi-material wall: piecewise vs smoothed conductivity at the interfaces, each as fixed case, interpolation and extrapolation |
| [`4_autoclave_2d_heat/`](4_autoclave_2d_heat/) | 2D axisymmetric autoclave: `inconel_nh3.py` (Inconel + NH₃), `inconel_nh3_one_gan_part.py`, `inconel_nh3_two_gan_parts_dd.py` (domain-decomposed, 7 networks) |
| [`5_navier_stokes_wip/`](5_navier_stokes_wip/) | **Work in progress:** steady Boussinesq Navier–Stokes. 2D axisymmetric NH₃, 2D Inconel conduction coupled to NH₃ flow, 3D cylinder with hemispherical bottom (autoclave and water/air test cases) |
| [`archive/`](archive/) | Earlier versions and drafts, kept for reference |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt

# reference data must be placed in pinn_autoclave/data/ (see data/README.md)
python 1_verification_1d_analytical/three_layer_constant_k.py   # needs no data
python 4_autoclave_2d_heat/inconel_nh3.py
```

Outputs (figures, CSV metrics, loss histories) are written to an `output*/` folder in the current working directory.
A GPU is recommended for the 2D cases.

## Results

<!-- TODO (SK): add 2–3 figures to pinn_autoclave/figures/ and one line each, e.g.
![2D temperature field: PINN vs CrysMAS](figures/inconel_nh3_2d_comparison.png)
*Inconel + NH₃: max. relative error vs CrysMAS = … %*
-->
