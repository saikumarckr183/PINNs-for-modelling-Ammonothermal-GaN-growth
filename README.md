# Scientific Machine Learning for Ammonothermal GaN Crystal Growth

**Saikumar Kasumurthy Rajendra** · M.Sc. Clean Energy Processes, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

This repository collects my own work from my Master's thesis and research internship at the
Chair of Electron Devices (LEB), FAU. Both projects deal with **ammonothermal growth of gallium nitride (GaN)**,
where GaN crystals grow from supercritical ammonia (NH₃) inside a sealed high-pressure autoclave.

| Project | What it is | Tools |
|---|---|---|
| [**1. PINNs for heat transfer in the ammonothermal autoclave**](pinn_autoclave/) | Physics-informed neural networks for the temperature field in an Inconel autoclave filled with supercritical NH₃ and GaN, checked against finite-volume (CrysMAS) reference simulations | Python, JAX, [ScimBa](https://www.scimba.org/), Optax |
| [**2. GaN solubility in supercritical ammonia**](gan_solubility/) | Thermodynamic analysis of supercritical NH₃ and machine-learning modelling of GaN solubility data from the literature | Python, pandas, scikit-learn |

---

## 1. PINNs for heat transfer in the ammonothermal autoclave

*Master's thesis: "Physics-Informed Neural Networks for Modelling Ammonothermal Gallium Nitride Growth"*

The temperature field inside the autoclave controls where GaN dissolves and where it crystallises.
I built PINN models of the axisymmetric autoclave (Inconel 718 wall, NH₃ cavity, GaN parts) using
the JAX version of the ScimBa library. The models were developed step by step:

1. **Verification:** a 1D three-layer conduction problem with an analytical solution
2. **Single material, temperature-dependent conductivity k(T):** Inconel and NH₃, written in ScimBa *and* as a plain JAX/Optax implementation
3. **Multi-material 1D:** piecewise and smoothed conductivity at material interfaces
4. **2D axisymmetric autoclave:** Inconel + NH₃, then with one and two GaN parts, using one network per material region with interface coupling
5. **Navier–Stokes (in progress):** steady Boussinesq flow of NH₃ in 2D axisymmetric and 3D geometries

Each stage covers a fixed case, **interpolation** and **extrapolation** to unseen boundary temperatures.

<!-- TODO (SK): add one key figure here, e.g. PINN vs CrysMAS temperature field:
![PINN vs FV reference](pinn_autoclave/figures/inconel_nh3_comparison.png) -->

→ Details, how to run the scripts and the file overview: [`pinn_autoclave/README.md`](pinn_autoclave/)

## 2. GaN solubility in supercritical ammonia

My contribution to a manuscript (in preparation) consolidating the literature data on GaN solubility in
supercritical ammonia: the **thermodynamic analysis** of supercritical NH₃ (reduced properties, phase boundary,
fill-level P–T analysis) and the **machine-learning modelling** of the solubility data.

→ [`gan_solubility/`](gan_solubility/)

---

## Acknowledgements

- Supervisors: **Dr. Saskia Schimmel** (LEB, FAU), **Victor Michel-Dansac** (MoCo team, IRMA, Université de Strasbourg) and **Markus Zenk**
- The PINN models are built on **[ScimBa](https://gitlab.com/scimba/scimba)** (MIT licence), developed by Emmanuel Franck, Victor Michel-Dansac and colleagues. ScimBa is **not** included in this repository; it is installed as a dependency.
- My research visit to Strasbourg was funded by the *Technologies d'Avenir* mobility grant.

## Licence

My code in this repository is released under the [MIT licence](LICENSE).
Reference data from the chair's simulations are not included (see [`pinn_autoclave/data/`](pinn_autoclave/data/)).
