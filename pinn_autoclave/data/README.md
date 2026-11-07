# Reference data (not included)

The finite-volume reference data come from CrysMAS simulations at the Chair of Electron Devices (LEB), FAU.
They are not published in this repository.

To run the scripts, place the files here with these names:

| File / folder | Used by |
|---|---|
| `Inconel_Data good nbs.csv` | `2_single_material_1d/*/inconel_*.py` |
| `NH3_Data.csv` | `2_single_material_1d/*/nh3_*.py` |
| `Mixed_Materials_Data.csv` | `3_multi_material_1d/*.py` |
| `2D Inconel + Ammonia K(T) correct data/2D_Heat_data.csv` | `4_autoclave_2d_heat/inconel_nh3*.py` |
| `2D Inconel + NH3 + 1 GaN K(T) extended/2D_Heat_data.csv` | `4_autoclave_2d_heat/one_gan_*.py` |
| `2D Inconel + NH3 + 2 GaN K(T) extended/2D_Heat_data.csv` | `4_autoclave_2d_heat/two_gan_*.py` |
| `Transient CrysMAS data/Autoclave_stable.csv` (stable case, 0 to 5330.6 s) | `5_autoclave_2d_transient/transient_5330s_*.py` |
| `Autoclave_stable/Autoclave_stable.csv` (stable case, 0 to 7200 s) | `5_autoclave_2d_transient/transient_7200s_*.py`, `6_navier_stokes_wip/transient_7200s_*.py` |
| `Autoclave_unstable/Autoclave_unstable.csv` (unstable case, 0 to 7200 s) | `5_autoclave_2d_transient/transient_7200s_global_parametric_K_unknown.py`, `6_navier_stokes_wip/transient_7200s_global_parametric_navier_stokes.py` |
| `complete_data_inc_nh3.xlsx` | `6_navier_stokes_wip/ns2d_inconel_nh3_coupled.py` |

The scripts in `1_verification_1d_analytical/` compare against analytical solutions and need no data.
