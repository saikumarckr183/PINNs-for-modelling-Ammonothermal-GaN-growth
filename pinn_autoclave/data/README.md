# Reference data (not included)

The finite-volume reference data come from CrysMAS simulations at the Chair of Electron Devices (LEB), FAU.
They are not published in this repository.

To run the scripts, place the files here with these names:

| File / folder | Used by |
|---|---|
| `Inconel_Data good nbs.csv` | `2_single_material_1d/*/inconel_*.py` |
| `NH3_data.csv` (plain JAX) / `NH3_Data.csv` (ScimBa) | `2_single_material_1d/*/nh3_*.py` |
| `Mixed_Materials_Data.csv` | `3_multi_material_1d/*.py` |
| `2D Inconel + Ammonia K(T) correct data/2D_Heat_data.csv` | `4_autoclave_2d_heat/inconel_nh3.py` |
| `2D Inconel + Ammonia + 1 GaN K(T) correct data/2D_Heat_data.csv` | `4_autoclave_2d_heat/inconel_nh3_one_gan_part.py` |
| `complete_data_two_gan.xlsx` | `4_autoclave_2d_heat/inconel_nh3_two_gan_parts_dd.py` |
| `complete_data_inc_nh3.xlsx` | `5_navier_stokes_wip/ns2d_inconel_nh3_coupled.py` |

The scripts in `1_verification_1d_analytical/` compare against analytical solutions and need no data.
