# high_bg_order_sol

Python workflow for generating high-background-order stellar solutions and L=2 source terms.

## What this folder contains

- TOV background solver and series initial conditions.
- H00 and `u = h00/r^2` solvers.
- `nu`/`lambda` grid builder from TOV backgrounds.
- Full and analytic source-term evaluators (parsed from Mathematica expressions).
- Source concatenation utility (analytic near center, numerical outside).
- Notebooks and sweep/testing utilities.

## Main files

- `eps_h2npz.py`: writes `config/h_eps_config.pkl` (step size, start radius, model count, EOS parameter).
- `solver_tov.py`: solves TOV for a pressure grid and writes `PMSolGrid_*.pkl` and `RGrid_*.pkl`.
- `solver_h00.py`: solves `h00` and `u = h00/r^2`; writes `H00grid_*.pkl` and `U00grid_*.pkl`.
- `solver_mu_lambda.py`: computes matched `NuGrid_*.pkl` and `LambGrid_*.pkl`.
- `solver_source.py`: computes full source term and writes `SourceGrid_full_*.pkl`.
- `source_concat.py`: blends analytic `r^7` source with numerical full source; writes `SourceGrid_concat_*.pkl`.
- `tov_series_aux.py`, `h00_aux.py`: high-order center series expansions.
- `source_aux.py`: parser/evaluator for `source expression/*.txt` Mathematica expressions.

## Dependencies

Install at least:

```bash
pip install numpy matplotlib tqdm sympy
```

## Configuration

`config/h_eps_config.pkl` controls core run parameters:

- `h`: radial step size (string in current config format, e.g. `"1e-4"`)
- `eps`: start radius (e.g. `"1e-3"`)
- `d_num`: number of central-pressure models (e.g. `25`)
- `K_cgs`: polytrope constant

Regenerate config with:

```bash
python high_bg_order_sol/eps_h2npz.py
```

## Run order

From repository root:

```bash
python high_bg_order_sol/eps_h2npz.py
python high_bg_order_sol/solver_tov.py
python high_bg_order_sol/solver_h00.py
python high_bg_order_sol/solver_mu_lambda.py
python high_bg_order_sol/solver_source.py
python high_bg_order_sol/source_concat.py
```

Outputs are written to:

- `high_bg_order_sol/numpy_grid/h{h}_eps{eps}/`

Typical files in that folder:

- `PMSolGrid_h..._eps....pkl`
- `RGrid_h..._eps....pkl`
- `H00grid_h..._eps....pkl`
- `U00grid_h..._eps....pkl`
- `NuGrid_h..._eps....pkl`
- `LambGrid_h..._eps....pkl`
- `SourceGrid_full_h..._eps....pkl`
- `SourceGrid_concat_h..._eps....pkl`

## Testing / parameter sweeps

- Notebooks in `testing/`:
  - `test_mass_radius.ipynb`
  - `test_source_terms.ipynb`
  - `test_love_c_relation.ipynb`
  - `test_nu_surface.ipynb`
- Sweep script:

```bash
python high_bg_order_sol/testing/sweep_u_center.py
```

This writes timestamped outputs under:

- `high_bg_order_sol/testing/output/u_center_sweep_YYYYMMDD_HHMMSS/`

including `metadata.json`, `summary.csv`, `ranking.csv`, figures, and `.npz` files.

## Notes

- Current solvers use absolute paths rooted at this machine path:
  `/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/...`
- If you move the project, update those path literals (or refactor to use relative paths via `Path(__file__)`).
- Some scripts accept both `RGrid_*` and legacy `Rgrid_*` filenames.
