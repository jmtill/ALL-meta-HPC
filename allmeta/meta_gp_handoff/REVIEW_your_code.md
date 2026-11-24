# Review of your `src/allmeta/task_gp.py` and `scaml.py` (2026-11-23)

Found by running your files (copied, unedited) against the checks. Your files were not changed.

## task_gp.py

### 1. `se_ard_kernel`: the outputscale prior and bounds are silently dropped

The closing parenthesis of `RBFKernel(...)` is after `outputscale_constraint=...`, so both outputscale arguments go to
`RBFKernel`, which accepts and ignores them. Evidence from running it:

```
ScaleKernel priors: ['base_kernel.lengthscale_prior']      # no outputscale_prior
ScaleKernel outputscale constraint: Positive()             # not Interval(1e-4, 1e2)
```

No error is raised; the outputscale is simply unregularized and unbounded. Fix: close `RBFKernel(` after
`lengthscale_constraint=...`, so the outputscale arguments belong to `ScaleKernel(`.

### 2. `fit_map`: only the first start runs

`model.load_state_dict(best_state)`, `model.eval()` and `return best` are indented inside the `for` loop, so the
function returns after the first start. Evidence: with `n_starts=5`, `fit_gpytorch_mll` ran **1 of 5** times.
Fix: dedent those three lines to the function's level.

### 3. Unused import

`SingleTaskVariationalGP` is imported and not used. Harmless; remove when convenient.

## scaml.py

### 4. `source_moments`: `p.distribtion`

Typo for `p.distribution`. Raises `AttributeError: 'MultivariateNormal' object has no attribute 'distribtion'`
on the first call.

### 5. `forward`: `self.cover_t`

Typo for `self.covar_t`. Would raise `AttributeError` once bug 4 is fixed.

## Not bugs

- `self. scale = ...` (space after the dot) is legal Python, but unusual; `self.scale` reads better.
- `return None` at the end of methods is redundant (Python returns `None` anyway) but harmless.

After fixing 1, 2, 4 and 5, `checks/check_reference.py` can be pointed at your files instead of `reference/models/`
by changing the `sys.path` line at its top.
