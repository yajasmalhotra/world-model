# Storage Estimate

## Current measured footprint (this machine)
- `term-project-wm` conda env: `~1.4 GB`
- Project directory (`world_model_project`): `~8.5 MB`
- Smoke run artifacts (`runs/`): `~7.9 MB`
- Aggregated outputs (`results/`): `~68 KB`

## Expected footprint during normal development
- Environment + dependencies: `1.4 - 2.5 GB`
- Source code + configs + docs: `<100 MB`
- Training/eval runs (checkpoints + logs + media): `1 - 4 GB` depending on retained runs
- Demo assets and report figures: `0.1 - 1 GB`

## Practical total budget
- Typical working footprint: `3 - 7 GB`
- Comfortable free-space headroom target: `>= 12 GB`

## Cleanup policy
- Keep only `checkpoints/best.pt` per run.
- Delete run `media/` folders except for runs used in the paper/demo.
- Keep only the latest evaluation and aggregated tables/plots.
