# Fresh Agent Prompt: Hidden-Trajectory Calibration

Use this prompt in a new agent session when continuing the project.

```text
I have an existing repo at:
`/Users/yajasmalhotra/Documents/OMSCS/COGS/Term Project/world_model_project`

It currently implements a CPU-feasible 2D synthetic object-centric world model:
- generated visual scenes with 2-4 objects and static rectangular occluders
- object state `[x, y, vx, vy, visible, occluded, size, shape_id, color_id, object_id]`
- pixel-to-state encoder
- state dynamics model
- direct pixel baseline
- evaluation metrics for occlusion RMSE, reappearance RMSE, identity consistency, frame MSE, and counterfactual locality
- Streamlit demo with counterfactual rollouts

Important repo paths:
- `src/data/scene_generator.py`
- `src/models/state_dynamics.py`
- `src/models/encoder.py`
- `src/eval/metrics.py`
- `configs/default.yaml`
- `app.py`

There may be an existing unstaged `app.py` change adding a state-first / pixel-path selector. Do not overwrite it. Check `git status --short --branch` first.

New project direction:

**Hidden-Trajectory Calibration for 3D Object-Permanence World Models**

I want to evolve the project from deterministic 2D object-state prediction into a 3D hidden-trajectory calibration benchmark. The model should not predict only one hidden object location during occlusion. It should maintain a calibrated belief distribution over possible hidden 3D object locations, render that belief as Gaussian splats or translucent probability fields, and quantitatively compare the belief against the true hidden 3D trajectory.

Research thesis:
Object permanence should not be evaluated only by final reappearance accuracy. A world model should maintain a physically plausible, calibrated belief distribution over hidden object locations throughout occlusion, and that belief should correlate with the true hidden trajectory.

Novelty constraints / nearby work:
- Broad `3D belief world model under partial observability` is not novel. 3D-Belief is close: https://arxiv.org/abs/2605.11367
- Plain `object permanence + Gaussian splats + physics` is not novel. PersistGS is close: https://arxiv.org/abs/2606.03479
- Particle-filter object permanence exists: https://arxiv.org/abs/2403.08231
- Object-centric video world models exist, e.g. SlotFormer: https://arxiv.org/abs/2210.05861
- Violation-of-expectation/intuitive-physics benchmarks exist: https://arxiv.org/abs/2308.10441 and https://arxiv.org/abs/2506.09849
- Uncertainty-aware Gaussian splatting exists: https://arxiv.org/abs/2605.30342 and https://arxiv.org/abs/2510.12768

Therefore, the defensible novelty is not splats, object permanence, or generic 3D belief inference alone. The novelty is the measurement lens: frame-by-frame calibration between a model's hidden belief and the true hidden object trajectory.

Target 3D setup:
- synthetic 3D scenes in a bounded box
- 2-5 simple objects: spheres, cubes, capsules, maybe low-poly meshes later
- occluders: boxes/planes, eventually SDF geometry
- state: `[x, y, z, vx, vy, vz, visible, occluded, size, shape_id, color_id, object_id]`
- observations: one or more low-res camera views, e.g. 64x64 or 96x96
- belief: particles, Gaussian mixtures, or sparse splats over `(x, y, z)`
- renderer: simple projection renderer first; optional PyTorch3D, nvdiffrast, or Three.js later
- compute target: Kaggle/Colab friendly, generated from seeds, no giant cached datasets

Core metrics:
- Negative log likelihood of the true hidden 3D location under belief: `-log B_t(x_true, y_true, z_true)`
- Belief mass around truth within radius `r`
- Expected 3D distance under belief
- Calibration curve: whether credible regions contain the true hidden object at the expected rate
- Occlusion degradation curve: metrics vs occlusion duration
- Surprise score for impossible reappearances outside high-probability belief regions
- Baselines: deterministic state model, constant-velocity physics prior, particle physics prior, direct pixel baseline if feasible

Compute constraints:
Assume I can use Kaggle and Google Colab. Keep v1 free or cheap. Avoid full NeRF/3DGS reconstruction, diffusion, large video pretraining, or learned object discovery as the first milestone. The first version should be state-supervised and small enough for free Kaggle GPUs.

Desired deliverables:
1. Index the repo and propose a careful implementation plan.
2. Add or extend `configs/belief3d_smoke.yaml` and `configs/belief3d.yaml` when implementing.
3. Add 3D data generation without breaking the existing 2D pipeline.
4. Add belief metrics and plots.
5. Add a Streamlit or notebook demo showing observed view, rendered belief splats, true hidden trajectory toggle, and surprise heatmap.
6. Add Kaggle-friendly training/evaluation scripts or notebook instructions.
7. Update README and docs.

Start with documentation and skeletons before major code changes. Keep edits scoped.
```
