# Kaggle Workflow for 3D Belief Experiments

This project direction is designed to be trainable on free Kaggle or Colab GPU sessions. The first implementation step is a physics-only particle belief baseline, which can run on CPU for smoke tests before any learned model is added.

## Target Constraints

- Keep generated observations at 64x64 for most experiments.
- Generate scenes from seed manifests instead of storing large datasets.
- Keep individual training jobs short enough to survive notebook limits.
- Save checkpoints, metrics, and plots every epoch.
- Treat Kaggle as the main free experiment runner; use paid GPU only if quota becomes the bottleneck.

## Suggested Notebook Flow

1. Clone or attach the repo.
2. Install minimal dependencies from `requirements.txt` or `environment.yml`.
3. Build or load 3D seed manifests.
4. Run a smoke scene generation cell and visualize a few frames.
5. Evaluate the physics-only belief baseline.
6. Train the learned belief model once the baseline is stable.
7. Evaluate belief metrics on validation/test splits.
8. Export plots, GIFs, and checkpoint artifacts.

## Config Tiers

Use three tiers once the 3D path exists:

- `belief3d_smoke.yaml`: tiny run for shape checks and CPU fallback.
- `belief3d_small.yaml`: quick Kaggle GPU run for iteration.
- `belief3d.yaml`: main experiment with multiple seeds and evaluation splits.
- `belief3d_large.yaml`: local stress-test target with 128px frames, 40 steps, up to 10 objects, and 512 particles/object.

Current smoke commands:

```bash
python scripts/build_manifests3d.py --config configs/belief3d_smoke.yaml
python scripts/evaluate_belief3d.py --config configs/belief3d_smoke.yaml
python scripts/train_belief3d.py --config configs/belief3d_smoke.yaml
python scripts/evaluate_belief3d.py --config configs/belief3d_smoke.yaml --mode all
```

## Cost Strategy

A respectable first version should cost `$0` if Kaggle GPU quota is available. If free quotas become annoying, rent a modest L4/A5000/RTX GPU for final sweeps rather than redesigning around large models.

## Artifact Checklist

Each notebook run should save:

- `config.yaml`
- `metrics.csv`
- `summary.json`
- `checkpoints/best.pt`
- belief-vs-truth plots
- occlusion degradation plots
- calibration plots
- demo GIFs or MP4s

## Failure Modes to Avoid

- Do not pre-render massive datasets.
- Do not train NeRF/3DGS reconstruction models for v1.
- Do not make learned object discovery a dependency of the first milestone.
- Do not rely on one very long training session.
