# Methodology

## 1. Purpose and Research Question

This project studies a concrete cognitive science question:

**Can a world model maintain persistent object representations through occlusion (object permanence), and use those representations for accurate future prediction and counterfactual simulation?**

The implementation uses simple synthetic 2D scenes so that the internal object state is known exactly. This makes it possible to evaluate not just visual prediction quality, but representational quality (identity continuity, behavior during occlusion, and controlled "what-if" interventions).


## 2. Cognitive Science Framing (Plain Language)

The experiment is built around three cognitive ideas:

1. **Persistent mental representations**  
   Objects should still be represented when they are temporarily not visible.

2. **Object files / object identity tracking**  
   The system should preserve "which object is which" after occlusion.

3. **Counterfactual reasoning**  
   If we intervene on one object's latent state (for example velocity), future outcomes should change in a targeted and interpretable way.

In this codebase, these ideas are operationalized with fixed object slots and explicit state variables per object.


## 3. Experimental Design Overview

The pipeline has two primary paths:

- **State-first path**: use ground-truth object state from observed frames -> world dynamics model -> future state rollout -> rendered future frames.
- **Pixel-input path**: observed pixels -> learned pixel-to-state encoder -> world dynamics model -> future state rollout -> rendered future frames.

A separate **direct pixel baseline** predicts future frames without structured object state, for comparison.


## 4. Synthetic Data Generation

Data generation code: [src/data/scene_generator.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/src/data/scene_generator.py)

### 4.1 Scene format

- Resolution: `64 x 64`, RGB
- Sequence length (default): `20` frames
- Observation window (default): first `8` frames
- Prediction window (default): remaining `12` frames
- Objects per scene: `2-4`
- Occluders per scene: `1-2` static rectangles
- Shapes: circle, square, triangle
- Colors: fixed 6-color palette
- Motion: straight-line motion with wall bounces
- No object-object collisions are modeled

### 4.2 Ground-truth state per object per timestep

State vector (`state_dim=10`):

`[x, y, vx, vy, visible, occluded, size, shape_id, color_id, object_id]`

All object states are represented in normalized coordinates. `visible` and `occluded` are recomputed each timestep from object position and occluder geometry.

### 4.3 Deterministic split construction

Manifest builder: [scripts/build_manifests.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/scripts/build_manifests.py)

Default split sizes:

- `train`: 600
- `val`: 120
- `test`: 120

Generalization splits:

- `test_long_occlusion`: longer sequence (`seq_len=28`, `obs_len=8`)
- `test_unseen_speed`: higher speed scale (`velocity_scale x 1.6`)
- `test_unseen_occluders`: edge-biased occluder placement

Each manifest row stores a seed and overrides, so the same scene can be regenerated exactly.


## 5. Model Components

### 5.1 Object-centric dynamics model

Code: [src/models/state_dynamics.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/src/models/state_dynamics.py)

Input:

- final observed state: `[B, O, F]`
- object mask: `[B, O]`
- occluders: `[B, K, 4]`

Architecture:

- state embedding MLP
- pairwise object interaction MLP (message passing)
- occluder-context MLP
- GRUCell recurrent update
- delta head predicting updates for dynamic variables `[x, y, vx, vy]`

Rollout:

1. Update hidden states with object interactions + context.
2. Predict state deltas.
3. Clamp position and velocity ranges.
4. Recompute visibility from predicted position and occluders.
5. Concatenate dynamic + visibility + static fields.

Output:

- predicted future state sequence `[B, H, O, F]`

### 5.2 Pixel-to-state encoder

Code: [src/models/encoder.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/src/models/encoder.py)

Input:

- observed frame sequence `[B, T_obs, C, H, W]`

Architecture:

- CNN per frame (3 conv layers + adaptive average pooling)
- temporal GRU across observed timesteps
- MLP head to `max_objects x state_dim`

Post-processing:

- `x, y` via sigmoid
- `vx, vy` via tanh-scaled range
- visibility via sigmoid, occluded as `1 - visibility`
- static features via sigmoid

Output:

- inferred initial object state `[B, O, F]`

### 5.3 Joint model

The joint model composes:

- encoder -> initial state
- dynamics -> rollout

This is trained end-to-end with both initial-state and rollout losses.

### 5.4 Direct pixel baseline

Code: [src/models/pixel_baseline.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/src/models/pixel_baseline.py)

Architecture:

- frame encoder CNN
- observation GRU
- rollout GRUCell in latent space
- transposed-conv decoder for future frames

This baseline does not enforce object identity structure.


## 6. Training Protocol

### 6.1 Configuration and defaults

Default config: [configs/default.yaml](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/configs/default.yaml)

Key defaults:

- optimizer: Adam
- learning rate: `1e-3`
- weight decay: `0`
- grad clip: `1.0`
- epochs:
  - dynamics: `25`
  - encoder: `20`
  - joint: `20`
  - pixel baseline: `15`

### 6.2 Device policy

Device selection (`auto`) prioritizes:

1. Apple MPS (if available)
2. CUDA (if available)
3. CPU fallback

Utility code: [src/train/utils.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/src/train/utils.py)

### 6.3 Losses

Loss code: [src/train/losses.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/src/train/losses.py)

World-model losses are weighted masked MSE:

- position loss
- velocity loss
- visibility loss
- static-feature loss

Total:

- `pos + 0.5*vel + 0.25*vis + 0.1*static`

Pixel baseline uses plain frame-level MSE.

### 6.4 Run artifact structure

Each run writes to:

`runs/<timestamp>_<experiment_name>/`

with:

- `config.yaml`
- `metrics.csv`
- `summary.json`
- `checkpoints/best.pt`
- `media/`


## 7. Evaluation Protocol

Evaluation scripts:

- [scripts/evaluate.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/scripts/evaluate.py)
- [src/eval/runner.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/src/eval/runner.py)
- [src/eval/metrics.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/src/eval/metrics.py)

### 7.1 World-model metrics

For state, pixel-encoder, and joint modes:

- `rollout_rmse`: position RMSE over rollout
- `occluded_rmse`: position RMSE on timesteps where object is occluded
- `reappearance_rmse`: position RMSE at first reappearance after occlusion
- `identity_consistency`: fraction of reappearances preserving slot identity (nearest-position proxy)
- `frame_mse`: MSE between rendered predicted frames and ground-truth frames
- `counterfactual_locality`: targeted-object motion change divided by average non-target change after intervention

### 7.2 Pixel baseline metric

- `pixel_rollout_mse`: frame MSE over predicted future frames

### 7.3 Counterfactual test

By default, evaluation applies a velocity intervention (`+0.04` on `vx`) to object slot 0 and compares rollout changes to the unmodified rollout.


## 8. Result Aggregation and Reporting

Aggregation script: [scripts/aggregate_results.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/scripts/aggregate_results.py)

Outputs:

- [results/run_summaries.csv](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/results/run_summaries.csv)
- [results/eval_metrics.csv](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/results/eval_metrics.csv)
- [results/eval_rollout_rmse_table.csv](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/results/eval_rollout_rmse_table.csv)
- [results/eval_rollout_rmse.png](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/results/eval_rollout_rmse.png)


## 9. Demo Methodology

Demo app: [app.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/app.py)

Behavior:

1. Load checkpoints (joint preferred, or separate encoder/dynamics).
2. Generate deterministic scene from a user-selected seed.
3. Infer initial state from observed frames.
4. Roll out predicted future.
5. Apply user-configured counterfactual (`dvx`, `dvy`) to selected object slot.
6. Render and display:
   - observed frames
   - predicted future
   - counterfactual future
   - trajectory comparison plot

Export script for paper/demo assets:

- [scripts/export_demo_assets.py](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/scripts/export_demo_assets.py)


## 10. Non-Smoke Run Used for Current Results

A full pass was executed with `configs/default.yaml` and explicitly selected checkpoints:

- dynamics: `runs/20260405_180736_train_dynamics/checkpoints/best.pt`
- encoder: `runs/20260405_181546_train_encoder/checkpoints/best.pt`
- joint: `runs/20260405_181650_train_joint/checkpoints/best.pt`
- pixel baseline: `runs/20260405_182431_train_pixel_baseline/checkpoints/best.pt`
- evaluation run: `runs/20260405_182630_evaluate`

High-level outcomes:

- State model is best on structured object-tracking metrics.
- Pixel and joint paths are functional but currently weaker on identity consistency.
- Counterfactual sensitivity is measurable in all structured variants.

Detailed values are in [results/eval_metrics.csv](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/results/eval_metrics.csv).


## 11. Limitations and Threats to Validity

1. **Synthetic world simplification**  
   No object-object collisions, simple motion, and limited visual complexity.

2. **Identity metric approximation**  
   Identity consistency uses nearest-position matching at reappearance, which is a pragmatic proxy rather than full assignment-based identity analysis.

3. **Encoder bottleneck**  
   Pixel-to-state quality limits joint performance in current configuration.

4. **No uncertainty model**  
   Dynamics are deterministic; no probabilistic prediction over future trajectories.

5. **Rendering-based frame evaluation**  
   World-model frame MSE is measured on deterministic rendering from state, not a learned photorealistic decoder.


## 12. Reproducibility Instructions

From project root:

```bash
conda env create -f environment.yml
conda activate term-project-wm

python scripts/build_manifests.py --config configs/default.yaml
python scripts/train_dynamics.py --config configs/default.yaml
python scripts/train_encoder.py --config configs/default.yaml
python scripts/train_joint.py --config configs/default.yaml
python scripts/train_pixel_baseline.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --mode all
python scripts/aggregate_results.py --runs-dir runs --output-dir results
streamlit run app.py
```

Fast verification:

```bash
python scripts/build_manifests.py --config configs/smoke.yaml
python scripts/train_dynamics.py --config configs/smoke.yaml
python scripts/train_encoder.py --config configs/smoke.yaml
python scripts/train_joint.py --config configs/smoke.yaml
python scripts/train_pixel_baseline.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml --mode all
python scripts/aggregate_results.py --runs-dir runs --output-dir results
```


## 13. Storage and Compute Notes

- The project generates scenes on the fly from seeds, which keeps disk use low.
- Most disk use comes from:
  - conda environment
  - checkpoints
  - optional media exports
- Keep only `best.pt` checkpoints per run for space efficiency.

For measured footprint and cleanup policy, see:

- [docs/STORAGE_ESTIMATE.md](/Users/yajasmalhotra/Documents/OMSCS/COGS/Term%20Project/world_model_project/docs/STORAGE_ESTIMATE.md)

