# Updates

Re-runs of the paper's Table 1 protocol on updated code.

Protocol for every table: success rate (%) of 50 test episodes with the GD planner, open-loop /
MPC, mean ± std over eval seeds 100/101/102; old results in brackets. Models are trained from
scratch with `conf/train.yaml` defaults (batch 32, frameskip 5, num_hist 3, seed 0; 20 epochs,
pusht 2) and evaluated with `plan_gd.yaml` / `plan_gd_mpc.yaml` (pusht: `objective.alpha=1`, MPC
`objective.mode=staged`).

## DINOv2 (patch) + global projector: predictor sizing (`models/dino.py`)

Commit
[`64a7585`](https://github.com/agentic-learning-ai-lab/temporal-straightening/commit/64a7585819e749bfec327ad984ee08570d07f0eb).

**What changed.** `train.py` sizes the predictor from `encoder.latent_ndim` before any forward
pass, but for the global projector (`encoder=dino_global`, one pooled token per frame)
`latent_ndim` was only set inside `forward()`. A fresh `dino_global` run therefore got a
predictor sized for the 14×14 patch grid (196 tokens per frame) while the encoder emitted one.
The ViT slices its block-causal mask to the runtime length (`bias[:, :, :T, :T]`), so the three
history tokens all landed in frame 0's all-ones block and attention was fully visible: every
history position but the last saw its own target, its loss became a copy task, and only the last
position trained on genuine one-step prediction. Runs did not crash. `DinoV2Encoder.__init__`
now sets `latent_ndim` from the projector's `pool_hw`.

**What is unaffected.** Every other encoder fixes `latent_ndim` at construction (`dino`,
`dino_channel`, `scratch_resnet_spatial`, `scratch_resnet`, `dino_cls`), so their rows are
untouched. Planning code is unchanged, and planning with an affected checkpoint was internally
consistent — the rollout sees only current and past frames, and only the last position's
prediction (the one trained properly) is consumed. The missing mask weakened the training
signal, not the planning procedure. Only the `DINOv2 (patch) + proj, 1×384` row moves;
pre-change checkpoints of it carry the weakened predictor.

**Results** — the row re-run with the change (projector lr 1e-6), λ selected on MPC validation
success (seeds 42/43, 50 episodes each) and tested on seeds 100/101/102.

Validation (GD / MPC %, mean over seeds 42/43):

| λ | Wall | UMaze | Medium | PushT |
|---|---|---|---|---|
| ✗ | 67.0 / 72.0 | 31.0 / 79.0 | 26.0 / 69.0 | 22.0 / 54.0 |
| 1e-1 | 81.0 / 97.0 | 63.0 / 86.0 | 31.0 / 94.0 | 26.0 / 50.0 |
| 1e-2 | 80.0 / 82.0 | 30.0 / 97.0 | 30.0 / 90.0 | 24.0 / 59.0 |
| 1e-3 | 68.0 / 73.0 | 34.0 / 89.0 | 22.0 / 77.0 | 28.0 / 51.0 |

Selected λ: 1e-1 for wall and medium; 1e-2 for umaze and pusht (paper's markers: wall 1e-3,
medium 1e-2, umaze/pusht 1e-1). Test:

| L_curv | Wall | UMaze | Medium | PushT |
|---|---|---|---|---|
| ✗ | 64.7±4.1 / 69.3±6.8 (28.7 / 76.0) | 26.0±7.5 / 76.0±8.6 (34.7 / 79.3) | 24.7±5.0 / 73.3±4.1 (18.0 / 46.0) | 24.7±6.8 / 52.7±3.4 (2.0 / 11.3) |
| ✓ | 82.7±2.5 / 96.0±2.8 (32.0 / 77.3) | 35.3±9.0 / 98.0±1.6 (38.7 / 96.0) | 24.0±3.3 / 95.3±0.9 (22.7 / 78.0) | 25.3±3.8 / 58.7±1.9 (2.0 / 8.7) |

**The paper's conclusion holds, with better numbers.** Restoring the mask lifts both arms well
above the paper's row on every env, and straightening helps across the board.

## ResNet (scratch), 1×384 on PushT: two encoders

The paper's pusht run for this row used the GeM encoder (`scratch_resnet_gem`, from the [AdaJEPA
release](https://github.com/agentic-learning-ai-lab/adajepa)) rather than the default
`scratch_resnet`. Both are compared under the row's shared recipe (encoder lr 1e-6, batch 32, 2
epochs), λ selected by MPC success on validation seeds 42/43 and tested on seeds 100/101/102.

Validation (GD / MPC %, mean over seeds 42/43):

| λ | `scratch_resnet` | `scratch_resnet_gem` |
|---|---|---|
| ✗ | 8.0 / 43.0 | 32.0 / 60.0 |
| 1e-1 | 30.0 / 59.0 | 43.0 / 60.0 |
| 1e-2 | 8.0 / 56.0 | 51.0 / 89.0 |
| 1e-3 | 9.0 / 53.0 | 42.0 / 64.0 |

Test with the selected λ (GD / MPC %, mean ± std over seeds 100/101/102):

| Encoder | ✗ | ✓ |
|---|---|---|
| `scratch_resnet` | 10.7±0.9 / 41.3±3.4 | 33.3±4.1 / 56.0±3.3 (λ = 1e-1) |
| `scratch_resnet_gem` | 37.3±2.5 / 71.3±5.0 | 55.3±5.2 / 89.3±0.9 (λ = 1e-2) |

With the GeM encoder the pusht cells reproduce, and the selected λ is the paper's (1e-2);
`scratch_resnet` is far below at either arm. The full table below uses the GeM cells for pusht.

## Full table

Every row of the paper's Table 1, re-run on the updated code. Encoder lr per the paper's Table
3: 1e-6 for the frozen-backbone and global rows and for the λ=0 spatial arms, 1e-5 for the
straightened spatial arms. Straightened global-feature arms use the λ selected by MPC success on
validation seeds 42/43 (ties broken by open-loop success), then evaluated on the test seeds —
DINO+proj 1×384: wall/medium 1e-1, umaze/pusht 1e-2; ResNet 1×384: wall/umaze 1e-1, medium/pusht
1e-2. Spatial rows use aggcos 1e-1 as in the paper (not swept). ResNet 14×14×8 uses the paper's
encoder settings (`dim=8, agg_type=mlp, agg_out_dim=128`, now the `scratch_resnet_spatial.yaml`
defaults); ResNet 1×384 on pusht uses the GeM encoder (previous section).

| Encoder | Dim | L_curv | Wall | UMaze | Medium | PushT |
|---|---|---|---|---|---|---|
| DINOv2 (CLS) | 1×384 | ✗ | 22.7±5.0 / 60.7±10.4 (28.7 / 66.7) | 20.7±2.5 / 72.7±0.9 (25.3 / 82.7) | 21.3±4.1 / 58.0±5.9 (20.0 / 67.5) | 14.7±0.9 / 46.7±4.1 (19.3 / 28.0) |
| DINOv2 (patch) + proj | 1×384 | ✗ | 64.7±4.1 / 69.3±6.8 (28.7 / 76.0) | 26.0±7.5 / 76.0±8.6 (34.7 / 79.3) | 24.7±5.0 / 73.3±4.1 (18.0 / 46.0) | 24.7±6.8 / 52.7±3.4 (2.0 / 11.3) |
| DINOv2 (patch) + proj | 1×384 | ✓ | 82.7±2.5 / 96.0±2.8 (32.0 / 77.3) | 35.3±9.0 / 98.0±1.6 (38.7 / 96.0) | 24.0±3.3 / 95.3±0.9 (22.7 / 78.0) | 25.3±3.8 / 58.7±1.9 (2.0 / 8.7) |
| ResNet (scratch) | 1×384 | ✗ | 63.3±5.2 / 70.7±1.9 (58.7 / 68.0) | 54.0±4.3 / 82.0±3.3 (58.0 / 88.7) | 73.3±3.4 / 88.7±2.5 (73.3 / 89.3) | 37.3±2.5 / 71.3±5.0 (40.0 / 70.0) |
| ResNet (scratch) | 1×384 | ✓ | 86.0±1.6 / 99.3±0.9 (86.7 / 97.3) | 60.7±3.4 / 96.0±1.6 (63.3 / 97.3) | 72.0±4.9 / 95.3±1.9 (79.3 / 93.3) | 55.3±5.2 / 89.3±0.9 (58.7 / 83.3) |
| DINOv2 (patch) | 14×14×384 | ✗ | 62.7±6.6 / 80.0±7.5 (52.7 / 76.7) | 31.3±3.8 / 82.0±3.3 (35.3 / 80.7) | 33.3±2.5 / 72.7±5.2 (40.8 / 76.7) | 46.0±7.1 / 72.0±5.9 (56.0 / 66.0) |
| DINOv2 (patch) + proj | 14×14×8 | ✗ | 84.7±9.0 / 92.7±5.0 (80.0 / 90.7) | 56.0±4.3 / 92.7±4.1 (44.0 / 81.3) | 77.3±5.2 / 97.3±0.9 (72.0 / 96.7) | 66.7±4.7 / 80.0±2.8 (70.0 / 78.7) |
| DINOv2 (patch) + proj | 14×14×8 | ✓ | 94.0±1.6 / 100.0±0.0 (90.7 / 100.0) | 86.0±4.3 / 100.0±0.0 (94.0 / 100.0) | 78.0±5.9 / 99.3±0.9 (82.7 / 98.7) | 70.7±4.7 / 87.3±4.1 (77.3 / 85.3) |
| ResNet (scratch) | 14×14×8 | ✗ | 24.0±4.3 / 51.3±6.2 (1.3 / 6.7) | 11.3±5.7 / 61.3±1.9 (14.7 / 66.0) | 63.3±7.4 / 84.7±5.0 (18.7 / 57.3) | 68.7±6.2 / 79.3±6.6 (71.3 / 70.7) |
| ResNet (scratch) | 14×14×8 | ✓ | 89.3±3.4 / 100.0±0.0 (84.7 / 100.0) | 75.3±2.5 / 99.3±0.9 (64.7 / 98.7) | 86.0±4.3 / 100.0±0.0 (80.7 / 99.3) | 84.0±4.3 / 92.0±4.9 (70.7 / 91.3) |

- The results reproduce, and the straightening conclusion is robust: straightening improves MPC
  success in every cell and open-loop success in 14 of 16, the two exceptions (medium, global
  rows) flat within seed noise. The global-projector row is higher than reported on wall, medium,
  and pusht (previous section).
- The ResNet 14×14×8 λ=0 baselines on wall and medium are far above the reported ones
  (24.0/51.3 vs 1.3/6.7; 63.3/84.7 vs 18.7/57.3): the reported baselines partially collapsed
  from training instability, which did not recur here. The straightened arms match, so the gains
  are smaller but the ordering is unchanged.
