# Run with, e.g.:
#   python train_idm.py idm.pretrained_run_dir=/path/to/world_model/checkpoints/2026-01-01/12-00-00

defaults:
  - _self_

idm:
  # Output directory of a completed world-model training run: must
  # contain hydra.yaml and checkpoints/model_latest.pth.
  pretrained_run_dir: ???
  pretrained_ckpt_name: model_latest.pth

  hidden_dim: 512
  num_layers: 3
  dropout: 0.0
  pooling: mean
  lr: 1e-4

  total_steps: 2000
  log_every_x_steps: 1
  val_every_x_steps: 10
  val_batches: 50      
  save_every_x_steps: 1000

  # Optional overrides -- default to whatever the pretrained run used:
  # num_hist: 3
  # num_pred: 1
  # frameskip: 5
  # img_size: 224

training:
  seed: 0
  batch_size: 64
  mixed_precision: "no"
  num_workers: 2

debug: false
