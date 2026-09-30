"""
Usage:
    python train_idm.py idm.pretrained_run_dir=/path/to/world_model/run
"""

import itertools
import logging
import os
import warnings
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict
from torchvision import transforms
from tqdm import tqdm

from idm import InverseDynamicsModel
from utils import cfg_to_dict, seed
import custom_resolvers  # noqa: F401  # Registers OmegaConf resolvers at import time.

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class IDMTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
        log.info(f"IDM saved dir: {cfg.saved_folder}")

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info("Multirun setup begin...")
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                torch.distributed.init_process_group(
                    backend="nccl", init_method="env://", timeout=timedelta(minutes=5)
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()

        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=cfg.training.get("mixed_precision", "no"),
        )
        self.device = self.accelerator.device
        log.info(f"rank: {self.accelerator.local_process_index}  device: {self.device}")

        seed(cfg.training.seed)

        self.pretrained_cfg, self.pretrained_dir = self._load_pretrained_cfg()

        # Episode-shape hyperparams default to whatever the pretrained
        # world model used; override under `idm:` in your yaml if you
        # want the IDM to train on a different history/pred/frameskip.
        self.num_hist = cfg.idm.get("num_hist", self.pretrained_cfg.num_hist)
        self.num_pred = cfg.idm.get("num_pred", self.pretrained_cfg.num_pred)
        self.frameskip = cfg.idm.get("frameskip", self.pretrained_cfg.frameskip)
        self.img_size = cfg.idm.get("img_size", self.pretrained_cfg.img_size)

        log.info(
            f"Loading dataset from {self.pretrained_cfg.env.dataset.data_path} ..."
        )
        self.datasets, traj_dsets = hydra.utils.call(
            self.pretrained_cfg.env.dataset,
            num_hist=self.num_hist,
            num_pred=self.num_pred,
            frameskip=self.frameskip,
        )
        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"batch_size={cfg.training.batch_size} "
            f"num_processes={self.accelerator.num_processes}."
        )
        gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        num_workers = cfg.training.get(
            "num_workers", self.pretrained_cfg.env.get("num_workers", 4)
        )
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=gpu_batch_size,
                shuffle=False,  # already shuffled in TrajSlicerDataset
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
            for x in ["train", "valid"]
        }
        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )
        log.info(f"dataloader batch size (per gpu): {gpu_batch_size}")

        self._build_frozen_encoders()
        self._build_idm()
        self._init_optimizer()

        self.step = 0
        self.total_steps = cfg.idm.total_steps

        idm_ckpt = Path(cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if idm_ckpt.exists():
            self._load_idm_ckpt(idm_ckpt)
            log.info(f"Resuming IDM training from step {self.step}: {idm_ckpt}")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg.get("wandb_run_id", None)
            wandb_dict = cfg_to_dict(cfg)
            wandb_dict["pretrained_run_dir"] = str(self.pretrained_dir)
            self.wandb_run = wandb.init(
                project=f"idm_{self.pretrained_cfg.env.name}",
                config=wandb_dict,
                id=wandb_run_id,
                resume="allow",
            )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        self.step_log = OrderedDict()

    # ------------------------------------------------------------------
    # pretrained (frozen) world-model components
    # ------------------------------------------------------------------
    def _load_pretrained_cfg(self):
        pretrained_dir = Path(self.cfg.idm.pretrained_run_dir)
        cfg_path = pretrained_dir / "hydra.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} not found -- idm.pretrained_run_dir must be the "
                "output directory of a completed world-model training run "
                "(it must contain hydra.yaml and checkpoints/)."
            )
        return OmegaConf.load(cfg_path), pretrained_dir

    def _build_frozen_encoders(self):
        pcfg = self.pretrained_cfg
        ckpt_name = self.cfg.idm.get("pretrained_ckpt_name", "model_latest.pth")
        ckpt_path = self.pretrained_dir / "checkpoints" / ckpt_name
        ckpt = {}
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            log.info(f"Loaded pretrained world-model checkpoint from {ckpt_path}")
        else:
            log.warning(
                f"No checkpoint at {ckpt_path}; instantiating encoder/proprio_encoder "
                "fresh from the pretrained run's config. This is only correct if "
                "those modules were fully frozen (train_encoder=False) during "
                "pretraining -- otherwise you're missing their trained weights."
            )

        # --- visual encoder ---
        self.encoder = ckpt.get("encoder", None)
        if self.encoder is None:
            encoder_kwargs = {}
            if getattr(pcfg.encoder, "projector_config", None) is not None:
                encoder_kwargs["projector_config"] = hydra.utils.instantiate(
                    pcfg.encoder.projector_config
                )
            self.encoder = hydra.utils.instantiate(pcfg.encoder, **encoder_kwargs)
            log.info("encoder: not in checkpoint -> instantiated fresh (frozen backbone).")
        else:
            log.info("encoder: loaded trained weights from checkpoint.")

        # --- proprio encoder ---
        self.proprio_encoder = ckpt.get("proprio_encoder", None)
        if self.proprio_encoder is None:
            self.proprio_encoder = hydra.utils.instantiate(
                pcfg.proprio_encoder,
                in_chans=self.datasets["train"].proprio_dim,
                emb_dim=pcfg.proprio_emb_dim,
            )
            log.info("proprio_encoder: not in checkpoint -> instantiated fresh.")
        else:
            log.info("proprio_encoder: loaded trained weights from checkpoint.")

        # capture dims before moving/freezing
        self.visual_emb_dim = self.encoder.emb_dim
        self.proprio_emb_dim = self.proprio_encoder.emb_dim

        self.encoder = self.encoder.to(self.device).eval()
        self.proprio_encoder = self.proprio_encoder.to(self.device).eval()
        for p in itertools.chain(
            self.encoder.parameters(), self.proprio_encoder.parameters()
        ):
            p.requires_grad = False

        # mirror VWorldModel's dino-specific resize so the frozen encoder
        # sees exactly the input distribution it was pretrained/used with
        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae, matches VWorldModel/Trainer
            num_side_patches = self.img_size // decoder_scale
            encoder_image_size = num_side_patches * self.encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(encoder_image_size)]
            )
        else:
            self.encoder_transform = lambda x: x

    @torch.no_grad()
    def encode_obs(self, obs):
        """Mirrors VWorldModel.encode_obs using the frozen pretrained encoders."""
        visual = obs["visual"].to(self.device)
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)
        proprio_emb = self.proprio_encoder(obs["proprio"].to(self.device))
        return {"visual": visual_embs, "proprio": proprio_emb}

    # ------------------------------------------------------------------
    # IDM model + optimizer
    # ------------------------------------------------------------------
    def _build_idm(self):
        cfg = self.cfg
        action_dim = self.datasets["train"].action_dim
        self.idm = InverseDynamicsModel(
            visual_emb_dim=self.visual_emb_dim,
            proprio_emb_dim=self.proprio_emb_dim,
            action_dim=action_dim,
            hidden_dim=cfg.idm.hidden_dim,
            num_layers=cfg.idm.num_layers,
            dropout=cfg.idm.get("dropout", 0.0),
            pooling=cfg.idm.get("pooling", "mean"),
        )
        self.idm = self.accelerator.prepare(self.idm)
        self.idm_raw = self.accelerator.unwrap_model(self.idm)
        if self.accelerator.is_main_process:
            total = sum(p.numel() for p in self.idm.parameters())
            log.info(f"[idm] total params: {total} (action_dim={action_dim})")

    def _init_optimizer(self):
        self.idm_optimizer = torch.optim.AdamW(
            self.idm.parameters(), lr=self.cfg.idm.lr
        )
        self.idm_optimizer = self.accelerator.prepare(self.idm_optimizer)

    # ------------------------------------------------------------------
    # checkpointing (the IDM's own training checkpoints, separate from
    # the pretrained world-model checkpoint loaded above)
    # ------------------------------------------------------------------
    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = {
                "step": self.step,
                "idm": self.accelerator.unwrap_model(self.idm),
                "idm_optimizer": self.idm_optimizer.state_dict(),
            }
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_step{self.step}.pth")
            log.info(f"Saved IDM checkpoint (step {self.step}) to {os.getcwd()}")

    def _load_idm_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.step = ckpt.get("step", 0)
        if "idm" in ckpt:
            self.idm = self.accelerator.prepare(ckpt["idm"])
            self.idm_raw = self.accelerator.unwrap_model(self.idm)
        if "idm_optimizer" in ckpt:
            try:
                self.idm_optimizer.load_state_dict(ckpt["idm_optimizer"])
                log.info("Loaded IDM optimizer state from checkpoint.")
            except Exception as e:
                log.warning(f"Failed to load IDM optimizer state: {e}")

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def _next_train_batch(self):
        """
        Pulls the next batch, transparently restarting the iterator when
        the dataset is exhausted. Deliberately NOT itertools.cycle, which
        caches every yielded item the first time through -- fine for a
        small dataset, a memory blow-up for a large one.
        """
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.dataloaders["train"])
            return next(self._train_iter)

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def compute_loss(self, obs, act):
        """
        obs["visual"]: (b, T, 3, H, W)
        act: (b, T, action_dim) or (b, T-1, action_dim) -- we only ever
             consume the first T-1 entries.

        ASSUMPTION (see module docstring, point 3): act[:, t] drives the
        transition frame t -> frame t+1. If your dataset instead aligns
        act[:, t] as the action *arriving at* frame t, swap the two lines
        marked below (use state[:, 1:], state[:, :-1] and act[:, 1:T]).
        """
        z_obs = self.encode_obs(obs)                    # visual:(b,T,p,d) proprio:(b,T,d)
        state = self.idm_raw.state_feat(z_obs)           # (b, T, state_dim)

        z_t = state[:, :-1, :]                            # <- assumption
        z_tp1 = state[:, 1:, :]                           # <- assumption
        T_minus_1 = z_t.shape[1]
        target_act = act[:, :T_minus_1, :]

        pred_act = self.idm(z_t, z_tp1)
        loss = nn.functional.mse_loss(pred_act, target_act)
        return loss

    # ------------------------------------------------------------------
    # validation (capped number of batches -- not a full pass, since the
    # dataset is assumed large)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def val(self):
        self.idm.eval()
        max_batches = self.cfg.idm.get("val_batches", 50) or None  # None/0 -> full pass
        val_iter = iter(self.dataloaders["valid"])
        total, count = 0.0, 0
        for i, data in enumerate(val_iter):
            if max_batches is not None and i >= max_batches:
                break
            obs, act, state = data
            loss = self.compute_loss(obs, act)
            loss = self.accelerator.gather_for_metrics(loss).mean()
            total += loss.item()
            count += 1
        val_loss = total / max(count, 1)
        log.info(f"step {self.step}  val_idm_loss: {val_loss:.4f} (over {count} batches)")
        if self.accelerator.is_main_process:
            self.wandb_run.log({"val_idm_loss": val_loss, "step": self.step})
        self.idm.train()

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def logs_update(self, logs):
        for key, value in logs.items():
            length = len(value)
            count, total = self.step_log.get(key, (0, 0.0))
            self.step_log[key] = (count + length, total + sum(value))

    def logs_flash(self):
        flat = OrderedDict()
        for key, (count, total) in self.step_log.items():
            flat[key] = total / count
        flat["step"] = self.step
        log.info(
            f"step {self.step}  train_idm_loss: "
            f"{flat.get('train_idm_loss', float('nan')):.4f}"
        )
        if self.accelerator.is_main_process:
            self.wandb_run.log(flat)
        self.step_log = OrderedDict()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self):
        self._train_iter = iter(self.dataloaders["train"])
        self.idm.train()
        pbar = tqdm(
            total=self.total_steps,
            initial=self.step,
            desc="IDM train",
            disable=not self.accelerator.is_main_process,
        )
        while self.step < self.total_steps:
            obs, act, state = self._next_train_batch()
            loss = self.compute_loss(obs, act)

            self.idm_optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.idm_optimizer.step()

            self.step += 1
            pbar.update(1)

            loss_val = self.accelerator.gather_for_metrics(loss).mean().item()
            self.logs_update({"train_idm_loss": [loss_val]})
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if self.step % self.cfg.idm.log_every_x_steps == 0:
                self.logs_flash()

            if self.step % self.cfg.idm.val_every_x_steps == 0:
                self.accelerator.wait_for_everyone()
                self.val()
                self.accelerator.wait_for_everyone()

            if self.step % self.cfg.idm.save_every_x_steps == 0:
                self.save_ckpt()

        pbar.close()
        self.logs_flash()
        self.save_ckpt()


@hydra.main(config_path="conf", config_name="train_idm")
def main(cfg):
    trainer = IDMTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
