import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from idm import inverse_dynamics_consistency_loss


def create_objective_fn(
    alpha,
    base,
    mode="last",
    idm=None,
    idm_weight=0.0,
):
    metric = nn.MSELoss(reduction="none")

    def add_idm(goal_loss, z_obs_pred, z_obs_tgt, actions):
        if idm is None or idm_weight == 0.0:
            return goal_loss

        z_goal_feat = idm.state_feat(z_obs_tgt).squeeze(1)

        idm_loss = inverse_dynamics_consistency_loss(
            idm=idm,
            z_obses=z_obs_pred,
            actions=actions,
            z_goal_feat=z_goal_feat,
        )

        return goal_loss + idm_weight * idm_loss

    def objective_fn_last(
        z_obs_pred,
        z_obs_tgt,
        step=None,
        actions=None,
    ):
        loss_visual = metric(
            z_obs_pred["visual"][:, -1:],
            z_obs_tgt["visual"],
        ).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )

        loss_proprio = metric(
            z_obs_pred["proprio"][:, -1:],
            z_obs_tgt["proprio"],
        ).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )

        goal_loss = loss_visual + alpha * loss_proprio

        return add_idm(
            goal_loss,
            z_obs_pred,
            z_obs_tgt,
            actions,
        )

    def objective_fn_all(
        z_obs_pred,
        z_obs_tgt,
        step=None,
        coeffs=None,
        base=base,
        actions=None,
    ):
        if coeffs is None:
            coeffs = np.array(
                [base**i for i in range(z_obs_pred["visual"].shape[1])],
                dtype=np.float32,
            )
            coeffs = torch.tensor(
                coeffs / np.sum(coeffs),
                device=z_obs_pred["visual"].device,
            )
        else:
            coeffs = coeffs.to(z_obs_pred["visual"].device)

        loss_visual = metric(
            z_obs_pred["visual"],
            z_obs_tgt["visual"],
        ).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )

        loss_proprio = metric(
            z_obs_pred["proprio"],
            z_obs_tgt["proprio"],
        ).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )

        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)

        goal_loss = loss_visual + alpha * loss_proprio

        return add_idm(
            goal_loss,
            z_obs_pred,
            z_obs_tgt,
            actions,
        )

    def objective_fn_staged(
        z_obs_pred,
        z_obs_tgt,
        step=None,
        actions=None,
    ):
        if step is None:
            return objective_fn_all(
                z_obs_pred=z_obs_pred,
                z_obs_tgt=z_obs_tgt,
                actions=actions,
            )

        if step < z_obs_pred["visual"].shape[1] - 1:
            return objective_fn_last(
                z_obs_pred=z_obs_pred,
                z_obs_tgt=z_obs_tgt,
                actions=actions,
            )

        return objective_fn_all(
            z_obs_pred=z_obs_pred,
            z_obs_tgt=z_obs_tgt,
            actions=actions,
        )

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    elif mode == "staged":
        return objective_fn_staged
    else:
        raise NotImplementedError
