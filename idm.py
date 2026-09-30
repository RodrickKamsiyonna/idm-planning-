import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseDynamicsModel(nn.Module):
    def __init__(
        self,
        visual_emb_dim,
        proprio_emb_dim,
        action_dim,
        hidden_dim=512,
        num_layers=3,
        dropout=0.0,
        pooling="mean",
    ):
        super().__init__()
        assert num_layers >= 1
        assert pooling == "mean", f"Unsupported pooling '{pooling}'"
        self.pooling = pooling
        self.visual_emb_dim = visual_emb_dim
        self.proprio_emb_dim = proprio_emb_dim
        self.action_dim = action_dim
        self.state_dim = visual_emb_dim + proprio_emb_dim

        in_dim = self.state_dim * 2  # z_t concatenated with z_{t+1}
        layers = []
        prev = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def state_feat(self, z_obs):
        visual = z_obs["visual"].mean(dim=-2)  # pool patch dimension
        proprio = z_obs["proprio"]
        return torch.cat([visual, proprio], dim=-1)

    def forward(self, z_t, z_tp1):
        """
        z_t, z_tp1: (..., state_dim) pre-pooled state features (see
        state_feat above). Concatenates the pair and regresses the action.
        """
        x = torch.cat([z_t, z_tp1], dim=-1)
        return self.net(x)


def inverse_dynamics_consistency_loss(idm, z_obses, actions, z_goal_feat=None):
    state = idm.state_feat(z_obses)          # (b, H+1, state_dim)
    if z_goal_feat is not None:
        state = torch.cat([state[:, :-1, :], z_goal_feat.unsqueeze(1)], dim=1)
    z_t, z_tp1 = state[:, :-1, :], state[:, 1:, :]
    pred_actions = idm(z_t, z_tp1)            # (b, H, action_dim)
    return F.mse_loss(pred_actions, actions)
