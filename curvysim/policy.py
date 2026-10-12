"""Actor-critic network shared by training, evaluation and the live bot.

Inputs: the feature vector (always), an egocentric local occupancy map
(``map_size > 0``) and a board-aligned whole-board map (``global_map``).
  * vector only: separate MLPs for policy and value.
  * with maps  : a CNN per map; embeddings are concatenated with the feature
    vector and fed to a shared MLP trunk.
"""
import torch
import torch.nn as nn

from .core import GRID
from .obs import OBS_DIM, MAP_CHANNELS, GLOBAL_CHANNELS

N_ACTIONS = 3  # index 0, 1, 2 -> move -1 (left), 0 (straight), +1 (right)


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


class MapEncoder(nn.Module):
    def __init__(self, channels, map_size, out_dim=256):
        super().__init__()
        layers = [nn.Conv2d(channels, 32, 5, stride=2, padding=2), nn.ReLU()]
        c, s = 32, (map_size + 1) // 2
        while s > 6:   # halve the resolution down to ~4-6 cells
            layers += [nn.Conv2d(c, 64, 3, stride=2, padding=1), nn.ReLU()]
            c, s = 64, (s + 1) // 2
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            n = self.net(torch.zeros(1, channels, map_size, map_size)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, out_dim), nn.ReLU())

    def forward(self, maps):
        return self.fc(self.net(maps.float() / 255.0))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=(512, 512, 256), map_size=0, map_embed=256,
                 global_map=False):
        super().__init__()
        self.config = {'obs_dim': obs_dim, 'hidden': list(hidden), 'map_size': map_size,
                       'map_embed': map_embed, 'global_map': global_map}
        self.map_size = map_size
        self.global_map = global_map
        self.shared = map_size > 0 or global_map
        if self.shared:
            width = obs_dim
            if map_size > 0:
                self.encoder = MapEncoder(MAP_CHANNELS, map_size, map_embed)
                width += map_embed
            if global_map:
                self.global_encoder = MapEncoder(GLOBAL_CHANNELS, GRID, map_embed)
                width += map_embed
            self.trunk = _mlp([width, *hidden])
        else:
            self.pi_body = _mlp([obs_dim, *hidden])
            self.v_body = _mlp([obs_dim, *hidden])
        self.pi_head = nn.Linear(hidden[-1], N_ACTIONS)
        self.v_head = nn.Linear(hidden[-1], 1)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, 2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi_head.weight, 0.01)
        nn.init.orthogonal_(self.v_head.weight, 1.0)

    def forward(self, obs, maps=None, gmaps=None):
        if self.shared:
            parts = [obs]
            if self.map_size > 0:
                parts.append(self.encoder(maps))
            if self.global_map:
                parts.append(self.global_encoder(gmaps))
            h = self.trunk(torch.cat(parts, -1))
            return self.pi_head(h), self.v_head(h).squeeze(-1)
        return self.pi_head(self.pi_body(obs)), self.v_head(self.v_body(obs)).squeeze(-1)

    def logits(self, obs, maps=None, gmaps=None):
        if self.shared:
            return self.forward(obs, maps, gmaps)[0]
        return self.pi_head(self.pi_body(obs))

    def value(self, obs, maps=None, gmaps=None):
        if self.shared:
            return self.forward(obs, maps, gmaps)[1]
        return self.v_head(self.v_body(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, maps=None, gmaps=None, greedy=False):
        logits = self.logits(obs, maps, gmaps)
        if greedy:
            return logits.argmax(-1)
        return torch.distributions.Categorical(logits=logits).sample()


def save_policy(path, model, extra=None):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({'config': model.config, 'state_dict': state, **(extra or {})}, path)


def expand_obs_inputs(state_dict, config, new_dim):
    """Adapt a checkpoint trained with fewer features (new ones are appended at the end of
    the observation) to ``new_dim``: the extra input weights are zero, so the network behaves
    exactly as before until training teaches it to use them."""
    old_dim = config['obs_dim']
    state_dict = dict(state_dict)
    first = (['trunk.0.weight'] if config.get('map_size', 0) > 0 or config.get('global_map')
             else ['pi_body.0.weight', 'v_body.0.weight'])
    for key in first:
        w = state_dict[key]
        extra = torch.zeros(w.shape[0], new_dim - old_dim, dtype=w.dtype)
        # observation first, then the map embeddings (shared trunk)
        state_dict[key] = torch.cat([w[:, :old_dim], extra, w[:, old_dim:]], dim=1)
    return state_dict, {**config, 'obs_dim': new_dim}


def load_policy(path, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    config, state = ckpt['config'], ckpt['state_dict']
    if config['obs_dim'] < OBS_DIM:
        state, config = expand_obs_inputs(state, config, OBS_DIM)
        ckpt = {**ckpt, 'config': config, 'state_dict': state, 'expanded_from': ckpt['config']['obs_dim']}
    model = ActorCritic(**config)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt
