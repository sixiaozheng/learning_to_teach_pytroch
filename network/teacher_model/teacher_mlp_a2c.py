import torch
import torch.nn as nn


class TeacherNetwork(nn.Module):

    def __init__(self, input_dim=25):
        super(TeacherNetwork, self).__init__()
        self.phi_body = nn.Sequential(nn.Linear(input_dim, 400),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(400,400),
                                      nn.ReLU(inplace=True))
        # self.actor_body = nn.Sequential()
        # self.critic_body = nn.Sequential()
        self.fc_action = self.layer_init(nn.Linear(400, 2), 1e-3)
        self.fc_critic = self.layer_init(nn.Linear(400,1), 1e-3)

    def layer_init(self, layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer

    def forward(self, obs, action=None):

        phi = self.phi_body(obs)
        # phi_a = self.actor_body(phi)
        # phi_v = self.critic_body(phi)
        logits = self.fc_action(phi)
        v = self.fc_critic(phi)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}



