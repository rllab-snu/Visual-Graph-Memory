import torch
import torch.nn as nn
from model.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.common.utils import CategoricalNet
from model.resnet import resnet
from model.resnet.resnet import ResNetEncoder
from .perception import Perception

class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)

class VGMPolicy(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            goal_sensor_uuid="pointgoal_with_gps_compass",
            hidden_size=512,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            resnet_baseplanes=32,
            backbone="resnet50",
            normalize_visual_inputs=True,
            cfg=None
    ):
        super().__init__()
        self.net = VGMNet(
            observation_space=observation_space,
            action_space=action_space,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            normalize_visual_inputs=normalize_visual_inputs,
            cfg=cfg
        )
        self.dim_actions = action_space.n

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def act(
            self,
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
            return_features=False,
            mask_stop=False
    ):

        features, rnn_hidden_states, preds, ffeatures, = self.net(
            observations, rnn_hidden_states, prev_actions, masks, return_features=return_features
        )
        distribution, x = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        # The shape of the output should be B * N * (shapes)\
        if return_features:
            return value, action, action_log_probs, rnn_hidden_states, x, preds, ffeatures
        return value, action, action_log_probs, rnn_hidden_states, x, preds, None

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, *_ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        value = self.critic(features)
        return value

    def evaluate_actions(
            self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states, preds, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution, x = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, x, preds[0], preds[1]


class VGMNet(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            goal_sensor_uuid,
            hidden_size,
            num_recurrent_layers,
            rnn_type,
            backbone,
            resnet_baseplanes,
            normalize_visual_inputs,
            cfg
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        self.num_category = 50
        self._n_input_goal = 0

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        self.perception_unit = Perception(cfg, None)
        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Linear(
                    cfg.features.visual_feature_dim * 3, hidden_size * 2
                ),
                nn.ReLU(True),
                nn.Linear(
                    hidden_size * 2, hidden_size
                ),
                nn.ReLU(True),
            )

        self.pred_aux1 = nn.Sequential(nn.Linear(cfg.features.visual_feature_dim, cfg.features.visual_feature_dim),
                                       nn.ReLU(),
                                       nn.Linear(cfg.features.visual_feature_dim, 1))
        self.pred_aux2 = nn.Sequential(nn.Linear(cfg.features.visual_feature_dim * 2, cfg.features.visual_feature_dim),
                                       nn.ReLU(),
                                       nn.Linear(cfg.features.visual_feature_dim, 1))
        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, mode='', return_features=False):
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        input_list = [observations['panoramic_rgb'].permute(0, 3, 1, 2) / 255.0,
                      observations['panoramic_depth'].permute(0, 3, 1, 2)]
        curr_tensor = torch.cat(input_list, 1)
        observations['curr_embedding'] = self.visual_encoder(curr_tensor).view(curr_tensor.shape[0], -1)

        goal_tensor = observations['target_goal'].permute(0, 3, 1, 2)
        observations['goal_embedding'] = self.visual_encoder(goal_tensor).view(goal_tensor.shape[0], -1)

        curr_context, goal_context, ffeatures = self.perception_unit(observations, masks,
                                                                     return_features=return_features)
        contexts = torch.cat((curr_context, goal_context), -1)
        feats = self.visual_fc(torch.cat((contexts, observations['curr_embedding']), 1))
        x = [feats, prev_actions]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        pred1 = self.pred_aux1(curr_context)
        pred2 = self.pred_aux2(contexts)

        return x, rnn_hidden_states, (pred1, pred2), ffeatures if return_features else None
