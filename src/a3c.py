import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class ActorCritic(nn.Module):

    def __init__(self, state_space, action_space):
        """ActorCritic algortihm.

        Args:
            state_space (int): input size of state.
            action_space (int): output size of action for policy.
        """
        super().__init__()
        self.dense1 = nn.Linear(state_space, 32)
        self.dense2 = nn.Linear(32, 32)
        self.policy = nn.Linear(32, action_space)
        self.value = nn.Linear(32, 1)

    def forward(self, state):
        """Forward propagation.

        Args:
            state (torch.nn.Tensor): state representation.
        """
        # Shared layers across actor and critic networks
        z1 = self.dense1(state)
        a1 = nn.ReLU()(z1)
        z2 = self.dense2(a1)
        a2 = nn.ReLU()(z2)

        # Softmax output for each action and its log probabilities
        logit = self.policy(a2)
        proba = nn.Softmax(dim=-1)(logit)
        log_proba = nn.LogSoftmax(dim=-1)(logit)

        # Value function
        value = self.value(a2)

        return value, proba, log_proba
