from collections import namedtuple

import math
import random

from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, hf_num):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(hf_num[0], hf_num[1])  # head_feature_num: 0 in_feature_num, 1 out_feature_num

    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNInterface:
    def __init__(self, liner_param, hyper_param, path):
        self.use_cuda = torch.cuda.is_available()
        self.policy_net = DQN(liner_param)
        self.target_net = DQN(liner_param)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set this network as evaluation mode

        if self.use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()
            self.float_tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
            self.long_tensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
            self.byte_tensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
            self.tensor = self.float_tensor

        self.hyper_param = hyper_param
        self.path = path

        self._optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.episode_done = 0
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.hyper_param[3] + (self.hyper_param[2] - self.hyper_param[3]) * \
                                               math.exp(-1. * self.steps_done / self.hyper_param[4])
        self.steps_done += 1
        if sample > eps_threshold:
            data_in = Variable(state, volatile=True).type(self.float_tensor)
            data_out = self.policy_net(data_in)
            return data_out.data.max(1)[1].view(1, 1)
        else:
            return self.long_tensor([[random.randrange(2)]])

    def set_done(self, done):
        if done[2]:
            self.episode_done = done[0]
        else:
            self.episode_done = 0
        self.steps_done = done[1]

    def train(self):
        if len(self.memory) < self.hyper_param[0]:
            return

        transitions = self.memory.sample(self.hyper_param[0])
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = self.byte_tensor(tuple(map(lambda s: s is not None, batch.next_state)))

        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]), volatile=True)

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.hyper_param[0]).type(self.tensor))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hyper_param[1]) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(expected_state_action_values.data)

        # Compute Huber loss
        loss = f.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self._optimizer.step()

        # Update the target network and save
        if self.episode_done % self.hyper_param[-1] == 0 and self.episode_done > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            torch.save(self.target_net.state_dict(), self.path[0] + self.path[1])
            print('Model has been saved to {}'.format(self.path[0]))
