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
    def __init__(self, liner_feature):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # head_feature_num: 0 in_feature_num, 1 out_feature_num
        self.head = nn.Linear(liner_feature[0], liner_feature[1])

    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNTrainInterface:
    def __init__(self, liner_param, optimize_param, path, use_trained=None):
        self.use_cuda = torch.cuda.is_available()

        self.policy_net = DQN(liner_param)
        self.target_net = DQN(liner_param)

        # Parallel should happen before loading model
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            self.policy_net = nn.DataParallel(self.policy_net)
            self.target_net = nn.DataParallel(self.target_net)

        if use_trained is not None:
            file_name = path[0] + path[1] + use_trained
            self.policy_net.load_state_dict(torch.load(file_name))

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set this network as evaluation mode

        if self.use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()
            self.float_tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
            self.long_tensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
            self.byte_tensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
            self.tensor = self.float_tensor

        self.optimize_param = optimize_param
        self.path = path

        self._optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=1e-5)
        self.memory = ReplayMemory(10000)

        self.episode_done = 0
        self.steps_done = 0

        self.loss = None

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.optimize_param[3] + ((self.optimize_param[2] - self.optimize_param[3]) *
                                                  math.exp(-1. * self.steps_done / self.optimize_param[4]))
        self.steps_done += 1
        if sample > eps_threshold:
            data_in = Variable(state, volatile=True).type(self.float_tensor)
            data_out = self.policy_net(data_in)
            return data_out.data.max(1)[1].view(1, 1), True
        else:
            return self.long_tensor([[random.randrange(2)]]), False

    def set_reward(self, reward):
        return self.tensor([reward]), reward

    def train(self):
        if len(self.memory) < self.optimize_param[0]:
            return

        transitions = self.memory.sample(self.optimize_param[0])
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = self.byte_tensor(tuple(map(lambda s: s is not None, batch.next_state)))

        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]), volatile=True)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.optimize_param[0]).type(self.tensor))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (self.optimize_param[1] * next_state_values) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(expected_state_action_values.data)

        # Compute Huber loss
        self.loss = f.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self._optimizer.zero_grad()
        self.loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self._optimizer.step()

    def update(self, is_last_step, counter):
        # Update the target network and save
        if self.episode_done % self.optimize_param[-2] == 0 and is_last_step:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.episode_done % self.optimize_param[-1] == 0 and is_last_step:
            torch.save(self.target_net.state_dict(), self.path[0] + self.path[1] + str(counter))
            print('Model {} has been saved to {}'.format(self.path[1] + str(counter), self.path[0]))
        if is_last_step:
            self.steps_done = 0
            self.episode_done += 1


class DQNEvalInterface:
    def __init__(self, liner_feature, file_name):
        self.use_cuda = torch.cuda.is_available()
        self.policy_net = DQN(liner_feature)

        if torch.cuda.device_count() > 1:
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            self.policy_net = nn.DataParallel(self.policy_net)

        self.policy_net.load_state_dict(torch.load(file_name))
        # self.policy_net.eval()

        if self.use_cuda:
            self.policy_net.cuda()
            self.float_tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
            self.long_tensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
            self.byte_tensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
            self.tensor = self.float_tensor

    def generate_action(self, state):
        data_in = Variable(state, volatile=True).type(self.float_tensor)
        data_out = self.policy_net(data_in)
        action = data_out.data.max(1)[1].view(1, 1)[0, 0]
        return action
