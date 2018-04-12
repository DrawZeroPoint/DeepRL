import gym
import numpy as np

import torch
import torchvision.transforms as t

from gym_interface import gym_info
from PIL import Image


class EnvInterface:

    """
    Functions defined with __X__ should be called by Python rather than user.
    """
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped

        self.episode_num = 20
        self.step_num = 0

        self.episode_count = 0
        self.step_count = 0

        self.observation = self.env.reset()
        self.finished = False

        self.prev_screen = None
        self.curr_screen = None

        self.resize = t.Compose([t.ToPILImage(),
                                 t.Resize(40, interpolation=Image.CUBIC),
                                 t.ToTensor()])

        self.float_tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def get_progress(self):
        return [self.episode_count, self.step_count, self.finished]

    def _get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))  # transpose into torch order (CHW)
        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]

        view_width = 320
        cart_location = self.get_location()
        if cart_location < view_width // 2:  # Floor division
            slice_range = slice(view_width)
        elif cart_location > (600 - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]

        screen = np.ascontiguousarray(screen, dtype=np.float32) * 0.0039215686  # 0.0039215686 = 1 / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).type(self.float_tensor)

    def get_location(self):
        world_width = self.env.x_threshold * 2  # x_threshold equal to 2.4 meter
        scale = 600 / world_width
        return int(self.env.state[0] * scale + 600 / 2.0)  # MIDDLE OF CART

    def get_state(self):
        if self.step_count == 0:
            self.prev_screen = self._get_screen()
            self.curr_screen = self._get_screen()
        else:
            self.prev_screen = self.curr_screen
            self.curr_screen = self._get_screen()
        return self.curr_screen - self.prev_screen

    def _next_episode(self):
        self.episode_count += 1
        self.step_count = 0
        self.env.reset()

    def reset_env(self):
        self.episode_count = 0
        self.step_count = 0
        self.observation = self.env.reset()
        self.finished = False

    # Similar with get_screen but faster
    def render_env(self):
        self.env.render()

    def set_up(self, e_num=20, s_num=0):
        self.episode_num = e_num
        self.step_num = s_num

    def step_once(self, act):
        if self.episode_count <= self.episode_num:
            self.observation = self.env.step(act)
            if self.step_num and self.step_count >= self.step_num:
                self._next_episode()
            else:
                if gym_info.get_observation_value(self.observation, 'done'):
                    if self.episode_count == self.episode_num:
                        self.finished = True
                    else:
                        self._next_episode()
                self.step_count += 1
        else:
            self.finished = True
