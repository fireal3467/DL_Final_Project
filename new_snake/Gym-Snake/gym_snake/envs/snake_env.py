import os, subprocess, time, signal
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller, Discrete

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = Discrete(4)
        self.random_init = random_init

    def step(self, action):
        self.last_obs, rewards, done, info = self.controller.step(action)
        return self.last_obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid
        return self.last_obs

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(self.last_obs)

        return self.viewer.isopen

    def seed(self, x):
        pass
