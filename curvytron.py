from __future__ import absolute_import, division, print_function

import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

import numpy as np

import client
import time

__all__ = ['CurvytronEnv']


class State:
    """The state of the game.
    
    It consists of the canvas, the position of the player, and the angle of the player.
    """
    def __init__(self, pixels, position, angle):
        self.angle = angle
        self.position = position
        self.pixels = pixels


class CurvytronEnv(gym.Env):
    """Custom Environment for Curvytron.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, server, room, **kwargs):
        # Connect to the curvytron server
        self.client = client.CurvytronClient(**kwargs)
        self.client.start()
        self.client.connect_to_server(server)
        time.sleep(1)   # needed to give the client time to connect before joining a room
        
        # Join the specified room (or create it if it doesn't exist)
        self.client.join_room(room)

        # There's only three actions: left, right, ahead.
        self.action_space = spaces.Discrete(3)

        # Need to set shape to whatever the dimensions of the observation are.
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.float32, shape=(128,128,3))
        self.action_set = {0:-1,1:0,2:1}

    def step(self, a):
        """
        Submits an action to the curvytron server and returns the state
        
        Args:
            a (int in [0, 1, 2]): action to submit (left, straight, right)
        
        Returns:
            state (tuple): canvas, position, angle
            reward (float): 0 if not done, -10 when episode ends
            done (boolean): False until episode ends
        """
        reward = -10.0
        action = self.action_set[a]
        done = not (self.client.player_alive and self.client.active_round)
        if not done:
            self.client.send_action(action)
            reward = 0.0

        state = State(self.client.get_canvas(), self.client.position, self.client.angle)
        return state, reward, done

    def reset(self):
        """Resets the environment and returns the initial state.

        The environment will stay in the reset state until the server starts a new round
        and the player is alive, at which point it will return the initial state of the
        new round.

        Returns:
            state (tuple): canvas, position, angle
        """
        sent_ready = False
        while not (self.client.active_round and self.client.player_alive):
            if not self.client.active_game and not sent_ready:
                self.client.send_ready()
                sent_ready=True
            time.sleep(0.5)
            continue    # TODO this continue is probably useless
        state = State(self.client.get_canvas(), self.client.position, self.client.angle)
        return state

    def render(self, mode='human', close=False):
        pass
