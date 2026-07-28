"""
module for retrieving reach_bottom_left reward function
from dm_control suite jaco
"""

import dmc
import numpy as np

env = dmc.make(name="jaco_reach_bottom_left")
reward_function = env._task.get_reward  # pylint: disable=protected-access


def distance_function(physics):
    """Euclidean distance (m) from the hand's tool-center-point to the task target."""
    hand_pos = physics.bind(env._task.hand.tool_center_point).xpos  # pylint: disable=protected-access
    target_pos = env._task._targets[0]  # pylint: disable=protected-access
    return float(np.linalg.norm(hand_pos - target_pos))
