"""module for retrieving backward flip reward function from custom_dmc_tasks walker"""

from custom_dmc_tasks.walker import flip_backward

reward_function = flip_backward()._task.get_reward  # pylint: disable=protected-access
