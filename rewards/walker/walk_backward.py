"""module for retrieving backward walk reward function from custom_dmc_tasks walker"""

from custom_dmc_tasks.walker import walk_backward

reward_function = walk_backward()._task.get_reward  # pylint: disable=protected-access
