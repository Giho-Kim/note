"""module for retrieving backward run reward function from custom_dmc_tasks walker"""

from custom_dmc_tasks.walker import run_backward

reward_function = run_backward()._task.get_reward  # pylint: disable=protected-access
