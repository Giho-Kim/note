"""module for retrieving legs_up reward function from custom_dmc_tasks.walker_yoga"""

from custom_dmc_tasks.walker_yoga import legs_up

reward_function = legs_up()._task.get_reward  # pylint: disable=protected-access
