"""module for retrieving arabesque reward function from custom_dmc_tasks.walker_yoga"""

from custom_dmc_tasks.walker_yoga import arabesque

reward_function = arabesque()._task.get_reward  # pylint: disable=protected-access
