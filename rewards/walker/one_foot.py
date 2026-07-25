"""module for retrieving one_foot reward function from custom_dmc_tasks.walker_yoga"""

from custom_dmc_tasks.walker_yoga import one_foot

reward_function = one_foot()._task.get_reward  # pylint: disable=protected-access
