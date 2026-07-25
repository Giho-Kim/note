"""module for retrieving high_kick reward function from custom_dmc_tasks.walker_yoga"""

from custom_dmc_tasks.walker_yoga import high_kick

reward_function = high_kick()._task.get_reward  # pylint: disable=protected-access
