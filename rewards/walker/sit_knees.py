"""module for retrieving sit_knees reward function from custom_dmc_tasks.walker_yoga"""

from custom_dmc_tasks.walker_yoga import sit_knees

reward_function = sit_knees()._task.get_reward  # pylint: disable=protected-access
