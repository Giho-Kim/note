"""module for retrieving lying_down reward function from custom_dmc_tasks.walker_yoga"""

from custom_dmc_tasks.walker_yoga import lying_down

reward_function = lying_down()._task.get_reward  # pylint: disable=protected-access
