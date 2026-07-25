"""module for retrieving lunge_pose reward function from custom_dmc_tasks.walker_yoga"""

from custom_dmc_tasks.walker_yoga import lunge_pose

reward_function = lunge_pose()._task.get_reward  # pylint: disable=protected-access
