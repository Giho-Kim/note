"""Static yoga-pose tasks for the Walker domain, built on top of dm_control's
stock `dm_control.suite.walker` (PlanarWalker/Physics/SUITE), not this repo's
own `custom_dmc_tasks.walker` module. Each pose is a hand-crafted reward over
body heights/positions (torso, thighs, legs, feet) -- no new physics model is
needed, `custom_dmc_tasks/walker.xml` already has all the referenced bodies.
"""
import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import walker
from dm_control.utils import rewards
from dm_control.utils import io as resources

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "custom_dmc_tasks")

_YOGA_STAND_HEIGHT = 1.0  # lower than stand height = 1.2
_YOGA_LIE_DOWN_HEIGHT = 0.1
_YOGA_FEET_UP_HEIGHT = 0.5
_YOGA_FEET_UP_LIE_DOWN_HEIGHT = 0.35
_YOGA_KNEE_HEIGHT = 0.25
_YOGA_KNEESTAND_HEIGHT = 0.75
_YOGA_SITTING_HEIGHT = 0.55
_YOGA_SITTING_LEGS_HEIGHT = 0.15


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, "walker.xml")), common.ASSETS


class YogaPlanarWalker(walker.PlanarWalker):
    """Planar walker rewarded for holding a static yoga pose."""

    def __init__(self, goal, random=None):
        super().__init__(move_speed=0, random=random)
        self._goal = goal

    def _arabesque_reward(self, physics):
        standing = rewards.tolerance(
            physics.torso_height(),
            bounds=(_YOGA_STAND_HEIGHT, float("inf")),
            margin=_YOGA_STAND_HEIGHT / 2,
        )
        left_foot_height = physics.named.data.xpos["left_foot", "z"]
        right_foot_height = physics.named.data.xpos["right_foot", "z"]
        max_foot = "right_foot" if right_foot_height > left_foot_height else "left_foot"
        min_foot = "right_foot" if right_foot_height <= left_foot_height else "left_foot"
        min_foot_height = physics.named.data.xpos[min_foot, "z"]
        max_foot_height = physics.named.data.xpos[max_foot, "z"]
        min_foot_down = rewards.tolerance(
            min_foot_height, bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        max_foot_up = rewards.tolerance(
            max_foot_height, bounds=(_YOGA_STAND_HEIGHT, float("inf")),
            margin=_YOGA_STAND_HEIGHT / 2)
        min_foot_x = physics.named.data.xpos[min_foot, "x"]
        max_foot_x = physics.named.data.xpos[max_foot, "x"]
        correct_foot_pose = 0.1 if max_foot_x > min_foot_x else 1.0
        feet_pose = (min_foot_down + max_foot_up * 2) / 3
        return standing * feet_pose * correct_foot_pose

    def _lying_down_reward(self, physics):
        torso_down = rewards.tolerance(
            physics.torso_height(), bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        horizontal = 1 - abs(physics.torso_upright())
        thigh_height = (physics.named.data.xpos["left_thigh", "z"]
                        + physics.named.data.xpos["right_thigh", "z"]) / 2
        thigh_down = rewards.tolerance(
            thigh_height, bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        leg_height = (physics.named.data.xpos["left_leg", "z"]
                     + physics.named.data.xpos["right_leg", "z"]) / 2
        leg_down = rewards.tolerance(
            leg_height, bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        feet_height = (physics.named.data.xpos["left_foot", "z"]
                       + physics.named.data.xpos["right_foot", "z"]) / 2
        feet_down = rewards.tolerance(
            feet_height, bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        return (3 * torso_down + horizontal + thigh_down + feet_down + leg_down) / 7

    def _legs_up_reward(self, physics):
        torso_down = rewards.tolerance(
            physics.torso_height(), bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        horizontal = 1 - abs(physics.torso_upright())
        torso_down = (3 * torso_down + horizontal) / 4
        feet_height = (physics.named.data.xpos["left_foot", "z"]
                       + physics.named.data.xpos["right_foot", "z"]) / 2
        feet_up = rewards.tolerance(
            feet_height, bounds=(_YOGA_FEET_UP_LIE_DOWN_HEIGHT, float("inf")),
            margin=_YOGA_FEET_UP_LIE_DOWN_HEIGHT / 2)
        return torso_down * feet_up

    def _high_kick_reward(self, physics):
        standing = rewards.tolerance(
            physics.torso_height(), bounds=(_YOGA_STAND_HEIGHT, float("inf")),
            margin=_YOGA_STAND_HEIGHT / 2)
        left_foot_height = physics.named.data.xpos["left_foot", "z"]
        right_foot_height = physics.named.data.xpos["right_foot", "z"]
        min_foot_height = min(left_foot_height, right_foot_height)
        max_foot_height = max(left_foot_height, right_foot_height)
        min_foot_down = rewards.tolerance(
            min_foot_height, bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        max_foot_up = rewards.tolerance(
            max_foot_height, bounds=(walker._STAND_HEIGHT, float("inf")),
            margin=walker._STAND_HEIGHT / 2)
        feet_pose = (3 * max_foot_up + min_foot_down) / 4
        return standing * feet_pose

    def _one_foot_reward(self, physics):
        standing = rewards.tolerance(
            physics.torso_height(), bounds=(_YOGA_STAND_HEIGHT, float("inf")),
            margin=_YOGA_STAND_HEIGHT / 2)
        left_foot_height = physics.named.data.xpos["left_foot", "z"]
        right_foot_height = physics.named.data.xpos["right_foot", "z"]
        min_foot_height = min(left_foot_height, right_foot_height)
        max_foot_height = max(left_foot_height, right_foot_height)
        min_foot_down = rewards.tolerance(
            min_foot_height, bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        max_foot_up = rewards.tolerance(
            max_foot_height, bounds=(_YOGA_FEET_UP_HEIGHT, float("inf")),
            margin=_YOGA_FEET_UP_HEIGHT / 2)
        return standing * max_foot_up * min_foot_down

    def _lunge_pose_reward(self, physics):
        standing = rewards.tolerance(
            physics.torso_height(), bounds=(_YOGA_KNEESTAND_HEIGHT, float("inf")),
            margin=_YOGA_KNEESTAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        torso = (3 * standing + upright) / 4
        left_leg_height = physics.named.data.xpos["left_leg", "z"]
        right_leg_height = physics.named.data.xpos["right_leg", "z"]
        min_leg_height = min(left_leg_height, right_leg_height)
        max_leg_height = max(left_leg_height, right_leg_height)
        min_leg_down = rewards.tolerance(
            min_leg_height, bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        max_leg_up = rewards.tolerance(
            max_leg_height, bounds=(_YOGA_KNEE_HEIGHT, float("inf")),
            margin=_YOGA_KNEE_HEIGHT / 2)
        max_thigh = "left_thigh" if max_leg_height == left_leg_height else "right_thigh"
        min_leg = "left_leg" if min_leg_height == left_leg_height else "right_leg"
        max_thigh_horiz = 1 - abs(physics.named.data.xmat[max_thigh, "zz"])
        min_leg_horiz = 1 - abs(physics.named.data.xmat[min_leg, "zz"])
        legs = (min_leg_down + max_leg_up + max_thigh_horiz + min_leg_horiz) / 4
        return torso * legs

    def _sit_knees_reward(self, physics):
        standing = rewards.tolerance(
            physics.torso_height(), bounds=(_YOGA_SITTING_HEIGHT, float("inf")),
            margin=_YOGA_SITTING_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        torso_up = (3 * standing + upright) / 4
        legs_height = (physics.named.data.xpos["left_leg", "z"]
                      + physics.named.data.xpos["right_leg", "z"]) / 2
        legs_down = rewards.tolerance(
            legs_height, bounds=(-float("inf"), _YOGA_SITTING_LEGS_HEIGHT),
            margin=_YOGA_SITTING_LEGS_HEIGHT * 1.5)
        feet_height = (physics.named.data.xpos["left_foot", "z"]
                       + physics.named.data.xpos["right_foot", "z"]) / 2
        feet_down = rewards.tolerance(
            feet_height, bounds=(-float("inf"), _YOGA_LIE_DOWN_HEIGHT),
            margin=_YOGA_LIE_DOWN_HEIGHT * 1.5)
        legs = (3 * legs_down + feet_down) / 4
        return torso_up * legs

    def get_reward(self, physics):
        return {
            "arabesque": self._arabesque_reward,
            "lying_down": self._lying_down_reward,
            "legs_up": self._legs_up_reward,
            "high_kick": self._high_kick_reward,
            "one_foot": self._one_foot_reward,
            "lunge_pose": self._lunge_pose_reward,
            "sit_knees": self._sit_knees_reward,
        }[self._goal](physics)


def _make_pose_task(goal):
    def task_fn(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
        physics = walker.Physics.from_xml_string(*get_model_and_assets())
        task = YogaPlanarWalker(goal=goal, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=walker._CONTROL_TIMESTEP, **environment_kwargs)
    task_fn.__name__ = goal
    task_fn.__doc__ = f"Returns the {goal.replace('_', ' ').title()} yoga pose task."
    return task_fn


POSES = ["arabesque", "lying_down", "legs_up", "high_kick", "one_foot",
         "lunge_pose", "sit_knees"]

for _goal in POSES:
    globals()[_goal] = walker.SUITE.add("custom")(_make_pose_task(_goal))
del _goal
