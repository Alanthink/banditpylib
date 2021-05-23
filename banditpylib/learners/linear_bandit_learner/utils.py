from typing import Optional

from banditpylib.bandits import LinearBandit
from banditpylib.learners import Learner


class LinearBanditLearner(Learner):
    """Abstract class for learners playing with linear bandit"""

    def __init__(self, arm_num: int, name: Optional[str]):
        """
        Args:
          arm_num: number of arms
          budget: total number of pulls
          name: alias name
        """
        super().__init__(name)
        if arm_num < 2:
            raise ValueError('Number of arms is expected at least 2. Got %d.' %
                             arm_num)
        self.__arm_num = arm_num

    @property
    def running_environment(self) -> type:
        return LinearBandit

    def arm_num(self) -> int:
        """
        Returns:
          number of arms
        """
        return self.__arm_num