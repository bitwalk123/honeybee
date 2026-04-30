import numpy as np

from structs.app_enum import ActionType


class AlgoModel:
    def __init__(self):
        pass

    @staticmethod
    def can_execute(action, masks: np.ndarray):
        """
        アクションが行動マスクで禁止されていないかチェック
        :param action:
        :param masks:
        :return:
        """
        return masks[action] == 1

    def predict(self, obj:dict, action_masks: np.ndarray) -> tuple[int, dict]:
        return ActionType.HOLD.value, {}
