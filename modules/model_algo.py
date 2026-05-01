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

    def predict(self, dict_obs, action_masks: np.ndarray) -> tuple[int, dict]:
        list_signal = dict_obs["position"]
        if list_signal[3] == 1.0 or list_signal[4] == 1.0:
            print(list_signal)
        return ActionType.HOLD.value, {}
