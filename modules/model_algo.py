import numpy as np

from structs.app_enum import ActionType, PositionType


class AlgoModel:
    def __init__(self):
        pass

    @staticmethod
    def can_execute(action: int, masks: np.ndarray):
        """
        アクションが行動マスクで禁止されていないかチェック
        :param action:
        :param masks:
        :return:
        """
        return masks[action] == 1

    def predict(self, dict_obs, action_masks: np.ndarray) -> tuple[int, dict]:
        arr_position = dict_obs["position"]
        idx = int(np.argmax(arr_position[:3]))
        position = PositionType(idx)
        ma_cross_golden = arr_position[3]
        ma_cross_dead = arr_position[4]
        if position == PositionType.NONE:
            if ma_cross_golden == 1.0 and self.can_execute(ActionType.BUY.value, action_masks):
                return ActionType.BUY.value, {'reason': 'golden_cross'}
            if ma_cross_dead == 1.0 and self.can_execute(ActionType.SELL.value, action_masks):
                return ActionType.SELL.value, {'reason': 'dead_cross'}
            return ActionType.HOLD.value, {}
        else:
            return ActionType.HOLD.value, {}
