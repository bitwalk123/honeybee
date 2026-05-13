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

        arr_signal = dict_obs["signal"]
        ma_cross_golden = arr_signal[0]
        ma_cross_dead = arr_signal[1]
        vwap_cross_golden = arr_signal[2]
        vwap_cross_dead = arr_signal[3]

        arr_position = dict_obs["position"]
        idx = int(np.argmax(arr_position))
        position = PositionType(idx)
        #if position == PositionType.NONE:
        """
        if ma_cross_golden == 1.0 and self.can_execute(ActionType.BUY.value, action_masks):
            return ActionType.BUY.value, {'reason': 'ma_golden_cross'}
        if ma_cross_dead == 1.0 and self.can_execute(ActionType.SELL.value, action_masks):
            return ActionType.SELL.value, {'reason': 'ma_dead_cross'}
        """
        if vwap_cross_golden == 1.0 and self.can_execute(ActionType.BUY.value, action_masks):
            return ActionType.BUY.value, {'reason': 'vwap_golden_cross'}
        if vwap_cross_dead == 1.0 and self.can_execute(ActionType.SELL.value, action_masks):
            return ActionType.SELL.value, {'reason': 'vwap_dead_cross'}
        return ActionType.HOLD.value, {}
        #else:
        #    return ActionType.HOLD.value, {}
