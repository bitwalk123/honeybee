import numpy as np

from structs.app_enum import PositionType


def position_to_onehot(pos: PositionType) -> np.ndarray:
    onehot = np.zeros(3, dtype=np.float32)
    onehot[int(pos)] = 1.0
    return onehot
