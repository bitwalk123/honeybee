from enum import Enum, IntEnum


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


"""
class PositionType(Enum):
    SHORT = -1
    NONE = 0
    LONG = 1
"""


class PositionType(IntEnum):
    SHORT = 0
    NONE = 1
    LONG = 2
