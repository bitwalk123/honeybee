from enum import Enum


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class PositionType(Enum):
    SHORT = -1
    NONE = 0
    LONG = 1
