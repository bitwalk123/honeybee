from enum import IntEnum


class ActionType(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


class PositionType(IntEnum):
    SHORT = 0
    NONE = 1
    LONG = 2
