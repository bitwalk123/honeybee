import math
import numpy as np
import pandas as pd

from funcs.conv import position_to_onehot
from modules.env_training import TrainingEnv
from structs.app_enum import ActionType, PositionType


class InferenceEnv(TrainingEnv):
    def __init__(self, code: str, df_tick: pd.DataFrame, render_mode=None) -> None:
        super().__init__(code, df_tick, render_mode)
