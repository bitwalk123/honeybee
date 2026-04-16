from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from structs.app_enum import PositionType


@dataclass
class EnvData:
    # ====== パラメータ ======
    # 売買系
    MAX_TRADE: int = 200  # 約定数上限（仮）
    # インジケータ系
    PERIOD_WARMUP: int = 300
    PERIOD_MA_1: int = 30
    PERIOD_MA_2: int = 300
    N_MINUS_MAX: int = 300
    # 報酬系
    REWARD_CROSS_ENTRY: float = 0.5  # クロス・シグナル時のエントリで報酬
    RATIO_PROFIT_HOLD: float = 0.01  # HOLD（建玉あり）時の含み損益からの報酬比率
    RATIO_PROFIT_CHANGE_HOLD: float = 0.005  # HOLD（建玉あり）時の含み損益変化度からの報酬比率
    COST_CONTRACT: float = 1.0  # 約定手数料（スリッページ相当）
    NUMERATOR_TERMINATION: float = 1.e3  # 早期終了時のペナルティ（分子/ステップ数）

    # インスタンス変数系（初期値が自明な変数のみ）
    row: int = 0  # ティックデータの行位置
    position: PositionType = PositionType.NONE  # ポジション
    n_trade: int = 0  # 約定回数
    count_negative: int = 0  # 含み損の継続カウンタ
    pnl_total = 0  # エピソードにおける総報酬
    dict_reward = defaultdict(list)  # 報酬保持用辞書 → 最後にデータフレーム化

    ts_open: float = 0.0
    price_open: float = 0.0
    diff_ma_pre: float = 0.0
    diff_vwap_pre: float = 0.0
    profit_pre: float = 0.0  # 一つ前の含み損益

    # フラグ関連
    flag_losscut_consecutive = False

    # ====== マスク処理関連 ======
    MASK_HOLD_ONLY = np.array([True, False, False], dtype=np.bool_)
    # 取りうるアクション: HOLD, BUY, SELL
    MASK_ALL = np.array([True, True, True], dtype=np.bool_)
    # 取りうるアクション: HOLD, SELL
    MASK_LONG = np.array([True, False, True], dtype=np.bool_)
    # 取りうるアクション: HOLD, BUY
    MASK_SHORT = np.array([True, True, False], dtype=np.bool_)

    POSITION_MASKS = {
        # 建玉なし
        PositionType.NONE: MASK_ALL,
        # LONG
        PositionType.LONG: MASK_LONG,
        # SHORT
        PositionType.SHORT: MASK_SHORT,
    }
