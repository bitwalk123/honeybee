from collections import defaultdict
from dataclasses import dataclass, field

import math
import numpy as np
import pandas as pd

from funcs.conv import position_to_onehot
from structs.app_enum import PositionType


@dataclass
class EnvData:
    # ====== パラメータ ======
    # 売買系
    MAX_TRADE: int = 200  # 約定数上限（仮）
    # インジケータ系
    PERIOD_WARMUP: int = 300  # インジケータのウォームアップ期間（ティック数）
    PERIOD_HOLD: int = 5  # 約定後に HOLD に固定する期間（ティック数）
    PERIOD_MA_1: int = 90  # 移動平均線の期間1
    PERIOD_MA_2: int = 900  # 移動平均線の期間2
    N_MINUS_MAX: int = 300  # 連続含み損の最大カウント数
    LOSSCUT_1: float = -25.0  # 単純ロスカット
    DD_RATIO_MAX: float = 0.5  # ドローダウン利確の最大比率
    DD_THRESHOLD: float = 10.0  # ドローダウン利確の閾値
    # 報酬系
    REWARD_CROSS_ENTRY: float = 0.5  # クロス・シグナル時のエントリで報酬
    RATIO_PROFIT_HOLD: float = 0.025  # HOLD（建玉あり）時の含み損益からの報酬比率
    RATIO_PROFIT_CHANGE_HOLD: float = 0.0025  # HOLD（建玉あり）時の含み損益変化度からの報酬比率
    COST_CONTRACT: float = 1.0  # 約定コスト（スリッページ相当）
    NUMERATOR_TERMINATION: float = 1.e3  # 早期終了時のペナルティ（分子/ステップ数）
    # 学習用ティックデータの報酬分布用の列名
    COL_CROSS_MA_GOLDEN: str = "cross_ma_golden"
    COL_CROSS_MA_DEAD: str = "cross_ma_dead"

    # インスタンス変数系（初期値が自明な変数のみ）
    row: int = 0  # ティックデータの行位置
    position: PositionType = PositionType.NONE  # ポジション
    n_trade: int = 0  # 約定回数
    count_negative: int = 0  # 含み損の継続カウンタ
    count_post_contract: int = 0  # 約定後の HOLD カウント用
    pnl_total: float = 0  # エピソードにおける総報酬
    # dict_reward = defaultdict(list)  # 報酬保持用辞書 → 最後にデータフレーム化
    dict_reward: dict = field(default_factory=lambda: defaultdict(list))

    ts: float = 0.0
    price: float = 0.0
    ma1: float = 0.0
    ma2: float = 0.0
    diff_ma: float = 0.0
    vwap: float = 0.0
    diff_vwap: float = 0.0
    profit: float = 0.0  # 含み損益
    profit_max: float = 0.0  # 最大含み損益
    dd_ratio: float = 0.0  # ドローダウン比率

    ts_open: float = 0.0
    price_open: float = 0.0
    volume_open: float = 0.0
    # diff_ma_pre: float = 0.0
    # diff_vwap_pre: float = 0.0
    profit_pre: float = 0.0  # 一つ前の含み損益

    # フラグ関連
    flag_losscut_consecutive: bool = False

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

    def does_take_profit(self) -> bool:
        if self.DD_THRESHOLD < self.profit and self.DD_RATIO_MAX < self.update_dd_ratio():
            return True
        else:
            return False

    def get_masks(self):
        """
        行動マスク
        【マスク】
        - ウォーミングアップ期間 → 強制 HOLD
        - ナンピン取引の禁止

        :return: mask
        """
        if self.row < self.PERIOD_WARMUP:
            # ウォーミングアップ期間 → 強制 HOLD
            return self.MASK_HOLD_ONLY

        if self.count_post_contract < self.PERIOD_HOLD:
            # 約定後の HOLD 期間 → 強制 HOLD
            return self.MASK_HOLD_ONLY

        try:
            return self.POSITION_MASKS[self.position]
        except KeyError:
            raise TypeError(f"Unknown PositionType: {self.position}")

    def get_obs(self) -> dict:
        """
        観測空間の算出
        :return:
        """
        """
        日毎に生じる絶対値のズレを少しでも抑えたい。
        そのため、株価に関連する特徴量に対して、始値で割っている。
        """
        if self.price_open > 0:
            market = np.array(
                [
                    self.ma1 / self.price_open,
                    self.profit,
                    self.profit_max,
                ],
                dtype=np.float32
            )
        else:
            market = np.array(
                [
                    1.0,
                    self.profit,
                    self.profit_max,
                ],
                dtype=np.float32
            )

        cross = np.array(
            [
                self.diff_ma,
                self.diff_vwap,
            ],
            dtype=np.float32
        )
        counter = np.array(
            [
                self.n_trade,
                self.count_negative,
                self.count_post_contract,
                self.dd_ratio,
            ],
            dtype=np.float32
        )
        position = position_to_onehot(self.position).astype(np.float32)
        obs = {
            "market": market,
            "cross": cross,
            "counter": counter,
            "position": position,
        }
        return obs

    def get_n_trade_reward(self) -> float:
        # 約定回数に応じた報酬（n で極大, r_max が最高報酬）
        n = 25
        r_max = 5.0  # 4/17 → 4/18
        return r_max * self.n_trade * math.e ** (1 - self.n_trade / n) / n

    def get_penalty_negative(self) -> float:
        return - (float(self.count_negative) / self.N_MINUS_MAX) ** 2

    def get_reward(self) -> pd.DataFrame:
        """
        ステップ毎に辞書に保持していた報酬情報をデータフレームに変換
        :return:
        """
        return pd.DataFrame(self.dict_reward)

    def get_reward_unrealized_profit(self) -> float:
        r = 0
        # 含み益があれば幾分かを報酬に
        r += self.profit * self.RATIO_PROFIT_HOLD
        # 含み益の増減に応じて幾分かを報酬に
        r += (self.profit - self.profit_pre) * self.RATIO_PROFIT_CHANGE_HOLD
        return r

    def get_technicals(self):
        return {
            "ts": self.ts,
            "price": self.price,
            "ma1": self.ma1,
            "ma2": self.ma2,
            "vwap": self.vwap,
            "profit": self.profit,
            "profit_max": self.profit_max,
            "dd_ratio": self.dd_ratio,
            "diff_ma": self.diff_ma,
            "diff_vwap": self.diff_vwap,
            "n_trade": self.n_trade,
            "count_negative": self.count_negative,
        }

    def inc_row(self):
        self.row += 1

    def get_count_post_contract(self) -> int:
        """ 約定後のカウント数を取得 """
        return self.count_post_contract

    def inc_count_post_contract(self) -> None:
        """ 約定後のカウント数をインクリメント """
        self.count_post_contract += 1

    def reset_count_post_contract(self) -> None:
        """ 約定後のカウント数をリセット """
        self.count_post_contract = 0

    def is_losscut(self) -> bool:
        return self.profit < self.LOSSCUT_1

    def reset_count_negative(self):
        self.count_negative = 0
        self.flag_losscut_consecutive = False

    def reset_profit_pre(self):
        self.profit_pre = 0.0

    def reset_profit_max(self):
        self.profit_max = 0.0

    def set_data(self, row):
        self.ts = row["Time"]
        self.price = row["Price"]
        self.ma1 = row["MA1"]
        self.ma2 = row["MA2"]
        self.diff_ma = row["DiffMA"]
        self.vwap = row["VWAP"]
        self.diff_vwap = row["DiffVWAP"]

    def set_data_open(self, row):
        self.ts_open = row["Time"]
        self.price_open = row["Price"]
        self.volume_open = row["Volume"]

    def update_count_negative(self):
        if self.profit < 0:
            self.count_negative += 1
        else:
            self.count_negative = 0

        if self.count_negative > self.N_MINUS_MAX:
            self.flag_losscut_consecutive = True
        else:
            self.flag_losscut_consecutive = False

    def update_dict_reward(self, reward) -> None:
        self.dict_reward["ts"].append(self.ts)
        self.dict_reward["reward"].append(reward)

    def update_dd_ratio(self) -> float:
        if 0 <= self.profit:
            if self.profit_max < self.profit:
                self.profit_max = self.profit
                self.dd_ratio = 0.0
            elif 0 < self.profit_max:
                # Drawdown Ratio
                self.dd_ratio = (self.profit_max - self.profit) / self.profit_max
            else:
                self.dd_ratio = 0.0
        else:
            self.dd_ratio = 0.0

        return self.dd_ratio

    def update_profit_pre(self):
        self.profit_pre = self.profit  # 一つ前の含み益の更新
