"""
Reference:
https://gymnasium.farama.org/introduction/create_custom_env/
"""
from typing import Any

import gymnasium as gym
import pandas as pd
from gymnasium import spaces
import numpy as np

from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType


class TrainingEnv(gym.Env):
    # metadata defines render modes and framerate
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, code: str, df: pd.DataFrame, render_mode=None) -> None:
        super().__init__()
        self.df: pd.DataFrame = df
        self.render_mode = render_mode

        # 報酬関連
        self.pnl_total = 0
        self.ratio_profit_hold = 0.1  # HOLD 時の含み損益から報酬
        self.cost_contract = 1  # 約定手数料（スリッページ相当）

        # ポジション・マネージャ
        self.posman = posman = PositionManager()
        posman.initPosition([code])

        # インスタンス変数の初期化
        self.code: str = code
        self.row: int = 0
        self.position: PositionType = PositionType.NONE
        self.profit: float = 0.0

        # Define action_space（行動空間）
        n_action_space = len(ActionType)
        self.action_space = spaces.Discrete(n_action_space)

        # Define observation_space（観測値空間）
        """
        0. 株価［最小値, 最大値］
        1. 含み益［-株価レンジ, +株価レンジ］
        """
        price_max = df["Price"].max()
        price_min = df["Price"].min()
        price_rng = price_max - price_min
        self.observation_space = spaces.Box(
            low=np.array([price_min, -price_rng]),
            high=np.array([price_max, +price_rng]),
            shape=(2,),
            dtype=np.float32
        )

    def action_masks(self) -> np.ndarray:
        """
        行動マスク
        【マスク】
        - ナンピン取引の禁止
        :return:
        """
        if self.position == PositionType.NONE:
            # 建玉なし → 取りうるアクション: HOLD, BUY, SELL
            return np.array([1, 1, 1], dtype=np.int8)
        elif self.position == PositionType.LONG:
            # 建玉あり LONG → 取りうるアクション: HOLD, SELL
            return np.array([1, 0, 1], dtype=np.int8)
        elif self.position == PositionType.SHORT:
            # 建玉あり SHORT → 取りうるアクション: HOLD, BUY
            return np.array([1, 1, 0], dtype=np.int8)
        else:
            raise TypeError(f"Unknown PositionType: {self.position}")

    def get_ts_price(self, row: int) -> tuple[float, float, float]:
        """
        ティックデータから一行抽出
        :param row:
        :return:
        """
        return self.df.iloc[row][["Time", "Price", "Volume"]]

    def get_transaction_result(self) -> pd.DataFrame:
        """
        取引結果
        :return:
        """
        return self.posman.getTransactionResult()

    def init_status(self) -> None:
        """
        初期化処理
        :return:
        """
        self.row = 0
        self.position = PositionType.NONE
        self.profit: float = 0.0
        self.pnl_total: float = 0.0
        # ポジション・マネージャのリセットと初期化
        self.posman.reset()
        self.posman.initPosition([self.code])

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        環境のリセット処理
        :param seed:
        :param options:
        :return:
        """
        # Mandatory: seed the random number generator
        super().reset(seed=seed)

        # Initialize your state
        _, price, volume = self.get_ts_price(0)
        profit = 0
        observation = np.array([price, profit], dtype=np.float32)
        info = {}  # Additional debug info
        self.init_status()
        return observation, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        ステップ処理
        :param action:
        :return:
        """
        # タイムスタンプと株価
        ts, price, volume = self.get_ts_price(self.row)

        # 含み損益
        profit = self.posman.getProfit(self.code, price)

        # 観測値
        observation = np.array([price, profit], dtype=np.float32)

        # 報酬
        reward = 0

        # 建玉管理
        action_type = ActionType(action)
        if action_type == ActionType.BUY:
            if self.position == PositionType.NONE:
                # 建玉がなければ買建
                self.posman.openPosition(self.code, ts, price, action_type)
                self.position = PositionType.LONG  # ポジションを更新
                reward -= self.cost_contract  # 約定コスト
            elif self.position == PositionType.SHORT:
                # 売建（ショート）であれば（買って）返済
                self.posman.closePosition(self.code, ts, price)
                self.position = PositionType.NONE  # ポジションを更新
                reward -= self.cost_contract  # 約定コスト
                reward += profit  # 含み損益分そっくり報酬
            else:
                raise "trade rule violation!"
        elif action_type == ActionType.SELL:
            if self.position == PositionType.NONE:
                # 建玉がなければ売建
                self.posman.openPosition(self.code, ts, price, action_type)
                self.position = PositionType.SHORT  # ポジションを更新
                reward -= self.cost_contract  # 約定コスト
            elif self.position == PositionType.LONG:
                # 買建（ロング）であれば（売って）返済
                self.posman.closePosition(self.code, ts, price)
                self.position = PositionType.NONE  # ポジションを更新
                reward -= self.cost_contract  # 約定コスト
                reward += profit  # 含み損益分そっくり報酬
            else:
                raise "trade rule violation!"
        elif action_type == ActionType.HOLD:
            # 含み益があれば幾分かを報酬に
            reward += profit * self.ratio_profit_hold
        else:
            raise f"unknown action type {action_type}!"

        # エピソード終了判定
        terminated = False  # Task finished (e.g., goal reached)
        truncated = False  # Time limit reached
        info = {}
        if len(self.df) - 1 <= self.row:
            if self.posman.hasPosition(self.code):
                reward -= self.cost_contract  # 約定コスト
                reward += profit * (1 - self.ratio_profit_hold)  # 残りの含み損益分
                self.posman.closePosition(self.code, ts, price, "強制返済")
                self.position = PositionType.NONE  # ポジションを更新

            truncated = True  # ← ステップ数上限による終了
            info["done_reason"] = "truncated: last_tick"
            info["transaction"] = self.get_transaction_result()

        self.row += 1
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        # Implement visualization logic based on self.render_mode
        pass

    def close(self) -> None:
        # Cleanup resources (e.g., close windows)
        pass
