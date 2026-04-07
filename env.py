"""
Reference:
https://gymnasium.farama.org/introduction/create_custom_env/
"""
import datetime
from collections import defaultdict
from typing import Any

import gymnasium as gym
import pandas as pd
from gymnasium import spaces
import numpy as np

from modules.posman import PositionManager
from modules.technical import MovingAverage, VWAP
from structs.app_enum import ActionType, PositionType


def position_to_onehot(pos: PositionType) -> np.ndarray:
    onehot = np.zeros(3, dtype=np.float32)
    onehot[int(pos)] = 1.0
    return onehot


class TrainingEnv(gym.Env):
    # metadata defines render modes and framerate
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, code: str, df_tick: pd.DataFrame, render_mode=None) -> None:
        super().__init__()
        self.df_tick: pd.DataFrame = df_tick
        self.render_mode = render_mode

        # 報酬保持用辞書 → 最後にデータフレーム化
        self.dict_reward = defaultdict(list)

        # 報酬関連
        self.pnl_total = 0
        self.ratio_profit_hold = 0.01  # HOLD（建玉あり）時の含み損益からの報酬比率
        self.cost_contract = 1  # 約定手数料（スリッページ相当）

        # インスタンス変数の初期化
        self.code: str = code
        self.row: int = 0
        self.position: PositionType = PositionType.NONE
        self.profit: float = 0.0

        # ポジション・マネージャ
        self.posman = posman = PositionManager()
        posman.initPosition([self.code])

        # Define action_space（行動空間）
        n_action_space = len(ActionType)
        self.action_space = spaces.Discrete(n_action_space)

        # 必要な観測値を追加
        ma1 = MovingAverage(window_size=30)
        df_tick["MA1"] = [ma1.update(p) for p in df_tick["Price"]]
        vwap = VWAP()
        df_tick["VWAP"] = [vwap.update(p, v) for p, v in zip(df_tick["Price"], df_tick["Volume"])]
        df_tick["Diff"] = (df_tick["MA1"] - df_tick["VWAP"]) / df_tick["VWAP"]

        print(df_tick.tail())

        # Define observation_space（観測値空間）
        """
        【観測値】- VecNormalize Wrapper を使用する前提
        1. Price（株価）
        2. Profit（含み損益）
        3. Diff（乖離率 - (MA1 - VWAP) / VWAP）
        4. SHORT
        5. NONE
        6. LONG
        """
        """
        self.observation_space = spaces.Box(
            low=np.array([
                -np.float32('inf'),
                -np.float32('inf'),
                -np.float32('inf'),
                np.float32(0),
                np.float32(0),
                np.float32(0),
            ]),
            high=np.array([
                np.float32('inf'),
                np.float32('inf'),
                np.float32('inf'),
                np.float32(1),
                np.float32(1),
                np.float32(1),
            ]),
            shape=(6,),
            dtype=np.float32
        )
        """
        self.observation_space = spaces.Dict({
            "market": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),  # [price, diff, profit]
            "position": spaces.MultiBinary(3),  # one-hot
        })

    '''
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
    '''

    # TrainingEnv.action_masks
    def action_masks(self) -> np.ndarray:
        if self.position == PositionType.NONE:
            mask = np.array([True, True, True], dtype=np.bool_)
        elif self.position == PositionType.LONG:
            mask = np.array([True, False, True], dtype=np.bool_)
        elif self.position == PositionType.SHORT:
            mask = np.array([True, True, False], dtype=np.bool_)
        else:
            raise TypeError(f"Unknown PositionType: {self.position}")
        return mask

    def get_data(self, row: int) -> tuple:
        """
        ティックデータから一行抽出
        :param row:
        :return:
        """
        return self.df_tick.iloc[row][["Time", "Price", "Diff"]]

    def get_reward(self) -> pd.DataFrame:
        df = pd.DataFrame(self.dict_reward)
        df["DateTime"] = [datetime.datetime.fromtimestamp(t) for t in df["ts"]]
        return df[["DateTime", "reward"]]

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

    def reset(self, seed=None, options=None):
        """
        環境のリセット処理
        :param seed:
        :param options:
        :return:
        """
        # Mandatory: seed the random number generator
        super().reset(seed=seed)

        # Initialize your state
        _, price, diff = self.get_data(0)
        profit = 0
        """
        observation = np.array([price, diff, profit], dtype=np.float32)
        pos_onehot = position_to_onehot(PositionType.NONE)
        obs = np.concatenate([observation, pos_onehot])
        """
        market = np.array([price, diff, profit], dtype=np.float32)
        position = position_to_onehot(self.position).astype(np.float32)  # shape (3,)
        obs = {"market": market, "position": position}
        info = {}  # Additional debug info
        self.init_status()
        return obs, info

    def step(self, action):
        """
        ステップ処理
        :param action:
        :return:
        """
        # データを一行分取得
        ts, price, diff = self.get_data(self.row)

        # 含み損益
        profit = self.posman.getProfit(self.code, price)

        # 観測値
        # observation = np.array([price, diff, profit], dtype=np.float32)

        # 報酬
        reward = 0

        # 建玉管理
        action_type = ActionType(action)
        if action_type == ActionType.BUY:
            if self.position == PositionType.NONE:
                # 【買建】建玉がなければ買建
                self.posman.openPosition(self.code, ts, price, action_type)
                self.position = PositionType.LONG  # ポジションを更新
                reward -= self.cost_contract  # 約定コスト
                # 買建用 VWAP 判定
                reward -= diff  # diff が負の時に買建すれば報酬
            elif self.position == PositionType.SHORT:
                # 【返済】売建（ショート）であれば（買って）返済
                self.posman.closePosition(self.code, ts, price)
                self.position = PositionType.NONE  # ポジションを更新
                reward -= self.cost_contract  # 約定コスト
                reward += profit  # 含み損益分そっくり報酬
            else:
                #raise "trade rule violation!"
                raise RuntimeError("trade rule violation!")
        elif action_type == ActionType.SELL:
            if self.position == PositionType.NONE:
                # 【売建】建玉がなければ売建
                self.posman.openPosition(self.code, ts, price, action_type)
                self.position = PositionType.SHORT  # ポジションを更新
                reward -= self.cost_contract  # 約定コスト
                # 売建用 VWAP 判定
                reward += diff  # diff が正の時に売建すれば報酬
            elif self.position == PositionType.LONG:
                # 【返済】買建（ロング）であれば（売って）返済
                self.posman.closePosition(self.code, ts, price)
                self.position = PositionType.NONE  # ポジションを更新
                reward -= self.cost_contract  # 約定コスト
                reward += profit  # 含み損益分そっくり報酬
            else:
                raise "trade rule violation!"
        elif action_type == ActionType.HOLD:
            if self.position != PositionType.NONE:
                # 含み益があれば幾分かを報酬に
                reward += profit * self.ratio_profit_hold
        else:
            raise f"unknown action type {action_type}!"

        # エピソード終了判定
        terminated = False  # Task finished (e.g., goal reached)
        truncated = False  # Time limit reached
        info = {}
        if len(self.df_tick) - 1 <= self.row:
            if self.posman.hasPosition(self.code):
                reward -= self.cost_contract  # 約定コスト
                reward += profit * (1 - self.ratio_profit_hold)  # 残りの含み損益分
                self.posman.closePosition(self.code, ts, price, "強制返済")
                self.position = PositionType.NONE  # ポジションを更新

            truncated = True  # ← ステップ数上限による終了
            info["done_reason"] = "truncated: last_tick"
            info["transaction"] = self.get_transaction_result()
            info["reward"] = self.get_reward()

        # モデル報酬の保持
        self.dict_reward["ts"].append(ts)
        self.dict_reward["reward"].append(reward)

        self.row += 1
        """
        pos_onehot = position_to_onehot(self.position)
        obs = np.concatenate([observation, pos_onehot])
        """
        # 既存ロジックで price,diff,profit を計算
        market = np.array([price, diff, profit], dtype=np.float32)
        position = position_to_onehot(self.position).astype(np.float32)
        obs = {"market": market, "position": position}
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        # Implement visualization logic based on self.render_mode
        pass

    def close(self) -> None:
        # Cleanup resources (e.g., close windows)
        pass
