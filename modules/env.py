"""
Reference:
https://gymnasium.farama.org/introduction/create_custom_env/
"""
from collections import defaultdict

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
        self.CODE: str = code  # 銘柄コード
        self.df_tick: pd.DataFrame = df_tick
        self.render_mode = render_mode

        # ====== 報酬パラメータ ======
        self.PERIOD_WARMUP: int = 300
        self.PERIOD_MA_1: int = 30
        self.N_MINUS_MAX: int = 300
        self.RATIO_PROFIT_HOLD: float = 0.015  # HOLD（建玉あり）時の含み損益からの報酬比率
        self.COST_CONTRACT: float = 1.0  # 約定手数料（スリッページ相当）
        self.NUMERATOR_TERMINATION: float = 1.e3  # 早期終了時のペナルティ（分子/ステップ数）

        # 定数
        self.MAX_TRADE: int = 200  # 約定数上限

        # インスタンス変数の初期化
        self.row: int = 0  # ティックデータの行位置
        # 寄り付き価格の取得
        _, self.price0, _, _ = self.get_data(0)
        self.position: PositionType = PositionType.NONE  # ポジション
        self.profit: float = 0.0  # 含み損益
        self.n_trade: int = 0  # 約定回数
        self.count_negative: int = 0  # 含み損の継続カウンタ
        # 報酬系
        self.pnl_total = 0  # エピソードにおける総報酬
        self.dict_reward = defaultdict(list)  # 報酬保持用辞書 → 最後にデータフレーム化

        # ポジション・マネージャ
        self.posman = posman = PositionManager()
        posman.initPosition([self.CODE])

        # ====== Define action_space（行動空間） ======
        n_action_space = len(ActionType)
        self.action_space = spaces.Discrete(n_action_space)

        # 必要な観測値を追加
        # 短周期移動平均 MA1
        ma1 = MovingAverage(window_size=self.PERIOD_MA_1)
        df_tick["MA1"] = [ma1.update(p) for p in df_tick["Price"]]
        # 出来高加重平均価格
        vwap = VWAP()
        df_tick["VWAP"] = [vwap.update(p, v) for p, v in zip(df_tick["Price"], df_tick["Volume"])]
        # 乖離度 (MA1 - VWAP) / VWAP
        df_tick["DiffVWAP"] = (df_tick["MA1"] - df_tick["VWAP"]) / df_tick["VWAP"]

        print(df_tick.tail())

        # ====== Define observation_space（観測値空間） ======
        """
        【観測値】- VecNormalize Wrapper を使用する前提
        [market]
        1. Price（株価）
        2. MA1（短周期移動平均）
        3. DiffVWAP（乖離率 - (MA1 - VWAP) / VWAP）
        4. Profit（含み損益）
        5. penalty_negative（含み損保持ペナルティ）
        [position]
        a. SHORT
        b. NONE
        c. LONG
        """
        self.observation_space = spaces.Dict({
            "market": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "position": spaces.MultiBinary(3),  # one-hot
        })

    def action_masks(self) -> np.ndarray:
        """
        行動マスク
        【マスク】
        - ウォーミングアップ期間 → 強制 HOLD
        - ナンピン取引の禁止

        （参考）
        class ActionType(Enum):
            HOLD = 0
            BUY = 1
            SELL = 2

        :return: mask
        """
        if self.row < self.PERIOD_WARMUP:
            # ウォーミングアップ期間 → 強制 HOLD
            mask = np.array([True, False, False], dtype=np.bool_)
        elif self.position == PositionType.NONE:
            # 建玉なし → 取りうるアクション: HOLD, BUY, SELL
            mask = np.array([True, True, True], dtype=np.bool_)
        elif self.position == PositionType.LONG:
            # 建玉あり LONG → 取りうるアクション: HOLD, SELL
            mask = np.array([True, False, True], dtype=np.bool_)
        elif self.position == PositionType.SHORT:
            # 建玉あり SHORT → 取りうるアクション: HOLD, BUY
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
        list_name = ["Time", "Price", "MA1", "DiffVWAP"]
        return tuple(self.df_tick.iloc[row][list_name])

    def get_reward(self) -> pd.DataFrame:
        """
        ステップ毎に辞書に保持していた報酬情報をデータフレームに変換
        :return:
        """
        # df = pd.DataFrame(self.dict_reward)
        # タイムスタンプを datetime.datetime 型に変換
        # df["DateTime"] = [datetime.datetime.fromtimestamp(t) for t in df["ts"]]
        # return df[["DateTime", "reward"]]
        return pd.DataFrame(self.dict_reward)

    def get_transaction_result(self) -> pd.DataFrame:
        """
        取引結果
        :return:
        """
        # ポジション・マネージャから取引明細をデーテフレームで取得
        return self.posman.getTransactionResult()

    def init_status(self) -> None:
        """
        初期化処理
        :return:
        """
        # インスタンス変数の初期化
        self.row: int = 0  # ティックデータの行位置
        # 寄り付き価格の取得
        _, self.price0, _, _ = self.get_data(0)
        self.position: PositionType = PositionType.NONE  # ポジション
        self.profit: float = 0.0  # 含み損益
        self.n_trade: int = 0  # 約定回数
        self.count_negative: int = 0  # 含み損の継続カウンタ
        # 報酬系
        self.pnl_total = 0  # エピソードにおける総報酬
        self.dict_reward = defaultdict(list)  # 報酬保持用辞書 → 最後にデータフレーム化

        # ポジション・マネージャのリセットと初期化
        self.posman.reset()
        self.posman.initPosition([self.CODE])

    def reset(self, seed=None, options=None):
        """
        環境のリセット処理
        :param seed: 乱数シードの設定
        :param options: 追加オプション
        :return: observation, info
        """

        # Gymnasiumの仕様に従ってseedを設定し、乱数生成器を取得
        super().reset(seed=seed)

        # 環境の初期化（常に寄り付きから開始）
        self.init_status()

        # データフレームの最初の行のデータを取得
        _, price, ma1, diff_vwap = self.get_data(0)
        # 含み損益
        profit = 0

        # ====== 観測値（状態） ======
        market = np.array(
            [
                price - self.price0,
                ma1 - self.price0,
                diff_vwap,
                profit,
                0.0
            ],
            dtype=np.float32
        )
        position = position_to_onehot(self.position).astype(np.float32)  # shape (3,)
        obs = {"market": market, "position": position}

        info = {}  # Additional debug info
        return obs, info

    def step(self, action):
        """
        ステップ処理
        :param action:
        :return:
        """
        # データフレームからデータを一行分取得
        ts, price, ma1, diff_vwap = self.get_data(self.row)
        # 含み損益の取得
        profit = self.posman.getProfit(self.CODE, price)
        # 初期報酬
        reward = 0

        # ====== 建玉管理 ======
        action_type = ActionType(action)
        if action_type == ActionType.BUY:
            if self.position == PositionType.NONE:
                # 【買建】建玉がなければ買建
                self.posman.openPosition(self.CODE, ts, price, action_type)
                self.position = PositionType.LONG  # ポジションを更新
                self.n_trade += 1  # 取引回数の更新
                # 【報酬】
                reward -= self.COST_CONTRACT  # 約定コスト
                # 買建用 VWAP 判定
                reward -= diff_vwap  # diff_vwap が負の時に買建すれば報酬
            elif self.position == PositionType.SHORT:
                # 【返済】売建（ショート）であれば（買って）返済
                self.posman.closePosition(self.CODE, ts, price)
                self.position = PositionType.NONE  # ポジションを更新
                self.n_trade += 1  # 取引回数の更新
                # 【報酬】
                reward -= self.COST_CONTRACT  # 約定コスト
                reward += profit  # 含み損益分そっくり報酬
            else:
                raise RuntimeError("Trade rule violation!")
        elif action_type == ActionType.SELL:
            if self.position == PositionType.NONE:
                # 【売建】建玉がなければ売建
                self.posman.openPosition(self.CODE, ts, price, action_type)
                self.position = PositionType.SHORT  # ポジションを更新
                self.n_trade += 1  # 取引回数の更新
                # 【報酬】
                reward -= self.COST_CONTRACT  # 約定コスト
                # 売建用 VWAP 判定
                reward += diff_vwap  # diff_vwap が正の時に売建すれば報酬
            elif self.position == PositionType.LONG:
                # 【返済】買建（ロング）であれば（売って）返済
                self.posman.closePosition(self.CODE, ts, price)
                self.position = PositionType.NONE  # ポジションを更新
                self.n_trade += 1  # 取引回数の更新
                # 【報酬】
                reward -= self.COST_CONTRACT  # 約定コスト
                reward += profit  # 含み損益分そっくり報酬
            else:
                raise RuntimeError("Trade rule violation!")
        elif action_type == ActionType.HOLD:
            if self.position != PositionType.NONE:
                # 含み益があれば幾分かを報酬に
                reward += profit * self.RATIO_PROFIT_HOLD
        else:
            raise TypeError(f"Unknown ActionType: {action_type}!")

        # ====== 含み益評価 ======
        if profit < 0:
            self.count_negative += 1
        else:
            self.count_negative = 0
        penalty_negative = - (float(self.count_negative) / self.N_MINUS_MAX) ** 2
        reward += penalty_negative

        # ====== エピソード終了判定 ======
        terminated = False  # Task finished (e.g., goal reached)
        truncated = False  # Time limit reached
        info = {}
        if len(self.df_tick) - 1 <= self.row:
            # ティックデータの末尾
            if self.posman.hasPosition(self.CODE):
                # 建玉があれば強制返済
                self.posman.closePosition(self.CODE, ts, price, "強制返済")
                self.position = PositionType.NONE  # ポジションを更新
                self.n_trade += 1  # 取引回数の更新
                # 【報酬】
                reward -= self.COST_CONTRACT  # 約定コスト
                reward += profit * (1 - self.RATIO_PROFIT_HOLD)  # 残りの含み損益分

            truncated = True  # ← ステップ数上限による終了
            info["done_reason"] = "truncated: last_tick"
            # 取引情報（データフレーム）
            info["transaction"] = self.get_transaction_result()
            # 報酬情報（データフレーム）
            info["reward"] = self.get_reward()
            print(f"約定回数 : {self.n_trade}")

        # モデル報酬の保持（分析用）
        self.dict_reward["ts"].append(ts)
        self.dict_reward["reward"].append(reward)

        # ステップ（データフレームの行）更新
        self.row += 1

        # ====== 観測値（状態） ======
        market = np.array(
            [
                price - self.price0,
                ma1 - self.price0,
                diff_vwap,
                profit,
                penalty_negative
            ],
            dtype=np.float32
        )
        position = position_to_onehot(self.position).astype(np.float32)
        obs = {"market": market, "position": position}

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        # Implement visualization logic based on self.render_mode
        pass

    def close(self) -> None:
        # Cleanup resources (e.g., close windows)
        pass
