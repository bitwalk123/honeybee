import gymnasium as gym
import math
import numpy as np
import pandas as pd
from gymnasium import spaces

from funcs.conv import position_to_onehot
from modules.env_data import EnvData
from modules.posman import PositionManager
from modules.technical import MovingAverage, VWAP
from structs.app_enum import ActionType, PositionType


class TrainingEnv(gym.Env):
    # metadata defines render modes and framerate
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, code: str, df_tick: pd.DataFrame, render_mode=None) -> None:
        super().__init__()
        self.CODE: str = code  # 銘柄コード
        self.df_tick: pd.DataFrame = df_tick
        self.render_mode = render_mode

        # データクラスのインスタンスを定義
        self.s = EnvData()

        # ====== データフレームに必要な観測値を追加 ======
        self._prep_observations()

        # ポジション・マネージャ
        self.posman = posman = PositionManager()
        posman.initPosition([self.CODE])

        # ====== 行動空間 action_space の定義 ======
        n_action_space = len(ActionType)
        self.action_space = spaces.Discrete(n_action_space)

        # ====== 観測（特徴量）空間 observation_space の定義 ======
        """
        【観測値】- VecNormalize Wrapper を使用する前提
        [market] - VecNormalize Wrapper で標準化
        1. Price（株価）
        2. MA1（短周期移動平均）
        3. MA2（長周期移動平均）
        4. VWAP（VWAP）
        5. Profit（含み損益）
        [cross] - 符号が重要であるため標準化しない
        1. DiffMA（乖離率 : (MA1 - MA2) / MA2）
        2. DiffVWAP（乖離率 : (MA1 - VWAP) / VWAP）
        [counter] - VecNormalize Wrapper で標準化
        1. n_trade（約定回数）
        2. count_negative（含み損の継続カウンタ）
        [position] - 標準化不要
        1. SHORT
        2. NONE
        3. LONG
        """
        self.observation_space = spaces.Dict({
            "market": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "cross": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "counter": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32),
            "position": spaces.MultiBinary(3),  # one-hot
        })

        # デバッグ用観測値
        self.obs = {}

    def _prep_observations(self):
        # 短周期移動平均 MA1
        ma1 = MovingAverage(window_size=self.s.PERIOD_MA_1)
        self.df_tick["MA1"] = [ma1.update(p) for p in self.df_tick["Price"]]
        ma2 = MovingAverage(window_size=self.s.PERIOD_MA_2)
        self.df_tick["MA2"] = [ma2.update(p) for p in self.df_tick["Price"]]
        self.df_tick["DiffMA"] = (self.df_tick["MA1"] - self.df_tick["MA2"]) / self.df_tick["MA2"]
        # 出来高加重平均価格
        vwap = VWAP()
        self.df_tick["VWAP"] = [vwap.update(p, v) for p, v in zip(self.df_tick["Price"], self.df_tick["Volume"])]
        # 乖離度 (MA1 - VWAP) / VWAP
        self.df_tick["DiffVWAP"] = (self.df_tick["MA1"] - self.df_tick["VWAP"]) / self.df_tick["VWAP"]

    def action_masks(self) -> np.ndarray:
        """
        行動マスク
        【マスク】
        - ウォーミングアップ期間 → 強制 HOLD
        - ナンピン取引の禁止

        :return: mask
        """
        if self.s.row < self.s.PERIOD_WARMUP:
            # ウォーミングアップ期間 → 強制 HOLD
            return self.s.MASK_HOLD_ONLY
        try:
            return self.s.POSITION_MASKS[self.s.position]
        except KeyError:
            raise TypeError(f"Unknown PositionType: {self.s.position}")

    def get_data(self, row: int) -> tuple:
        """
        ティックデータから一行抽出
        :param row:
        :return:
        """
        list_name = ["Time", "Price", "MA1", "MA2", "DiffMA", "VWAP", "DiffVWAP"]
        return tuple(self.df_tick.iloc[row][list_name])

    def get_obs(self):
        return self.obs

    def get_reward(self) -> pd.DataFrame:
        """
        ステップ毎に辞書に保持していた報酬情報をデータフレームに変換
        :return:
        """
        return pd.DataFrame(self.s.dict_reward)

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
        # データクラスのインスタンスを再定義
        self.s = EnvData()

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
        super().reset(seed=int(np.random.rand() * 1000))

        # 環境の初期化（常に寄り付きから開始）
        self.init_status()

        # ====== 観測値（状態） ======
        market = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32
        )
        cross = np.array([0, 0], dtype=np.float32)
        counter = np.array([0, 0], dtype=np.float32)
        position = position_to_onehot(self.s.position).astype(np.float32)
        obs = {"market": market, "cross": cross, "counter": counter, "position": position}

        info = {}  # Additional debug info
        return obs, info

    def step(self, action):
        """
        ステップ処理
        :param action:
        :return:
        """
        # ====== データフレームからデータを一行分取得 ======
        ts, price, ma1, ma2, diff_ma, vwap, diff_vwap = self.get_data(self.s.row)
        # 含み損益の取得
        profit = self.posman.getProfit(self.CODE, price)
        # 初期報酬
        reward = 0
        # 情報用辞書
        info = {}

        # ====== 建玉管理 ======
        action_type = ActionType(action)
        if action_type == ActionType.BUY:
            if self.s.position == PositionType.NONE:
                # 【買建】建玉がなければ買建
                self.posman.openPosition(self.CODE, ts, price, action_type)
                self.s.position = PositionType.LONG  # ポジションを更新
                self.s.n_trade += 1  # 取引回数の更新
                self.s.profit_pre = 0.0  # 一つ前の含み益

                # 【報酬・ペナルティ】
                reward -= self.s.COST_CONTRACT  # 約定コスト

                # ゴールデン・クロス時のエントリ報酬（約定コスト相殺＋α）
                if self.s.diff_ma_pre < 0 < diff_ma:
                    reward += self.s.COST_CONTRACT + self.s.REWARD_CROSS_ENTRY
                if self.s.diff_vwap_pre < 0 < diff_vwap:
                    reward += self.s.COST_CONTRACT + self.s.REWARD_CROSS_ENTRY

            elif self.s.position == PositionType.SHORT:
                # 【返済】売建（ショート）であれば（買って）返済
                reward += self.position_close(ts, price, profit)

            else:
                raise RuntimeError("Trade rule violation!")

        elif action_type == ActionType.SELL:
            if self.s.position == PositionType.NONE:
                # 【売建】建玉がなければ売建
                self.posman.openPosition(self.CODE, ts, price, action_type)
                self.s.position = PositionType.SHORT  # ポジションを更新
                self.s.n_trade += 1  # 取引回数の更新
                self.s.profit_pre = 0.0  # 一つ前の含み益

                # 【報酬・ペナルティ】
                reward -= self.s.COST_CONTRACT  # 約定コスト

                # デッド・クロス時のエントリ報酬（約定コスト相殺＋α）
                if diff_ma < 0 < self.s.diff_ma_pre:
                    reward += self.s.COST_CONTRACT + self.s.REWARD_CROSS_ENTRY
                if diff_vwap < 0 < self.s.diff_vwap_pre:
                    reward += self.s.COST_CONTRACT + self.s.REWARD_CROSS_ENTRY

            elif self.s.position == PositionType.LONG:
                # 【返済】買建（ロング）であれば（売って）返済
                reward += self.position_close(ts, price, profit)

            else:
                raise RuntimeError("Trade rule violation!")

        elif action_type == ActionType.HOLD:
            if self.s.position != PositionType.NONE:
                # 【報酬・ペナルティ】
                # 含み益があれば幾分かを報酬に
                reward += profit * self.s.RATIO_PROFIT_HOLD
                # 含み益の増減に応じて幾分かを報酬に
                reward += (profit - self.s.profit_pre) * self.s.RATIO_PROFIT_CHANGE_HOLD
        else:
            raise TypeError(f"Unknown ActionType: {action_type}!")

        # ====== 連続含み益評価 ======
        if profit < 0:
            self.s.count_negative += 1
        else:
            self.s.count_negative = 0
        penalty_negative = - (float(self.s.count_negative) / self.s.N_MINUS_MAX) ** 2
        reward += penalty_negative
        if self.s.count_negative > self.s.N_MINUS_MAX:
            self.s.flag_losscut_consecutive = True
        else:
            self.s.flag_losscut_consecutive = False

        # ====== エピソード終了判定 ======
        terminated = False  # Task finished (e.g., goal reached)
        truncated = False  # Time limit reached

        if len(self.df_tick) - 1 <= self.s.row:
            # ティックデータの末尾
            if self.posman.hasPosition(self.CODE):
                # 建玉があれば強制返済
                reward += self.position_close_force(ts, price, profit)

            # 約定回数に応じた報酬（n で極大, r_max が最高報酬）
            n = 25
            r_max = 10.0
            reward += r_max * self.s.n_trade * math.e ** (1 - self.s.n_trade / n) / n

            truncated = True  # ← ステップ数上限による終了
            info["done_reason"] = "truncated: last_tick"
            # 取引情報（データフレーム）
            info["transaction"] = self.get_transaction_result()
            # 報酬情報（データフレーム）
            info["reward"] = self.get_reward()
            print(f"約定回数 : {self.s.n_trade}")

        # ====== 観測値（状態） ======
        market = np.array(
            [
                price,
                ma1,
                ma2,
                vwap,
                profit,
            ],
            dtype=np.float32
        )
        cross = np.array(
            [
                diff_ma,
                diff_vwap,
            ],
            dtype=np.float32
        )
        counter = np.array(
            [
                self.s.n_trade,
                self.s.count_negative
            ],
            dtype=np.float32
        )
        position = position_to_onehot(self.s.position).astype(np.float32)
        self.obs = obs = {
            "market": market,
            "cross": cross,
            "counter": counter,
            "position": position,
        }

        # 一つ前の値を更新
        self.s.profit_pre = profit  # 一つ前の含み益の更新
        self.s.diff_ma_pre = diff_ma
        self.s.diff_vwap_pre = diff_vwap

        # ステップ（データフレームの行）更新
        self.s.row += 1

        # ====== モデル報酬の保持（分析用） ======
        self.s.dict_reward["ts"].append(ts)
        self.s.dict_reward["reward"].append(reward)

        # ====== テクニカル分析用の情報 ======
        dict_technical = {
            "ts": ts,
            "price": price,
            "ma1": ma1,
            "ma2": ma2,
            "vwap": vwap,
            "profit": profit,
            "diff_ma": diff_ma,
            "diff_vwap": diff_vwap,
            "n_trade": self.s.n_trade,
            "count_negative": self.s.count_negative,
        }
        info["technical"] = dict_technical

        return obs, reward, terminated, truncated, info

    def position_close(self, ts: float, price: float, profit: float) -> int:
        """
        ポジション・クローズ
        :param ts:
        :param price:
        :param profit:
        :return:
        """
        # ポジション管理
        self.posman.closePosition(self.CODE, ts, price)
        self.s.position = PositionType.NONE  # ポジションを更新
        self.s.n_trade += 1  # 取引回数の更新
        self.s.profit_pre = 0.0  # 一つ前の含み益
        # 【報酬】
        r = 0
        r -= self.s.COST_CONTRACT  # 約定コスト
        r += profit  # 含み損益分そっくり報酬
        # 連続含み損
        if self.s.flag_losscut_consecutive:
            # ロスカットに対して約定コストを相殺＋αの報酬
            r += self.s.COST_CONTRACT + 0.5
        self.s.flag_losscut_consecutive = False
        self.s.count_negative = 0
        return r

    def position_close_force(self, ts: float, price: float, profit: float) -> int:
        """
        ポジション・クローズ（強制）
        :param ts:
        :param price:
        :param profit:
        :return:
        """
        # ポジション管理
        self.posman.closePosition(self.CODE, ts, price, "強制返済")
        self.s.position = PositionType.NONE  # ポジションを更新
        self.s.n_trade += 1  # 取引回数の更新
        self.s.profit_pre = 0.0  # 一つ前の含み益
        # 【報酬】
        r = 0
        r -= self.s.COST_CONTRACT  # 約定コスト
        r += profit  # 含み損益分そっくり報酬
        return r

    def render(self) -> None:
        # Implement visualization logic based on self.render_mode
        pass

    def close(self) -> None:
        # Cleanup resources (e.g., close windows)
        pass
