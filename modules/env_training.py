import gymnasium as gym
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
        self.get_data_open()  # 始値の取得

        # ====== データフレームに必要な観測値を追加 ======
        self._prep_observations()
        # ====== 事前に定義できる報酬を算出 ======
        self._prep_rewards()

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
        1. MA1（短周期移動平均）
        2. Profit（含み損益）
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
            "market": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
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

        # 長周期移動平均 MA2
        ma2 = MovingAverage(window_size=self.s.PERIOD_MA_2)
        self.df_tick["MA2"] = [ma2.update(p) for p in self.df_tick["Price"]]

        # 乖離度 (MA1 - MA2) / MA2
        self.df_tick["DiffMA"] = (self.df_tick["MA1"] - self.df_tick["MA2"]) / self.df_tick["MA2"] * 100.

        # 出来高加重平均価格
        vwap = VWAP()
        self.df_tick["VWAP"] = [vwap.update(p, v) for p, v in zip(self.df_tick["Price"], self.df_tick["Volume"])]

        # 乖離度 (MA1 - VWAP) / VWAP
        self.df_tick["DiffVWAP"] = (self.df_tick["MA1"] - self.df_tick["VWAP"]) / self.df_tick["VWAP"] * 100.

    def _prep_rewards(self):
        colname1 = self.s.COL_CROSS_MA_GOLDEN
        colname2 = self.s.COL_CROSS_MA_DEAD
        n: int = len(self.df_tick)
        w: int = 120
        p: float = 5.0
        # クロス・ポイント
        diff_ma_pre: float | None = None
        for r in range(n):
            diff_ma = self.df_tick.at[r, "DiffMA"]
            if diff_ma_pre is None:
                self.df_tick.at[r, colname1] = 0.0
            elif diff_ma_pre < 0 <= diff_ma:
                self.df_tick.at[r, colname1] = p
            else:
                self.df_tick.at[r, colname1] = 0.0

            if diff_ma_pre is None:
                self.df_tick.at[r, colname2] = 0.0
            elif diff_ma <= 0 < diff_ma_pre:
                self.df_tick.at[r, colname2] = p
            else:
                self.df_tick.at[r, colname2] = 0.0

            diff_ma_pre = diff_ma

        # クロス・ポイント前後
        for r in range(n):
            v1 = self.df_tick.at[r, colname1]
            v2 = self.df_tick.at[r, colname2]
            if v1 == p:
                for i in range(1, w):
                    denom = (i * 2) ** 2
                    r_pre = r - i
                    if 0 <= r_pre:
                        self.df_tick.at[r_pre, colname1] += p / denom
                    r_post = r + i
                    if r_post < n - 1:
                        # クロスを超えたら急峻に
                        self.df_tick.at[r_post, colname1] += p / denom / denom
            if v2 == p:
                for i in range(1, w):
                    denom = (i * 2) ** 2
                    r_pre = r - i
                    if 0 <= r_pre:
                        self.df_tick.at[r_pre, colname2] += p / denom
                    r_post = r + i
                    if r_post < n - 1:
                        # クロスを超えたら急峻に
                        self.df_tick.at[r_post, colname2] += p / denom / denom

        # トレーニング用データの保存
        # print("特徴量などを追加したデータを保存しました。")
        # self.df_tick.to_csv("traning_data.csv")

    def action_masks(self) -> np.ndarray:
        """
        行動マスク
        【マスク】
        - ウォーミングアップ期間 → 強制 HOLD
        - ナンピン取引の禁止

        :return: mask
        """
        return self.s.get_masks()

    def close(self) -> None:
        # Cleanup resources (e.g., close windows)
        pass

    def get_data(self) -> None:
        """
        ティックデータから一行抽出
        :param row:
        :return:
        """
        list_name = ["Time", "Price", "MA1", "MA2", "DiffMA", "VWAP", "DiffVWAP"]
        row = self.df_tick.iloc[self.s.row][list_name]
        self.s.set_data(row)

    def get_data_open(self) -> None:
        """
        始値情報の取得
        """
        list_name = ["Time", "Price", "Volume"]
        row = self.df_tick.iloc[0][list_name]
        self.s.set_data_open(row)

    def get_reward_cross_ma_dead(self):
        return self.df_tick.iloc[self.s.row][self.s.COL_CROSS_MA_DEAD]

    def get_reward_cross_ma_golden(self):
        return self.df_tick.iloc[self.s.row][self.s.COL_CROSS_MA_GOLDEN]

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

    def position_open(self, action_type: ActionType) -> float:
        self.s.position = self.posman.openPosition(
            self.CODE, self.s.ts, self.s.price, action_type
        )
        self.s.n_trade += 1  # 取引回数の更新
        self.s.profit_pre = 0.0  # 一つ前の含み益
        # 【報酬・ペナルティ】
        r = 0.0
        r -= self.s.COST_CONTRACT  # 約定コスト
        return r

    def position_close(self, note="") -> float:
        """
        ポジション・クローズ
        :return:
        """
        # ポジション管理
        self.s.position = self.posman.closePosition(
            self.CODE, self.s.ts, self.s.price, note=note
        )
        self.s.n_trade += 1  # 取引回数の更新
        self.s.profit_pre = 0.0  # 一つ前の含み益
        # 【報酬】
        r = 0.0
        r -= self.s.COST_CONTRACT  # 約定コスト
        r += self.s.profit  # 含み損益分そっくり報酬
        self.s.reset_count_negative()
        return r

    def position_close_force(self) -> float:
        """
        ポジション・クローズ（強制）
        :param profit:
        :return:
        """
        return self.position_close(note="強制返済")

    def render(self) -> None:
        # Implement visualization logic based on self.render_mode
        pass

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
                1.0,
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
        self.get_data()
        # 含み損益の取得
        self.s.profit = self.posman.getProfit(self.CODE, self.s.price)
        # 初期報酬
        reward = 0
        # 情報用辞書
        info = {}

        # ====== 建玉管理 ======
        action_type = ActionType(action)
        reward_cross_ma_golden = self.get_reward_cross_ma_golden()
        reward_cross_ma_dead = self.get_reward_cross_ma_dead()
        if action_type == ActionType.BUY:
            if self.s.position == PositionType.NONE:
                # 【買建】建玉がなければ買建
                reward += self.position_open(action_type)
                # ゴールデン・クロス時のエントリに対する報酬
                reward += reward_cross_ma_golden
                # デッド・クロス時のエントリに対するペナルティ
                reward -= reward_cross_ma_dead
            elif self.s.position == PositionType.SHORT:
                # 【返済】売建（ショート）であれば（買って）返済
                reward += self.position_close()
            else:
                raise RuntimeError("Trade rule violation!")
        elif action_type == ActionType.SELL:
            if self.s.position == PositionType.NONE:
                # 【売建】建玉がなければ売建
                reward += self.position_open(action_type)
                # ゴールデン・クロス時のエントリに対するペナルティ
                reward -= reward_cross_ma_golden
                # デッド・クロス時のエントリに対する報酬
                reward += reward_cross_ma_dead
            elif self.s.position == PositionType.LONG:
                # 【返済】買建（ロング）であれば（売って）返済
                reward += self.position_close()
            else:
                raise RuntimeError("Trade rule violation!")
        elif action_type == ActionType.HOLD:
            if self.s.position == PositionType.NONE:
                # クロス・シグナルに応じた僅かなペナルティ
                reward_sum = reward_cross_ma_golden + reward_cross_ma_dead
                denom = 1000.0
                reward -= reward_sum / denom
            else:
                # 【報酬・ペナルティ】
                # 含み益があれば幾分かを報酬に
                reward += self.s.get_reward_unrealized_profit()
        else:
            raise TypeError(f"Unknown ActionType: {action_type}!")

        # ====== 連続含み損評価 ======
        self.s.update_count_negative()
        reward += self.s.get_penalty_negative()
        if self.s.flag_losscut_consecutive:
            # 【ロスカット】
            if self.posman.hasPosition(self.CODE):
                _ = self.position_close_force()
        """
        if self.s.is_losscut():
            # 【ロスカット】
            if self.posman.hasPosition(self.CODE):
                _ = self.position_close_force()
        """

        # ====== エピソード終了判定 ======
        terminated = False  # Task finished (e.g., goal reached)
        truncated = False  # Time limit reached

        if len(self.df_tick) - 1 <= self.s.row:
            # ティックデータの末尾
            if self.posman.hasPosition(self.CODE):
                # 建玉があれば強制返済
                reward += self.position_close_force()

            # 約定回数に応じた報酬
            reward += self.s.get_n_trade_reward()

            truncated = True  # ← ステップ数上限による終了
            info["done_reason"] = "truncated: last_tick"
            # 取引情報（データフレーム）
            info["transaction"] = self.get_transaction_result()
            # 報酬情報（データフレーム）
            info["reward"] = self.s.get_reward()
            print(f"約定回数 : {self.s.n_trade}")

        # 一つ前の含み益の更新
        self.s.update_profit_pre()

        # ====== 観測値（状態） ======
        obs = self.s.get_obs()

        # ステップ（データフレームの行）更新
        self.s.inc_row()

        # ====== モデル報酬の保持（分析用） ======
        self.s.update_dict_reward(reward)
        # ====== テクニカル情報（分析用） ======
        info["technical"] = self.s.get_technicals()

        return obs, reward, terminated, truncated, info
