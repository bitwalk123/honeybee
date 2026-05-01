import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from funcs.conv import position_to_onehot
from modules.env_data import EnvData
from modules.posman import PositionManager
from modules.technical import MovingAverage, VWAP, RSI, Momentum, EMA
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
        2. MA2（長周期移動平均）
        3. Momentum（モメンタム）
        4. Profit（含み損益）
        5. ProfitMax（最大含み損益）
        6. n_trade（約定回数）
        7. count_negative（含み損の継続カウンタ）
        8. 約定コスト
        9. dd_ratio（ドローダウン率）
        [cross] - 符号が重要であるため標準化しない (-1, 1)
        1. DiffMA（乖離率 : (MA1 - MA2) / MA2）
        2. DiffVWAP（乖離率 : (MA1 - VWAP) / VWAP）
        3. RSI
        [position] - 標準化不要
        1. SHORT
        2. NONE
        3. LONG
        4. MA Golden Cross
        5. MA Dead Cross
        """
        self.observation_space = spaces.Dict({
            "market": spaces.Box(
                low=np.array([
                    -np.float32('inf'),  # 1. MA1（短周期移動平均）
                    -np.float32('inf'),  # 2. MA2（長周期移動平均）
                    -np.float32('inf'),  # 3. Momentum（モメンタム）
                    -np.float32('inf'),  # 4. Profit（含み損益）
                    -np.float32('inf'),  # 5. ProfitMax（最大含み損益）
                    np.float32(0),  # 6. n_trade（約定回数）
                    np.float32(0),  # 7. count_negative（含み損の継続カウンタ）
                    -np.float32('inf'),  # 8. 約定コスト
                    np.float32(0),  # 9. dd_ratio（ドローダウン率）
                ]),
                high=np.array([
                    np.float32('inf'),  # 1. MA1（短周期移動平均）
                    np.float32('inf'),  # 2. MA2（長周期移動平均）
                    np.float32('inf'),  # 3. Momentum（モメンタム）
                    np.float32('inf'),  # 4. Profit（含み損益）
                    np.float32('inf'),  # 5. ProfitMax（最大含み損益）
                    np.float32(1),  # 6. n_trade（約定回数）
                    np.float32(1),  # 7. count_negative（含み損の継続カウンタ）
                    np.float32(-self.s.COST_CONTRACT),  # 8. 約定コスト
                    np.float32('inf'),  # 9. dd_ratio（ドローダウン率）
                ]),
                shape=(9,),
                dtype=np.float32
            ),
            "cross": spaces.Box(
                low=np.array([
                    np.float32(-5),  # 1. DiffMA（乖離率 : (MA1 - MA2) / MA2）
                    np.float32(-5),  # 2. DiffVWAP（乖離率 : (MA1 - VWAP) / VWAP）
                    np.float32(0),  # 3. RSI
                ]),
                high=np.array([
                    np.float32(5),  # 1. DiffMA（乖離率 : (MA1 - MA2) / MA2）
                    np.float32(5),  # 2. DiffVWAP（乖離率 : (MA1 - VWAP) / VWAP）
                    np.float32(1),  # 3. RSI
                ]),
                shape=(3,),
                dtype=np.float32
            ),
            "position": spaces.MultiBinary(5),  # one-hot
        })

        # デバッグ用観測値
        self.obs = {}

    def _prep_observations(self):
        # 短周期移動平均 MA1
        # ma1 = MovingAverage(window_size=self.s.PERIOD_MA_1)
        ma1 = EMA(window_size=self.s.PERIOD_MA_1)
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

        # RSI
        rsi = RSI(window_size=self.s.PERIOD_RSI)
        self.df_tick["RSI"] = [rsi.update(p) for p in self.df_tick["Price"]]

        # Momentum
        mom = Momentum(window_size=self.s.PERIOD_MOM)
        self.df_tick["Momentum"] = [mom.update(p) for p in self.df_tick["Price"]]

    def _prep_rewards(self):
        colname1 = self.s.COL_CROSS_MA_GOLDEN
        colname2 = self.s.COL_CROSS_MA_DEAD
        for colname in [colname1, colname2]:
            self.df_tick[colname] = 0
        n: int = len(self.df_tick)
        w: int = 120
        p: float = 5.0
        # クロス・ポイント
        diff_ma_pre: float | None = None
        for r in range(n - 1):
            diff_ma = self.df_tick.at[r, "DiffMA"]
            # 報酬付与は次のステップになる
            # --- ゴールデン・クロス ---
            if diff_ma_pre is None:
                self.df_tick.at[r + 1, colname1] = 0.0
            elif diff_ma_pre <= 0 < diff_ma:
                self.df_tick.at[r + 1, colname1] = p
            else:
                self.df_tick.at[r + 1, colname1] = 0.0
            # --- デッド・クロス ---
            if diff_ma_pre is None:
                self.df_tick.at[r + 1, colname2] = 0.0
            elif diff_ma < 0 <= diff_ma_pre:
                self.df_tick.at[r + 1, colname2] = p
            else:
                self.df_tick.at[r + 1, colname2] = 0.0

            diff_ma_pre = diff_ma

        """
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
        """

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
        :return:
        """
        row = self.df_tick.iloc[self.s.row][self.s.list_col_name]
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
        self.s.reset_profit_pre()  # 一つ前の含み益のリセット
        # 【報酬・ペナルティ】
        r = 0.0
        r += self.s.add_contract_cost()  # 約定コスト
        # print("open", datetime.datetime.fromtimestamp(self.s.ts), self.s.count_post_contract, r)
        self.s.reset_count_post_contract()  # 約定後の経過カウンタのリセット
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
        self.s.reset_profit_pre()  # 一つ前の含み益のリセット
        self.s.reset_profit_max()  # 最大含み益のリセット
        # 【報酬】
        r = 0.0
        r += self.s.add_contract_cost()  # 約定コスト
        # print("close", datetime.datetime.fromtimestamp(self.s.ts), self.s.count_post_contract, r, self.s.profit)
        self.s.reset_count_post_contract()  # 約定後の経過カウンタのリセット
        r += self.s.profit  # 含み損益分そっくり報酬
        self.s.reset_count_negative()
        return r

    def position_close_force(self, note="強制返済") -> float:
        """
        ポジション・クローズ（強制）
        :param note:
        :return:
        """
        return self.position_close(note)

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
                1,  # 1. MA1（短周期移動平均）
                1,  # 2. MA2（長周期移動平均）
                0,  # 3. Momentum（モメンタム）
                0,  # 4. Profit（含み損益）
                0,  # 5. ProfitMax（最大含み損益）
                0,  # 6. n_trade（約定回数）
                0,  # 7. count_negative（含み損の継続カウンタ）
                0,  # 8. 約定コスト
                0,  # 9. dd_ratio（ドローダウン率）
            ],
            dtype=np.float32
        )
        cross = np.array([0, 0, 0], dtype=np.float32)
        position = np.concatenate([
            position_to_onehot(self.s.position).astype(np.float32),
            np.array([False, False], dtype=np.float32)
        ])
        obs = {"market": market, "cross": cross, "position": position}

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
        self.s.update_profit_max()  # 最大含み益の更新
        # 初期報酬
        reward = 0
        # 情報用辞書
        info = {}

        # === 連続含み損評価 ===
        self.s.update_count_negative()
        reward += self.s.get_penalty_negative()

        """
        # 強制的な利確・ロスカットすると学習に影響するようだ
        flag_not_action_yet = True  # ポジション変更のアクション済か確認用フラグ
        # 1. 利確判定
        if self.s.does_take_profit():
            # 建玉返済
            reward += self.position_close_force(note="ドローダウン利確")
            flag_not_action_yet = False

        # 2. 連続含み損評価・判定
        if self.s.flag_losscut_consecutive:
            if flag_not_action_yet and self.posman.hasPosition(self.CODE):
                reward += self.position_close_force(note="連続含み損")
                flag_not_action_yet = False
            else:
                reward += self.s.get_penalty_negative()

        # 3. 含み益→含み損ロスカット判定
        if flag_not_action_yet and 5 < self.s.profit_max and self.s.profit < -10:
            reward += self.position_close_force(note="益→損ロスカット")
            flag_not_action_yet = False

        # 4. 単純ロスカット判定
        if flag_not_action_yet and self.s.is_losscut():
            reward += self.position_close_force(note="単純ロスカット")
            flag_not_action_yet = False
        """

        # if flag_not_action_yet:
        # ====== 建玉管理 ======
        reward_dead = self.get_reward_cross_ma_dead()
        reward_golden = self.get_reward_cross_ma_golden()
        action_type = ActionType(action)
        if action_type == ActionType.BUY:
            if self.s.position == PositionType.NONE:
                # 【買建】建玉がなければ買建
                reward += self.position_open(action_type)
                # ゴールデン・クロスで買いなら報酬（報酬分布より）
                reward += reward_golden
                # デッド・クロスで買いならペナルティ（報酬分布より）
                reward -= reward_dead
            elif self.s.position == PositionType.SHORT:
                # 【返済】売建（ショート）であれば（買って）返済
                reward += self.position_close()
            else:
                raise RuntimeError("Trade rule violation!")
        elif action_type == ActionType.SELL:
            if self.s.position == PositionType.NONE:
                # 【売建】建玉がなければ売建
                reward += self.position_open(action_type)
                # ゴールデン・クロスで売りならペナルティ（報酬分布より）
                reward -= reward_golden
                # デッド・クロスで売りなら報酬（報酬分布より）
                reward += reward_dead
            elif self.s.position == PositionType.LONG:
                # 【返済】買建（ロング）であれば（売って）返済
                reward += self.position_close()
            else:
                raise RuntimeError("Trade rule violation!")
        elif action_type == ActionType.HOLD:
            if self.s.position == PositionType.NONE:
                # 何もしない時は微小の正負のノイズを報酬に加える
                reward += (np.random.rand() - 0.5) / 1_000_000.0
            else:
                # 【報酬・ペナルティ】
                # 含み益があれば幾分かを報酬に
                reward += self.s.get_reward_unrealized_profit()
        else:
            raise TypeError(f"Unknown ActionType: {action_type}!")

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

        # ====== 観測値（状態） ======
        obs = self.s.get_obs()

        # 一つ前の含み益の更新
        self.s.update_profit_pre()

        # 一つ前の特徴量の更新
        self.s.update_feature_pre()

        # ステップ（データフレームの行）更新
        self.s.inc_row()

        # ====== モデル報酬の保持（分析用） ======
        self.s.update_dict_reward(reward)
        # ====== テクニカル情報（分析用） ======
        info["technical"] = self.s.get_technicals()

        return obs, reward, terminated, truncated, info
