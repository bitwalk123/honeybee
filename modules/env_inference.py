import pandas as pd

from modules.env_training import TrainingEnv
from structs.app_enum import ActionType, PositionType


class InferenceEnv(TrainingEnv):
    def __init__(self, code: str, df_tick: pd.DataFrame, render_mode=None) -> None:
        super().__init__(code, df_tick, render_mode)

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
        # 情報用辞書
        info = {}

        # ====== 建玉管理 ======
        action_type = ActionType(action)
        if action_type == ActionType.BUY:
            if self.s.position == PositionType.NONE:
                # 【買建】建玉がなければ買建
                _ = self.position_open(action_type)
            elif self.s.position == PositionType.SHORT:
                # 【返済】売建（ショート）であれば（買って）返済
                _ = self.position_close()
            else:
                raise RuntimeError("Trade rule violation!")
        elif action_type == ActionType.SELL:
            if self.s.position == PositionType.NONE:
                # 【売建】建玉がなければ売建
                _ = self.position_open(action_type)
            elif self.s.position == PositionType.LONG:
                # 【返済】買建（ロング）であれば（売って）返済
                _ = self.position_close()
            else:
                raise RuntimeError("Trade rule violation!")

        elif action_type == ActionType.HOLD:
            pass
        else:
            raise TypeError(f"Unknown ActionType: {action_type}!")

        # ====== 連続含み損評価 ======
        self.s.update_count_negative()
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
                _ = self.position_close_force()

            truncated = True  # ← ステップ数上限による終了
            info["done_reason"] = "truncated: last_tick"
            # 取引情報（データフレーム）
            info["transaction"] = self.get_transaction_result()
            print(f"約定回数 : {self.s.n_trade}")

        # ====== 観測値（状態） ======
        obs = self.s.get_obs()

        # ステップ（データフレームの行）更新
        self.s.inc_row()

        # ====== テクニカル情報（分析用） ======
        info["technical"] = self.s.get_technicals()

        return obs, 0, terminated, truncated, info
