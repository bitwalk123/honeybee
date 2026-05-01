import pandas as pd

from modules.env_training import TrainingEnv
from structs.app_enum import ActionType, PositionType


class InferenceEnv(TrainingEnv):
    def __init__(self, code: str, df_tick: pd.DataFrame, render_mode=None) -> None:
        super().__init__(code, df_tick, render_mode)

    def _execute_forced_close_if_needed(self) -> bool:
        """
        リスク管理ルールに基づいて建玉の強制決済を判定・実行

        優先順位:
        1. ドローダウン利確
        2. 連続含み損による損切り
        3. 単純ロスカット

        Returns:
            bool: 強制決済を実行した場合True
        """
        # 1. 利確判定
        if self.s.does_take_profit():
            self.position_close_force(note="ドローダウン利確")
            return True

        # 2. 連続含み損評価・判定
        self.s.update_count_negative()
        if self.s.flag_losscut_consecutive:
            self.position_close_force(note="連続含み損")
            return True

        # 3. 含み益→含み損ロスカット判定
        if 10 < self.s.profit_max and self.s.profit < -10:
            self.position_close_force(note="益→損ロスカット")
            return True

        # 4. 単純ロスカット判定
        if self.s.is_losscut():
            self.position_close_force(note="単純ロスカット")
            return True

        return False

    def _execute_model_action(self, action: int) -> None:
        """
        強化学習モデルから受け取ったアクションを実行

        Args:
            action: モデルが出力したアクション値
        """
        action_type = ActionType(action)
        if action_type == ActionType.HOLD:
            return

        if action_type == ActionType.BUY:
            self._handle_buy_action()
        elif action_type == ActionType.SELL:
            self._handle_sell_action()
        else:
            raise TypeError(f"Unknown ActionType: {action_type}!")

    def _handle_buy_action(self) -> None:
        """買いアクションの処理"""
        if self.s.position == PositionType.NONE:
            if self.s.check_valid_entry(ActionType.BUY):  # エントリーの妥当性をチェック
                # ポジションなし → 買建
                self.position_open(ActionType.BUY)
        elif self.s.position == PositionType.SHORT:
            if self.s.check_valid_repayment():  # 返済の妥当性をチェック
                # ショートポジション保有 → 買い返済
                self.position_close()
        else:
            # ロングポジション保有時に買いアクション → ルール違反
            raise RuntimeError("Trade rule violation: Cannot BUY while holding LONG position!")

    def _handle_sell_action(self) -> None:
        """売りアクションの処理"""
        if self.s.position == PositionType.NONE:
            if self.s.check_valid_entry(ActionType.SELL):  # エントリーの妥当性をチェック
                # ポジションなし → 売建
                self.position_open(ActionType.SELL)
        elif self.s.position == PositionType.LONG:
            if self.s.check_valid_repayment():  # 返済の妥当性をチェック
                # ロングポジション保有 → 売り返済
                self.position_close()
        else:
            # ショートポジション保有時に売りアクション → ルール違反
            raise RuntimeError("Trade rule violation: Cannot SELL while holding SHORT position!")

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
        # 情報用辞書
        info = {}

        # ====== 強制決済判定フェーズ ======
        # 建玉がある場合、リスク管理ルールに基づいて強制決済を検討
        forced_close_executed = False

        if self.posman.hasPosition(self.CODE):
            forced_close_executed = self._execute_forced_close_if_needed()

        # ====== 通常アクション実行フェーズ ======
        # 強制決済が実行されなかった場合のみ、モデルのアクションを実行
        if not forced_close_executed:
            self._execute_model_action(action)

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

        # 一つ前の特徴量の更新
        self.s.update_feature_pre()

        # ステップ（データフレームの行）更新
        self.s.inc_row()

        # ====== テクニカル情報（分析用） ======
        info["technical"] = self.s.get_technicals()

        return obs, 0, terminated, truncated, info
