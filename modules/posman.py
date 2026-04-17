import pandas as pd

from structs.app_enum import ActionType, PositionType


class PositionManager:
    def __init__(self) -> None:
        self.order: int = 0  # 注文番号
        self.unit: int = 1  # 売買単位

        self.dict_price: dict[str, float] = {}
        self.dict_total: dict[str, float] = {}
        self.dict_action: dict[str, ActionType] = {}

        # 取引履歴用辞書
        self.records: dict[str, list] = {
            "注文番号": [],
            "注文日時": [],
            "銘柄コード": [],
            "売買": [],
            "約定単価": [],
            "約定数量": [],
            "損益": [],
            "備考": [],
        }

    def initPosition(self, list_code: list[str]) -> None:
        for code in list_code:
            self.dict_price[code] = 0.0  # 建玉取得時の株価
            self.dict_total[code] = 0.0  # 銘柄毎の収益
            self.dict_action[code] = ActionType.HOLD

    def openPosition(self, code: str, ts: float, price: float, action: ActionType, note: str = "") -> PositionType:
        """
        ポジションをオープン（建玉取得）
        :param code:
        :param ts:
        :param price:
        :param action:
        :param note:
        :return:
        """
        self.dict_price[code] = price
        self.dict_action[code] = action

        # 取引履歴
        self.order += 1
        self.records["注文番号"].append(self.order)
        self.records["注文日時"].append(ts)
        self.records["銘柄コード"].append(code)
        if action == ActionType.BUY:
            self.records["売買"].append("買建")
            position = PositionType.LONG
        elif action == ActionType.SELL:
            self.records["売買"].append("売建")
            position = PositionType.SHORT
        else:
            msg: str = f"Invalid action type {action} for code {code}"
            raise ValueError(msg)
        self.records["約定単価"].append(price)
        self.records["約定数量"].append(self.unit)
        self.records["損益"].append(None)
        self.records["備考"].append(note)

        return position

    def closePosition(self, code: str, ts: float, price: float, note: str = "") -> PositionType:
        """
        ポジションをクローズ（建玉返済）
        :param code:
        :param ts:
        :param price:
        :param note:
        :return:
        """
        action = self.dict_action[code]

        # 損益計算
        if action == ActionType.BUY:
            profit = (price - self.dict_price[code]) * self.unit
            position = PositionType.NONE
        elif action == ActionType.SELL:
            profit = (self.dict_price[code] - price) * self.unit
            position = PositionType.NONE
        else:
            msg: str = f"Invalid action type {action} for code {code}"
            raise ValueError(msg)

        self.dict_total[code] += profit

        # 取引履歴
        self.order += 1
        self.records["注文番号"].append(self.order)
        self.records["注文日時"].append(ts)
        self.records["銘柄コード"].append(code)
        if action == ActionType.BUY:
            self.records["売買"].append("売埋")
        else:  # ActionType.SELL
            self.records["売買"].append("買埋")
        self.records["約定単価"].append(price)
        self.records["約定数量"].append(self.unit)
        self.records["損益"].append(profit)
        self.records["備考"].append(note)

        # 売買状態のリセット
        self.dict_price[code] = 0.0
        self.dict_action[code] = ActionType.HOLD

        return position

    def getProfit(self, code: str, price: float) -> float:
        if price == 0.0:
            return 0.0
        if self.dict_action[code] == ActionType.BUY:
            return (price - self.dict_price[code]) * self.unit
        elif self.dict_action[code] == ActionType.SELL:
            return (self.dict_price[code] - price) * self.unit
        else:
            return 0.0

    def getTotal(self, code: str) -> float:
        return self.dict_total[code]

    def getTransactionResult(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        """
        float 型の UNIX タイムスタンプを UTC として解釈し、
        JST (Asia/Tokyo) に変換した後、
        タイムゾーン情報を取り除いて tz-naive な datetime64[ns] にする
        """
        df["注文日時"] = (
            pd.to_datetime(df["注文日時"], unit="s", utc=True)
            .dt.tz_convert("Asia/Tokyo")
            .dt.tz_localize(None)
        )
        return df

    def hasPosition(self, code:str) -> bool:
        if self.dict_action[code] == ActionType.HOLD:
            return False
        else:
            return True

    def reset(self):
        self.__init__()