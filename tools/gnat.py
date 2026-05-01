import datetime
import os

import pandas as pd
from matplotlib import (
    font_manager as fm,
    pyplot as plt,
)

from funcs.plot import (
    plot_diff_ma,
    plot_diff_vwap,
    plot_main,
    plot_profit,
)
from funcs.tide import get_tse_x_range

from modules.agent_algo import AlgoAgent


class Gnat:
    def __init__(self):
        # 銘柄コード
        self.code = "9984"
        # エージェントのインスタンス生成
        self.agent = AlgoAgent(self.code)

        # テクニカルデータを格納する辞書
        self.dict_technical = {}
        self.dict_result = {}
        self.list_cols = ['注文番号', '銘柄コード', '売買', '約定単価', '約定数量', '損益', '備考']

    def run(self, file_excel: str):
        # ====== 推論 ======
        # テクニカルデータを格納する辞書
        self.dict_result, self.dict_technical = self.agent.infer(file_excel)
        self.show_transaction(file_excel)

    def plot(self):
        df_trans = self.dict_result["transaction"]
        df_trans.index = [pd.to_datetime(t) for t in df_trans["注文日時"]]
        df_trans.index.name = "注文日時"
        df_trans = df_trans[self.list_cols]
        pnl = df_trans["損益"].sum()
        n_trade = len(df_trans)
        xlabel = f"実現損益 : {pnl} 円/株, 約定回数 : {n_trade} 回"

        df = pd.DataFrame(self.dict_technical)
        df.index = [datetime.datetime.fromtimestamp(ts) for ts in df["ts"]]
        # print(df)

        # プロットの x軸の範囲を算出（左右10分のマージン）
        dt_date, dt_left, dt_right = get_tse_x_range(df)

        # Matplotlib の共通設定
        FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
        fm.fontManager.addfont(FONT_PATH)

        # FontPropertiesオブジェクト生成（名前の取得のため）
        font_prop = fm.FontProperties(fname=FONT_PATH)
        font_prop.get_name()

        plt.rcParams["font.family"] = font_prop.get_name()

        fig = plt.figure(figsize=(6.8, 6))
        ax = dict()
        n = 4
        gs = fig.add_gridspec(
            n, 1,
            wspace=0.0, hspace=0.0,
            height_ratios=[2 if i == 0 else 1 for i in range(n)]
        )
        for i, axis in enumerate(gs.subplots(sharex="col")):
            ax[i] = axis
            ax[i].grid()
            for t in df_trans.index:
                ax[i].axvline(x=t, color="red", linewidth=0.25, zorder=100)

        # 株価
        i = 0
        title = f"{dt_date} : {self.code} ─ 疑似モデルのパフォーマンス"
        ax[i].set_xlim(dt_left, dt_right)
        plot_main(ax[i], df, title)

        # MA乖離率
        i += 1
        plot_diff_ma(ax[i], df)

        # VWAP乖離率
        i += 1
        plot_diff_vwap(ax[i], df)

        # 含み損益
        i += 1
        plot_profit(ax[i], df)

        ax[i].set_xlabel(xlabel)

        plt.tight_layout()
        output = "technical.png"
        plt.savefig(output)
        plt.show()

    def show_transaction(self, f: str):
        df = self.dict_result["transaction"]
        pnl = df["損益"].sum()
        n_contract = len(df)
        # 取引結果を標準出力
        print(df)
        print(f"{os.path.basename(f)}, 損益 : {pnl} 円, 約定回数 : {n_contract} 回")
