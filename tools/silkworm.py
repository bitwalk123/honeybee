import os

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
    ticker as ticker,
)


class SilkWorm:
    def __init__(self, name_doe: str, df: pd.DataFrame):
        self.name_doe = name_doe
        self.df = df

        self.n_doe = len(self.df)
        print(f"DOE 条件数 : {self.n_doe}")

        list_col = df.columns.tolist()
        # DOE 因子（大文字のみ）の抽出
        list_factor = [c for c in list_col if c.isupper()]
        # 応答特性
        list_response = [c for c in list_col if c.islower()]
        # DOE 因子のうち、条件が複数あるものを抽出
        list_factor_doe = list()
        for factor in list_factor:
            if len(self.df[factor].unique()) > 1:
                list_factor_doe.append(factor)
        self.list_factor_doe = sorted(list_factor_doe)
        print("実験因子", self.list_factor_doe)
        print("応答特性", list_response)

    def mulreg(self):
        # 説明変数(X)と目的変数(y)の準備
        X = self.df[self.list_factor_doe]
        y = self.df["pnl"]

        # 定数項（切片）の追加
        X = sm.add_constant(X)

        # モデルの構築と適合
        model = sm.OLS(y, X)
        results = model.fit()

        print(results.summary())

    def main_effect(self):
        # Matplotlib の共通設定
        FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
        fm.fontManager.addfont(FONT_PATH)

        # FontPropertiesオブジェクト生成（名前の取得のため）
        font_prop = fm.FontProperties(fname=FONT_PATH)
        font_prop.get_name()

        plt.rcParams["font.family"] = font_prop.get_name()

        for factor in self.list_factor_doe:
            # フィギュアと軸の準備
            fig, ax = plt.subplots(figsize=(3, 2))
            # 散布図を描画
            sns.scatterplot(data=self.df, x=factor, y="pnl", ax=ax)

            positions = sorted(list(self.df[factor].unique()))
            labels = [str(int(x)) for x in positions]
            plt.xticks(positions, labels)

            ax.plot(
                positions,
                [self.df[self.df[factor] == x]["pnl"].mean() for x in positions]
            )

            ax.grid(axis="y")
            plt.tight_layout()

            name_img = os.path.join("doe", self.name_doe, f"main_effect_{factor}.png")
            print(name_img)
            plt.savefig(name_img)
            plt.show()
