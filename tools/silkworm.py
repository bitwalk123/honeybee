import operator
import os
from functools import reduce
from itertools import product

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

        # 実験数
        self.n_doe = len(self.df)
        print(f"DOE 条件数 : {self.n_doe}")

        # DOE 因子（大文字のみ）の抽出
        list_col = df.columns.tolist()
        list_factor = [c for c in list_col if c.isupper()]

        # 応答特性
        list_response = [c for c in list_col if c.islower()]

        # DOE 因子のうち、条件が複数あるものを抽出
        list_factor_doe = list()
        for factor in list_factor:
            if len(self.df[factor].unique()) > 1:
                # 水準が複数ある因子を抽出
                list_factor_doe.append(factor)

        self.list_factor_doe = sorted(list_factor_doe)
        print("実験因子", self.list_factor_doe)
        print("応答特性", list_response)

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
            labels = [str(int(x)) if x > 1 else x for x in positions]
            plt.xticks(positions, labels)

            ax.plot(
                positions,
                [self.df[self.df[factor] == x]["pnl"].mean() for x in positions]
            )

            ax.set_xlabel(f"{factor}, n={self.n_doe}")
            ax.set_ylabel("PnL")
            ax.grid(axis="y")
            plt.tight_layout()

            name_img = os.path.join("doe", self.name_doe, f"main_effect_{factor}.png")
            print(name_img)
            plt.savefig(name_img)
            plt.show()

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

    def ranking(self):
        rows=[]
        levels = [sorted(list(self.df[factor].unique())) for factor in self.list_factor_doe]
        for combo in product(*levels):
            condition = dict(zip(self.list_factor_doe, combo))
            mask = reduce(operator.and_, (self.df[k] == v for k, v in condition.items()))
            total = self.df[mask]["pnl"].sum()
            row = {**condition, "total": total}
            rows.append(row)

        # 最後に DataFrame 化
        df = pd.DataFrame(rows)

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
            sns.scatterplot(data=df, x=factor, y="total", ax=ax)

            positions = sorted(list(df[factor].unique()))
            labels = [str(int(x)) if x > 1 else x for x in positions]
            plt.xticks(positions, labels)

            ax.plot(
                positions,
                [df[df[factor] == x]["total"].mean() for x in positions]
            )

            ax.set_xlabel(f"{factor}, n={len(df)}")
            ax.set_ylabel("Total PnL")
            ax.grid(axis="y")
            plt.tight_layout()

            name_img = os.path.join("doe", self.name_doe, f"total_effect_{factor}.png")
            print(name_img)
            plt.savefig(name_img)
            plt.show()

        df_ranking = df.sort_values('total', ascending=False, ignore_index=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_ranking)

