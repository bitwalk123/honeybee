import pandas as pd
import statsmodels.api as sm


class SilkWorm:
    def __init__(self, df: pd.DataFrame):
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
