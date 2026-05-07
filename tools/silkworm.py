import pandas as pd


class SilkWorm:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        print(df)
