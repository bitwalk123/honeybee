# 疑似モデルを利用した推論（全ファイル × DOE条件）
import glob
import os
import sys

import pandas as pd

from tools.gnat import Gnat

if __name__ == "__main__":
    name_doe = "doe-002"

    dict_setting = {}
    df_doe = pd.read_csv(os.path.join("doe", name_doe, "doe.csv"))  # DOE条件のCSVファイルを読み込み
    csv_result = os.path.join("doe", name_doe, "result.csv")  # 結果用
    df_result = pd.DataFrame()

    # 学習に使用するティックデータ
    home = os.path.expanduser("~")
    path_excel = os.path.join(home, "MyProjects", "kabuto", "collection", "*.xlsx")
    list_file_excel = sorted(glob.glob(path_excel))

    for file_excel in list_file_excel:
        for r in range(len(df_doe)):
            row = df_doe.iloc[r]
            r2 = len(df_result)
            for colname in df_doe.columns:
                dict_setting[colname] = row[colname]
                df_result.loc[r2, colname] = row[colname]
            print(dict_setting)

            # 毎回、インスタンスを作成
            obj = Gnat(dict_setting)

            print("推論対象ファイル")
            print(file_excel)
            dict_result, dict_technical = obj.run(file_excel)
            df_transaction = dict_result["transaction"]
            df_result.loc[r2, "file"] = os.path.basename(file_excel)
            df_result.loc[r2, "pnl"] = df_transaction["損益"].sum()
            df_result.loc[r2, "contracts"] = len(df_transaction)

    print(df_result)
    df_result.to_csv(csv_result, index=False)
