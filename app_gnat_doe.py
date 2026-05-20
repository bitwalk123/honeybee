# 疑似モデルを利用した推論（全ファイル × DOE条件）
import glob
import os
import shutil
import sys

import pandas as pd

from tools.gnat import Gnat

if __name__ == "__main__":
    name_doe = "doe-006"

    dict_setting = {}
    try:
        df_doe = pd.read_csv(os.path.join("doe", name_doe, "doe.csv"))  # DOE条件のCSVファイルを読み込み
    except FileNotFoundError:
        print(f"DOE条件のCSVファイルが見つかりません: doe/{name_doe}/doe.csv")
        sys.exit(1)

    # 学習に使用するティックデータ
    home = os.path.expanduser("~")
    dir_excel = os.path.join(home, "MyProjects", "kabuto", "collection")
    path_excel = os.path.join(dir_excel, "*.xlsx")
    list_file_excel = sorted(glob.glob(path_excel))

    # 結果用のファイル
    csv_result = os.path.join("doe", name_doe, "result.csv")  # 結果用
    if os.path.exists(csv_result):
        print(f"結果ファイルが既に存在します: {csv_result}")
        csv_result_bak = os.path.join("doe", name_doe, "result.bak")  # バックアップ用

        df_result_pre = pd.read_csv(csv_result)
        list_file_excel_pre = list(set(df_result_pre["file"]))
        list_file_excel_body = [os.path.basename(x) for x in list_file_excel]
        list_file_excel_new = [x for x in list_file_excel_body if x not in list_file_excel_pre]
        if len(list_file_excel_new) == 0:
            print("新しいファイルがありません。終了します")
            sys.exit(0)
        list_file_excel = sorted([os.path.join(dir_excel, x) for x in list_file_excel_new])

        # ファイルを csv_result から csv_result_bak にコピー
        shutil.copy(csv_result, csv_result_bak)
        print(f"結果ファイル {csv_result} を {csv_result_bak} にコピーしました。")
    else:
        df_result_pre = pd.DataFrame()

    print("集計するファイル")
    for file_excel in list_file_excel:
        print(file_excel)

    df_result = pd.DataFrame()
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

    df_result = pd.concat([df_result_pre, df_result], ignore_index=True)
    print(df_result)
    df_result.to_csv(csv_result, index=False)
