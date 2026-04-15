import datetime
import glob
import os
import sys
from collections import defaultdict

import pandas as pd

from funcs.plot import plot_performance
from funcs.tide import get_dt_from_excel
from modules.agent import MyPPOAgent


def get_transaction(f:str, d: dict) -> None:
    df = dict_result["transaction"]
    print(df)
    filename = os.path.basename(f)
    pnl = df["損益"].sum()
    n_contract = len(df)
    print(f"{filename}, 損益 : {pnl} 円, 約定係数 : {n_contract} 回")

    d["file"].append(filename)
    d["code"].append(code)
    d["pnl"].append(pnl)
    d["contracts"].append(n_contract)


if __name__ == "__main__":
    # 銘柄コード
    code = "9984"
    # モデル名
    name_model = "model_test.zip"
    # ログフォルダ
    dir_logs = "./logs/"
    # TensorBoard 用ログ
    tb_logs = "./tb_logs/"

    # ====== 学習用エージェント ======
    agent = MyPPOAgent(code, name_model, dir_logs, tb_logs)

    # 学習に使用するティックデータ
    home = os.path.expanduser("~")
    path_excel = os.path.join(home, "MyProjects", "kabuto", "collection", "*.xlsx")
    list_excel_all = sorted(glob.glob(path_excel))
    list_excel = list_excel_all[-20:]

    # 推論に渡すExcelリストが確かにリストになっているか確認
    if type(list_excel) is not list:
        print(f"list_excel is not list!")
        sys.exit()
    else:
        # 推論対象ファイル
        print("推論対象ファイル")
        for file_excel in list_excel:
            print(file_excel)
        print()

    # ====== 推論 ======
    # Excelファイル名、報酬、収益、約定回数 用 辞書
    dict_transaction = defaultdict(list)
    # テクニカルデータを格納する辞書
    dict_technical = {}
    # Excelファイル毎のループ
    for file_excel in list_excel:
        dict_result, dict_technical = agent.infer(file_excel)
        if "transaction" in dict_result:
            get_transaction(file_excel, dict_transaction)

    # 辞書 → データフレーム
    df_transaction = pd.DataFrame(dict_transaction)
    # インデックスは Excelファイル名から割り出した日付
    df_transaction.index = [get_dt_from_excel(f) for f in dict_transaction["file"]]
    df_transaction.to_pickle("transaction.pkl")

    df_technical = pd.DataFrame(dict_technical)
    df_technical.index = [datetime.datetime.fromtimestamp(ts) for ts in df_technical["ts"]]
    df_technical.to_pickle("technical.pkl")
    print(df_technical)

    # ====== プロット ======
    # 念の為、銘柄コードが一つしか存在しないことを確認
    list_code = list(set(df_transaction["code"]))
    if len(list_code) == 1:
        code = list_code[0]
    else:
        raise ValueError(f"len(list_code) is not 1!")
    # トレンド表示
    plot_performance(code, df_transaction)
