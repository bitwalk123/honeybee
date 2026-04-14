import glob
import os
import sys
from collections import defaultdict

import pandas as pd

from funcs.tide import get_dt_from_excel
from modules.agent import MyPPOAgent

if __name__ == "__main__":
    code = "9984"

    name_model = "model_test.zip"

    # ログフォルダ
    dir_logs = "./logs/"

    # TensorBoard 用ログ
    tb_logs = "./tb_logs/"

    # 学習用エージェント
    agent = MyPPOAgent(code, name_model, dir_logs, tb_logs)

    # 学習に使用するティックデータ
    home = os.path.expanduser("~")
    path_excel = os.path.join(home, "MyProjects", "kabuto", "collection", "*.xlsx")
    list_excel_all = sorted(glob.glob(path_excel))
    list_excel = list_excel_all[-40:]

    # 推論に渡す Excel リストが確かにリストになっているか確認
    if type(list_excel) is not list:
        print(f"list_excel is not list!")
        sys.exit()
    else:
        # 推論対象ファイル
        print("推論対象ファイル")
        for file_excel in list_excel:
            print(file_excel)

    # 推論
    dict_perf = defaultdict(list)
    for file_excel in list_excel:
        dict_result = agent.infer(file_excel)
        if "transaction" in dict_result:
            df = dict_result["transaction"]
            print(df)
            pnl = df["損益"].sum()
            n_contract = len(df)
            print(f"損益 : {pnl} 円, 約定係数 : {n_contract} 回")

            dict_perf["file"].append(os.path.basename(file_excel))
            dict_perf["code"].append(code)
            dict_perf["pnl"].append(pnl)
            dict_perf["contracts"].append(n_contract)

    df_perf = pd.DataFrame(dict_perf)
    df_perf.index = [get_dt_from_excel(f) for f in dict_perf["file"]]
    print(df_perf)
    df_perf.to_pickle("performance.pkl")