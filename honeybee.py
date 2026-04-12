import glob
import os
from enum import Enum, auto

from modules.agent import MyPPOAgent

if __name__ == "__main__":
    code = "9984"

    name_model = "model_test.zip"

    # ログフォルダ
    dir_logs = "./logs/"

    # TensorBoard 用ログ
    tb_logs = "./tb_logs/"

    # 学習用エージェント
    agent = MyPPOAgent(code, name_model, dir_logs, tb_logs, flag_new=True)

    # 学習に使用するティックデータ
    # list_csv: list[str] = ["20260408_9984.csv", "20260409_9984.csv", "20260410_9984.csv"]
    # list_csv: list[str] = ["20260410_9984.csv"]
    home = os.path.expanduser("~")
    path_excel = os.path.join(home, "MyProjects", "kabuto", "collection", "*.xlsx")
    list_excel_all = sorted(glob.glob(path_excel))
    list_excel = list_excel_all[-6:-1]

    # 学習
    n_episode = 25  # 概ねのエピソード数
    for file_excel in list_excel:
        agent.train(file_excel, n_episode)

    # 推論
    file_excel = list_excel[-1]
    agent.infer(file_excel)
