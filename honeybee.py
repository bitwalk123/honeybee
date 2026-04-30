"""
全ティックデータで学習
ファイルリストをシャッフルして学習を実施
"""
import glob
import os

from modules.agent import PPOAgent

if __name__ == "__main__":
    code = "9984"

    name_model = "model_test.zip"

    # ログフォルダ
    dir_logs = "./logs/"

    # TensorBoard 用ログ
    tb_logs = "./tb_logs/"

    # 学習用エージェント
    agent = PPOAgent(code, name_model, dir_logs, tb_logs, flag_new=True)

    # 学習に使用するティックデータ
    home = os.path.expanduser("~")
    path_excel = os.path.join(home, "MyProjects", "kabuto", "collection", "*.xlsx")
    list_excel = sorted(glob.glob(path_excel))
    # list_excel = list_excel_all

    # 1 日あたりのエピソード
    episodes_per_day = 20

    # ティックデータ数 × episodes_per_day エピソード分のリストを作る
    list_excel_episode = list_excel * episodes_per_day

    # 学習
    agent.train(list_excel_episode)

    # 推論（確認用）
    # file_excel = list_excel[-1]
    # agent.infer(file_excel)
