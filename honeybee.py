import glob
import os
import sys

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
    home = os.path.expanduser("~")
    path_excel = os.path.join(home, "MyProjects", "kabuto", "collection", "*.xlsx")
    list_excel_all = sorted(glob.glob(path_excel))
    list_excel = list_excel_all[-20:]

    # 学習に渡す Excel リストが確かにリストになっているか確認
    if type(list_excel) is not list:
        print(f"list_excel is not list!")
        sys.exit()
    else:
        # 学習対象ファイル
        print("学習対象ファイル")
        for file_excel in list_excel:
            print(file_excel)

    """
    file_excel = list_excel[0]
    agent.train(file_excel, 1)
    sys.exit()
    """

    # 学習
    for i, file_excel in enumerate(list_excel):
        # 概ねのエピソード数
        if i > 1:
            n_episode = 50
        else:
            n_episode = 100
        agent.train(file_excel, n_episode)

    # 推論（確認用）
    file_excel = list_excel[-1]
    agent.infer(file_excel)
