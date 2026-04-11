import os

from modules.agent import MyPPOAgent

if __name__ == "__main__":
    path_model = os.path.join("models", "model_test.zip")

    # 学習に使用するティックデータ
    list_csv: list[str] = ["20260408_9984.csv", "20260409_9984.csv", "20260410_9984.csv"]
    # list_csv: list[str] = ["20260410_9984.csv"]

    # ログフォルダ
    dir_logs = "./logs/"

    # TensorBoard 用ログ
    tb_logs = "./tb_logs/"

    # 学習用エージェント
    agent = MyPPOAgent(dir_logs, tb_logs)

    # 学習
    n_episode = 100  # 概ねのエピソード数
    for file_csv in list_csv:
        agent.train(path_model, file_csv, n_episode)

    # 推論（現在は同じファイルで）
    file_csv = list_csv[-1]
    agent.infer(path_model, file_csv)
