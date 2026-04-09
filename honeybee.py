from modules.agent import MyPPOAgent

if __name__ == "__main__":
    # 学習に使用するティックデータ
    file_csv: str = "20260409_9984.csv"

    # ログフォルダ
    dir_logs = "./logs/"

    # TensorBoard 用ログ
    tb_logs = "./tb_logs/"

    # 学習用エージェント
    agent = MyPPOAgent(dir_logs, tb_logs)

    # 学習
    n_episode = 100  # 概ねのエピソード数
    agent.train(file_csv, n_episode)

    # 推論（現在は同じファイルで）
    agent.infer(file_csv)
