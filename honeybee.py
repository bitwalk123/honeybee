from modules.agent import MyPPOAgent

if __name__ == "__main__":
    # 学習に使用するティックデータ
    file_csv: str = "20260401_9984.csv"

    # ログフォルダ
    dir_logs = "./logs/"

    # TensorBoard 用ログ
    tb_logs = "./tb_logs/"

    # 概ねのエピソード数
    n_episode = 3
    agent = MyPPOAgent(file_csv, dir_logs, tb_logs, n_episode)

    # 学習
    agent.train()

    # 推論（現在は同じファイルで）
    agent.infer()






