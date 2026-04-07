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
    agent = MyPPOAgent(dir_logs, tb_logs)

    # 学習
    agent.train(file_csv)

    # 推論（現在は同じファイルで）
    agent.infer(file_csv)






