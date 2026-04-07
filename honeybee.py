from modules.agent import MyPPOAgent

if __name__ == "__main__":
    # 使用するティックデータ
    file_csv: str = "20260401_9984.csv"
    # ログフォルダ
    dir_logs = "./logs/"
    # TensorBoard 用ログ
    tb_logs = "./tb_logs/"

    # エピソード数
    n_episode = 3
    agent = MyPPOAgent(file_csv, dir_logs, tb_logs, n_episode)
    agent.train()






