import os
from enum import Enum, auto

from modules.agent import MyPPOAgent

class AgentActionType(Enum):
    TRAIN = auto()
    INFER = auto()
    BOTH = auto()

if __name__ == "__main__":
    agent_action_type = AgentActionType.INFER

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

    if agent_action_type == AgentActionType.TRAIN:
        # 学習
        n_episode = 10  # 概ねのエピソード数
        for file_csv in list_csv:
            agent.train(path_model, file_csv, n_episode)

    # 推論
    if agent_action_type == AgentActionType.INFER:
        file_csv = list_csv[-1]
        agent.infer(path_model, file_csv)
