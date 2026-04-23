import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class InfoCallback(BaseCallback):
    def __init__(self, dir_logs: str, verbose=0):
        super().__init__(verbose)
        self.dir_logs = dir_logs
        self.n_episode = 0

    def _on_step(self) -> bool:
        # VecEnv なので複数環境分の info が入る
        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info and "transaction" in info and "reward" in info:
                # 総モデル報酬
                episode_reward = info["episode"]["r"]
                # 売買履歴
                df_transaction: pd.DataFrame = info["transaction"]
                # 損益
                pnl = df_transaction["損益"].sum()
                # 約定回数
                n_transaction = len(df_transaction)
                # 標準出力
                print("取引結果:")
                print(df_transaction)
                print(
                    f"モデル報酬 : {episode_reward},\n"
                    f"損益 : {pnl} 円, 約定係数 : {n_transaction} 回"
                )

                """
                # 売買履歴のデータフレームを CSV 形式で保存
                path_csv_transaction = os.path.join(
                    self.dir_logs,
                    f"transaction_{self.n_episode:06d}.csv"
                )
                df_transaction.to_csv(path_csv_transaction, index=False)

                # 報酬
                df_reward: pd.DataFrame = info["reward"]
                # 報酬のデータフレームを CSV 形式で保存
                path_csv_reward = os.path.join(
                    self.dir_logs,
                    f"reward_{self.n_episode:06d}.csv"
                )
                df_reward.to_csv(path_csv_reward, index=False)
                """

                self.n_episode += 1

                # TensorBoard にスカラーとして記録
                self.logger.record("custom/episode_reward", episode_reward)
                self.logger.record("custom/pnl", pnl)
                self.logger.record("custom/transactions", n_transaction)

        return True
