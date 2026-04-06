from stable_baselines3.common.callbacks import BaseCallback


class InfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # VecEnv なので複数環境分の info が入る
        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info and "transaction" in info:
                total_reward = info["episode"]["r"]
                df = info["transaction"]
                pnl = df['損益'].sum()
                n_transaction = len(df)
                print("取引結果:")
                print(df)
                print(
                    f"モデル報酬 : {total_reward},\n"
                    f"損益 : {pnl} 円, 約定係数 : {n_transaction} 回"
                )
                # TensorBoard にスカラーとして記録
                self.logger.record("custom/episode_reward", total_reward)
                self.logger.record("custom/pnl", pnl)
                self.logger.record("custom/transactions", n_transaction)

        return True
