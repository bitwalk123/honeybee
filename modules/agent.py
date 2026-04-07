import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import TrainingEnv
from funcs.io import get_sample_data, prep_dir_logs_monitor, update_new_dir
from funcs.plot import learning_curve
from modules.agent_auxiliary import InfoCallback


class MyPPOAgent:
    def __init__(self, file_csv: str, dir_logs: str, tb_logs: str, n_episode: int) -> None:
        self.file_csv = file_csv
        self.dir_logs = dir_logs
        self.tb_logs = tb_logs

        # 銘柄コードとティックデータのデータフレームを取得
        self.code, self.df = get_sample_data(file_csv)
        unit_episode = len(self.df)
        # 学習用ステップ数の設定
        self.timesteps = unit_episode * n_episode

        # Monitor 用ログの準備
        self.file_log = prep_dir_logs_monitor(self.dir_logs)
        # TensorBoard 用ログの準備
        update_new_dir(self.tb_logs)

        self.model: MaskablePPO | None = None
        # VecNormalizeの内部状態の保存用
        self.file_pkl = "vecnormalize.pkl"

    def make_env(self):
        # 1. Gymnasium 継承の環境クラスのインスタンス
        env_gym = TrainingEnv(self.code, self.df)
        # 2. Monitor Wrapper
        env_mon = Monitor(env_gym, self.dir_logs)

        return env_mon

    def train(self):
        """
        学習（訓練）
        :return:
        """

        # ====== 環境 ======
        # 3. DummyVecEnv Wrapper
        env_dummy = DummyVecEnv([self.make_env])

        # 4. VecNormalize Wrapper
        env_train = VecNormalize(
            env_dummy,
            norm_obs=True,
            norm_reward=True,
            norm_obs_keys=["market"]
        )

        # ====== モデル生成 ======
        self.model = MaskablePPO(
            "MultiInputPolicy",
            env_train,
            verbose=1,
            tensorboard_log=self.tb_logs,
        )

        # ====== 学習実施 ======
        print("Begin training...")
        callback = InfoCallback(dir_logs=self.dir_logs)
        self.model.learn(
            total_timesteps=self.timesteps,
            callback=callback,
        )
        # 推論時に利用できるように VecNormalize の内部状態を保存
        env_train.save(self.file_pkl)

        # ====== 報酬トレンド/学習曲線 ======
        # 学習ログを読込（最初の行の読み込みを除外）
        df_reward = pd.read_csv(self.file_log, skiprows=[0])
        learning_curve(df_reward, self.file_csv)

    def infer(self):
        if self.model is None:
            # モデルが空でないかチェック
            print("no trained model available!")
            return

        # ====== 学習後の推論用環境の準備 ======
        # 3. DummyVecEnv Wrapper
        env_dummy = DummyVecEnv([self.make_env])

        # 4. VecNormalize Wrapper
        env_infer = VecNormalize.load(self.file_pkl, env_dummy)  # 学習情報を読み込む
        env_infer.training = False
        env_infer.norm_reward = False  # 推論時は報酬正規化を無効化

        # 特定環境を指定するインデックス
        idx = 0  # 環境は 1 つのみなので、インデックスは常に 0

        # 環境のリセット
        obs = env_infer.reset()
        # assert env_inf.observation_space.contains(obs), "observation_space mismatch"
        print(f"Initial observation:\n{obs}")
        episode_over = False
        total_reward = 0

        # ====== 推論実施 ======
        print("Begin inference...")
        info = []
        while not episode_over:
            # VecEnv では action_masks を env_method で取得する
            raw_mask = env_infer.env_method("action_masks")[idx]  # 1D mask
            action_masks = np.array([raw_mask], dtype=np.bool_)  # バッチ次元を付与
            # マスク情報付きで推論
            action, _states = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            # 環境でステップ処理
            action = np.array([action])  # VecEnv では複数環境分の配列
            obs, reward, done, info = env_infer.step(action)
            total_reward += reward[idx]
            episode_over = done[idx]
        else:
            dict_info = info[idx]
            # 取引結果を出力
            if "transaction" in dict_info:
                df = dict_info["transaction"]
                print(df)
                print(
                    f"モデル報酬 : {total_reward},\n"
                    f"損益 : {df['損益'].sum()} 円, 約定係数 : {len(df)} 回"
                )

        # 環境の終了処理
        env_infer.close()
