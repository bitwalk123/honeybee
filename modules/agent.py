import os.path
import shutil

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from funcs.excel import get_excel_sheet
from modules.env import TrainingEnv
from funcs.io import get_sample_data, prep_dir_logs_monitor, update_new_dir
from funcs.plot import learning_curve
from modules.agent_auxiliary import InfoCallback


class MyPPOAgent:
    def __init__(
            self,
            code: str,
            name_model: str,
            dir_logs: str,
            tb_logs: str,
            flag_new: bool = False
    ) -> None:
        self.code: str = code
        self.name_model = name_model
        self.dir_logs = dir_logs
        self.tb_logs = tb_logs

        self.file_csv: str = ""
        self.df: pd.DataFrame = pd.DataFrame()

        # Monitor 用ログの準備
        # self.file_log = prep_dir_logs_monitor(self.dir_logs)
        # TensorBoard 用ログの準備
        # update_new_dir(self.tb_logs)

        # モデルが格納されるディレクトリ
        dir_model = "models"
        os.makedirs(dir_model, exist_ok=True)

        # モデルのフル・パス
        self.path_model = os.path.join(dir_model, name_model)

        # VecNormalizeの内部状態の保存用
        name_body = os.path.splitext(os.path.basename(name_model))[0]
        self.path_normalize = os.path.join(dir_model, f"{name_body}_vecnormalize.pkl")

        # print(self.path_model, self.path_normalize)
        if flag_new:
            print("deleting existing dir_logs...")
            if os.path.exists(self.dir_logs):
                shutil.rmtree(self.dir_logs)
                print(f"deleted {self.dir_logs}.")

            print("deleting existing tb_logs...")
            if os.path.exists(self.tb_logs):
                shutil.rmtree(self.tb_logs)
                print(f"deleted {self.tb_logs}.")

            print("deleting existing model...")
            if os.path.exists(self.path_model):
                os.remove(self.path_model)
                print(f"deleted {self.path_model}.")
            if os.path.exists(self.path_normalize):
                os.remove(self.path_normalize)
                print(f"deleted {self.path_normalize}.")

    def make_env(self):
        # 1. Gymnasium 継承の環境クラスのインスタンス
        env_gym = TrainingEnv(self.code, self.df)
        # 2. Monitor Wrapper
        env_mon = Monitor(env_gym, self.dir_logs)

        return env_mon

    def train(self, file_excel: str, n_episode: int = 3):
        """
        学習（訓練）
        :return:
        """
        # 指定銘柄コードのティックデータのデータフレームを取得
        self.df = get_excel_sheet(file_excel, self.code)
        unit_episode = len(self.df)
        # 学習用ステップ数の設定
        timesteps = unit_episode * n_episode

        # ====== 環境 ======
        # 3. DummyVecEnv Wrapper
        env_dummy = DummyVecEnv([self.make_env])

        # 4. VecNormalize Wrapper
        if os.path.exists(self.path_normalize):
            env_train = VecNormalize.load(
                self.path_normalize,
                env_dummy,
            )
        else:
            env_train = VecNormalize(
                env_dummy,
                norm_obs=True,
                norm_reward=True,
                norm_obs_keys=["market"]
            )

        if os.path.exists(self.path_model):
            # ====== モデル・ロード ======
            model = MaskablePPO.load(
                self.path_model,
                env=env_train,
                verbose=1,
                tensorboard_log=self.tb_logs,
            )
            print(f"model is loaded from {self.path_model}.")
        else:
            # ====== モデル生成 ======
            model = MaskablePPO(
                "MultiInputPolicy",
                env=env_train,
                verbose=1,
                tensorboard_log=self.tb_logs,
            )
            print(f"new model is created.")

        # ====== 学習実施 ======
        print("Begin training...")
        callback = InfoCallback(dir_logs=self.dir_logs)
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
        )
        # モデルの保存
        model.save(self.path_model)
        print(f"model is saved to {self.path_model}.")
        # 推論時に利用できるように VecNormalize の内部状態を保存
        env_train.save(self.path_normalize)
        print(f"VecNormalize is saved to {self.path_normalize}.")

        # ====== 報酬トレンド/学習曲線 ======
        # 学習ログを読込（最初の行の読み込みを除外）
        # df_reward = pd.read_csv(self.file_log, skiprows=[0])
        # try:
        #    learning_curve(df_reward, self.file_csv)
        # except ValueError as e:
        #    print(e)

    def infer(self, file_excel: str):
        # 指定銘柄コードのティックデータのデータフレームを取得
        self.df = get_excel_sheet(file_excel, self.code)

        # ====== 学習後の推論用環境の準備 ======
        # 2. DummyVecEnv Wrapper
        env_dummy = DummyVecEnv([self.make_env])

        # 3. VecNormalize Wrapper
        env_infer = VecNormalize.load(self.path_normalize, env_dummy)  # 学習情報を読み込む
        env_infer.training = False
        env_infer.norm_reward = False  # 推論時は報酬正規化を無効化

        if os.path.exists(self.path_model):
            model = MaskablePPO.load(
                self.path_model,
                env=env_infer,
                verbose=1,
            )
            print(f"model is loaded from {self.path_model}.")
        else:
            print(f"{self.path_model} does not exist!")
            return

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
            # print(raw_mask)
            action_masks = np.array([raw_mask], dtype=np.bool_)  # バッチ次元を付与
            # マスク情報付きで推論
            action, _states = model.predict(
                obs,
                action_masks=action_masks,
                deterministic=True
            )
            # 環境でステップ処理
            action = np.array([action])  # VecEnv では複数環境分の配列
            obs, reward, done, info = env_infer.step(action)
            # print(obs, reward, done, info)
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
