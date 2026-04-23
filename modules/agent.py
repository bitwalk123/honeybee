import os.path
import random
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from funcs.excel import get_excel_sheet
from modules.env_inference import InferenceEnv
from modules.env_training import TrainingEnv
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

    def make_env_inference(self):
        # 1. 環境クラス継承の推論用環境クラスのインスタンス
        env_gym = InferenceEnv(self.code, self.df)
        # 2. Monitor Wrapper
        env_mon = Monitor(env_gym, self.dir_logs)

        return env_mon

    def make_env_training(self):
        # 1. Gymnasium 継承の環境クラスのインスタンス
        env_gym = TrainingEnv(self.code, self.df)
        # 2. Monitor Wrapper
        env_mon = Monitor(env_gym, self.dir_logs)

        return env_mon

    def train(self, list_excel: list):
        """
        学習（訓練）
        :return:
        """
        # 1 本の長いファイル・リストをシャッフル
        random.shuffle(list_excel)
        print("学習対象ファイル")
        for file_excel in list_excel:
            print(file_excel)
        print(f"{len(list_excel)} files")

        model = None
        env_train = None
        for n_episode, file_excel in enumerate(list_excel):
            # 指定銘柄コードのティックデータのデータフレームを取得
            self.df = get_excel_sheet(file_excel, self.code)
            # unit_episode = len(self.df)
            # 学習用ステップ数の設定
            timesteps = len(self.df)

            # ====== 環境 ======
            # 3. DummyVecEnv Wrapper
            env_dummy = DummyVecEnv([self.make_env_training])

            # 4. VecNormalize Wrapper
            if env_train is None:
                if os.path.exists(self.path_normalize):
                    env_train = VecNormalize.load(self.path_normalize, env_dummy)
                    env_train.training = True
                    env_train.norm_reward = True
                    env_train.norm_obs = True
                else:
                    env_train = VecNormalize(
                        env_dummy,
                        norm_obs=True,
                        norm_reward=True,
                        norm_obs_keys=["market", "counter"]
                    )
            else:
                # 2 回目以降は環境だけ差し替える
                # env_train.set_venv(env_dummy)
                # 2回目以降：新しい VecNormalize を作り直す
                new_env = VecNormalize(
                    env_dummy,
                    norm_obs=True,
                    norm_reward=True,
                    norm_obs_keys=["market", "counter"]
                )
                new_env.obs_rms = env_train.obs_rms
                new_env.ret_rms = env_train.ret_rms
                new_env.clip_obs = env_train.clip_obs
                new_env.clip_reward = env_train.clip_reward
                new_env.training = True
                env_train = new_env

            if model is None:
                # ====== モデル生成 ======
                model = MaskablePPO(
                    "MultiInputPolicy",
                    env=env_train,
                    verbose=1,
                    tensorboard_log=self.tb_logs,
                )
                print(f"new model is created.")
            else:
                # ====== 環境を更新 ======
                model.set_env(env_train)

            # ====== 学習実施 ======
            print(f"Training in episode {n_episode}")
            callback = InfoCallback(dir_logs=self.dir_logs)
            try:
                model.learn(
                    total_timesteps=timesteps,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
            except ValueError as e:
                import traceback
                traceback.print_exc()
                print("learn failed:", e)

                # model から VecEnv を取得（VecNormalize / DummyVecEnv が返る）
                venv = model.get_env()

                # 1) VecEnv 経由で安全に呼ぶ（推奨）
                # indices=0 で 0 番目の環境だけ呼ぶ
                try:
                    obs_list = venv.env_method('get_obs', indices=0)
                    print("env_method get_obs:", obs_list)
                except Exception as ex:
                    print("env_method failed:", ex)

        # VecNormalize の内部状態を保存
        env_train.save(self.path_normalize)
        # print(f"VecNormalize is saved to {self.path_normalize}.")

        # モデルの保存
        model.save(self.path_model)
        print(f"model is saved to {self.path_model}.")

        # env_train.close()

    def infer(self, file_excel: str) -> tuple:
        # 指定銘柄コードのティックデータのデータフレームを取得
        self.df = get_excel_sheet(file_excel, self.code)

        # ====== 学習後の推論用環境の準備 ======
        # 2. DummyVecEnv Wrapper
        env_dummy = DummyVecEnv([self.make_env_inference])

        # 3. VecNormalize Wrapper
        if os.path.exists(self.path_normalize):
            env_infer = VecNormalize.load(
                self.path_normalize,
                env_dummy
            )  # 学習情報を読み込む
            env_infer.training = False
            env_infer.norm_reward = False  # 推論時は報酬正規化を無効化
        else:
            print(f"{self.path_normalize} does not exist!")
            return {}, {}

        if os.path.exists(self.path_model):
            model = MaskablePPO.load(
                self.path_model,
                env=env_infer,
                verbose=1,
            )
            print(f"model is loaded from {self.path_model}.")
        else:
            print(f"{self.path_model} does not exist!")
            return {}, {}

        # 特定環境を指定するインデックス
        idx = 0  # 環境は 1 つのみなので、インデックスは常に 0

        # 環境のリセット
        obs = env_infer.reset()
        # assert env_inf.observation_space.contains(obs), "observation_space mismatch"
        # print(f"Initial observation:\n{obs}")
        episode_over = False
        total_reward = 0

        # ====== 推論実施 ======
        print("Begin inference...")
        dict_result = dict()
        dict_technical = defaultdict(list)

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
            if "technical" in info[idx]:
                d = info[idx]["technical"]
                for key in d.keys():
                    dict_technical[key].append(d[key])

        else:
            dict_info = info[idx]
            # 取引結果を出力
            if "transaction" in dict_info:
                dict_result["transaction"] = dict_info["transaction"]

        # 環境の終了処理
        env_infer.close()
        return dict_result, dict_technical
