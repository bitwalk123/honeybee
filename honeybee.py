import os

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import TrainingEnv
from funcs.io import get_sample_data
from funcs.plot import learning_curve

if __name__ == "__main__":
    # 使用するティックデータ
    file_csv: str = "20260401_9984.csv"

    # 銘柄コードとティックデータのデータフレームを取得
    code, df = get_sample_data(file_csv)

    # ログフォルダの準備
    dir_log = "./logs/"
    os.makedirs(dir_log, exist_ok=True)
    file_log = os.path.join(dir_log, "monitor.csv")

    # VecNormalizeの内部状態の保存用
    file_pkl = "vecnormalize.pkl"

    # 学習用ステップ数の設定
    # timesteps = 1_000_000
    timesteps = 30_000


    def make_env():
        # 1. Gymnasium 継承の環境クラスのインスタンス
        env_gym = TrainingEnv(code, df)
        # 2. Monitor Wrapper
        env_mon = Monitor(env_gym, dir_log)
        return env_mon


    # ====== 学習環境の準備 ======

    # 3. DummyVecEnv Wrapper
    env_dummy = DummyVecEnv([make_env])

    # 4. VecNormalize Wrapper
    env_train = VecNormalize(env_dummy, norm_obs=True, norm_reward=True)

    # モデルの準備
    model = MaskablePPO("MlpPolicy", env_train, verbose=1)

    # ====== 学習実施 ======
    print("Begin training...")
    model.learn(total_timesteps=timesteps)

    # 推論時に利用できるように VecNormalize の内部状態を保存
    env_train.save(file_pkl)

    # ====== 報酬トレンド/学習曲線 ======
    # 最初の行の読み込みを除外
    df_reward = pd.read_csv(file_log, skiprows=[0])
    learning_curve(df_reward, file_csv)

    # ====== 推論環境の準備 ======

    env_inf = DummyVecEnv([make_env])
    env_inf = VecNormalize.load(file_pkl, env_inf)
    env_inf.training = False
    env_inf.norm_reward = False  # 推論時は報酬正規化を無効化
    idx = 0  # 環境は 1 つのみなので、インデックスは常に 0

    # 環境のリセット
    obs = env_inf.reset()
    print(f"Initial observation: {obs}")
    episode_over = False
    total_reward = 0

    # ====== 推論実施 ======
    print("Begin inference...")
    info = []
    while not episode_over:
        # VecEnv では action_masks を env_method で取得する
        action_masks = env_inf.env_method("action_masks")[idx]
        # マスク情報付きで推論
        action, _states = model.predict(obs, action_masks=action_masks)
        action = np.array([action]) # VecEnv では複数環境分の配列
        obs, reward, done, info = env_inf.step(action)
        total_reward += reward[idx]
        episode_over = done[idx]
    else:
        dict_info = info[idx]
        # 取引結果を表示
        if "transaction" in dict_info:
            df = dict_info["transaction"]
            print(df)
            print(
                f"モデル報酬 : {total_reward},\n"
                f"損益 : {df['損益'].sum()} 円, 約定係数 : {len(df)} 回"
            )

    # 環境の終了処理
    env_train.close()
