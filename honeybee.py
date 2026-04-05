import os

import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from env import TrainingEnv
from funcs.io import get_sample_data
from funcs.plot import learning_curve

if __name__ == "__main__":
    file_csv: str = "20260401_9984.csv"
    code, df = get_sample_data(file_csv)
    timesteps = 1_000_000

    # ログフォルダの準備
    dir_log = "./logs/"
    os.makedirs(dir_log, exist_ok=True)

    # 学習環境の準備
    env0 = TrainingEnv(code, df)
    env = Monitor(env0, dir_log)  # Monitorの利用

    # モデルの準備
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)

    # 環境のリセット
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    episode_over = False
    total_reward = 0
    # 推論の実行
    print("Begin inference...")
    counter = 0
    while not episode_over:
        # ラッピング前の閑居いうインスタンスから行動マスクの取得
        action_masks = env0.action_masks()
        # マスク情報付きで推論
        action, _ = model.predict(obs, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

    # 結果の表示
    print(f"Episode finished!\nTotal reward: {total_reward}")
    df = env0.get_transaction_result()
    print(df)
    print(f"損益 : {df['損益'].sum()} 円, 約定係数 : {len(df)} 回")

    # 環境の終了処理
    env.close()

    # 報酬トレンド/学習曲線
    name_log = "monitor.csv"
    # 最初の行の読み込みを除外
    df_monitor = pd.read_csv(os.path.join(dir_log, name_log), skiprows=[0])
    learning_curve(df_monitor, file_csv)
