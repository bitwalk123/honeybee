import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kalman_filter(data, process_variance, measurement_variance):
    """
    1次元カルマンフィルタ
    data: ティックデータのリスト
    process_variance (Q): プロセス誤差（価格がどれくらい変動しうるか。小さいほど滑らかになる）
    measurement_variance (R): 観測誤差（ティックのノイズがどれくらい大きいか）
    """
    n_iter = len(data)
    sz = (n_iter,)

    # 配列の初期化
    estimates = np.zeros(sz)  # 推定された「真の価格」
    error_est = np.zeros(sz)  # 推定誤差の不確かさ

    # 初期値の設定
    estimates[0] = data[0]
    error_est[0] = 1.0  # 初期の不確かさ（適当な値でOK）

    for k in range(1, n_iter):
        # 1. 予測ステップ
        prediction = estimates[k - 1]
        error_prediction = error_est[k - 1] + process_variance

        # 2. 更新ステップ（カルマンゲインの計算）
        kalman_gain = error_prediction / (error_prediction + measurement_variance)

        # 実測値（ティックデータ）を取り入れて推定値を更新
        estimates[k] = prediction + kalman_gain * (data[k] - prediction)

        # 推定誤差の更新
        error_est[k] = (1 - kalman_gain) * error_prediction

    return estimates


# --- サンプルデータの生成 (2秒間隔を想定) ---
np.random.seed(42)
n = 100
true_price = 100 + np.cumsum(np.random.normal(0, 0.2, n))  # 真のトレンド
tick_data = true_price + np.random.normal(0, 0.5, n)  # ノイズの乗ったティック

# --- カルマンフィルタの適用 ---
# process_varianceを小さくすると「より滑らか」に、大きくすると「より機敏」になります
kf_estimates = kalman_filter(tick_data, process_variance=0.01, measurement_variance=0.25)

# --- プロット ---
plt.figure(figsize=(12, 6))
plt.plot(tick_data, label='Raw Tick Data', color='silver', marker='o', markersize=4, alpha=0.6)
plt.plot(true_price, label='True Path (Hidden)', color='black', linestyle='--')
plt.plot(kf_estimates, label='Kalman Filter Estimate', color='green', lw=2)

plt.title('Stock Price Smoothing with Kalman Filter')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
